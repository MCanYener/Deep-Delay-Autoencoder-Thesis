import os
import time
import datetime
import torch
import pickle
import numpy as np
import pandas as pd
from torch.optim import Adam
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from SINDyCode.SINDyPreamble import library_size, sindy_library
from SINDyCode.SINDyPreamble import sindy_library as preamble_sindy_library
from AutoencoderCode.Autoencoder import PreSVD_Sindy_Autoencoder, RfeUpdateCallback, SindyCall, TrajectoryWindowDataset


class TrainModel:
    def __init__(self, data, params, device=None, hankel=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = data
        self.params = self.fix_params(params)
        self.savename = self.get_name()
        self.hankel = torch.tensor(hankel, dtype=torch.float32) if hankel is not None else None
        self.model = self.get_model().to(self.device)

        # ---- ODE-net inspection (after model is fully built and moved) ----
        print("\n🧪 DEBUG: ODE Net (inside SINDy)")
        if hasattr(self.model, "sindy") and hasattr(self.model.sindy, "net_model") and self.model.sindy.net_model is not None:
            print(self.model.sindy.net_model)
            for i, layer in enumerate(self.model.sindy.net_model):
                if isinstance(layer, nn.Linear):
                    print(f"  Layer {i}: Linear(in={layer.in_features}, out={layer.out_features})")
                else:
                    print(f"  Layer {i}: {layer.__class__.__name__}")
        else:
            print("  (no ODE net; ode_net=False)")

        # ---- quick shape sanity check ----
        try:
            with torch.no_grad():
                dummy = torch.randn(8, self.model.latent_dim, device=self.device)
                theta = self.model.sindy.theta(dummy)  # (8, library_dim)
                print(f"[DEBUG] dummy z: {dummy.shape}, theta(z): {theta.shape}, coeffs: {self.model.sindy.coefficients.shape}")
                out = theta @ self.model.sindy.coefficients  # (8, latent_dim)
                print(f"[DEBUG] (theta @ coeffs): {out.shape}  # should be (8, {self.model.latent_dim})")
        except Exception as e:
            print(f"[DEBUG] ODE-net/Theta shape check skipped due to error: {e}")

        self.history = defaultdict(list)

    def get_name(self, include_date=True):
        pre = "results"
        post = f"{self.params['model_tag']}"
        if include_date:
            name = f"{pre}_{datetime.datetime.now().strftime('%Y%m%d%H%M')}_{post}"
        else:
            name = f"{pre}_{post}"
        return name

    def fix_params(self, params):
        input_dim = params["input_dim"]
        observable_dim = params['observable_dim']
        if observable_dim == 1:
            widths = params["widths_1d"]
        else:
            widths = params["widths_2d"]

        if "coefficient_threshold" not in params:
            params["coefficient_threshold"] = 1e-2
        params["rfe_threshold"] = params["coefficient_threshold"]

        print("[DEBUG] Final coefficient_threshold:", params["coefficient_threshold"])
        print("[DEBUG] Final rfe_threshold:", params["rfe_threshold"])

        # Only considering Lorenz model
        params["library_dim"] = library_size(
            params["latent_dim"], params["poly_order"], params["include_sine"]
        )

        # Set actual coefficients from data
        params["actual_coefficients"] = self.data.sindy_coefficients

        # ✅ Always compute sparse weighting if not already set
        if params.get("sparse_weighting") is None:
            try:
                _, _, sparse_weights = preamble_sindy_library(
                    self.data.z[:100, :],
                    params["poly_order"],
                    include_sparse_weighting=True,
                    include_names=True,  # <-- add this line
                    include_sine=params.get("include_sine", False),
                    exact_features=params.get("exact_features", False)
                )
                params["sparse_weighting"] = sparse_weights
                print("[DEBUG] Sparse weighting initialized with shape:", sparse_weights.shape)
            except ValueError:
                print("⚠️ Warning: sparse_weights not returned from sindy_library.")

        return params

    #def get_data(self):
        # By default, just use x and dx
        #train_x, test_x = train_test_split(self.data.x, train_size=self.params["train_ratio"], shuffle=False)
        #train_dx, test_dx = train_test_split(self.data.dx, train_size=self.params["train_ratio"], shuffle=False)

        #return (train_x, train_dx), (test_x, test_dx)

    def get_model(self):
        # 🔍 Debug: Check fixed mask and values
        fm = self.params.get("sindy_fixed_mask")
        fv = self.params.get("sindy_fixed_values")

        print("\n[DEBUG] SINDy fixed_mask in params:")
        if fm is None:
            print("❌ sindy_fixed_mask is None")
        else:
            print(np.array(fm).astype(int))  # use int for readability (0/1)

        print("[DEBUG] SINDy fixed_values in params:")
        if fv is None:
            print("❌ sindy_fixed_values is None")
        else:
            print(np.array(fv))
        print("[DEBUG] rfe_threshold at model creation:", self.params.get("rfe_threshold"))
        return PreSVD_Sindy_Autoencoder(self.params, hankel = self.hankel)

    def fit(self):
        #train_data, test_data = self.get_data()
        self.save_params()

        os.makedirs(os.path.join(self.params["data_path"], self.savename), exist_ok=True)

        # === Separate parameter groups ===
        ae_params = [p for name, p in self.model.named_parameters() if p.requires_grad and "sindy.coefficients" not in name]
        sindy_params = [p for name, p in self.model.named_parameters() if p.requires_grad and "sindy.coefficients" in name]

        # === Optimizers ===
        optimizer_ae = torch.optim.Adam(ae_params, lr=self.params["learning_rate"])
        optimizer_sindy = torch.optim.Adam(sindy_params, lr=self.params.get("sindy_learning_rate", self.params["learning_rate"]))

        scheduler_ae = torch.optim.lr_scheduler.LambdaLR(
            optimizer_ae, lr_lambda=lambda epoch: self.params["lr_decay_ae"] ** epoch
        )
        scheduler_sindy = torch.optim.lr_scheduler.LambdaLR(
            optimizer_sindy, lr_lambda=lambda epoch: self.params["lr_decay_sindy"] ** epoch
        )

        # === Training data setup ===
        # === Sequence training data (B, T, D) without crossing boundaries ===
        window_T      = int(self.params.get("window_T", 20))   # pick your horizon
        window_stride = int(self.params.get("window_stride", 1))

        lengths = getattr(self.data, "lengths", None)
        if lengths is None:
            # fallback if lengths not provided (assume equal splits by n_ics)
            n_ics = int(self.params.get("n_ics", 1))
            per = self.data.x.shape[0] // max(n_ics, 1)
            lengths = [per] * n_ics

        # --- quick guard ---
        min_len = min(lengths)
        if window_T > min_len:
            raise ValueError(
                f"window_T={window_T} is larger than the shortest trajectory ({min_len}). "
                f"Lower window_T or generate longer trajectories."
)

        full_seq_ds = TrajectoryWindowDataset(
            X=self.data.x, dX=self.data.dx, lengths=lengths,
            T=window_T, stride=window_stride
        )

        # split by windows, not raw rows
        n_total = len(full_seq_ds)
        n_train = int(self.params["train_ratio"] * n_total)
        train_indices = list(range(n_train))
        val_indices   = list(range(n_train, n_total))

        from torch.utils.data import Subset
        train_dataset = Subset(full_seq_ds, train_indices)
        val_dataset   = Subset(full_seq_ds, val_indices)

        train_loader = DataLoader(
            train_dataset, batch_size=self.params["batch_size"], shuffle=True, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.params["batch_size"], shuffle=False, pin_memory=True
        )

        # quick sanity print
        xb, dxb = next(iter(train_loader))
        print(f"[SEQ DEBUG] batch shapes: x {xb.shape}, dx {dxb.shape}")  # -> (B, T, D)

        if self.params.get("use_sindycall", True):
            self.sindy_caller = SindyCall(
                model=self.model,
                params=self.params,
                threshold=self.params["coefficient_threshold"],
                update_freq=self.params.get("update_freq", 10),
                x=self.data.x,
                t=self.data.t,
                lengths = lengths
            )
        else:
            self.sindy_caller = None

        rfe_callback = RfeUpdateCallback(
            self.model,
            rfe_frequency=self.params.get("rfe_frequency", 10),
            print_frequency=self.params.get("sindy_print_rate", 10),
            rfe_start=self.params.get("rfe_start", 200),
        )

        sindy_trainable = self.params.get("train_sindy", True)
        sindy_train_start = self.params.get("sindy_train_start", 0)

        print("\n🧪 DEBUG: Starting fit()")
        print(f"self.data.x.shape: {self.data.x.shape}")
        print(f"input_dim: {self.params['input_dim']}")
        print(f"observable_dim: {self.params['observable_dim']}")
        print(f"svd_dim: {self.params['svd_dim']}")
        print(f"Expecting encoder input size: {self.model.encoder[0].in_features}")

        # === Training loop ===
        for epoch in range(self.params["max_epochs"]):
            self.model.train()

            # per-epoch aggregators
            total_sum = 0.0
            n_batches = 0
            comp_sums = defaultdict(float)  # sums for each component loss
            comp_counts = defaultdict(int)  # how many batches produced that loss key

            for batch_x, batch_dx in train_loader:

                batch_x = batch_x.to(self.device, non_blocking=True)   # (B,T,D)
                batch_dx = batch_dx.to(self.device, non_blocking=True)
                optimizer_ae.zero_grad()
                optimizer_sindy.zero_grad()

                loss, batch_losses = self.model.get_loss(
                    batch_x, batch_x, batch_dx, batch_x, batch_x, batch_dx, epoch=epoch
                )
                loss.backward()

                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"❌ NaN/Inf in gradient for {name}")
                            raise RuntimeError("Aborting due to NaN or Inf in gradients.")
                        param.grad.data = torch.nan_to_num(param.grad.data, nan=0.0, posinf=1e3, neginf=-1e3)
                        param.grad.data.clamp_(-500.0, 500.0)

                optimizer_ae.step()
                if sindy_trainable and epoch >= sindy_train_start:
                    optimizer_sindy.step()

                # accumulate
                total_sum += float(loss.detach().cpu())
                n_batches += 1
                for k, v in batch_losses.items():
                    comp_sums[k] += float(v.detach().cpu())
                    comp_counts[k] += 1

            # === Validation ===
            val_total = 0.0
            val_batches = 0
            val_comp_sums = defaultdict(float)
            val_comp_counts = defaultdict(int)

            self.model.eval()
            with torch.no_grad():
                for vx, vdx in val_loader:
                    vx  = vx.to(self.device, non_blocking=True)   # (B,T,D)
                    vdx = vdx.to(self.device, non_blocking=True)

                    with torch.enable_grad():
                        vloss, vlosses = self.model.get_loss(vx, vx, vdx, vx, vx, vdx, epoch=epoch)
                        
                    val_total += float(vloss.detach().cpu())
                    val_batches += 1
                    for k, v in vlosses.items():
                        val_comp_sums[k]  += float(v.detach().cpu())
                        val_comp_counts[k] += 1


            # === Logging (epoch-averaged) ===
            all_loss_names = ["rec","sindy_z","sindy_x","integral","x0","l1","l1_group","sindy_x0","sindy_y0"]

            # --- Train ---
            epoch_avg_total = total_sum / max(n_batches, 1)
            self.history["train_total"].append(epoch_avg_total)
            for name in all_loss_names:
                if comp_counts[name] > 0:
                    self.history[f"train_{name}"].append(comp_sums[name] / comp_counts[name])
                else:
                    self.history[f"train_{name}"].append(0.0)

            # --- Validation ---
            self.history["val_total"].append(val_total / max(val_batches, 1))
            for name in all_loss_names:
                if val_comp_counts[name] > 0:
                    self.history[f"val_{name}"].append(val_comp_sums[name] / val_comp_counts[name])
                else:
                    self.history[f"val_{name}"].append(0.0)  # or float("nan")

            scheduler_ae.step()
            scheduler_sindy.step()
            # === Periodic prints ===
            if epoch % self.params.get("sindy_print_rate", 10) == 0:
                ae_lr = scheduler_ae.get_last_lr()[0]
                sindy_lr = scheduler_sindy.get_last_lr()[0]
                print(f"Epoch {epoch}: Loss = {epoch_avg_total:.6f}")
                print(f"AE LR: {ae_lr:.6f} | SINDy LR: {sindy_lr:.6f}")
                print("--- SINDy Coefficient Matrix (Sanity Check) ---")
                print(self.model.sindy.coefficients.detach().cpu().numpy())

                if self.params.get("train_sindy", True):
                    start = self.params.get("sindy_train_start", 0)
                    ramp = self.params.get("sindy_ramp_duration", 500)
                    if epoch >= start:
                        sindy_scale = min((epoch - start) / float(ramp), 1.0)
                    else:
                        sindy_scale = 0.0
                    print(f"SINDy scale factor: {sindy_scale:.3f}")

            if (
                self.sindy_caller
                and self.sindy_caller.last_model is not None
                and self.sindy_caller.last_epoch == epoch
            ):
                print("--- PySINDy Discovered Model ---")
                self.sindy_caller.last_model.print()

            rfe_callback.update_rfe(epoch)

            if self.sindy_caller:
                self.sindy_caller.update_sindy(epoch)

        # === Save results ===
        self.save_results()
        
    def save_params(self):
        os.makedirs(self.params["data_path"], exist_ok=True)
        df = pd.DataFrame([self.params])
        df.to_pickle(os.path.join(self.params["data_path"], self.savename + "_params.pkl"))

    def save_results(self):

        # Create folders
        os.makedirs(self.params["data_path"], exist_ok=True)
        model_dir = os.path.join(self.params["data_path"], self.savename)
        os.makedirs(model_dir, exist_ok=True)

        # Save loss and coefficients
        results_dict = {
            "losses": self.history,
            "sindy_coefficients": self.model.sindy.coefficients.detach().cpu().numpy()
        }
        df = pd.DataFrame([results_dict])
        df.to_pickle(os.path.join(self.params["data_path"], self.savename + "_results.pkl"))
        torch.save(self.model.state_dict(), os.path.join(model_dir, "model.pth"))

        # === Generate Basis Terms Consistently ===
        latent_sample = self.model.encoder(torch.tensor(self.data.x[:100], dtype=torch.float32).to(self.device)).detach().cpu().numpy()

        _, basis = sindy_library(
            latent_sample,
            poly_order=self.params["poly_order"],
            include_sine=self.params.get("include_sine", False),
            include_names=True,
            exact_features=self.params.get("exact_features", False)
        )

        # === Print Final SINDy Equations ===
        coefs = self.model.sindy.coefficients.detach().cpu().numpy()
        library_size, dim = coefs.shape

        if len(basis) != library_size:
            print(f"[WARNING] Length of basis ({len(basis)}) does not match coefficient shape ({library_size})")

        print("\nFinal SINDy Model:")
        for eq in range(dim):
            terms = [f"{coefs[i, eq]:.4f} * {basis[i]}" for i in range(library_size) if abs(coefs[i, eq]) > 1e-5]
            equation = f"dz{eq}/dt = " + " + ".join(terms) if terms else f"dz{eq}/dt = 0"
            print(equation)

    def get_callbacks(self, params, savename):
        callback_list = []
        
        if params["coefficient_threshold"] is not None:
            callback_list.append(RfeUpdateCallback(rfe_frequency=params["threshold_frequency"]))

        if params["use_sindycall"]:
            print("Generating data for SindyCall...")
            params2 = params.copy()
            params2["tend"] = 200
            params2["n_ics"] = 1

            # Copy data for SindyCall
            data2 = self.data.copy()
            data2.run_sim(params2["n_ics"], params2["tend"], params2["dt"])

            print("Done.")
            x = data2.x
            callback_list.append(SindyCall(threshold=params2["sindy_threshold"], update_freq=params2["sindycall_freq"], x=x))

        return callback_list
    
    def simulate_sindy_trajectory(self, z0, timesteps=3200, dt=0.01):
        z0 = torch.tensor(z0, dtype=torch.float32).to(self.device)
        z_traj = [z0]

        def f(z):
            theta_z = self.model.sindy.theta(z.unsqueeze(0))  # shape [1, library_dim]
            dz = torch.matmul(theta_z, self.model.sindy.coefficients).squeeze(0)
            return dz

        z_curr = z0
        for _ in range(timesteps - 1):
            k1 = f(z_curr)
            k2 = f(z_curr + 0.5 * dt * k1)
            k3 = f(z_curr + 0.5 * dt * k2)
            k4 = f(z_curr + dt * k3)
            z_next = z_curr + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            z_traj.append(z_next)
            z_curr = z_next

        return torch.stack(z_traj, dim=0)