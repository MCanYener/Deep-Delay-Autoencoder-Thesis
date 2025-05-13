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

from SINDyCode.SINDyPreamble import library_size, sindy_library
from SINDyCode.SINDyPreamble import sindy_library as preamble_sindy_library
from AutoencoderCode.Autoencoder import PreSVD_Sindy_Autoencoder, RfeUpdateCallback, SindyCall


class TrainModel:
    def __init__(self, data, params, device=None, hankel=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = data
        self.params = self.fix_params(params)
        self.savename = self.get_name()
        self.hankel = torch.tensor(hankel, dtype=torch.float32) if hankel is not None else None
        self.model = self.get_model().to(self.device)
        self.history = []

    def get_name(self, include_date=True):
        pre = "results"
        post = f"{self.params['model']}_{self.params['case']}"
        if include_date:
            name = f"{pre}_{datetime.datetime.now().strftime('%Y%m%d%H%M')}_{post}"
        else:
            name = f"{pre}_{post}"
        return name

    def fix_params(self, params):
        input_dim = params["input_dim"]
        params["widths"] = [int(i * input_dim) for i in params["widths_ratios"]]

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

    def get_data(self):
        # By default, just use x and dx
        train_x, test_x = train_test_split(self.data.x, train_size=self.params["train_ratio"], shuffle=False)
        train_dx, test_dx = train_test_split(self.data.dx, train_size=self.params["train_ratio"], shuffle=False)

        return (train_x, train_dx), (test_x, test_dx)

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
        train_data, test_data = self.get_data()
        self.save_params()

        os.makedirs(os.path.join(self.params["data_path"], self.savename), exist_ok=True)

        optimizer = Adam(self.model.parameters(), lr=self.params["learning_rate"])
        decay_rate = self.params["lr_decay"]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: decay_rate ** epoch)

        train_x = torch.tensor(train_data[0], dtype=torch.float32)
        train_dx = torch.tensor(train_data[1], dtype=torch.float32)
        test_x = torch.tensor(test_data[0], dtype=torch.float32).to(self.device)
        test_dx = torch.tensor(test_data[1], dtype=torch.float32).to(self.device)

        train_dataset = TensorDataset(train_x.to(self.device), train_dx.to(self.device))
        train_loader = DataLoader(train_dataset, batch_size=self.params["batch_size"], shuffle=True)

        self.history = defaultdict(list)

        print("\nActive Loss Weights:")
        for k in self.params:
            if "loss_weight" in k and self.params[k] > 0:
                print(f"{k}: {self.params[k]}")

        # ✅ SINDyCall logic (ON by default unless explicitly disabled)
        if self.params.get("use_sindycall", True):
            self.sindy_caller = SindyCall(
                model=self.model,
                params=self.params,
                threshold=self.params["coefficient_threshold"],
                update_freq=self.params.get("update_freq", 10),
                x=self.data.x,
                t=self.data.t
            )
        else:
            self.sindy_caller = None
        
        rfe_callback = RfeUpdateCallback(self.model, 
                                 rfe_frequency=self.params.get("rfe_frequency", 10),
                                 print_frequency=self.params.get("sindy_print_rate", 10))
        
        print("Launching training with coefficient_threshold =", self.params["coefficient_threshold"])

        for epoch in range(self.params["max_epochs"]):
            self.model.train()
            epoch_loss = 0.0
            if epoch == 0:
                self.model.sindy.print_trainability_report("(After Epoch 0)")

            for batch_x, batch_dx in train_loader:
                optimizer.zero_grad()

                inputs = (batch_x, batch_x, batch_dx)
                outputs = (batch_x, batch_x, batch_dx)

                loss, losses = self.model.get_loss(*inputs, *outputs)

                loss.backward()

                optimizer.step()

                with torch.no_grad():
                    # Strict enforcement of fixed entries
                    fixed_mask = self.model.sindy.fixed_mask
                    fixed_vals = self.model.sindy.fixed_values
                    coeffs = self.model.sindy.coefficients

                    # Reset any drifted fixed values (in case of numerical error)
                    coeffs[fixed_mask] = fixed_vals[fixed_mask]

                    # Optional: enforce soft sparsity mask manually (RFE style)
                    if self.params.get("coefficient_threshold") is not None:
                        mask = coeffs.abs() > self.params["coefficient_threshold"]
                        coeffs *= mask.float()  # zero out small coefficients
                        self.model.sindy.coefficients_mask = mask

                    # Double check that constraints hold
                    if not torch.allclose(coeffs[fixed_mask], fixed_vals[fixed_mask], atol=1e-6):
                        print("❌ Fixed coefficient mismatch!")
                        diff = coeffs[fixed_mask] - fixed_vals[fixed_mask]
                        print("Max difference:", diff.abs().max().item())
                        print("Full matrix:\n", coeffs.cpu().numpy())
                        print("Expected fixed values:\n", fixed_vals.cpu().numpy())

                epoch_loss += loss.item()

                # Log all sub-losses
                self.history["train_rec"].append(losses.get("rec", torch.tensor(0.0)).detach().cpu().item())
                self.history["train_sindy_z"].append(losses.get("sindy_z", torch.tensor(0.0)).detach().cpu().item())
                self.history["train_sindy_x"].append(losses.get("sindy_x", torch.tensor(0.0)).detach().cpu().item())
                self.history["train_integral"].append(losses.get("integral", torch.tensor(0.0)).detach().cpu().item())
                self.history["train_x0"].append(losses.get("x0", torch.tensor(0.0)).detach().cpu().item())
                self.history["train_l1"].append(losses.get("l1", torch.tensor(0.0)).detach().cpu().item())

            epoch_loss /= len(train_loader)
            self.history["train_total"].append(epoch_loss)

            scheduler.step()

            # 🔍 LOG average coefficient stats (ADD HERE)
            trainable_mask = ~fixed_mask
            mean_all = coeffs[trainable_mask].abs().mean().item()
            mean_per_column = []
            for j in range(coeffs.shape[1]):
                col_mask = trainable_mask[:, j]
                if col_mask.sum() > 0:
                    mean_per_column.append(coeffs[:, j][col_mask].abs().mean().item())
                else:
                    mean_per_column.append(float('nan'))

            self.history['mean_coef_all'].append(mean_all)
            self.history['mean_coef_per_col'].append(mean_per_column)

            if epoch % self.params["loss_print_rate"] == 0:
                print(f"Epoch {epoch}: Loss = {epoch_loss:.6f} | LR = {optimizer.param_groups[0]['lr']:.6f}")

            if epoch % self.params["sindy_print_rate"] == 0:
                print("--- SINDy Coefficient Matrix (Sanity Check) ---")
                print(self.model.sindy.coefficients.detach().cpu().numpy())

            with torch.no_grad():
                coeff_matrix = self.model.sindy.coefficients
                if torch.isnan(coeff_matrix).any():
                    print("❌ SINDy matrix contains NaN entities! Stopping training.")
                    raise RuntimeError("SINDy matrix contains NaN entities!")
            
            rfe_callback.update_rfe(epoch)

            # ✅ Trigger SINDyCall if active
            if self.sindy_caller:
                self.sindy_caller.update_sindy(epoch)

                # --- Plot average coefficient magnitude ---
        plt.figure(figsize=(10, 4))
        plt.plot(self.history['mean_coef_all'], label='Mean of All Trainable Coefficients')
        plt.xlabel("Epoch")
        plt.ylabel("Mean |Coefficient|")
        plt.title("Average Magnitude of Trainable Coefficients (All)")
        plt.grid(True)
        plt.legend()
        plt.show()

        # --- Plot per column average coefficient magnitude ---
        mean_per_col = list(zip(*self.history['mean_coef_per_col']))  # transpose to [col][epoch]
        for j, col_vals in enumerate(mean_per_col):
            plt.plot(col_vals, label=f'Column {j}')
        plt.xlabel("Epoch")
        plt.ylabel("Mean |Coefficient|")
        plt.title("Average Coefficient Magnitude per Output Column")
        plt.grid(True)
        plt.legend()
        plt.show()

        self.save_results()

    def save_params(self):
        os.makedirs(self.params["data_path"], exist_ok=True)
        df = pd.DataFrame([self.params])
        df.to_pickle(os.path.join(self.params["data_path"], self.savename + "_params.pkl"))

    def save_results(self):
        # Create general folder if missing
        os.makedirs(self.params["data_path"], exist_ok=True)
        
        # Create model-specific folder if missing
        model_dir = os.path.join(self.params["data_path"], self.savename)
        os.makedirs(model_dir, exist_ok=True)

        results_dict = {
            "losses": self.history,
            "sindy_coefficients": self.model.sindy.coefficients.detach().cpu().numpy()
        }
        df = pd.DataFrame([results_dict])
        df.to_pickle(os.path.join(self.params["data_path"], self.savename + "_results.pkl"))
        torch.save(self.model.state_dict(), os.path.join(model_dir, "model.pth"))
        # --- Print final SINDy model ---
        coefs = self.model.sindy.coefficients.detach().cpu().numpy()
        library_size = coefs.shape[0]
        dim = coefs.shape[1]    
        terms = []

        # Reconstruct library terms (assuming poly order up to 2 for simplicity)
        basis = ['1'] + [f'z{i}' for i in range(self.model.latent_dim)]
        if self.params['poly_order'] >= 2:
            basis += [f'z{i}z{j}' for i in range(self.model.latent_dim) for j in range(i, self.model.latent_dim)]

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
    
    def simulate_sindy_trajectory(self, z0, timesteps=1000, dt=0.01):
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