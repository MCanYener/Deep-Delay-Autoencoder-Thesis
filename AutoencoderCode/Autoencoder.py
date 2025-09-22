import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchmetrics
from torch.autograd import grad
import pysindy as ps
from pysindy.differentiation import FiniteDifference
import traceback
from sklearn.exceptions import NotFittedError
from math import comb

# This callback is used to update mask, i.e. apply recursive feature elimination in Sindy at the beginning of each epoch
class RfeUpdateCallback:
    def __init__(self, model, rfe_frequency=10, print_frequency=10, rfe_start=200):
        self.model = model
        self.rfe_frequency = rfe_frequency
        self.print_frequency = print_frequency
        self.rfe_start = rfe_start

    def update_rfe(self, epoch):
        if not self.model.sindy.coefficients.requires_grad:
            print("[RFE] Skipped — SINDy coefficients are not trainable.")
            return

        if epoch > self.rfe_start and epoch % self.rfe_frequency == 0:
            self.model.sindy.update_mask()
            print(f"[RFE] Pruned coefficients below {self.model.sindy.rfe_threshold}")

# This class is used to fit a SINDy model to the latent space representation of the data.
class SindyCall:
    def __init__(self, model, params, threshold, update_freq, x, t, lengths = None):
        self.model = model
        self.params = params
        self.threshold = threshold
        self.update_freq = update_freq
        self.t = t
        self.x = x
        self.poly_order = params.get("poly_order", 3)
        self.dt = params.get("dt", 0.01)
        self.last_model = None
        self.last_epoch = -1
        self.lengths = lengths

    def _split_by_lengths(self, A):
        if self.lengths is None:
            # fallback: assume equal-length segments if n_ics provided
            n_traj = self.params.get("n_ics", 1)
            K = A.shape[0] // n_traj
            return [A[i*K:(i+1)*K] for i in range(n_traj)]
        idx = np.cumsum([0] + list(self.lengths))
        return [A[idx[i]:idx[i+1]] for i in range(len(self.lengths))]

    def update_sindy(self, epoch):
        if epoch < self.params.get("sindycall_start", 60):
            return
        if epoch % self.update_freq != 0:
            return

        device = next(self.model.parameters()).device
        x_tensor = torch.tensor(self.x, dtype=torch.float32).to(device)

        with torch.no_grad():
            z_latent = self.model.encoder(x_tensor).cpu().numpy()

        if z_latent.ndim == 3:
            z_latent = z_latent.squeeze(0)
        elif z_latent.ndim != 2:
            raise ValueError(f"[ERROR] Unexpected z_latent shape: {z_latent.shape}")

        #z_latent = (z_latent - np.mean(z_latent, axis=0)) / (np.std(z_latent, axis=0) + 1e-6)  #normalization 
        z_latent = np.nan_to_num(z_latent, nan=0.0, posinf=1e3, neginf=-1e3)
        z_latent = np.clip(z_latent, -1e3, 1e3)
        z_dot = np.gradient(z_latent, self.dt, axis=0)

        z_list = self._split_by_lengths(z_latent)
        z_dot_list = self._split_by_lengths(z_dot)
        if any(arr.shape[0] < 3 for arr in z_list):
            return  # too short to fit safely
        if any(np.isnan(a).any() or np.isinf(a).any() for a in z_list + z_dot_list):
            return

        library = ps.PolynomialLibrary(degree=self.poly_order)
        library.fit(z_latent)
        n_features = library.transform(z_latent).shape[1]
        n_outputs = z_latent.shape[1]

        fixed_mask = self.model.sindy.fixed_mask.detach().cpu().numpy()
        fixed_values = self.model.sindy.fixed_values.detach().cpu().numpy()
        constraint_rows = np.argwhere(fixed_mask)
        constraint_lhs = np.zeros((len(constraint_rows), n_features * n_outputs))
        constraint_rhs = []

        for idx, (i, j) in enumerate(constraint_rows):
            constraint_lhs[idx, j * n_features + i] = 1
            constraint_rhs.append(fixed_values[i, j])
        constraint_rhs = np.array(constraint_rhs)

        optimizer = ps.ConstrainedSR3(
            threshold=self.threshold,
            nu=self.params.get("constraint_nu", 1e-2),
            constraint_lhs=constraint_lhs,
            constraint_rhs=constraint_rhs,
            max_iter=10000,
            tol=1e-10,
            thresholder="l0"
        )

        sindy_model = ps.SINDy(
            optimizer=optimizer,
            feature_library=library,
            feature_names=[f"z{i}" for i in range(n_outputs)],
            discrete_time=False
        )

        if np.isnan(z_latent).any() or np.isinf(z_latent).any():
            return
        if np.isnan(z_dot).any() or np.isinf(z_dot).any():
            return

        try:
            sindy_model.fit(z_list, x_dot=z_dot_list, t=self.dt, ensemble=True, quiet=True, multiple_trajectories=True) #consider fitting to the original data as opposed to the latent data
            self.last_model = sindy_model
            self.last_epoch = epoch
        except Exception as e:
            print(f"[ERROR] Exception during SINDy fit: {e}")
            traceback.print_exc()
            return

        with torch.no_grad():
            coefs_np = sindy_model.coefficients()
            if coefs_np is None or np.isnan(coefs_np).any():
                return

            new_coefs = torch.from_numpy(coefs_np.T.astype(np.float32)).to(self.model.sindy.coefficients.device)

            # Clamp / sanitize ONLY if training enabled
            if self.params.get("sindy_trainable", True):
                new_coefs = torch.nan_to_num(new_coefs, nan=0.0, posinf=1e2, neginf=-1e2)
                new_coefs = torch.clamp(new_coefs, -100.0, 100.0)
                new_coefs[self.model.sindy.fixed_mask] = self.model.sindy.fixed_values[self.model.sindy.fixed_mask]

            if torch.isnan(new_coefs).any() or torch.isinf(new_coefs).any():
                return

            self.model.sindy.coefficients.data.copy_(new_coefs)

            if self.params.get("sindy_trainable", True):
                self.model.sindy.coefficients_mask = (new_coefs.abs() > 1e-5).float()

            self.model.sindy.initialized = True

# This class is initialized by the Autoencoder Network to create the SINDy model.
class Sindy(nn.Module):
    def __init__(self, library_dim, state_dim, poly_order, model_tag='lorenz', 
                 initializer='constant', actual_coefs=None, rfe_threshold=None, 
                 include_sine=False, exact_features=False, fix_coefs=False, 
                 sindy_pert=0.0, ode_net=False, ode_net_widths=[1.5, 2.0],
                 coefficient_init_val=0.0,
                 fixed_mask=None, fixed_values=None, params=None):
        super(Sindy, self).__init__()

        self.library_dim = library_dim
        self.state_dim = state_dim
        self.poly_order = poly_order
        self.include_sine = include_sine
        self.rfe_threshold = rfe_threshold
        self.exact_features = exact_features
        self.actual_coefs = actual_coefs
        self.fix_coefs = fix_coefs  # << flag controlling constraint use
        self.sindy_pert = sindy_pert
        self.model_tag = model_tag
        self.ode_net = ode_net
        self.ode_net_widths = ode_net_widths

        self.l2 = params.get("loss_weight_layer_l2", 1e-6) if params else 1e-6
        self.l1 = params.get("loss_weight_layer_l1", 0.0) if params else 0.0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Fixed mask and values ---
        if self.fix_coefs:
            self.fixed_mask = torch.tensor(fixed_mask, dtype=torch.bool, device=device) if fixed_mask is not None else torch.zeros((library_dim, state_dim), dtype=torch.bool, device=device)
            self.fixed_values = torch.tensor(fixed_values, dtype=torch.float32, device=device) if fixed_values is not None else torch.zeros((library_dim, state_dim), dtype=torch.float32, device=device)
        else:
            self.fixed_mask = torch.zeros((library_dim, state_dim), dtype=torch.bool, device=device)
            self.fixed_values = torch.zeros((library_dim, state_dim), dtype=torch.float32, device=device)

        # --- Initialize coefficient matrix ---
        if params.get("sindy_trainable", True):
            self.coefficients = nn.Parameter(
                torch.full((library_dim, state_dim),
                           coefficient_init_val,
                           dtype=torch.float32,
                           device=device),
                requires_grad=True
            )
        else:
            self.register_buffer("coefficients", torch.full(
                (library_dim, state_dim),
                coefficient_init_val,
                dtype=torch.float32,
                device=device
            ))

        # Apply fixed values to fixed locations (if enabled)
        if self.fix_coefs:
            self.coefficients.data[self.fixed_mask] = self.fixed_values[self.fixed_mask]

        # Apply gradient masking (if enabled)
        if self.fix_coefs and params.get("sindy_trainable", True):
            def mask_grad(grad):
                return grad * (~self.fixed_mask).float()
            self.coefficients.register_hook(mask_grad)

        # Optional neural net for theta library
        if self.ode_net:
            self.net_model = self.make_theta_network(self.library_dim, self.ode_net_widths)
            
    def forward(self, z):
        theta_z = self.theta(z)
        return torch.matmul(theta_z, self.coefficients.to(z.device))

    def theta(self, z):
        if self.ode_net:
            return self.net_model(z)
        else:
            return self.sindy_library(z, z.shape[1], self.poly_order,
                                      include_sine=self.include_sine,
                                      exact_features=self.exact_features,
                                      model=self.model_tag)

    def make_theta_network(self, output_dim, hidden_sizes):  # absolute sizes
        layers = []
        in_dim = self.state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, int(h)))
            layers.append(nn.ReLU())
            in_dim = int(h)
        layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers)

    def sindy_library(self, z, latent_dim, poly_order, include_sine=False, exact_features=False, model_tag='lorenz'):
        library = [torch.ones(z.shape[0], dtype=torch.float32, device=z.device)]
        for i in range(latent_dim):
            library.append(z[:, i])

        if poly_order > 1:
            for i in range(latent_dim):
                for j in range(i, latent_dim):
                    library.append(z[:, i] * z[:, j])

        if poly_order > 2:
            for i in range(latent_dim):
                for j in range(i, latent_dim):
                    for k in range(j, latent_dim):
                        library.append(z[:, i] * z[:, j] * z[:, k])

        if poly_order > 3:
            for i in range(latent_dim):
                for j in range(i, latent_dim):
                    for k in range(j, latent_dim):
                        for p in range(k, latent_dim):
                            library.append(z[:, i] * z[:, j] * z[:, k] * z[:, p])

        if poly_order > 4:
            for i in range(latent_dim):
                for j in range(i, latent_dim):
                    for k in range(j, latent_dim):
                        for p in range(k, latent_dim):
                            for q in range(p, latent_dim):
                                library.append(z[:, i] * z[:, j] * z[:, k] * z[:, p] * z[:, q])

        if include_sine:
            for i in range(latent_dim):
                library.append(torch.sin(z[:, i]))

        return torch.stack(library, dim=1)

    def reset_parameters(self):
        with torch.no_grad():
            self.coefficients.data.uniform_(-0.1, 0.1)
            self.coefficients.data[self.fixed_mask] = self.fixed_values[self.fixed_mask]

    def update_mask(self):
        if not self.coefficients.requires_grad:
            print("[RFE] Skipping mask update: coefficients are not trainable.")
            return

        if self.rfe_threshold is not None:
            with torch.no_grad():
                new_mask = (self.coefficients.abs() > self.rfe_threshold)

                # 🔒 Failsafe: skip update if all coefficients fall below threshold
                if new_mask.sum() == 0:
                    print("⚠️ RFE skipped — all coefficients below threshold.")
                    return

                # ✅ Proceed with mask update
                self.coefficients_mask = new_mask.float()
                self.coefficients.data *= new_mask.float()  # Zero out small entries

                # Reapply fixed constraints
                self.coefficients.data[self.fixed_mask] = self.fixed_values[self.fixed_mask]

            # 🔄 Replace existing gradient mask hook
            if hasattr(self, "_rfe_hook"):
                self._rfe_hook.remove()

            def gradient_mask_hook(grad):
                return grad * new_mask.float() * (~self.fixed_mask).float()

            self._rfe_hook = self.coefficients.register_hook(gradient_mask_hook)

    def print_trainability_report(self, label=""):
        print(f"\n--- SINDy Coefficient Trainability Matrix {label}---")
        coeffs = self.coefficients.detach().cpu().numpy()
        mask = self.fixed_mask.detach().cpu().numpy()

        for i in range(self.library_dim):
            row = []
            for j in range(self.state_dim):
                val = coeffs[i, j]
                marker = "X" if mask[i, j] else "O"
                row.append(f"{val:7.3f}{marker}")
            print(" | ".join(row))

        print("Legend: O = trainable, X = fixed")
        print(f"Total trainable: {(~self.fixed_mask).sum().item()} / {self.library_dim * self.state_dim}")

    def debug_shapes(self, z):
        with torch.no_grad():
            theta = self.theta(z)
            print(f"[DEBUG] z.shape={z.shape}")
            print(f"[DEBUG] theta(z).shape={theta.shape}  # should be (batch, library_dim) in learned-library mode")
            print(f"[DEBUG] coefficients.shape={self.coefficients.shape}  # (library_dim, state_dim)")
            out = theta @ self.coefficients
            print(f"[DEBUG] (theta @ coeffs).shape={out.shape}  # should be (batch, state_dim)")

    
# This class is the main autoencoder model that combines the encoder, decoder, and SINDy model. Used with and SVD input.
class PreSVD_Sindy_Autoencoder(nn.Module):
    def __init__(self, params, hankel=None):
        super(PreSVD_Sindy_Autoencoder, self).__init__()
        self.params = params
        self.latent_dim = params['latent_dim']
        self.input_dim = params['input_dim']
        self.observable_dim = params['observable_dim']
        self.svd_dim = params['svd_dim']
        self.library_dim = params['library_dim']
        self.poly_order = params['poly_order']
        self.include_sine = params['include_sine']
        self.epochs = params['max_epochs']
        self.rfe_threshold = params['rfe_threshold']
        self.rfe_frequency = params['threshold_frequency']
        self.print_frequency = params['print_frequency']
        self.sindy_pert = params['sindy_pert']
        self.use_bias = params['use_bias']
        self.l2 = params['loss_weight_layer_l2']
        self.l1 = params['loss_weight_layer_l1']

        self.total_loss_metric = torchmetrics.MeanMetric()
        self.rec_loss_metric = torchmetrics.MeanMetric()
        self.sindy_z_loss_metric = torchmetrics.MeanMetric()
        self.sindy_x_loss_metric = torchmetrics.MeanMetric()
        self.integral_loss_metric = torchmetrics.MeanMetric()
        self.x0_loss_metric = torchmetrics.MeanMetric()
        self.l1_loss_metric = torchmetrics.MeanMetric()

        # ✅ Compute the number of polynomial features
        def compute_library_dim(latent_dim, poly_order):
            return sum([comb(latent_dim + i - 1, i) for i in range(1, poly_order + 1)]) + 1  # +1 for constant term

        num_library_terms = compute_library_dim(self.latent_dim, self.poly_order)

        # ✅ Hankel embedding (if used)
        self.hankel = torch.tensor(hankel, dtype=torch.float32) if hankel is not None else None

        # ✅ Writable matrix for SINDyCall
        self.coefficients_matrix = torch.zeros((self.latent_dim, num_library_terms))

        # Define time constant
        self.time = torch.linspace(0.0, params['dt'] * params['svd_dim'], params['svd_dim'])
        self.dt = torch.tensor(params['dt'], dtype=torch.float32)

        # Define encoder, decoder, and Sindy model
        # Determine effective input/output size and width list
        if self.observable_dim == 1:
            net_input_dim = self.input_dim
            widths = params['widths_1d']
        else:
            net_input_dim = self.input_dim * self.observable_dim
            widths = params['widths_2d']

        # Construct encoder and decoder
        self.encoder = self.make_network(net_input_dim, self.latent_dim, widths)
        self.decoder = self.make_network(self.latent_dim, net_input_dim, list(reversed(widths)))
        print("\n🧪 DEBUG: Encoder network")
        print(self.encoder)
        print(f"Input dim to encoder: {self.encoder[0].in_features}")
        print("\n🧪 DEBUG: Decoder network")
        print(self.decoder)
        print(f"Input dim to encoder: {self.decoder[0].in_features}")

        # Optional fixed mask and values for constraining SINDy coefficients
        fixed_mask = params.get("sindy_fixed_mask", None)
        fixed_values = params.get("sindy_fixed_values", None)

        # Create SINDy module with optional constraints
        self.sindy = Sindy(
            library_dim=self.library_dim,
            state_dim=self.latent_dim,
            poly_order=self.poly_order,
            actual_coefs=params.get('actual_coefficients'),
            initializer=params['coefficient_initialization'],
            coefficient_init_val=params.get('coefficient_initialization_constant', 1e-2),
            rfe_threshold=self.rfe_threshold,
            include_sine=self.include_sine,
            exact_features=params.get('exact_features', False),
            fix_coefs=params.get('fix_coefs', False),
            sindy_pert=self.sindy_pert,
            ode_net=params.get("ode_net", False),
            ode_net_widths=params.get("ode_net_widths", [1.5, 2.0]),
            fixed_mask=fixed_mask,
            fixed_values=fixed_values,
            params=params,  # for l1/l2 regularization
        )
        print("\n🧪 DEBUG: ODE Net (inside SINDy)")

        if hasattr(self.sindy, "net_model") and self.sindy.net_model is not None:
            print(self.sindy.net_model)

            # Layer-by-layer summary
            for i, layer in enumerate(self.sindy.net_model):
                if isinstance(layer, nn.Linear):
                    print(f"  Layer {i}: Linear(in={layer.in_features}, out={layer.out_features})")
                else:
                    print(f"  Layer {i}: {layer.__class__.__name__}")
        else:
            print("  (no ODE net; ode_net=False)")

        def grad_hook(grad):
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                print("❌ NaN or Inf detected in encoder gradient — clamping.")
                return torch.nan_to_num(grad, nan=0.0, posinf=1e2, neginf=-1e2)
            return grad

        self.encoder[0].weight.register_hook(grad_hook)


    def make_network(self, input_dim, output_dim, hidden_layers, name=""):
        layers = []
        layer_dims = [input_dim] + hidden_layers + [output_dim]

        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            linear = nn.Linear(in_dim, out_dim, bias=self.use_bias)
            nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu' if out_dim != output_dim else 'linear')
            if self.use_bias:
                nn.init.zeros_(linear.bias)
            layers.append(linear)
            if out_dim != output_dim:
                layers.append(nn.ReLU())  # Swap for Tanh if needed

        return nn.Sequential(*layers)
    
    def train_step(self, data, optimizer, sindy_optimizer):
        inputs, outputs = data
        x, v, dv_dt = inputs
        x_out, v_out, dv_dt_out = outputs

        optimizer.zero_grad()
        sindy_optimizer.zero_grad()

        loss, losses = self.get_loss(x, v, dv_dt, x_out, v_out, dv_dt_out)
        
        loss.backward()

        optimizer.step()
        sindy_optimizer.step()

    def split_observables(self, v):
        """Splits the input into [x, y] based on observable_dim."""
        if self.observable_dim == 1:
            return v, None
        elif self.observable_dim == 2:
            return v[:, :self.input_dim], v[:, self.input_dim:]
        else:
            raise NotImplementedError("Only observable_dim=1 or 2 is supported.")


    @torch.no_grad()  # Disable gradient computation
    def test_step(self, data):
        inputs, outputs = data
        x, v, dv_dt = inputs
        x_out, v_out, dv_dt_out = outputs
        

        loss, losses = self.get_loss(x, v, dv_dt, x_out, v_out, dv_dt_out)

        # Update loss metrics
        self.update_losses(loss, losses)
        
        return {name: metric.item() for name, metric in self.metrics.items()}
    
    def update_losses(self, loss, losses):
        self.total_loss_metric.update(loss)

        self.rec_loss_metric.update(losses['rec'])
        
        if self.params['loss_weight_sindy_z'] > 0:
            self.sindy_z_loss_metric.update(losses['sindy_z'])
        
        if self.params['loss_weight_sindy_x'] > 0:
            self.sindy_x_loss_metric.update(losses['sindy_x'])
        
        if self.params['loss_weight_integral'] > 0:
            self.integral_loss_metric.update(losses['integral'])
        
        if self.params['loss_weight_x0'] > 0:
            self.x0_loss_metric.update(losses['x0'])
        
        if self.params['loss_weight_sindy_regularization'] > 0:
            self.l1_loss_metric.update(losses['l1'])
    @property
    def metrics(self):
        metrics_dict = {
            'total': self.total_loss_metric,
            'rec': self.rec_loss_metric,
        }

        if self.params['loss_weight_sindy_z'] > 0.0:
            metrics_dict['sindy_z'] = self.sindy_z_loss_metric

        if self.params['loss_weight_sindy_x'] > 0.0:
            metrics_dict['sindy_x'] = self.sindy_x_loss_metric

        if self.params['loss_weight_integral'] > 0.0:
            metrics_dict['integral'] = self.integral_loss_metric

        if self.params['loss_weight_x0'] > 0.0:
            metrics_dict['x0'] = self.x0_loss_metric

        if self.params['loss_weight_sindy_regularization'] > 0.0:
            metrics_dict['l1'] = self.l1_loss_metric

        return metrics_dict

    def get_loss(self, x, v, dv_dt, x_out, v_out, dv_dt_out, epoch=0):
        losses = {}
        total  = 0.0

        # ---- phase & ramps ----
        train_sindy = self.params.get("train_sindy", True)
        start_sindy = self.params.get("sindy_train_start", 0)
        use_sindy   = train_sindy and (epoch >= start_sindy)

        ramp = self.params.get("sindy_ramp_duration", 500)
        sindy_scale = 0.0 if epoch < start_sindy else min((epoch - start_sindy)/float(ramp), 1.0)

        l1_start    = self.params.get("sindy_l1_ramp_start", 0)
        l1_duration = self.params.get("sindy_l1_ramp_duration", 500)
        l1_scale    = min(max(epoch - l1_start, 0)/float(l1_duration), 1.0)

        # ---- sanitize derivatives ----
        dv_dt     = torch.nan_to_num(dv_dt,     nan=0.0, posinf=1e3, neginf=-1e3).clamp(-100,100)
        dv_dt_out = torch.nan_to_num(dv_dt_out, nan=0.0, posinf=1e3, neginf=-1e3).clamp(-100,100)

        seq_mode = (v.dim() == 3)  # (B,T,D)

        if not seq_mode:
            # ================= pointwise =================
            v.requires_grad_(True)
            z  = self.encoder(v)
            xh = self.decoder(z)

            # rec in observation space
            vx, vy   = self.split_observables(v)
            vhx, vhy = self.split_observables(xh)
            if self.params['loss_weight_rec'] > 0:
                rec = F.mse_loss(vhx, vx) + (F.mse_loss(vhy, vy) if self.observable_dim==2 else 0.0)
                total += self.params['loss_weight_rec'] * rec
                losses['rec'] = rec

            # x0 latent (align z[:,0] to reference latent target; here we use hankel’s first column or x[:,0] if you pass latent)
            if self.params.get('loss_weight_x0', 0.0) > 0.0:
                pass

            if use_sindy:
                # dz/dt SINDy
                z_dot_sindy = torch.nan_to_num(self.sindy(z), nan=0.0, posinf=1e3, neginf=-1e3).clamp(-1e3,1e3)

                # dz/dt auto: (dz/dx) * (dx/dt)
                if self.params['loss_weight_sindy_z'] > 0.0:
                    B, L = z.shape
                    z_dot_auto = torch.zeros_like(z)
                    ones = torch.ones(B, device=v.device)
                    for i in range(L):
                        grads = torch.autograd.grad(z[:,i], v, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
                        z_dot_auto[:, i] = (grads * dv_dt).sum(dim=1)
                    dz = F.mse_loss(z_dot_auto, z_dot_sindy)
                    total += sindy_scale * self.params['loss_weight_sindy_z'] * dz
                    losses['sindy_z'] = dz

                # dx/dt via decoder Jacobian
                if self.params['loss_weight_sindy_x'] > 0.0:
                    z = z.requires_grad_(True)
                    vh = self.decoder(z)
                    B, Din = vh.shape
                    v_dot_auto = torch.zeros_like(vh)
                    ones = torch.ones(B, device=z.device)
                    for i in range(Din):
                        grads = torch.autograd.grad(vh[:,i], z, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
                        v_dot_auto[:, i] = (grads * z_dot_sindy).sum(dim=1)
                    dx = F.mse_loss(torch.nan_to_num(v_dot_auto, nan=0.0, posinf=1e2, neginf=-1e2), dv_dt_out)
                    total += sindy_scale * self.params['loss_weight_sindy_x'] * dx
                    losses['sindy_x'] = dx

        else:
            # ================= sequence (teacher forcing) =================
            B, T, Din = v.shape
            v_flat = v.reshape(B*T, Din).requires_grad_(True)

            z_flat = self.encoder(v_flat)                     # (B*T, L)
            L      = z_flat.shape[1]
            z_seq  = z_flat.view(B, T, L)                     # (B, T, L)

            xh_flat = self.decoder(z_flat)
            # rec on all steps
            vx, vy   = self.split_observables(v.reshape(B*T, Din))
            vhx, vhy = self.split_observables(xh_flat)
            if self.params['loss_weight_rec'] > 0:
                rec = F.mse_loss(vhx, vx) + (F.mse_loss(vhy, vy) if self.observable_dim==2 else 0.0)
                total += self.params['loss_weight_rec'] * rec
                losses['rec'] = rec

            # x0 latent (z_seq vs encoded target’s first step)
            if self.params.get('loss_weight_x0', 0.0) > 0.0:
                z0     = z_seq[:, 0, :]              # (B, L)
                x0_rec = self.decoder(z0)            # (B, D)
                v0     = v[:, 0, :]                  # (B, D) true first observation(s)
                x0_loss = F.mse_loss(x0_rec, v0)
                total += self.params['loss_weight_x0'] * x0_loss
                losses['x0'] = x0_loss

            if use_sindy:
                # SINDy dz/dt at all steps
                z_dot_sindy_flat = torch.nan_to_num(self.sindy(z_flat), nan=0.0, posinf=1e3, neginf=-1e3).clamp(-1e3,1e3)

                # dz/dt auto across all steps: (dz/dx)*(dx/dt)
                if self.params['loss_weight_sindy_z'] > 0.0:
                    z_dot_auto_flat = torch.zeros_like(z_flat)
                    ones = torch.ones(B*T, device=v_flat.device)
                    for i in range(L):
                        grads = torch.autograd.grad(z_flat[:,i], v_flat, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
                        z_dot_auto_flat[:, i] = (grads * dv_dt.reshape(B*T, -1)).sum(dim=1)
                    dz = F.mse_loss(z_dot_auto_flat, z_dot_sindy_flat)
                    total += sindy_scale * self.params['loss_weight_sindy_z'] * dz
                    losses['sindy_z'] = dz

                # Integral (rollout vs latent sequence)
                if self.params.get('loss_weight_integral', 0.0) > 0.0:
                    z0   = z_seq[:, 0, :]
                    zsim = [z0]
                    for _ in range(T-1):
                        zc = zsim[-1]
                        k1 = self.sindy(zc)
                        k2 = self.sindy(zc + self.dt/2 * k1)
                        k3 = self.sindy(zc + self.dt/2 * k2)
                        k4 = self.sindy(zc + self.dt    * k3)
                        zn = zc + (self.dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
                        zn = torch.nan_to_num(zn, nan=0.0, posinf=1e3, neginf=-1e3).clamp(-500,500)
                        zsim.append(zn)
                    zsim = torch.stack(zsim, dim=1)             # (B,T,L)

                    # latent teacher-forcing target = encoded latent sequence
                    int_loss = F.mse_loss(zsim[:,1:,:], z_seq[:,1:,:])
                    total += sindy_scale * self.params['loss_weight_integral'] * int_loss
                    losses['integral'] = int_loss

                # (Optional) sindy_x0 in observation space using the rollout’s first step
                if self.params.get("loss_weight_sindy_x0", 0.0) > 0.0:
                    z0     = z_seq[:, 0, :]      # (B, L)
                    dec0   = self.decoder(z0)     # (B, D)
                    v0     = v[:, 0, :]           # (B, D) true first obs
                    sx0    = F.mse_loss(dec0, v0)
                    total += self.params['loss_weight_sindy_x0'] * sx0
                    losses['sindy_x0'] = sx0

                if self.params['loss_weight_sindy_x'] > 0.0:

                    z_flat_req = z_flat.detach().requires_grad_(True)
                    xh_flat_req = self.decoder(z_flat_req)  # (B*T, D)

                    BT, Din = xh_flat_req.shape
                    v_dot_auto_flat = torch.zeros_like(xh_flat_req)

                    ones = torch.ones(B*T, device=z_flat_req.device)
                    for i in range(Din):
                        grads = torch.autograd.grad(
                            outputs=xh_flat_req[:, i],
                            inputs=z_flat_req,
                            grad_outputs=ones,
                            create_graph=True,
                            retain_graph=True
                        )[0]  # (B*T, L)
                        v_dot_auto_flat[:, i] = (grads * z_dot_sindy_flat).sum(dim=1)

                    dx = F.mse_loss(
                        torch.nan_to_num(v_dot_auto_flat, nan=0.0, posinf=1e2, neginf=-1e2),
                        dv_dt.reshape(B*T, Din)
                    )
                    total += sindy_scale * self.params['loss_weight_sindy_x'] * dx
                    losses['sindy_x'] = dx

        # Regularizers
        if use_sindy and self.params.get("loss_weight_sindy_regularization", 0.0) > 0.0:
            l1 = self.sindy.coefficients.abs().mean()
            total += l1_scale * self.params['loss_weight_sindy_regularization'] * l1
            losses['l1'] = l1
        if use_sindy and self.params.get("loss_weight_sindy_group_l1", 0.0) > 0.0:
            g = self.sindy.coefficients.norm(p=2, dim=1).sum()
            total += l1_scale * self.params['loss_weight_sindy_group_l1'] * g
            losses['l1_group'] = g

        return total, losses

    def forward(self, x):
        z = self.encoder(x)
        z = torch.clamp(z, -1e3, 1e3)  # Clamp latent value
        return self.decoder(z)
    

class TrajectoryWindowDataset(Dataset):
    """
    Builds (T,D) windows from concatenated trajectories X (N,D) and dX (N,D),
    using a `lengths` list so windows do not cross trajectory boundaries.
    """
    def __init__(self, X, dX, lengths, T, stride=1, device=None):
        assert X.shape == dX.shape
        self.X = X
        self.dX = dX
        self.T = int(T)
        self.stride = int(stride)
        self.device = device

        # cumulative offsets to map (traj, local_idx) -> global row index
        self.lengths = list(map(int, lengths))
        self.offsets = [0]
        for L in self.lengths:
            self.offsets.append(self.offsets[-1] + L)

        # precompute all valid window starts per trajectory
        self.index = []  # list of (traj_id, start_local)
        for tidx, L in enumerate(self.lengths):
            max_start = L - self.T
            if max_start < 0:
                continue
            for s in range(0, max_start + 1, self.stride):
                self.index.append((tidx, s))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        tidx, s = self.index[i]
        g0 = self.offsets[tidx] + s
        g1 = g0 + self.T
        x_seq  = self.X[g0:g1, :]     # (T, D)
        dx_seq = self.dX[g0:g1, :]    # (T, D)

        # return CPU tensors here; DataLoader pin_memory=True will be fast, and you can .to(device) in the train loop
        return torch.from_numpy(x_seq).float(), torch.from_numpy(dx_seq).float()

# This class is the main autoencoder model that combines the encoder, decoder, and SINDy model. Used with and SVD input.
class SindyAutoencoder(nn.Module):
    def __init__(self, params):
        super(SindyAutoencoder, self).__init__()
        self.params = params
        self.latent_dim = params['latent_dim']
        self.input_dim = params['input_dim']
        self.widths = params['widths']
        self.activation = getattr(F, params['activation'])
        self.library_dim = params['library_dim']
        self.poly_order = params['poly_order']
        self.include_sine = params['include_sine']
        self.initializer = params['coefficient_initialization']
        self.epochs = params['max_epochs']
        self.rfe_threshold = params['coefficient_threshold']
        self.rfe_frequency = params['threshold_frequency']
        self.print_frequency = params['print_frequency']
        self.sindy_pert = params['sindy_pert']
        self.fixed_coefficient_mask = None
        self.actual_coefs = torch.tensor(params['actual_coefficients'], dtype=torch.float32)
        self.use_bias = params['use_bias']
        self.l2 = params['loss_weight_layer_l2']
        self.l1 = params['loss_weight_layer_l1']
        self.sparse_weights = torch.tensor(params['sparse_weighting'], dtype=torch.float32) if params['sparse_weighting'] else None
        if params['fixed_coefficient_mask']:
            self.fixed_coefficient_mask = (torch.abs(self.actual_coefs) > 1e-10).float()
        
        self.dt = torch.tensor(params['dt'], dtype=torch.float32)
        
        self.encoder = self.make_network(self.input_dim, self.latent_dim, self.widths, name='encoder')
        print("Creating encoder with input_dim =", self.input_dim)

        self.decoder = self.make_network(self.latent_dim, self.input_dim, self.widths[::-1], name='decoder')
        
        if not params['trainable_auto']:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
        
        self.sindy = Sindy(self.library_dim, self.latent_dim, self.poly_order, model_tag=params['model_tag'],
                            initializer=self.initializer, actual_coefs=self.actual_coefs, rfe_threshold=self.rfe_threshold,
                            include_sine=self.include_sine, exact_features=params['exact_features'],
                            fix_coefs=params['fix_coefs'], sindy_pert=self.sindy_pert, ode_net=params['ode_net'],
                            ode_net_widths=params['ode_net_widths'])
    
    def make_network(self, input_dim, output_dim, widths, name):
        layers = []
        in_dim = input_dim

        for i, w in enumerate(widths):
            linear = nn.Linear(in_dim, w, bias=self.use_bias)
            nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')  # <-- Explicit init
            if self.use_bias:
                nn.init.zeros_(linear.bias)  # optional: init bias to 0
            layers.append(linear)
            layers.append(nn.ReLU()) #ReLU can be changed for Tanh or ELU if needed
            in_dim = w

        final_layer = nn.Linear(in_dim, output_dim, bias=self.use_bias)
        nn.init.kaiming_uniform_(final_layer.weight, nonlinearity='linear')
        if self.use_bias:
            nn.init.zeros_(final_layer.bias)

        layers.append(final_layer)
        return nn.Sequential(*layers)
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    def compute_loss(self, x, dx_dt, x_out, dx_dt_out):
        losses = {}
        loss = 0.0
        
        z = self.encoder(x)
        xh = self.decoder(z)
        dz_dt_sindy = self.sindy(z).unsqueeze(-1)
        
        if self.params['loss_weight_sindy_x'] > 0.0:
            dxh_dz = torch.autograd.functional.jacobian(lambda z_: self.decoder(z_), z)
            dxh_dt = torch.matmul(dxh_dz, dz_dt_sindy)
            loss_dx = F.mse_loss(dxh_dt, dx_dt_out)
            loss += self.params['loss_weight_sindy_x'] * loss_dx
            losses['sindy_x'] = loss_dx
        
        loss_rec = F.mse_loss(xh, x_out)
        loss_l1 = torch.mean(torch.abs(self.sindy.coefficients)) if self.sparse_weights is None else \
                  torch.mean(torch.abs(self.sparse_weights * self.sindy.coefficients))
        loss += self.params['loss_weight_rec'] * loss_rec + self.params['loss_weight_sindy_regularization'] * loss_l1
        
        losses['rec'] = loss_rec
        losses['l1'] = loss_l1
        
        return loss, losses
    
    def train_step(self, optimizer, data):
        inputs, outputs = data
        x, dx_dt = inputs[0], inputs[1].unsqueeze(-1)
        x_out, dx_dt_out = outputs[0], outputs[1].unsqueeze(-1)
        
        optimizer.zero_grad()
        loss, losses = self.compute_loss(x, dx_dt, x_out, dx_dt_out)
        loss.backward()
        optimizer.step()
        
        return {key: val.item() for key, val in losses.items()}
