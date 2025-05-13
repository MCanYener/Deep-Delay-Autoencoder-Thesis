import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics
from torch.autograd import grad
import pysindy as ps
from pysindy.differentiation import FiniteDifference
import traceback
from sklearn.exceptions import NotFittedError
from math import comb

# This callback is used to update mask, i.e. apply recursive feature elimination in Sindy at the beginning of each epoch
class RfeUpdateCallback:
    def __init__(self, model, rfe_frequency=10, print_frequency=10):
        """
        Parameters:
        - model: PreSVD_Sindy_Autoencoder model
        - rfe_frequency: How often to apply RFE (zero out small coefficients permanently)
        - print_frequency: How often to print coefficients
        """
        self.model = model
        self.rfe_frequency = rfe_frequency
        self.print_frequency = print_frequency

    def update_rfe(self, epoch):
        if epoch % self.print_frequency == 0:
            print('--- SINDy Coefficient Matrix ---')
            print(self.model.sindy.coefficients.detach().cpu().numpy())

        if epoch > 200 and epoch % self.rfe_frequency == 0:
            self.model.sindy.update_mask()
            print(f"[RFE] Pruned coefficients below {self.model.sindy.rfe_threshold}")


# This class is used to fit a SINDy model to the latent space representation of the data.
class SindyCall:
    def __init__(self, model, params, threshold, update_freq, x, t):
        """
        PyTorch-compatible SINDy callback using SR3 and explicit derivatives.

        Parameters:
        - model: PreSVD_Sindy_Autoencoder model
        - params: full parameter dictionary
        - threshold: SR3 sparsity threshold
        - update_freq: how often to update coefficients (in epochs)
        - x: input time-series data (training)
        - t: time vector corresponding to x
        """
        self.model = model
        self.params = params
        self.threshold = threshold
        self.update_freq = update_freq
        self.t = t
        self.x = x
        self.poly_order = params.get("poly_order", 3)
        self.dt = params.get("dt", 0.01)

    def update_sindy(self, epoch):
        if epoch < self.params.get("sindycall_start", 60):
            return
        if epoch % self.update_freq != 0:
            return

        print("\n--- Running SINDy (Explicit Derivatives, Unconstrained) ---")

        # Step 1: Get latent trajectory
        device = next(self.model.parameters()).device
        x_tensor = torch.tensor(self.x, dtype=torch.float32).to(device)

        with torch.no_grad():
            z_latent = self.model.encoder(x_tensor).cpu().numpy()

        if z_latent.ndim == 3:
            z_latent = z_latent.squeeze(0)
        elif z_latent.ndim != 2:
            raise ValueError(f"[ERROR] Unexpected z_latent shape: {z_latent.shape}")

        print(f"[DEBUG] z_latent.shape = {z_latent.shape}")
        print(f"[DEBUG] std(z) = {np.std(z_latent, axis=0)}")

        # Step 2: Compute explicit derivatives
        z_dot = np.gradient(z_latent, self.dt, axis=0)

        # Step 3: Define SINDy model
        #if epoch < 100:
        #    optimizer = ps.SR3(threshold=0.01, nu=1e-3)
        #else:
        #    optimizer = ps.SR3(threshold=self.threshhold, nu=1e-2)
        optimizer = ps.STLSQ(threshold=self.threshold)
        library = ps.PolynomialLibrary(degree=self.poly_order)

        sindy_model = ps.SINDy(
            optimizer=optimizer,
            feature_library=library,
            feature_names=[f"z{i}" for i in range(z_latent.shape[1])],
            discrete_time=False
        )

        # Step 4: Fit model with explicit derivatives
        try:
            sindy_model.fit(z_latent, x_dot=z_dot, t=self.dt, quiet=True)
            print("✅ SINDy fit successful.")
            sindy_model.print()
        except Exception as e:
            print(f"[ERROR] Exception during SINDy fit: {e}")
            traceback.print_exc()
            return

        # Step 5: Safely transplant coefficients with constraints
        with torch.no_grad():
            new_coefs = torch.tensor(sindy_model.coefficients().T, dtype=torch.float32).to(device)

            # Re-impose fixed constraints
            fixed_mask = self.model.sindy.fixed_mask.to(device)
            fixed_values = self.model.sindy.fixed_values.to(device)

            new_coefs[fixed_mask] = fixed_values[fixed_mask]

            # In-place overwrite of 2D parameter
            self.model.sindy.coefficients.data.copy_(new_coefs)

            print("[DEBUG] coefficients updated in-place from SINDy fit.")

            # Optional: update mask for logging or RFE tracking
            self.model.sindy.coefficients_mask = (new_coefs.abs() > 1e-5).float()


# This class is initialized by the Autoencoder Network to create the SINDy model.
class Sindy(nn.Module):
    def __init__(self, library_dim, state_dim, poly_order, model='lorenz', 
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
        print("[DEBUG] Sindy rfe_threshold set to:", self.rfe_threshold)
        self.exact_features = exact_features
        self.actual_coefs = actual_coefs
        self.fix_coefs = fix_coefs
        self.sindy_pert = sindy_pert
        self.model = model
        self.ode_net = ode_net
        self.ode_net_widths = ode_net_widths

        self.l2 = params.get("loss_weight_layer_l2", 1e-6) if params else 1e-6
        self.l1 = params.get("loss_weight_layer_l1", 0.0) if params else 0.0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Fixed and trainable mask ---
        self.fixed_mask = torch.tensor(fixed_mask, dtype=torch.bool, device=device) if fixed_mask is not None else torch.zeros((library_dim, state_dim), dtype=torch.bool, device=device)
        self.fixed_values = torch.tensor(fixed_values, dtype=torch.float32, device=device) if fixed_values is not None else torch.zeros((library_dim, state_dim), dtype=torch.float32, device=device)

        # --- Full 2D coefficient matrix ---
        self.coefficients = nn.Parameter(torch.full((library_dim, state_dim),
                                                    coefficient_init_val,
                                                    dtype=torch.float32,
                                                    device=device))
        # Initialize with fixed values where required
        self.coefficients.data[self.fixed_mask] = self.fixed_values[self.fixed_mask]

        # --- Hook to mask gradients at fixed locations ---
        def mask_grad(grad):
            return grad * (~self.fixed_mask).float()

        self.coefficients.register_hook(mask_grad)

        # Optional neural network for theta
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
                                      model=self.model)

    def make_theta_network(self, output_dim, widths):
        layers = []
        input_dim = self.state_dim
        for width in widths:
            layers.append(nn.Linear(input_dim, int(width * input_dim)))
            layers.append(nn.ELU())
            input_dim = int(width * input_dim)
        layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)

    def sindy_library(self, z, latent_dim, poly_order, include_sine=False, exact_features=False, model='lorenz'):
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
        if self.rfe_threshold is not None:
            with torch.no_grad():
                new_mask = (self.coefficients.abs() > self.rfe_threshold)
                self.coefficients_mask = new_mask.float()
                self.coefficients.data *= new_mask.float()  # Zero out small entries

                # Apply fixed constraint again
                self.coefficients.data[self.fixed_mask] = self.fixed_values[self.fixed_mask]

            # Replace existing hook with a new one
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

    
# This class is the main autoencoder model that combines the encoder, decoder, and SINDy model. Used with and SVD input.
class PreSVD_Sindy_Autoencoder(nn.Module):
    def __init__(self, params, hankel=None):
        super(PreSVD_Sindy_Autoencoder, self).__init__()
        self.params = params
        self.latent_dim = params['latent_dim']
        self.input_dim = params['input_dim']
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
        self.encoder = self.make_network(self.input_dim, self.latent_dim, params['widths'], name='encoder')
        self.decoder = self.make_network(self.latent_dim, self.svd_dim, list(reversed(params['widths'])), name='decoder')
        # Optional fixed mask and values for constraining SINDy coefficients
        fixed_mask = params.get("sindy_fixed_mask", None)
        fixed_values = params.get("sindy_fixed_values", None)

        # Create SINDy module with optional constraints
        self.sindy = Sindy(
            library_dim=self.library_dim,
            state_dim=self.latent_dim,
            poly_order=self.poly_order,
            model=params['model'],
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


    def make_network(self, input_dim, output_dim, widths, name):
        layers = []
        in_dim = input_dim
        for i, w in enumerate(widths):
            layers.append(nn.Linear(in_dim, w, bias=self.use_bias))
            layers.append(nn.ReLU())  # Change to whatever activation function is used
            in_dim = w
        layers.append(nn.Linear(in_dim, output_dim, bias=self.use_bias))
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


    @torch.no_grad()  # Disable gradient computation
    def test_step(self, data):
        inputs, outputs = data
        x, v, dv_dt = inputs
        x_out, v_out, dv_dt_out = outputs
        
        dv_dt = dv_dt.unsqueeze(2)  # Match TensorFlow expand_dims
        dv_dt_out = dv_dt_out.unsqueeze(2)

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

    def get_loss(self, x, v, dv_dt, x_out, v_out, dv_dt_out):
        losses = {}
        loss = 0.0

        # --- Forward pass ---
        v.requires_grad_(True)
        z = self.encoder(v)                     # Latent space [B, d_z]
        vh = self.decoder(z)                   # Reconstructed input [B, d_v]
        ref = self.hankel.to(z.device) if self.hankel is not None else x

        # --- Reconstruction loss: vh vs v_out ---
        loss_rec = F.mse_loss(vh, v_out)
        loss += self.params['loss_weight_rec'] * loss_rec
        losses['rec'] = loss_rec

        # --- SINDy prediction from latent ---
        if self.params['loss_weight_sindy_z'] > 0.0 or self.params['loss_weight_sindy_x'] > 0.0:
            z_dot_sindy = self.sindy(z)  # [B, d_z]

        # --- Ż loss: Automatic diff through encoder ---
        if self.params['loss_weight_sindy_z'] > 0.0:
            batch_size, latent_dim = z.shape
            z_dot_auto = torch.zeros_like(z)

            for i in range(latent_dim):
                grads = torch.autograd.grad(
                    outputs=z[:, i],
                    inputs=v,
                    grad_outputs=torch.ones(batch_size, device=v.device),
                    create_graph=True,
                    retain_graph=True
                )[0]  # shape: [batch_size, input_dim]
                
                z_dot_auto[:, i] = torch.sum(grads * dv_dt, dim=1)
            loss_dz = F.mse_loss(z_dot_auto, z_dot_sindy)
            loss += self.params['loss_weight_sindy_z'] * loss_dz
            losses['sindy_z'] = loss_dz

        # --- V̇ loss: Automatic diff through decoder ---
        if self.params['loss_weight_sindy_x'] > 0.0:
            z.requires_grad_(True)
            batch_size, input_dim = vh.shape
            v_dot_auto = torch.zeros_like(vh)

            for i in range(input_dim):
                grads = torch.autograd.grad(
                    outputs=vh[:, i],
                    inputs=z,
                    grad_outputs=torch.ones(batch_size, device=z.device),
                    create_graph=True,
                    retain_graph=True
                )[0]  # shape: [batch_size, latent_dim]

                v_dot_auto[:, i] = torch.sum(grads * z_dot_sindy, dim=1)
            loss_dx = F.mse_loss(v_dot_auto, dv_dt_out)
            loss += self.params['loss_weight_sindy_x'] * loss_dx
            losses['sindy_x'] = loss_dx

        # --- x0 loss: match first time step in latent ---
        if self.params['loss_weight_x0'] > 0.0:
            loss_x0 = F.mse_loss(z[:, 0], ref[:, 0])
            loss += self.params['loss_weight_x0'] * loss_x0
            losses['x0'] = loss_x0

        # --- SINDy consistency loss (RK4 integration) ---
        if self.params['loss_weight_integral'] > 0.0:
            sol = z
            loss_int = (sol[:, 0] - x[:, 0]).pow(2)
            for _ in range(len(self.time) - 1):
                k1 = self.sindy(sol)
                k2 = self.sindy(sol + self.dt / 2 * k1)
                k3 = self.sindy(sol + self.dt / 2 * k2)
                k4 = self.sindy(sol + self.dt * k3)
                sol = sol + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
                sol = torch.where(torch.abs(sol) > 500.0, torch.tensor(50.0, device=sol.device), sol)
                loss_int += (sol[:, 0] - ref[:, 1]).pow(2)

            loss += self.params['loss_weight_integral'] * loss_int.mean()
            losses['integral'] = loss_int.mean()

        # --- L1 regularization on SINDy coefficients ---
        loss_l1 = self.sindy.coefficients.abs().mean()
        loss += self.params['loss_weight_sindy_regularization'] * loss_l1
        losses['l1'] = loss_l1

        return loss, losses

    def forward(self, x):
        return self.decoder(self.encoder(x))
    

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
        
        self.sindy = Sindy(self.library_dim, self.latent_dim, self.poly_order, model=params['model'],
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
            layers.append(nn.ReLU())
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
