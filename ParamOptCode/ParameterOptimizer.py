import optuna
import copy
import torch
import numpy as np
import torch.nn.functional as F

from ParamOptCode.DefaultParams import params as default_params
from DataGenerationCode.DataGeneration import LorenzSystem
from AdditionalFunctionsCode.SolverFunctions import SynthData
from TrainingCode.TrainingCode import TrainModel


class HyperparameterOptimizer:
    def __init__(self, n_trials=50, n_jobs=1, study_name="sindy_opt", storage_path="sqlite:///optuna_study.db"):
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.models = {}

        # Optuna parallelization with SQLite storage
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage_path,
            direction="minimize",
            load_if_exists=True
        )

    def evaluate_static_loss(self, trainer, synth_data, 
                             weight_rec=1.0, 
                             weight_sindy_z=0.1, 
                             weight_sindy_x=0.1, 
                             weight_integral=0.1, 
                             weight_x0=0.1):
        device = trainer.device
        x = torch.tensor(synth_data.x, dtype=torch.float32).to(device)
        v = x
        dv_dt = torch.tensor(synth_data.dx, dtype=torch.float32).to(device)
        v.requires_grad_(True)

        # Forward pass
        z = trainer.model.encoder(v)
        vh = trainer.model.decoder(z)
        rec_loss = F.mse_loss(vh, v)

        # ż loss
        z_dot_sindy = trainer.model.sindy(z)
        batch_size, latent_dim = z.shape
        z_dot_auto = torch.zeros_like(z)
        for i in range(latent_dim):
            grads = torch.autograd.grad(
                outputs=z[:, i],
                inputs=v,
                grad_outputs=torch.ones(batch_size, device=v.device),
                create_graph=True,
                retain_graph=True
            )[0]
            z_dot_auto[:, i] = torch.sum(grads * dv_dt, dim=1)
        sindy_z_loss = F.mse_loss(z_dot_auto, z_dot_sindy)

        # v̇ loss
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
            )[0]
            v_dot_auto[:, i] = torch.sum(grads * z_dot_sindy, dim=1)
        sindy_x_loss = F.mse_loss(v_dot_auto, dv_dt)

        # x₀ loss
        x0_loss = F.mse_loss(z[:, 0], x[:, 0])

        # RK4 consistency loss
        sol = z
        loss_int = (sol[:, 0] - x[:, 0]).pow(2)
        for _ in range(len(trainer.model.time) - 1):
            k1 = trainer.model.sindy(sol)
            k2 = trainer.model.sindy(sol + trainer.model.dt / 2 * k1)
            k3 = trainer.model.sindy(sol + trainer.model.dt / 2 * k2)
            k4 = trainer.model.sindy(sol + trainer.model.dt * k3)
            sol = sol + (trainer.model.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            sol = torch.where(torch.abs(sol) > 500.0, torch.tensor(50.0, device=sol.device), sol)
            loss_int += (sol[:, 0] - x[:, 1]).pow(2)
        integral_loss = loss_int.mean()

        # Total loss using static weights
        total_loss = (
            weight_rec * rec_loss +
            weight_sindy_z * sindy_z_loss +
            weight_sindy_x * sindy_x_loss +
            weight_integral * integral_loss +
            weight_x0 * x0_loss
        )

        return total_loss.item()

    def objective(self, trial):
        # Start fresh
        trial_params = {}

        # --- Fixed params from your default ---
        trial_params.update({
            'model': 'lorenz',
            'case': 'synthetic_lorenz',
            'input_dim': 7,  # usually overwritten outside before calling
            'latent_dim': 3,
            'poly_order': 2,
            'include_sine': False,
            'fix_coefs': False,
            'svd_dim': 7,
            'delay_embedded': True,
            'scale': True,
            'coefficient_initialization': 'constant',
            'coefficient_initialization_constant': 1e-2,
            'widths_ratios': [1.0, 0.6],
            'max_epochs': 500,
            'patience': 20,
            'threshold_frequency': 5,
            'print_frequency': 10,
            'sindy_pert': 0.0,
            'ode_net': False,
            'ode_net_widths': [1.5, 2.0],
            'exact_features': True,
            'use_bias': True,
            'train_ratio': 0.8,
            'data_path': './trained_models',
            'save_checkpoints': False,
            'save_freq': 5,
            'learning_rate_sched': False,
            'use_sindycall': False,
            'sparse_weighting': None,
            'system_coefficients': None,
        })

        # --- Dynamic search parameters ---
        trial_params['loss_weight_rec'] = trial.suggest_float("loss_weight_rec", 1e-3, 1.0, log=True)
        trial_params['loss_weight_sindy_z'] = trial.suggest_float("loss_weight_sindy_z", 1e-6, 1e-1, log=True)
        trial_params['loss_weight_sindy_x'] = trial.suggest_float("loss_weight_sindy_x", 1e-6, 1e-1, log=True)
        trial_params['loss_weight_sindy_regularization'] = trial.suggest_float("loss_weight_sindy_regularization", 1e-8, 1e-2, log=True)
        trial_params['loss_weight_integral'] = trial.suggest_float("loss_weight_integral", 1e-3, 1.0, log=True)
        trial_params['loss_weight_x0'] = trial.suggest_float("loss_weight_x0", 1e-4, 1.0, log=True)
        trial_params['loss_weight_layer_l2'] = trial.suggest_float("loss_weight_layer_l2", 0.0, 1e-2)
        trial_params['loss_weight_layer_l1'] = trial.suggest_float("loss_weight_layer_l1", 0.0, 1e-2)
        trial_params['learning_rate'] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        trial_params['lr_decay'] = trial.suggest_float("lr_decay", 0.99, 1.0)
        trial_params['batch_size'] = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
        trial_params['coefficient_threshold'] = trial.suggest_float("coefficient_threshold", 1e-6, 1e-3, log=True)

        # --- Data generation ---
        lorenz = LorenzSystem()
        synth_data = SynthData(input_dim=7)
        z0 = np.array([1.0, 1.0, 1.0])
        synth_data.run_sim(tend=32, dt=0.01, z0=z0, apply_svd=False)

        trial_params['dt'] = 0.01
        trial_params['tend'] = synth_data.t[-1]
        trial_params['n_ics'] = 1

        trial_params['input_dim'] = trial_params['svd_dim']

        # --- Train ---
        trainer = TrainModel(synth_data, trial_params)
        trainer.fit()

        # Save the model and coefficients
        self.models[trial.number] = trainer.model
        trial.set_user_attr("sindy_coefficients", trainer.model.sindy.coefficients.detach().cpu().numpy().tolist())
        trial.set_user_attr("latent_dim", trainer.model.latent_dim)
        trial.set_user_attr("poly_order", trainer.model.poly_order)

        # --- Evaluate static loss ---
        return self.evaluate_static_loss(trainer, synth_data)
    
    def print_top_trials(self, top_k=5):

        print(f"\nTop {top_k} Trials by Static Loss:")
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        top_trials = sorted(completed_trials, key=lambda t: t.value)[:top_k]
        
        for i, trial in enumerate(top_trials, 1):
            print(f"\n📌 Trial {i} (Global Trial #{trial.number}):")
            print(f"  Static Loss: {trial.value:.6f}")
            print("  Params:")
            for key, val in trial.params.items():
                print(f"    {key}: {val}")

            # Print SINDy model
            print("\n  SINDy Model:")
            coefs = np.array(trial.user_attrs["sindy_coefficients"])
            latent_dim = trial.user_attrs["latent_dim"]
            poly_order = trial.user_attrs["poly_order"]

            basis = ['1'] + [f'z{i}' for i in range(latent_dim)]
            if poly_order >= 2:
                basis += [f'z{i}z{j}' for i in range(latent_dim) for j in range(i, latent_dim)]

            library_size, dim = coefs.shape

            for eq in range(dim):
                terms = [f"{coefs[i, eq]:.4f} * {basis[i]}" for i in range(library_size) if abs(coefs[i, eq]) > 1e-5]
                equation = f"    dz{eq}/dt = " + " + ".join(terms) if terms else f"    dz{eq}/dt = 0"
                print(equation)


    def run(self, top_k=5):
        self.study.optimize(self.objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
        self.print_top_trials(top_k=top_k)
        
        # Only consider completed trials
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            raise ValueError("No completed trials to select the best from.")
        
        return sorted(completed_trials, key=lambda t: t.value)[0]  # Return best completed trial