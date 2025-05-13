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

    def evaluate_trajectory_error(self):
        model = self.trainer.model
        synth_data = self.data
        device = self.device

        # --- True latent dynamics from simulation ---
        z_true = torch.tensor(synth_data.z, dtype=torch.float32).to(device)
        x_true = torch.tensor(synth_data.x, dtype=torch.float32).to(device)

        # --- Latent encoding of input ---
        x_input = x_true
        z_latent = model.encoder(x_input)

        # --- Safety check ---
        if torch.isnan(z_latent).any():
            print("❌ NaNs in z_latent! Returning inf.")
            return float('inf')

        # --- Truncate to matching length if needed ---
        min_len = min(z_latent.shape[0], z_true.shape[0])
        z_latent = z_latent[:min_len]
        z_true = z_true[:min_len]

        # --- Compute latent MSE ---
        latent_mse = F.mse_loss(z_latent, z_true)

        return latent_mse.item()

    def objective(self, trial):
        # --- Fixed params from your default ---
        trial_params = {
            'model': 'lorenz',
            'case': 'synthetic_lorenz',
            'input_dim': 7,
            'latent_dim': 3,
            'poly_order': 2,
            'include_sine': False,
            'fix_coefs': False,
            'svd_dim': 7,
            'delay_embedded': True,
            'scale': True,
            'coefficient_initialization': 'constant',
            'coefficient_initialization_constant': 0.0,
            'widths_ratios': [1.0, 6/7, 5/6, 4/5, 3/4],
            'max_epochs': 1000,
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
            'rfe_frequency': 10,
            'sindy_print_rate': 10,
            'loss_print_rate': 10,
        }

        # --- Fixed SINDy values ---
        trial_params["sindy_fixed_values"] = np.array([
            [0.,  0.,  0.],
            [0.,  0.,  0.],
            [0., -1.,  0.],
            [0.,  0.,  0.],
            [0.,  0.,  0.],
            [0.,  0.,  0.],
            [0., -1.,  0.],
            [0.,  0.,  0.],
            [0.,  0.,  0.],
            [0.,  0.,  0.],
        ], dtype=np.float32)

        # --- Choose between constrained and unconstrained SINDy mask ---
        mask_options = [
            np.array([
                [False, False, False],
                [False, False, False],
                [False,  True, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False,  True, False],
                [ True,  True,  True],
                [ True,  True,  True],
                [ True,  True,  True],
            ], dtype=bool),
            np.zeros((10, 3), dtype=bool)  # All False
        ]
        mask_choice = trial.suggest_categorical("sindy_mask_type", [0, 1])
        trial_params["sindy_fixed_mask"] = mask_options[mask_choice]

        # --- Dynamic hyperparameter search ---
        trial_params['loss_weight_rec'] = trial.suggest_float("loss_weight_rec", 1e-4, 1.0, log=True)
        trial_params['loss_weight_sindy_z'] = trial.suggest_float("loss_weight_sindy_z", 1e-6, 1e-1, log=True)
        trial_params['loss_weight_sindy_x'] = trial.suggest_float("loss_weight_sindy_x", 1e-6, 1e-1, log=True)
        trial_params['loss_weight_sindy_regularization'] = trial.suggest_float("loss_weight_sindy_regularization", 1e-6, 1e-1, log=True)
        trial_params['loss_weight_integral'] = trial.suggest_float("loss_weight_integral", 1e-4, 1.0, log=True)
        trial_params['loss_weight_x0'] = trial.suggest_float("loss_weight_x0", 1e-4, 1.0, log=True)
        trial_params['loss_weight_layer_l2'] = trial.suggest_float("loss_weight_layer_l2", 0.0, 1e-2)
        trial_params['loss_weight_layer_l1'] = trial.suggest_float("loss_weight_layer_l1", 0.0, 1e-2)
        trial_params['learning_rate'] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        trial_params['lr_decay'] = trial.suggest_float("lr_decay", 0.99, 1.0)
        trial_params['batch_size'] = trial.suggest_categorical("batch_size", [128, 256, 512])
        trial_params['coefficient_threshold'] = trial.suggest_float("coefficient_threshold", 1e-6, 1e-4, log=True)

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
        self.trainer = trainer  # for evaluate_trajectory_error()
        self.params = trial_params
        self.data = synth_data
        self.device = trainer.device
        trainer.fit()

        # Save model and metadata
        self.models[trial.number] = trainer.model
        trial.set_user_attr("sindy_coefficients", trainer.model.sindy.coefficients.detach().cpu().numpy().tolist())
        trial.set_user_attr("latent_dim", trainer.model.latent_dim)
        trial.set_user_attr("poly_order", trainer.model.poly_order)

        return self.evaluate_trajectory_error()
    
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