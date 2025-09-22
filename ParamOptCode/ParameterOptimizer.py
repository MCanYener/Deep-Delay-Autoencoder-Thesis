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

        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage_path,
            direction="minimize",
            load_if_exists=True
        )

    def evaluate_static_loss(self, trainer):
        history = trainer.history
        latest = {k: history[k][-1] if len(history[k]) > 0 else 0.0 for k in history}

        weights = {
            "train_rec": 1.0,
            "train_sindy_z": 1e-2,
            "train_sindy_x": 1e-2,
            "train_integral": 1e-1,
            "train_x0": 1e-2,
            "train_l1": 1e-2,
        }

        return sum(w * latest.get(k, 0.0) for k, w in weights.items())

    def objective(self, trial):
        # === Fixed Parameters ===
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
            'exact_features': True,
            'use_bias': True,
            'train_ratio': 0.8,
            'data_path': './trained_models',
            'save_checkpoints': False,
            'save_freq': 5,
            'learning_rate_sched': False,
            'use_sindycall': False,
            'rfe_frequency': 10,
            'sindy_print_rate': 10,
            'loss_print_rate': 10,
            'batch_size': 128,
        }

        # === SINDy constraints ===
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

        mask_options = [
            np.array([
                [True, True, True],
                [False, False, False],
                [False, True, False],
                [False, False, False],
                [True, True, True],
                [False, False, False],
                [False, True, False],
                [True, True, True],
                [False, False, False],
                [True, True, True],
            ], dtype=bool),
            np.zeros((10, 3), dtype=bool)
        ]
        trial_params["sindy_fixed_mask"] = mask_options[trial.suggest_categorical("sindy_mask_type", [0, 1])]

        # === Tuned parameters ===
        trial_params.update({
            'loss_weight_rec': trial.suggest_float("loss_weight_rec", 1e-4, 1.0, log=True),
            'loss_weight_sindy_z': trial.suggest_float("loss_weight_sindy_z", 1e-6, 1e-1, log=True),
            'loss_weight_sindy_x': trial.suggest_float("loss_weight_sindy_x", 1e-6, 1e-1, log=True),
            'loss_weight_sindy_regularization': trial.suggest_float("loss_weight_sindy_regularization", 1e-6, 1e-1, log=True),
            'loss_weight_integral': trial.suggest_float("loss_weight_integral", 1e-4, 1.0, log=True),
            'loss_weight_sindy_x0': trial.suggest_float("loss_weight_sindy_x0", 1e-4, 1.0, log=True),
            'loss_weight_x0': trial.suggest_float("loss_weight_x0", 1e-4, 1.0, log=True),
            'loss_weight_layer_l2': trial.suggest_float("loss_weight_layer_l2", 0.0, 1e-2),
            'loss_weight_layer_l1': trial.suggest_float("loss_weight_layer_l1", 0.0, 1e-2),
            'loss_weight_sindy_group_l1': trial.suggest_float("loss_weight_sindy_group_l1", 1e-7, 1e-3, log=True),
            'sindy_l1_ramp_start': trial.suggest_int("sindy_l1_ramp_start", 0, 500),
            'sindy_l1_ramp_duration': trial.suggest_int("sindy_l1_ramp_duration", 100, 1000),
            'learning_rate': trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            'ae_learning_rate': trial.suggest_float("ae_learning_rate", 1e-5, 1e-2, log=True),
            'sindy_learning_rate': trial.suggest_float("sindy_learning_rate", 1e-5, 1e-2, log=True),
            'lr_decay_ae': trial.suggest_float("lr_decay_ae", 0.98, 1.0),
            'lr_decay_sindy': trial.suggest_float("lr_decay_sindy", 0.98, 1.0),
            'coefficient_threshold': trial.suggest_float("coefficient_threshold", 1e-4, 1e-2, log=True),
            'ode_net': trial.suggest_categorical("ode_net", [True, False]),
        })

        # === Synthetic Data ===
        synth_data = SynthData(input_dim=7)
        synth_data.run_sim(tend=32, dt=0.01, z0=np.array([1.0, 1.0, 1.0]), apply_svd=False)
        trial_params['dt'] = 0.01
        trial_params['tend'] = synth_data.t[-1]
        trial_params['n_ics'] = 1
        trial_params['input_dim'] = trial_params['svd_dim']

        trainer = TrainModel(synth_data, trial_params)
        trainer.fit()

        trial.set_user_attr("sindy_coefficients", trainer.model.sindy.coefficients.detach().cpu().numpy().tolist())
        trial.set_user_attr("latent_dim", trainer.model.latent_dim)
        trial.set_user_attr("poly_order", trainer.model.poly_order)

        return self.evaluate_static_loss(trainer)

    def print_top_trials(self, top_k=5):
        print(f"\nTop {top_k} Trials by Static Evaluation Loss:")
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        top_trials = sorted(completed_trials, key=lambda t: t.value)[:top_k]

        for i, trial in enumerate(top_trials, 1):
            print(f"\n📌 Trial {i} (Global Trial #{trial.number}):")
            print(f"  Static Eval Loss: {trial.value:.6f}")
            print("  Params:")
            for key, val in trial.params.items():
                print(f"    {key}: {val}")

            print("\n  SINDy Model:")
            coefs = np.array(trial.user_attrs["sindy_coefficients"])
            latent_dim = trial.user_attrs["latent_dim"]
            poly_order = trial.user_attrs["poly_order"]
            basis = ['1'] + [f'z{i}' for i in range(latent_dim)]
            if poly_order >= 2:
                basis += [f'z{i}z{j}' for i in range(latent_dim) for j in range(i, latent_dim)]

            for eq in range(coefs.shape[1]):
                terms = [f"{coefs[i, eq]:.4f} * {basis[i]}" for i in range(len(basis)) if abs(coefs[i, eq]) > 1e-5]
                equation = f"    dz{eq}/dt = " + " + ".join(terms) if terms else f"    dz{eq}/dt = 0"
                print(equation)

    def run(self, top_k=5):
        self.study.optimize(self.objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
        self.print_top_trials(top_k=top_k)
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            raise ValueError("No completed trials to select the best from.")
        return sorted(completed_trials, key=lambda t: t.value)[0]