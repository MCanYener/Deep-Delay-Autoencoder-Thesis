import torch
import numpy as np
import matplotlib.pyplot as plt
from TrainingCode.TrainingCode import TrainModel
from AdditionalFunctionsCode.SolverFunctions import SynthData, get_hankel_svd
from DataGenerationCode.DataGeneration import LorenzSystem
from ParamOptCode.DefaultParams import params as default_params
import pickle
import optuna


class TrialEvaluator:
    def __init__(self, trial_params, tend=32, dt=0.01):
        self.params = trial_params.copy()
        self.model_tag = self.params.get("model_tag", "lorenz")  # Default to Lorenz
        self.tend = tend
        self.dt = dt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = None
        self.trainer = None

        self._set_model_defaults()  # <-- Determine input_dim etc.
        self._prepare_params()
        self._generate_data()

    def _set_model_defaults(self):
        if self.model_tag == "lorenz":
            self.input_dim = 7
            self.svd_dim = 7
            self.latent_dim = 3
        elif self.model_tag == "glyco":
            self.input_dim = 5
            self.svd_dim = 5
            self.latent_dim = 2
        else:
            raise ValueError(f"Unknown model tag: {self.model_tag}")

    def _prepare_params(self):
        # ---- core dims first (from _set_model_defaults) ----
        self.params["input_dim"]  = self.input_dim
        self.params["svd_dim"]    = self.svd_dim
        self.params["latent_dim"] = self.latent_dim

        # ---- resolve widths BEFORE writing them back to params ----
        if self.model_tag == "lorenz":
            default_w1 = [6, 5, 4]
            default_w2 = [10, 8, 6, 4]
        elif self.model_tag == "glyco":
            default_w1 = [4, 3]
            default_w2 = [8, 6, 4]
        else:
            raise ValueError(f"Unknown model tag: {self.model_tag}")

        # pull from incoming params if present, else use defaults
        w1 = self.params.get("widths_1d", default_w1)
        w2 = self.params.get("widths_2d", default_w2)

        # set instance attributes first so they exist
        self.widths_1d = w1
        self.widths_2d = w2

        # now mirror into params (safe now)
        self.params["widths_1d"] = self.widths_1d
        self.params["widths_2d"] = self.widths_2d

        # ---- timing / paths / observable config ----
        self.params["max_epochs"] = self.params.get("max_epochs", 1000)
        self.params["dt"]         = self.dt
        self.params["tend"]       = self.tend
        self.params["data_path"]  = self.params.get("data_path", "./trained_models")

        self.observable_dim = self.params.get("observable_dim", 1)
        self.params["observable_dim"] = self.observable_dim

        # ---- training / data defaults ----
        self.params["train_ratio"]         = self.params.get("train_ratio", 0.8)
        self.params["batch_size"]          = self.params.get("batch_size", 128)
        self.params["learning_rate"]       = self.params.get("learning_rate", 1e-3)
        self.params["sindy_learning_rate"] = self.params.get("sindy_learning_rate", 1e-4)
        self.params["lr_decay_ae"]         = self.params.get("lr_decay_ae", 0.999)
        self.params["lr_decay_sindy"]      = self.params.get("lr_decay_sindy", 0.999)

        self.params["train_sindy"]          = self.params.get("train_sindy", True)
        self.params["sindy_train_start"]    = self.params.get("sindy_train_start", 0)
        self.params["sindy_ramp_duration"]  = self.params.get("sindy_ramp_duration", 500)
        self.params["sindy_l1_ramp_start"]  = self.params.get("sindy_l1_ramp_start", 0)
        self.params["sindy_l1_ramp_duration"] = self.params.get("sindy_l1_ramp_duration", 500)

        # ---- windowing (sequence mode) ----
        self.params["window_T"]      = self.params.get("window_T", 20)
        self.params["window_stride"] = self.params.get("window_stride", 1)

        # ---- loss weights ----
        self.params["loss_weight_rec"]                   = self.params.get("loss_weight_rec", 1.0)
        self.params["loss_weight_sindy_z"]               = self.params.get("loss_weight_sindy_z", 0.0)
        self.params["loss_weight_sindy_x"]               = self.params.get("loss_weight_sindy_x", 0.0)
        self.params["loss_weight_integral"]              = self.params.get("loss_weight_integral", 1.0)
        self.params["loss_weight_x0"]                    = self.params.get("loss_weight_x0", 0.0)
        self.params["loss_weight_sindy_regularization"]  = self.params.get("loss_weight_sindy_regularization", 1e-4)
        self.params["loss_weight_sindy_group_l1"]        = self.params.get("loss_weight_sindy_group_l1", 0.0)

        # ---- callbacks ----
        self.params["use_sindycall"] = self.params.get("use_sindycall", True)
        self.params["update_freq"]   = self.params.get("update_freq", 10)
        self.params["rfe_frequency"] = self.params.get("rfe_frequency", 10)
        self.params["sindy_print_rate"] = self.params.get("sindy_print_rate", 10)

        # ---- n_ics as int ----
        self.n_ics = int(self.params.get("n_ics", 1))

    def _generate_data(self):
        synth_data = SynthData(
            input_dim=self.input_dim,
            model_tag=self.model_tag,
            params=self.params,
            observable_dim=self.observable_dim
        )
        synth_data.run_multi_sim(
            n_trajectories=self.n_ics,
            tend=self.tend,
            dt=self.dt,
            apply_svd=False  # matches your training path
        )
        self.data = synth_data

    def train_model(self):
        self.trainer = TrainModel(self.data, self.params, device=self.device)
        self.trainer.model.sindy.print_trainability_report("(Before Training)")
        print(f"Input Dim: {self.input_dim}, Latent Dim: {self.latent_dim}, SVD Dim: {self.svd_dim}")
        print(f"⚙️ model_tag in TrialEvaluator: {self.model_tag}")
        print(f"⚙️ model_tag in SynthData: {self.data.model_tag}")
        print(f"⚙️ model_tag in Sindy: {self.trainer.model.sindy.model_tag}")
        self.trainer.fit()

    def evaluate_and_plot(self):
        model = self.trainer.model
        synth_data = self.data
        device = self.device
        observable_dim = getattr(model, "observable_dim", 1)

        # === Setup ===
        with torch.no_grad():
            x_raw = torch.tensor(synth_data.x, dtype=torch.float32, device=device)
            z_latent = model.encoder(x_raw).cpu().numpy()
            x_reconstructed = model(x_raw).cpu().numpy()
        ground_truth = synth_data.z[:z_latent.shape[0]]

        num_traj = self.params['n_ics']
        traj_len = z_latent.shape[0] // num_traj

        # === Reconstruction plot (global) ===
        x_reconstructed = model(x_raw).detach().cpu().numpy()
        plt.figure(figsize=(10, 4))
        if observable_dim >= 1:
            plt.subplot(1, 2 if observable_dim == 2 else 1, 1)
            plt.plot(synth_data.x[:, 0], label="Original x")
            plt.plot(x_reconstructed[:, 0], label="Reconstructed x")
            plt.title("Observable x")
            plt.xlabel("Time step")
            plt.grid(True)
            plt.legend()

        if observable_dim == 2:
            plt.subplot(1, 2, 2)
            plt.plot(synth_data.x[:, 1], label="Original y")
            plt.plot(x_reconstructed[:, 1], label="Reconstructed y")
            plt.title("Observable y")
            plt.xlabel("Time step")
            plt.grid(True)
            plt.legend()

        plt.suptitle("Reconstruction vs Original (no SVD)")
        plt.tight_layout()
        plt.show()

        # === Latent space plots (per trajectory) ===
        for i in range(num_traj):
            start = i * traj_len
            end = (i + 1) * traj_len

            fig = plt.figure(figsize=(14, 6))
            if self.latent_dim == 3:
                ax1 = fig.add_subplot(1, 2, 1, projection='3d')
                ax1.plot(ground_truth[start:end, 0], ground_truth[start:end, 1], ground_truth[start:end, 2], color='blue')
                ax1.set_title(f"True Trajectory {i}")

                ax2 = fig.add_subplot(1, 2, 2, projection='3d')
                ax2.plot(z_latent[start:end, 0], z_latent[start:end, 1], z_latent[start:end, 2], color='purple')
                ax2.set_title(f"Latent Trajectory {i}")

            elif self.latent_dim == 2:
                ax1 = fig.add_subplot(1, 2, 1)
                ax1.plot(ground_truth[start:end, 0], ground_truth[start:end, 1], color='blue')
                ax1.set_title(f"True Trajectory {i}")

                ax2 = fig.add_subplot(1, 2, 2)
                ax2.plot(z_latent[start:end, 0], z_latent[start:end, 1], color='purple')
                ax2.set_title(f"Latent Trajectory {i}")

            else:
                print(f"[Warning] Latent dim = {self.latent_dim} not supported for trajectory plotting.")
            plt.tight_layout()
            plt.show()

        # === Simulate SINDy dynamics per trajectory ===
        for i in range(num_traj):
            print(f"\nSimulating SINDy model from latent z0 of Trajectory {i}...")
            start = i * traj_len
            end = (i + 1) * traj_len

            with torch.no_grad():
                z0 = torch.tensor(z_latent[start], dtype=torch.float32).to(device)
                z_sim = self.trainer.simulate_sindy_trajectory(z0, timesteps=traj_len, dt=self.params["dt"])
                z_sim_np = z_sim.detach().cpu().numpy()
                x_sim = model.decoder(z_sim.to(device)).cpu().numpy()

            # --- Plot SINDy-decoded observable vs true ---
            if observable_dim >= 1:
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 2 if observable_dim == 2 else 1, 1)
                plt.plot(synth_data.x[start:end, 0], label="True x")
                plt.plot(x_sim[:, 0], label="SINDy Decoded x", linestyle="--")
                plt.title(f"Observable x – True vs SINDy (Trajectory {i})")
                plt.xlabel("Time step")
                plt.grid(True)
                plt.legend()

                if observable_dim == 2:
                    plt.subplot(1, 2, 2)
                    plt.plot(synth_data.x[start:end, 1], label="True y")
                    plt.plot(x_sim[:, 1], label="SINDy Decoded y", linestyle="--")
                    plt.title(f"Observable y – True vs SINDy (Trajectory {i})")
                    plt.xlabel("Time step")
                    plt.grid(True)
                    plt.legend()

                plt.suptitle(f"Reconstruction from SINDy Latents (Trajectory {i})")
                plt.tight_layout()
                plt.show()

            # --- Plot SINDy-simulated vs. true latent ---
            plt.figure(figsize=(10, 4))
            plt.plot(z_latent[start:end, 0], label="True Latent z[0]")
            plt.plot(z_sim_np[:, 0], label="SINDy Simulated z[0]", linestyle="--")
            plt.legend()
            plt.title(f"Latent z[0] – True vs SINDy (Trajectory {i})")
            plt.xlabel("Time step")
            plt.ylabel("z[0] value")
            plt.grid(True)
            plt.show()

            # --- 3D or 2D phase plot ---
            fig = plt.figure(figsize=(14, 6))
            if self.latent_dim == 3:
                ax1 = fig.add_subplot(1, 2, 1, projection='3d')
                ax1.plot(z_latent[start:end, 0], z_latent[start:end, 1], z_latent[start:end, 2], color='blue')
                ax1.set_title(f"True Latent Trajectory {i}")

                ax2 = fig.add_subplot(1, 2, 2, projection='3d')
                ax2.plot(z_sim_np[:, 0], z_sim_np[:, 1], z_sim_np[:, 2], color='green')
                ax2.set_title(f"SINDy Simulated Trajectory {i}")

            elif self.latent_dim == 2:
                ax1 = fig.add_subplot(1, 2, 1)
                ax1.plot(z_latent[start:end, 0], z_latent[start:end, 1], color='blue')
                ax1.set_title(f"True Latent Trajectory {i}")

                ax2 = fig.add_subplot(1, 2, 2)
                ax2.plot(z_sim_np[:, 0], z_sim_np[:, 1], color='green')
                ax2.set_title(f"SINDy Simulated Trajectory {i}")

            else:
                print(f"[Warning] Latent dim = {self.latent_dim} not supported for plotting.")
            plt.tight_layout()
            plt.show()

class TopTrialSelector:
    def __init__(self, db_path, study_name, top_k=5):
        """
        Load the top-k successful trials from a given Optuna study/database.

        Args:
            db_path (str): Path to the SQLite database (e.g., "Databases/merged_optuna.db")
            study_name (str): Name of the Optuna study (e.g., "sindy_opt_merged")
            top_k (int): Number of top trials to extract
        """
        self.db_path = db_path
        self.study_name = study_name
        self.top_k = top_k
        self.study = self._load_study()
        self.top_trials = self._get_top_trials()

    def _load_study(self):
        print(f"📂 Loading study: {self.study_name} from {self.db_path}")
        storage_str = f"sqlite:///{self.db_path}"
        return optuna.load_study(study_name=self.study_name, storage=storage_str)

    def _get_top_trials(self):
        completed = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        top_trials = sorted(completed, key=lambda t: t.value)[:self.top_k]
        print(f"✅ Loaded {len(top_trials)} top trials.")
        return top_trials

    def save_top_params(self, save_path="top_trial_params.pkl"):
        """
        Save the parameter dictionaries of the top-k trials to a pickle file.

        Args:
            save_path (str): Where to save the file
        """
        top_params = []
        for trial in self.top_trials:
            p = trial.params.copy()
            p['input_dim'] = 80  # overwrite for consistency with Hankel matrix generation
            top_params.append(p)

        with open(save_path, 'wb') as f:
            pickle.dump(top_params, f)
        print(f"💾 Saved top-{self.top_k} trial parameters to {save_path}")

    def get_trial_params(self, idx):
        if not (1 <= idx <= self.top_k):
            raise IndexError(f"Index out of bounds. Choose between 1 and {self.top_k}.")
        trial_params = self.top_trials[idx - 1].params.copy()
        trial_params['input_dim'] = 80  # overwrite as before

        # Merge with defaults
        merged = default_params.copy()
        merged.update(trial_params)  # trial values take priority
        print("🔍 Merged Parameters:\n", merged.keys())
        return merged
    
    def plot_loss_curves(history, norm=False, sindy_start = 0):
        """
        Plots the training and validation loss curves.

        Parameters:
        - history (dict): Dictionary containing training and validation loss lists.
                        Keys expected: 'train_total', 'val_total', 'train_rec', etc.
        - norm (bool): If True, normalize all losses by their first value.
        """
        keys = list(history.keys())
        unique_loss_names = sorted(set(k.split('_', 1)[1] for k in keys if k.startswith('train_')))
        n_plots = len(unique_loss_names)

        ncols, nrows = 2, int(np.ceil(n_plots / 2))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
        axes = axes.flatten()

        for idx, loss_name in enumerate(unique_loss_names):
            ax = axes[idx]
            y = np.array(history.get(f"train_{loss_name}", []), dtype=float)
            x = np.arange(len(y))

            # mask NaNs so plotting starts when data exists
            m = ~np.isnan(y)
            if m.any():
                if norm and y[m][0] != 0:
                    y[m] = y[m] / y[m][0]
                ax.plot(x[m], y[m], label="train")
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

            ax.set_yscale("log")
            ax.set_title(f"{loss_name} loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True, which="both", alpha=0.3)

            if sindy_start > 0:
                ax.axvline(sindy_start, ls="--", alpha=0.4)

            ax.legend()

        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()