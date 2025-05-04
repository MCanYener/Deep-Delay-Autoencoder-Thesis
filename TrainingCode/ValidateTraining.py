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
    def __init__(self, trial_params, svd_dim=7, input_dim=7, tend=32, dt=0.01):
        self.params = trial_params.copy()
        self.input_dim = input_dim
        self.svd_dim = svd_dim
        self.tend = tend
        self.dt = dt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = None
        self.trainer = None

        self._prepare_params()
        self._generate_data()

    def _prepare_params(self):
        self.params['input_dim'] = self.svd_dim
        self.params['svd_dim'] = self.svd_dim
        self.params['max_epochs'] = 1000 #1000
        self.params['dt'] = self.dt
        self.params['tend'] = self.tend
        self.params['n_ics'] = 1
        self.params['data_path'] = './trained_models'

    def _generate_data(self):
        lorenz = LorenzSystem()
        synth_data = SynthData(input_dim=self.input_dim)
        z0 = np.array([1.0, 1.0, 1.0])
        synth_data.run_sim(tend=self.tend, dt=self.dt, z0=z0, apply_svd=False)
        self.data = synth_data

    def train_model(self):
        self.trainer = TrainModel(self.data, self.params, device=self.device)
        self.trainer.model.sindy.print_trainability_report("(Before Training)")
        self.trainer.fit()

    def evaluate_and_plot(self):
        model = self.trainer.model
        synth_data = self.data
        device = self.device

        # Convert to tensor
        x_raw = torch.tensor(synth_data.x, dtype=torch.float32).to(device)
        
        # Reconstruct
        x_reconstructed = model(x_raw).detach().cpu().numpy()

        # --- Reconstruction plot ---
        plt.figure(figsize=(10, 4))
        plt.plot(synth_data.x[:, 0], label="Original")
        plt.plot(x_reconstructed[:, 0], label="Reconstructed")
        plt.legend()
        plt.title("Reconstruction vs Original (no SVD)")
        plt.xlabel("Time step")
        plt.ylabel("State value")
        plt.grid(True)
        plt.show()

        # --- Latent space plot ---
        z_latent = model.encoder(x_raw).detach().cpu().numpy()
        ground_truth = synth_data.z[:z_latent.shape[0]]

        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], color='blue', linewidth=1.5)
        ax1.set_title("True Lorenz Phase Space")

        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.plot(z_latent[:, 0], z_latent[:, 1], z_latent[:, 2], color='purple', linewidth=1.5)
        ax2.set_title("Latent Space Phase Portrait")
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
    
    def plot_loss_curves(history, norm=False):
        """
        Plots the training and validation loss curves.

        Parameters:
        - history (dict): Dictionary containing training and validation loss lists.
                        Keys expected: 'train_total', 'val_total', 'train_rec', etc.
        - norm (bool): If True, normalize all losses by their first value.
        """
        keys = list(history.keys())
        unique_loss_names = sorted(set(k.split('_', 1)[1] for k in keys))  # e.g. 'total', 'rec', 'z_dot'
        n_plots = len(unique_loss_names)

        ncols = 2
        nrows = int(np.ceil(n_plots / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
        axes = axes.flatten()

        for idx, loss_name in enumerate(unique_loss_names):
            ax = axes[idx]
            train_key = f"train_{loss_name}"
            val_key = f"val_{loss_name}"

            train_loss = np.array(history.get(train_key, []))
            val_loss = np.array(history.get(val_key, []))

            if norm and len(train_loss) > 0 and len(val_loss) > 0:
                train_loss = train_loss / train_loss[0] if train_loss[0] != 0 else train_loss
                val_loss = val_loss / val_loss[0] if val_loss[0] != 0 else val_loss

            ax.plot(train_loss, label="training", color="blue")
            ax.plot(val_loss, label="validation", color="orange")
            ax.set_yscale("log")
            ax.set_title(f"{loss_name} loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True)
            ax.legend()

        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()