import optuna
import os

class OptunaStudyMerger:
    def __init__(self, db_paths, merged_db_path="sqlite:///merged_optuna.db", merged_study_name="sindy_opt_merged", top_k=5):
        self.db_paths = db_paths
        self.merged_db_path = merged_db_path
        self.merged_study_name = merged_study_name
        self.top_k = top_k
        self.merged_study = optuna.create_study(
            study_name=merged_study_name,
            storage=merged_db_path,
            direction="minimize",
            load_if_exists=True
        )

    def derive_study_name(self, db_path):
        filename = os.path.basename(db_path)
        if filename == "optuna_study.db":
            return "sindy_opt"
        elif filename.startswith("optuna_study_") and filename.endswith(".db"):
            suffix = filename.replace("optuna_study_", "").replace(".db", "")
            return f"sindy_opt_{suffix}"
        else:
            return None

    def merge_top_trials(self):
        for db_path in self.db_paths:
            study_name = self.derive_study_name(db_path)
            if study_name is None:
                print(f"❌ Unrecognized filename format: {db_path}")
                continue

            storage = f"sqlite:///{db_path}"
            try:
                temp_study = optuna.load_study(study_name=study_name, storage=storage)
            except Exception as e:
                print(f"❌ Failed to load {db_path}: {e}")
                continue

            complete_trials = [t for t in temp_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            failed_trials = [t for t in temp_study.trials if t.state != optuna.trial.TrialState.COMPLETE]

            print(f"\n📁 {db_path}")
            print(f"  ✅ Completed trials: {len(complete_trials)}")
            print(f"  ❌ Failed/Pruned trials: {len(failed_trials)}")

            top_trials = sorted(complete_trials, key=lambda t: t.value)[:self.top_k]

            for trial in top_trials:
                # Create a new trial and set parameters and user attrs
                def tell_trial(trial_to_set):
                    trial_to_set.params = trial.params
                    trial_to_set.user_attrs = trial.user_attrs
                    return trial.value

                self.merged_study.enqueue_trial(trial.params, user_attrs=trial.user_attrs)
                print(f"  ✔️ Enqueued trial with loss {trial.value:.6f}")
                
    def print_top_trials(self, top_k=5):
        print(f"\n🌟 Top {top_k} Trials in Merged Study:")
        completed_trials = [t for t in self.merged_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        top_trials = sorted(completed_trials, key=lambda t: t.value)[:top_k]

        for i, trial in enumerate(top_trials, 1):
            print(f"\n📌 Trial {i}:")
            print(f"  Static Loss: {trial.value:.6f}")
            print("  Params:")
            for k, v in trial.params.items():
                print(f"    {k}: {v}")