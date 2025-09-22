
import torch
import time
import os
from TrainingCode.TrainingCode import TrainModel  
from ParamOptCode.ParameterOptimizer import HyperparameterOptimizer

print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Device name(s):", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

# === SLURM-based Parallel Setup ===
n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
n_trials = 5 * n_jobs


print(f"SLURM_CPUS_PER_TASK detected: {n_jobs}")
print(f"Hyperparameter trials to run: {n_trials}")

# === Optional: Clear old Optuna DB ===
if os.path.exists("optuna_study.db"):
    os.remove("optuna_study.db")
    print("Old Optuna study database removed.")

# === Start HPO ===
start_time = time.time()

optimizer = HyperparameterOptimizer(n_trials=n_trials, n_jobs=n_jobs)
best = optimizer.run()

end_time = time.time()
print(f"\n✅ Best hyperparameters:\n{best.params}")
print(f"⏱️ HPO completed in {(end_time - start_time)/60:.2f} minutes")