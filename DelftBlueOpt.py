
import torch
import time
from TrainingCode.TrainingCode import TrainModel  
from ParamOptCode.DefaultParams import params
from AdditionalFunctionsCode.SolverFunctions import SynthData
from DataGenerationCode.DataGeneration import LorenzSystem
from ParamOptCode.ParameterOptimizer import HyperparameterOptimizer
import os

print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Device name(s):", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
# Get Slurm-allocated CPU count or fallback
n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
n_human = int(n_jobs/4)

#For safety reason when this code is to be run on my own PC, n_trails is set to 8, n_jobs is set to 4
#As a double check:
print(f"SLURM_CPUS_PER_TASK detected: {n_jobs}")
print(f"Hyperparameter trials to run: {2*n_jobs}")
os.remove("optuna_study.db")  # or your full path


start_time = time.time()
device = torch.device("cpu")
device_type = "GPU" if device.type == "cuda" else "CPU"
params["input_dim"] = params["svd_dim"]
optimizer = HyperparameterOptimizer(n_trials=2*4, n_jobs=4)
best = optimizer.run()
end_time = time.time()
print(f"Best hyperparameters: {best}")
print(f"Hyperparameter optimization completed in {(end_time - start_time)/60:.2f} minutes")