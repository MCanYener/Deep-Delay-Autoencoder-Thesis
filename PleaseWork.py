import pickle
from TrainingCode.ValidateTraining import TopTrialSelector, TrialEvaluator

# --- CONFIGURATION ---
DB_PATH = "Databases/optuna_study_1.db"
STUDY_NAME = "sindy_opt_1"
TOP_K = 5
TRIAL_INDEX_TO_RUN = 1  # <-- Change this (1 = best, 2 = second-best, etc.)

# --- STEP 1: Load and Save Top Trials ---
selector = TopTrialSelector(db_path=DB_PATH, study_name=STUDY_NAME, top_k=TOP_K)
selector.save_top_params("top_trial_params.pkl")

# --- STEP 2: Load Selected Trial Params ---
trial_params = selector.get_trial_params(TRIAL_INDEX_TO_RUN)

# --- STEP 3: Run Training + Plot Evaluation ---
evaluator = TrialEvaluator(trial_params)
evaluator.train_model()
evaluator.evaluate_and_plot()
