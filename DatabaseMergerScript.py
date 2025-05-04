from DatabaseMergerCode.DatabaseMerger import OptunaStudyMerger

merger = OptunaStudyMerger(
    db_paths=[
        "Databases/optuna_study.db",
        "Databases/optuna_study_1.db",
        "Databases/optuna_study_2.db",
        "Databases/optuna_study_3.db",
        "Databases/optuna_study_4.db",
        "Databases/optuna_study_5.db",
        "Databases/optuna_study_6.db",
        "Databases/optuna_study_7.db",
        "Databases/optuna_study_8.db",
        "Databases/optuna_study_9.db",
        "Databases/optuna_study_10.db",
        "Databases/optuna_study_11.db",
        "Databases/optuna_study_12.db",
        "Databases/optuna_study_13.db",
        "Databases/optuna_study_14.db",
        "Databases/optuna_study_15.db",
        "Databases/optuna_study_16.db",
        "Databases/optuna_study_17.db",
        "Databases/optuna_study_18.db",
        "Databases/optuna_study_19.db",
        "Databases/optuna_study_20.db",
        "Databases/optuna_study_21.db",
        "Databases/optuna_study_22.db",
        "Databases/optuna_study_23.db",
        "Databases/optuna_study_24.db",
    ],
    merged_db_path="sqlite:///merged_optuna.db",
    merged_study_name="sindy_opt_merged",
    top_k=5
)

merger.merge_top_trials()
merger.print_top_trials()