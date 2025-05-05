import numpy as np
params = {
    'model': 'lorenz',
    'case': 'synthetic_lorenz',
    'input_dim': 7,
    'latent_dim': 3,
    'poly_order': 2,
    'include_sine': False,
    'fix_coefs': False,
    'svd_dim':7, #60
    'delay_embedded': True,
    'scale': True,
    'coefficient_initialization': 'constant',
    'coefficient_initialization_constant': 0,
    'widths_ratios': [1.0, 6/7, 5/6, 4/5, 3/4],

    # Training
    'max_epochs': 1000,
    'patience': 20,
    'batch_size': 128,
    'learning_rate': 1e-3,
    "lr_decay": 0.999,

    # Loss Weights
    'loss_weight_rec': 0.3,
    'loss_weight_sindy_z': 0.001,
    'loss_weight_sindy_x': 0.001,
    'loss_weight_sindy_regularization': 1e-5,
    'loss_weight_integral': 0.1,
    'loss_weight_x0': 0.01,
    'loss_weight_layer_l2': 0.0,
    'loss_weight_layer_l1': 0.0,

    # SINDy
    'coefficient_threshold': 0.01,
    'threshold_frequency': 5,
    'print_frequency': 10,
    'sindy_pert': 0.0,
    'ode_net': False,
    'ode_net_widths': [1.5, 2.0],
    'exact_features': True,
    'use_bias': True,
    'tend': 32,
    'dt': 0.01,

    # Misc
    'train_ratio': 0.8,
    'data_path': './trained_models',
    'save_checkpoints': False,
    'save_freq': 5,
    'learning_rate_sched': False,
    'use_sindycall': False,
    'sindycall_start': 60,
    'sindy_print_rate': 10,
    'loss_print_rate': 10,
    'sparse_weighting': None,
    'system_coefficients': None,
    'update_freq': 10,

    #Constraints
    
    "sindy_fixed_mask": np.array([
        [False, False, False],
        [False, False, False],
        [False,  True, False],
        [False, False, False],
        [False, False, False],
        [False,  True, False],
        [False, False, False],
        [ True,  True,  True],
        [ True,  True,  True],
        [ True,  True,  True],
    ], dtype=bool),
#    "sindy_fixed_mask": np.array([
#       [False, False, False],
#       [False, False, False],
#        [False,  False, False],
#        [False, False, False],
#        [False, False, False],
#        [False,  False, False],
#        [False, False, False],
#        [ False,  False,  False],
#        [ False,  False,  False],
#        [ False,  False,  False],
#    ], dtype=bool),

    "sindy_fixed_values": np.array([
        [0.,  0.,  0.],
        [0.,  0.,  0.],
        [0., -1.,  0.],
        [0.,  0.,  0.],
        [0.,  0.,  0.],
        [0.,  1.,  0.],
        [0.,  0.,  0.],
        [0.,  0.,  0.],
        [0.,  0.,  0.],
        [0.,  0.,  0.],
    ], dtype=np.float32),
}