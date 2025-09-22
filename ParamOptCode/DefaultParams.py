import numpy as np
params = {
    'model_tag': 'lorenz',
    'case': 'synthetic_lorenz',
    'input_dim': 7,
    'latent_dim': 3,
    'poly_order': 2,
    'include_sine': False,
    'fix_coefs': True,
    'svd_dim':14, #60
    'delay_embedded': True,
    'scale': True,
    'coefficient_initialization': 'constant',
    'coefficient_initialization_constant': 0,
    'widths_1d': [6, 5, 4], 
    'widths_2d': [12, 10, 8, 6, 4], #[12, 10, 8, 6, 4]
    'observable_dim': 2,
    'sindy_compare_observables': True,
    'n_ics': 2,
    'skip_rows' : 1,
    'row_lag' : 1,

    # Training
    'max_epochs': 3000,
    'patience': 20,
    'batch_size': 128,
    'learning_rate': 1e-3,
    "lr_decay_ae": 0.9901043877553937,
    'lr_decay_sindy': 0.9991043877553937,
    'ae_learning_rate': 0.009962377354725672,
    'sindy_learning_rate': 0.0009962377354725672,
    'window_T' : 20,
    'window_stride': 1,

    # Loss Weights
    'loss_weight_rec': 0.3,
    'loss_weight_sindy_z': 0.001,
    'loss_weight_sindy_x': 0.001,
    'loss_weight_sindy_regularization': 1e-5,
    'loss_weight_integral': 0.1,
    'loss_weight_x0': 0.01,
    'loss_weight_layer_l2': 0.0,
    'loss_weight_layer_l1': 0.0,
    "loss_weight_sindy_group_l1": 1e-5,    
    'loss_weight_sindy_x0':0.008330912654371252,     #2 instead of 8 in first relevent digit     
    "sindy_l1_ramp_start": 50,
    "sindy_l1_ramp_duration": 100,

    # SINDy
    'coefficient_threshold': 0.1,
    'threshold_frequency': 5,
    'print_frequency': 10,
    'sindy_pert': 0.0,
    'ode_net': True,
    'ode_net_widths':[32, 20], #[4/3, 5/4] or [3/2, 4/3] depending on the model
    'exact_features': False,
    'use_bias': True,
    'tend': 32,
    'dt': 0.01,
    'rfe_frequency': 10,
    'sindy_print_rate': 10,
    'sindy_trainable': True,
    'sindy_train_start': 0,

    # Misc
    'train_ratio': 0.8,
    'data_path': './trained_models',
    'save_checkpoints': False,
    'save_freq': 5,
    'learning_rate_sched': False,
    'use_sindycall': False,
    'sindycall_start': 100,
    'rfe_start': 250,
    'sindy_print_rate': 10,
    'loss_print_rate': 10,
    'sparse_weighting': None,
    'system_coefficients': None,
    'update_freq': 2,
  

    #Constraints
#    'sindy_fixed_mask' : np.array([ #This set of constraints is Glyco Perfect
#    [ True, False],   # 1
#    [False,  True],   # z0
#    [False, False],   # z1
#    [ True,  True],   # z0^2
#    [ True,  True],   # z0*z1
#    [ True,  True],   # z1^2
#    [ True,  True],   # z0^3
#    [False, False],   # z0^2*z1
#    [ True,  True],   # z0*y^2
#    [ True,  True],   # z1^3
#    ], dtype=bool),

#    'sindy_fixed_mask' : np.array([ #This set of constraints is Glyco + Random
#    [False, False],   # 1
#    [False, False],   # z0
#    [False, False],   # z1
#    [False, False],   # z0^2
#    [False, False],   # z0*z1
#    [ True,  True],   # z1^2
#    [ True,  True],   # z0^3
#    [False, False],   # z0^2*z1
#    [ True,  True],   # z0*y^2
#    [ True,  True],   # z1^3
#    ], dtype=bool),

#        'sindy_fixed_mask' : ~np.array([ #This set of constraints is Glyco Entirely
#    [False, False],   # 1
#    [False, False],   # z0
#    [False, False],   # z1
#    [False, False],   # z0^2
#    [False, False],   # z0*z1
#    [False, False],   # z1^2
#    [False, False],   # z0^3
#    [False, False],   # z0^2*z1
#    [False, False],   # z0*y^2
#    [False, False],   # z1^3
#    ], dtype=bool),

    "sindy_fixed_mask": np.array([ #This set of constraints is Lorenz + Random 
        [True, True, True], #1
        [False, False, False], #x
        [False,  True, False], #y
        [False, False, False], #z
        [True, True, True], #x^2
        [False,  False, False], #xy
        [False, True, False], #xz
        [ True,  True,  True], #y^2
        [ False,  False,  False], #yz
        [ True,  True,  True], #z^2
    ], dtype=bool),


#    "sindy_fixed_mask": np.array([   #This set of constraints is None
#       [False, False, False],
#       [False, False, False],
#        [False,  False, False],
#        [False, False, False],
#        [False, False, False],
#        [False,  False, False],
#        [False, False, False],
#        [ False,  False,  False],
#       [ False,  False,  False],
#        [ False,  False,  False],
#    ], dtype=bool),

#    "sindy_fixed_mask": ~np.array([   #This set of constraints is All
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

#    "sindy_fixed_mask": ~np.array([   #This set of constraints is Lorenz
#        [False, False, False],
#        [True, True, False],
#        [True,  True, False],
#        [False, False, True],
#        [False, False, False],
#        [False,  False, True],
#        [False, True, False],
#        [ False,  False,  False],
#        [ False,  False,  False],
#        [ False,  False,  False],
#    ], dtype=bool),

#    "sindy_fixed_values": np.array([
#        [0.,  0.,  0.],
#        [0.,  0.,  0.],
#        [0., -1.,  0.],
#        [0.,  0.,  0.],
#        [0.,  0.,  0.],
#        [0.,  0.,  0.],
#        [0.,  -1.,  0.],
#        [0.,  0.,  0.],
#        [0.,  0.,  0.],
#        [0.,  0.,  0.],
#    ], dtype=np.float32),

#    'sindy_fixed_values' : np.array([
#        [0.0,   0.5],     # constant: used in dz1 only
#        [-1.0,  0.0],     # z0: used in dz0 only
#        [ 0.1,   -0.1 ],      # z1: both equations
#        [0.0,  0.0],      # z0^2
#        [0.0,  0.0],      # z0*z1
#        [0.0,  0.0],      # z1^2
#        [0.0,  0.0],      # z0^3
#        [1.0, -1.0],      # z0^2*z1: nonlinear term
#        [0.0,  0.0],      # z0*y^2
#        [0.0,  0.0],      # z1^3
#    ]),

    "sindy_fixed_values" : np.array([
    [0.0,  0.0,   0.0],   # 1
    [-10.0,  28.0,   0.0],   # x
    [10.0, -1.0,   0.0],   # y
    [0.0,  0.0,   -8/3],   # z
    [0.0,  0.0,   0.0],   # x^2
    [0.0, 0.0,   1.0],   # xy
    [0.0,  -1.0,   0.0],   # xz
    [0.0,  0.0,   0.0],   # y^2
    [0.0,  0.0,   0.0],   # yz
    [0.0,  0.0,  0.0]  # z^2
], dtype=np.float32)
}