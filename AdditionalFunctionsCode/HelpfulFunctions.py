import torch
import itertools
import random
import numpy as np

def get_hyperparameter_list(hyperparams):
    """
    Generates a shuffled list of all hyperparameter combinations.

    Args:
        hyperparams (dict): Dictionary where keys are parameter names and values are lists of possible values.

    Returns:
        list of dicts: Each dictionary represents a unique combination of hyperparameters.
    """
    def dict_product(dicts):
        return [dict(zip(dicts, x)) for x in itertools.product(*dicts.values())]
    
    hyperparams_list = dict_product(hyperparams)
    random.shuffle(hyperparams_list)
    return hyperparams_list

def get_hankel(x, dimension, delays, skip_rows=1):
    """
    Constructs a Hankel matrix from a time series.

    Args:
        x (np.ndarray): 1D time-series data.
        dimension (int): Number of rows in Hankel matrix.
        delays (int): Number of columns in Hankel matrix.
        skip_rows (int): Step size for delay embedding.

    Returns:
        np.ndarray: Hankel matrix of shape (dimension, delays).
    """
    if skip_rows > 1:
        delays = len(x) - delays * skip_rows
    
    H = np.zeros((dimension, delays), dtype=np.float32)
    
    for j in range(delays):
        H[:, j] = x[j * skip_rows:j * skip_rows + dimension]

    return H


def get_hankel_svd(H, reduced_dim):
    """
    Performs Singular Value Decomposition (SVD) on the Hankel matrix.

    Args:
        H (np.ndarray): Hankel matrix of shape (samples, features) — e.g., (2496, 80)
        reduced_dim (int): Number of singular values to retain.

    Returns:
        tuple: (U, s, VT, rec_v) - Decomposed matrices and reduced-rank reconstruction.
               rec_v has shape (samples, reduced_dim)
    """
    U, S, VT = np.linalg.svd(H, full_matrices=False)
    rec_v = U[:, :reduced_dim] @ np.diag(S[:reduced_dim])  # shape (samples, reduced_dim)
    return U, S, VT, rec_v