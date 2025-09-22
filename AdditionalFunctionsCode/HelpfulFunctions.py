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

def get_hankel(
    x,
    dimension,
    delays=None,             # kept for backward-compat; interpreted as "n_cols"
    skip_rows=1,             # column stride (gap between column starts)
    params=None,
    row_lag=1,               # delay between successive rows inside a column
    strict=True,             # if True, raise when requested n_cols > K_max
    return_transposed=True,  # keep True to match your current return H.T
    dtype=np.float64,
):
    """
    Build a trajectory (Hankel-like) matrix from a 1-D series x.

    Semantics:
      - dimension (m): window length = number of rows inside each column.
      - row_lag (tau): step between successive rows within a column.
      - skip_rows (s): column stride = step between starting indices of columns.
      - delays / n_cols (K): number of columns (optional). If None, use K_max.

    Indexing (before optional transpose):
      H[i, j] = x[i*row_lag + j*skip_rows],  i=0..m-1, j=0..K-1

    Returns:
      H.T with shape (K, m) if return_transposed=True (default),
      else H with shape (m, K).
    """
    if params is None:
        params = {}
    col_stride = params.get('skip_rows', skip_rows)  # preserve your params slot
    tau = row_lag
    m = int(dimension)

    # --- validation ---
    if m <= 0:
        raise ValueError("dimension (window length) must be positive.")
    if tau <= 0 or col_stride <= 0:
        raise ValueError("row_lag and skip_rows (column stride) must be positive integers.")

    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D; got shape {x.shape}")
    N = x.shape[0]

    # Need at least (m-1)*tau + 1 samples to form one column
    min_needed = (m - 1) * tau + 1
    if N < min_needed:
        raise ValueError(f"Not enough samples: need >= {min_needed}, got {N}.")

    # Maximum feasible number of columns
    K_max = 1 + (N - 1 - (m - 1) * tau) // col_stride

    # Respect user-provided 'delays' as requested number of columns
    if delays is None:
        K = K_max
    else:
        K_req = int(delays)
        if K_req <= 0:
            raise ValueError("delays (requested columns) must be positive.")
        if K_req > K_max:
            if strict:
                raise ValueError(
                    f"Requested delays={K_req} exceeds feasible K_max={K_max} "
                    f"for len(x)={N}, dimension={m}, row_lag={tau}, skip_rows={col_stride}."
                )
            K = K_max  # clip
        else:
            K = K_req

    # --- vectorized indexing ---
    i = np.arange(m)[:, None]             # (m, 1)
    j = np.arange(K)[None, :]             # (1, K)
    idx = i * tau + j * col_stride        # (m, K)
    H = x[idx].astype(dtype, copy=False)  # (m, K)

    return H.T if return_transposed else H


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