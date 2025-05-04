from scipy.special import binom
import numpy as np
from scipy.special import binom
from scipy.integrate import odeint, solve_ivp

def library_size(n, poly_order, use_sine=False, include_constant=True):
    """
    Computes the number of terms in a polynomial feature library.
    """
    l = 0
    for k in range(poly_order + 1):
        l += int(binom(n + k - 1, k))
    if use_sine:
        l += n
    if not include_constant:
        l -= 1
    return l

# Function to construct the feature names
def sindy_library_names(latent_dim, poly_order):
    """
    Generate names of SINDy library terms for a given latent dimension and polynomial order.
    
    Args:
        latent_dim (int): Number of latent variables.
        poly_order (int): Maximum polynomial order.

    Returns:
        list: Names of the feature library terms.
    """
    feature_names = ["1"]  # Constant term

    # Linear terms
    for i in range(latent_dim):
        feature_names.append(f"z{i+1}")

    # Quadratic terms
    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                feature_names.append(f"z{i+1} * z{j+1}")

    return feature_names

def sindy_library(X, poly_order, include_sine=False, include_names=False, exact_features=False, include_sparse_weighting=False):
    # Upgrade to combinations
    symbs = ['x', 'y', 'z', '4', '5', '6', '7']

    # GENERALIZE 
    if exact_features:
        # Check size (is first dimension batch?)
        library = np.ones((X.shape[0],5))
        library[:, 0] = X[:,0]
        library[:, 1] = X[:,1]
        library[:, 2] = X[:,2]
        library[:, 3] = X[:,0]* X[:,1]
        library[:, 4] = X[:,0]* X[:,2]
        names = ['x', 'y', 'z', 'xy', 'xz']
        
        if include_sparse_weighting:
            sparse_weights = np.ones((5, X.shape[1]))  # e.g., shape [5, 3] for 3 latent dims
    else:
        
        m, n = X.shape
        l = library_size(n, poly_order, include_sine, True)
        library = np.ones((m, l))
        sparse_weights = np.ones((l, n))
        
        index = 1
        names = ['1']

        for i in range(n):
            library[:,index] = X[:,i]
            sparse_weights[index, :] *= 1
            names.append(symbs[i])
            index += 1

        if poly_order > 1:
            for i in range(n):
                for j in range(i,n):
                    library[:,index] = X[:,i]*X[:,j]
                    sparse_weights[index, :] *= 2
                    names.append(symbs[i]+symbs[j])
                    index += 1

        if poly_order > 2:
            for i in range(n):
                for j in range(i,n):
                    for k in range(j,n):
                        library[:,index] = X[:,i]*X[:,j]*X[:,k]
                        sparse_weights[index, :] *= 3
                        names.append(symbs[i]+symbs[j]+symbs[k])
                        index += 1

        if poly_order > 3:
            for i in range(n):
                for j in range(i,n):
                    for k in range(j,n):
                        for q in range(k,n):
                            library[:,index] = X[:,i]*X[:,j]*X[:,k]*X[:,q]
                            sparse_weights[index, :] *= 4
                            names.append(symbs[i]+symbs[j]+symbs[k]+symbs[q])
                            index += 1

        if poly_order > 4:
            for i in range(n):
                for j in range(i,n):
                    for k in range(j,n):
                        for q in range(k,n):
                            for r in range(q,n):
                                library[:,index] = X[:,i]*X[:,j]*X[:,k]*X[:,q]*X[:,r]
                                sparse_weights[index, :] *= 5
                                names.append(symbs[i]+symbs[j]+symbs[k]+symbs[q]+symbs[r])
                                index += 1

        if include_sine:
            for i in range(n):
                library[:,index] = np.sin(X[:,i])
                names.append('sin('+symbs[i]+')')
                index += 1

       
    return_list = [library]
    if include_names:
        return_list.append(names)
    if include_sparse_weighting:
        return_list.append(sparse_weights)
    return return_list

def sindy_library_order2(X, dX, poly_order, include_sine=False):
    m,n = X.shape
    l = library_size(2*n, poly_order, include_sine, True)
    library = np.ones((m,l))
    index = 1

    X_combined = np.concatenate((X, dX), axis=1)

    for i in range(2*n):
        library[:,index] = X_combined[:,i]
        index += 1

    if poly_order > 1:
        for i in range(2*n):
            for j in range(i,2*n):
                library[:,index] = X_combined[:,i]*X_combined[:,j]
                index += 1

    if poly_order > 2:
        for i in range(2*n):
            for j in range(i,2*n):
                for k in range(j,2*n):
                    library[:,index] = X_combined[:,i]*X_combined[:,j]*X_combined[:,k]
                    index += 1

    if poly_order > 3:
        for i in range(2*n):
            for j in range(i,2*n):
                for k in range(j,2*n):
                    for q in range(k,2*n):
                        library[:,index] = X_combined[:,i]*X_combined[:,j]*X_combined[:,k]*X_combined[:,q]
                        index += 1
                    
    if poly_order > 4:
        for i in range(2*n):
            for j in range(i,2*n):
                for k in range(j,2*n):
                    for q in range(k,2*n):
                        for r in range(q,2*n):
                            library[:,index] = X_combined[:,i]*X_combined[:,j]*X_combined[:,k]*X_combined[:,q]*X_combined[:,r]
                            index += 1

    if include_sine:
        for i in range(2*n):
            library[:,index] = np.sin(X_combined[:,i])
            index += 1


def sindy_fit(RHS, LHS, coefficient_threshold):
    m,n = LHS.shape
    Xi = np.linalg.lstsq(RHS,LHS, rcond=None)[0]
    
    for k in range(10):
        small_inds = (np.abs(Xi) < coefficient_threshold)
        Xi[small_inds] = 0
        for i in range(n):
            big_inds = ~small_inds[:,i]
            if np.where(big_inds)[0].size == 0:
                continue
            Xi[big_inds,i] = np.linalg.lstsq(RHS[:,big_inds], LHS[:,i], rcond=None)[0]
    return Xi


def sindy_simulate(x0, t, Xi, poly_order, include_sine=False, exact_features=False):
    m = t.size
    n = x0.size
    f = lambda x,t : np.dot(sindy_library(np.array(x).reshape((1,n)), poly_order, include_sine, include_names=False, exact_features=exact_features), Xi).reshape((n,))
    x = odeint(f, x0, t)
    return x


def sindy_simulate_order2(x0, dx0, t, Xi, poly_order, include_sine):
    m = t.size
    n = 2*x0.size
    l = Xi.shape[0]

    Xi_order1 = np.zeros((l,n))
    for i in range(n//2):
        Xi_order1[2*(i+1),i] = 1.
        Xi_order1[:,i+n//2] = Xi[:,i]
    
    x = sindy_simulate(np.concatenate((x0,dx0)), t, Xi_order1, poly_order, include_sine)
    return x

def sindy_simulate_improved(x0, t, Xi, poly_order):
    """
    Improved simulation of the SINDy-recovered system using `solve_ivp`
    for better numerical stability.
    
    Args:
        x0 (array): Initial condition.
        t (array): Time array.
        Xi (array): SINDy coefficients.
        poly_order (int): Polynomial order of the library.

    Returns:
        array: Simulated trajectory.
    """
    def sindy_rhs(t, x):
        """Computes the right-hand side of the SINDy system."""
        x = x.reshape((1, -1))  # Ensure shape consistency
        dxdt = np.dot(sindy_library(x, poly_order), Xi)
        return dxdt.flatten()

    # Solve the system with a more robust integrator
    sol = solve_ivp(sindy_rhs, (t[0], t[-1]), x0, t_eval=t, method='RK45', rtol=1e-8, atol=1e-10)
    
    return sol.y.T  # Transpose to match shape (N,3)

def sindy_simulate_chunked(x0, t, Xi, poly_order, chunk_size=2000):
    """
    Simulates the SINDy system in chunks to reduce error accumulation.

    Args:
        x0 (np.ndarray): Initial condition.
        t (np.ndarray): Time array.
        Xi (np.ndarray): Learned SINDy coefficients.
        poly_order (int): Polynomial order used in SINDy.
        chunk_size (int): Number of time points per chunk.

    Returns:
        np.ndarray: Simulated state trajectory.
    """
    num_points = len(t)
    x_total = []
    x_current = x0

    start_idx = 0
    while start_idx < num_points:
        end_idx = min(start_idx + chunk_size, num_points)
        t_chunk = t[start_idx:end_idx]

        # Simulate this chunk
        x_chunk = sindy_simulate(x_current, t_chunk, Xi, poly_order)

        # Append to results
        x_total.append(x_chunk)

        # Update initial condition for next chunk
        x_current = x_chunk[-1]

        # Move to next chunk (overlap by 1 for continuity)
        start_idx += chunk_size - 1

    # Concatenate results
    return np.vstack(x_total)