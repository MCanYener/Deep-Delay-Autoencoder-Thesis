from SINDyCode.SINDyPreamble import library_size
import torch
import numpy as np

class LorenzSystem:
    def __init__(self, sigma=10.0, rho=28.0, beta=8/3):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

        # Suggested initial condition mean and std (from the original code)
        self.z0_mean = np.array([0, 0, 25])  # Mean IC
        self.z0_std = np.array([36, 48, 41])  # Std deviation of IC

        # Precompute SINDy coefficient matrix
        self.Xi = np.zeros((library_size(3, 2), 3))
        self.Xi[1, 0] = -self.sigma
        self.Xi[2, 0] = self.sigma
        self.Xi[1, 1] = self.rho
        self.Xi[2, 1] = -1
        self.Xi[6, 1] = -1
        self.Xi[3, 2] = -self.beta
        self.Xi[5, 2] = 1

    def dynamics(self, s, t=None):
        """
        Computes the time derivative ds/dt for the Lorenz system.
        
        Args:
            s (array-like): State vector [x, y, z].
            t (float): Time (unused but required for odeint).
        
        Returns:
            list: Time derivatives [dx/dt, dy/dt, dz/dt].
        """
        sigma, rho, beta = self.sigma, self.rho, self.beta
        x, y, z = s  # Unpack state variables

        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z

        return [dxdt, dydt, dzdt]


    def compute_sindy_coefficients(self):
        """
        Computes the known SINDy coefficient matrix for the Lorenz system.
        """
        dim = 3
        n = self.normalization
        poly_order = self.poly_order
        Xi = np.zeros((library_size(dim, poly_order), dim))

        # Manually set known Lorenz coefficients
        Xi[1, 0] = -self.sigma
        Xi[2, 0] = self.sigma * n[0] / n[1]
        Xi[1, 1] = self.rho * n[1] / n[0]
        Xi[2, 1] = -1
        Xi[6, 1] = -n[1] / (n[0] * n[2])
        Xi[3, 2] = -self.beta
        Xi[5, 2] = n[2] / (n[0] * n[1])

        return torch.tensor(Xi, dtype=torch.float32)

    def get_initial_conditions(self):
        """
        Returns suggested mean and standard deviation for initial conditions.
        """
        z0_mean_sug = torch.tensor([0.0, 0.0, 25.0])
        z0_std_sug = torch.tensor([36.0, 48.0, 41.0])
        return z0_mean_sug, z0_std_sug


