import numpy as np
from scipy.integrate import odeint, solve_ivp
import tqdm
from DataGenerationCode.DataGeneration import LorenzSystem  
from AdditionalFunctionsCode.HelpfulFunctions import get_hankel, get_hankel_svd

class SynthData:
    def __init__(self, input_dim=128, normalization=None):
        """
        Generates synthetic data using the Lorenz system and constructs a Hankel matrix.

        Args:
            input_dim (int): Number of time delays for Hankel matrix.
            normalization (np.ndarray or None): Normalization factors (default: ones).
        """
        self.input_dim = input_dim
        self.lorenz = LorenzSystem()  # Instantiate the Lorenz system
        self.normalization = np.ones(3) if normalization is None else np.array(normalization)

    def solve_ivp_chunked(self, z0, time, chunk_size=2000):
        """
        Solves the Lorenz system using solve_ivp in chunks.
        
        Args:
            z0 (np.ndarray): Initial condition.
            time (np.ndarray): Time array.
            chunk_size (int): Number of time steps per chunk.

        Returns:
            z (np.ndarray): Simulated state trajectory.
            dz (np.ndarray): Time derivative at each step.
        """
        f = self.lorenz.dynamics  # Ensure this function takes (t, z) and returns dz/dt
        num_steps = len(time)
        
        z_list = []
        dz_list = []

        # Iterate over chunks
        for start in range(0, num_steps, chunk_size):
            end = min(start + chunk_size, num_steps)
            t_chunk = time[start:end]

            # Solve for this chunk
            sol = solve_ivp(lambda t, z: f(z, t), (t_chunk[0], t_chunk[-1]), z0, t_eval=t_chunk,
                            method='RK45', rtol=1e-8, atol=1e-10)

            if not sol.success:
                raise RuntimeError(f"solve_ivp failed in chunk {start}-{end}: {sol.message}")

            # Store results
            z_list.append(sol.y.T)
            dz_chunk = np.array([f(sol.y[:, i], t_chunk[i]) for i in range(len(t_chunk))])
            dz_list.append(dz_chunk)

            # Update initial condition for the next chunk
            z0 = sol.y[:, -1]  # Use last point as initial condition

        # Concatenate all chunks
        z = np.vstack([chunk[1:] if i > 0 else chunk for i, chunk in enumerate(z_list)])
        dz = np.vstack([chunk[1:] if i > 0 else chunk for i, chunk in enumerate(dz_list)])

        return z, dz

        

    def solve_ivp(self, z0, time):
            """
            Solves the Lorenz system using scipy.odeint.

            Args:
                z0 (np.ndarray): Initial condition, shape (3,).
                time (np.ndarray): Time array.

            Returns:
                z (np.ndarray): Simulated state trajectory.
                dz (np.ndarray): Time derivative at each step.
            """
            f = self.lorenz.dynamics
            z = odeint(f, z0, time, rtol=1e-8, atol=1e-10)
            dz = np.array([f(z[i], time[i]) for i in range(len(time))])
            return z, dz

    def run_sim(self, tend, dt, z0, apply_svd=False, svd_dim=None, scale=True):
        """
        Runs the Lorenz system with chunked simulation and constructs the Hankel matrix.

        Args:
            tend (float): Simulation time duration.
            dt (float): Time step.
            z0 (np.ndarray): Initial condition, shape (3,).
            apply_svd (bool): If True, performs SVD on the Hankel matrix and replaces x/dx.
            svd_dim (int): Reduced SVD dimension (required if apply_svd=True).
            scale (bool): If True, standardizes the SVD output.
        """
        from sklearn.preprocessing import StandardScaler

        time = np.arange(0, tend, dt)

        print("Generating Lorenz system...")
        z, dz = self.solve_ivp(z0, time)

        z *= self.normalization
        dz *= self.normalization

        # Use only x(t)
        x = z[:, 0]
        dx = dz[:, 0]

        # Build Hankel matrix
        delays = len(time) - self.input_dim
        H = get_hankel(x, self.input_dim, delays).T  # shape (delays, input_dim)
        dH = get_hankel(dx, self.input_dim, delays).T  # same shape

        self.t = time
        self.z = z
        self.dz = dz
        self.xorig = H.copy()  # for safety/debug
        self.sindy_coefficients = np.array(self.lorenz.Xi, dtype=np.float32)

        if apply_svd:
            assert svd_dim is not None, "You must specify svd_dim if apply_svd=True"

            U, S, VT, rec_v = get_hankel_svd(H, reduced_dim=svd_dim)

            if scale:
                rec_v = StandardScaler().fit_transform(rec_v)

            self.x = rec_v
            self.dx = np.gradient(rec_v, dt, axis=0)
            self.U = U
            self.S = S
            self.VT = VT
            self.hankel = H
            self.dhankel = dH
            print("self.x (rec_v):", self.x.shape)
            print("self.dx:", self.dx.shape)

        else:
            self.x = H
            self.dx = dH
            self.hankel = H
            self.dhankel = dH

    def run_sim_chunked(self, tend, dt, z0):
        """
        Runs the Lorenz system with chunked simulation and constructs the Hankel matrix.

        Args:
            tend (float): Simulation time duration.
            dt (float): Time step.
            z0 (np.ndarray): Initial condition, shape (3,).
        """
        time = np.arange(0, tend, dt)

        # Solve in chunks
        print("Generating Lorenz system solution in chunks...")
        z, dz = self.solve_ivp_chunked(z0, time)

        # Apply normalization
        z *= self.normalization
        dz *= self.normalization

        # Extract first state variable for Hankel matrix
        x = z[:, 0]
        dx = dz[:, 0]

        # Construct Hankel matrices
        delays = len(time) - self.input_dim
        H = get_hankel(x, self.input_dim, delays)
        dH = get_hankel(dx, self.input_dim, delays)

        # Store results
        self.z = z
        self.dz = dz
        self.x = H
        self.dx = dH
        self.t = time
        self.sindy_coefficients = np.array(self.lorenz.Xi, dtype=np.float32)
        
