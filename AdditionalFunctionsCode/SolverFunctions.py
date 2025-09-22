import numpy as np
from scipy.integrate import odeint, solve_ivp
import tqdm
from DataGenerationCode.DataGeneration import LorenzSystem, GlycolysisSystem 
from AdditionalFunctionsCode.HelpfulFunctions import get_hankel, get_hankel_svd
from sklearn.preprocessing import StandardScaler


class SynthData:
    def __init__(self, input_dim=128, normalization=None, num_trajectories=1, model_tag='lorenz', params = None,  observable_dim=1):
        self.input_dim = input_dim
        self.num_trajectories = num_trajectories
        self.model_tag = model_tag.lower()
        self.observable_dim = observable_dim
        self.params = params

        # Choose system
        if self.model_tag == 'lorenz':
            self.system = LorenzSystem()
        elif self.model_tag == 'glyco':
            self.system = GlycolysisSystem()
        else:
            raise ValueError(f"Unknown system_name: {model_tag}")

        self.normalization = np.ones(self.system.z0_mean.shape) if normalization is None else np.array(normalization)

    def solve_ivp(self, z0, time):
        f = self.system.dynamics
        z = odeint(f, z0, time, rtol=1e-8, atol=1e-10)
        dz = np.array([f(z[i], time[i]) for i in range(len(time))])
        return z, dz

    def run_multi_sim(self, n_trajectories, tend, dt, apply_svd=False, svd_dim=None, scale=True):
        f = self.system.dynamics
        time = np.arange(0, tend, dt)
        z0_mean, z0_std = self.system.z0_mean, self.system.z0_std

        z_all, dz_all, H_all, dH_all, K_list = [], [], [], [], []

        # Read embedding controls from params (with defaults)
        skip_rows = self.params.get("skip_rows", 1)   # column stride
        row_lag   = self.params.get("row_lag", 1)     # row delay

        for _ in range(n_trajectories):
            z0 = z0_std * (np.random.rand(len(z0_std)) - 0.5) + z0_mean
            z, dz = self.solve_ivp(z0, time)
            z *= self.normalization
            dz *= self.normalization

            # Let get_hankel compute K_max internally -> set delays=None
            n_cols = None

            if self.observable_dim == 1:
                x  = z[:, 0]
                dx = dz[:, 0]

                H  = get_hankel(x,  self.input_dim, delays=n_cols, params=self.params, row_lag=row_lag, skip_rows=skip_rows)
                dH = get_hankel(dx, self.input_dim, delays=n_cols, params=self.params, row_lag=row_lag, skip_rows=skip_rows)

                z_all.append(z); dz_all.append(dz)
                H_all.append(H); dH_all.append(dH)
                K_list.append(H.shape[0])  
                

            elif self.observable_dim == 2:
                x, y  = z[:, 0], z[:, 1]
                dx, dy = dz[:, 0], dz[:, 1]

                Hx  = get_hankel(x,  self.input_dim, delays=n_cols, params=self.params, row_lag=row_lag, skip_rows=skip_rows)
                Hy  = get_hankel(y,  self.input_dim, delays=n_cols, params=self.params, row_lag=row_lag, skip_rows=skip_rows)
                dHx = get_hankel(dx, self.input_dim, delays=n_cols, params=self.params, row_lag=row_lag, skip_rows=skip_rows)
                dHy = get_hankel(dy, self.input_dim, delays=n_cols, params=self.params, row_lag=row_lag, skip_rows=skip_rows)

                # With Hx/Hy shape (K, m), hstack doubles the per-column features to (K, 2m) as intended.
                H  = np.hstack([Hx,  Hy])
                dH = np.hstack([dHx, dHy])

                z_all.append(z); dz_all.append(dz)
                H_all.append(H); dH_all.append(dH)
                K_list.append(H.shape[0])
            else:
                raise ValueError("Observable dimension must be 1 or 2.")

        # Keep your stacking: (num_traj*K, m or 2m)
        self.z       = np.concatenate(z_all, axis=0)
        self.dz      = np.concatenate(dz_all, axis=0)
        self.hankel  = np.vstack(H_all)
        self.dhankel = np.vstack(dH_all)
        self.K_per_traj = K_list
        self.t = time
        self.sindy_coefficients = np.array(self.system.Xi, dtype=np.float32)
        self.lengths = [H.shape[0] for H in H_all]

        if apply_svd:
            assert svd_dim is not None, "Specify svd_dim if apply_svd=True"
            U, S, VT, rec_v = get_hankel_svd(self.hankel, reduced_dim=svd_dim)
            if scale:
                rec_v = StandardScaler().fit_transform(rec_v)
            self.x  = rec_v
            self.dx = np.gradient(rec_v, dt, axis=0)
            self.U, self.S, self.VT = U, S, VT
        else:
            self.x  = self.hankel
            self.dx = self.dhankel

        print("\n🧪 DEBUG: Final data shapes from run_multi_sim")
        print(f"self.hankel.shape: {self.hankel.shape}")
        print(f"self.x.shape: {self.x.shape}")
        print(f"self.observable_dim: {self.observable_dim}, input_dim: {self.input_dim}")

