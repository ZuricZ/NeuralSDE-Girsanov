import numpy as np
from scipy.optimize import fsolve
import torch


def brodie_kaya(self):
    def conditional_func(v):
        D = np.exp(-self.kappa * self.dt)
        C = self.nu * (1 - D) / (4 * self.kappa)
        xi = (4 * self.kappa * D * v) / (self.nu ** 2 * (1 - D))
        dof = 4 * self.kappa * self.theta / (self.nu ** 2)

    pass

    def generate_paths_implicit(self, T, n_steps, N_paths, scheme=None, antithetic=False):
        dt = T / n_steps
        self.dt = dt

        price_path, vol_path = np.zeros((N_paths, n_steps + 1)), np.zeros((N_paths, n_steps + 1))
        price_path[:, 0], vol_path[:, 0] = np.log(self.S0), self.V0

        cov = np.array([[1, self.rho], [self.rho, 1]])

        discretize_price_func = lambda v, dB: (self.r - 0.5 * v) * dt + np.sqrt(v) * dB
        discretize_vol_func = lambda x: None

        if antithetic:
            dW = np.sqrt(dt) * np.random.multivariate_normal(np.zeros(2), cov=cov, size=(N_paths // 2, n_steps + 1))
            dW = np.concatenate([dW, -dW], axis=0)
        else:
            dW = np.sqrt(dt) * np.random.multivariate_normal(np.zeros(2), cov=cov, size=(N_paths, n_steps + 1))

        for i in range(n_steps):
            price_path[:, i + 1] = discretize_price_func(vol_path[:, i], dW[:, i, 0])

            vol_path[:, i + 1] = fsolve(discretize_vol_func, x0=vol_path[:, i], args=())

        price_path = self.S0 * np.exp(np.cumsum(price_path, axis=1))
        return price_path, vol_path, dW


def get_control_variate(self, strikes, maturities, price_path, vol_path, dW):
        # TODO: This is wrong. Has to be for each maturity separately
        # Use expand instead of repeat to preserve memory
        S = price_path[:, :, None, None].expand(-1, -1, strikes.shape[0], maturities.shape[0])
        V = vol_path[:, :, None, None].expand(-1, -1, strikes.shape[0], maturities.shape[0])
        # dW = dW[:, :, 0, None, None].expand(-1, -1, strikes.shape[0], maturities.shape[0])
        dist = torch.distributions.normal.Normal(torch.zeros_like(S), torch.ones_like(S))

        K = strikes[None, None, :, None].expand(*S.shape)
        T = maturities[None, None, None, :].expand(*S.shape)

        d1 = (torch.log(S / K) + (self.r + 0.5 * V ** 2) * T) / (V * torch.sqrt(T))
        delta = dist.cdf(d1)

        # shape: i=N, j=n, k=|K|, l=|T|
        return torch.einsum('ijkl,ij->kl', delta, dW[:, :, 0]) / S.shape[0]


def add(dW, device):
    # add column of zeros so V_i and dW_i are aligned
    dW = torch.cat([torch.zeros((dW.shape[0], 1, dW.shape[2]), device=device), dW], dim=1)


# def girsanov(self, vol_path, dW):
#     # vol_path: nn.tensor (N_paths,n_steps,1)
#     vol_path = vol_path.reshape(vol_path.shape[0], vol_path.shape[1], -1)
#     integral_1 = torch.sum(self.phi_1(vol_path) * dW[:, :, 0, None], dim=1)
#     integral_2 = torch.sum(self.phi_2(vol_path) * dW[:, :, 1, None], dim=1)
#     integral_12 = torch.sum(self.phi_1(vol_path) * self.phi_2(vol_path) * self.dt, dim=1)
#     integral_1122 = torch.sum((torch.pow(self.phi_1(vol_path), 2)
#                                + torch.pow(self.phi_2(vol_path), 2)) * self.dt, dim=1)
#     return torch.exp(- integral_1 - integral_2 - self.rho * integral_12 - 0.5 * integral_1122)

# def get_girsanov_matrix(self, maturity_idx, vol_path, dW):
    # girsanov_list = []
    # for idx in maturity_idx:
    #     girsanov_list.append(self.girsanov(vol_path[:, :idx], dW[:, :idx, :]))
    # return torch.cat(girsanov_list, dim=1)