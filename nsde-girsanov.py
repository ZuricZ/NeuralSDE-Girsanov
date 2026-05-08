import os
import numpy as np
import torch
import torch.nn as nn
from ignite.contrib.metrics.regression import MeanAbsoluteRelativeError
from sample_model.heston import Heston as SampleModel
from lib.networks import FFNN, ResFFNN
# from lib.options import Vanilla_Put, Vanilla_Call
from lib.utils import torch_max_func, searchsorted, init_weights
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
from lib.plot import plot_IV_slices
from lib.compute_iv import get_vega

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0.):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, new_loss, old_loss):
        if (new_loss - old_loss) >= self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


class NSDE(SampleModel):

    # HESTON: dS_t = (r - 0.5*V_t)*dt + sqrt(V_t)dB_t ; dV_t = kappa*(theta - V_t)*dt + sqrt(V_t)*dW_t
    #         a(v) = (r - 0.5*v) ; b(v) = sqrt(v) ; m(v) = kappa*(theta - v) ; s(v) = nu*sqrt(v)

    def __init__(self, sample_model, pre_train=False):
        super().__init__()
        vars(self).update(vars(sample_model))

        # nsde model coefficients
        self.nsde_kappa = FFNN(sizes=[1, 15, 15, 1], activation=nn.ReLU, output_activation=nn.Identity).to(device)
        # self.nsde_kappa = lambda v: nsde.kappa*(nsde.theta - v)
        # self.nsde_mu = ResFFNN(sizes=[1, 15, 15, 1]).to(device)
        self.nsde_params = list(itertools.chain(*[list(self.nsde_kappa.parameters()),
                                                  # list(self.nsde_mu.parameters())
                                                  ]))
        self.nsde_kappa.apply(init_weights)
        # self.nsde_mu.apply(init_weights)
        if pre_train:
            self.pre_train(n_epoch=5000)

        self.price_matrix = None

    # sample model coefficients
    def a(self, v):
        return self.r - 0.5 * v

    def b(self, v):
        return torch.sqrt(v)

    def m(self, v):
        return self.kappa * (self.theta - v)

    def s(self, v):
        return self.nu * torch.sqrt(v)

    def phi_1(self, v):
        return torch.zeros(v.shape, device=device)
        # return self.rho / np.sqrt(1-self.rho ** 2) * (self.nsde_mu(v) - self.a(v)) / self.s(v)

    def phi_2(self, v):
        rho_bar = np.sqrt(1 - self.rho ** 2)
        term1 = (self.nsde_kappa(v) - self.m(v)) / torch.clamp(rho_bar * self.s(v), min=1e-16)
        term2 = self.rho / rho_bar * self.phi_1(v)
        return term1 - term2

    def get_model_price(self, payoff_matrix, Z):
        # compute the sample mean vectorised.
        # By dimension: i=N_samples, j=n_strikes, k=n_maturities, m=extra_dim
        model_price = torch.einsum('ijk,ik->jk', payoff_matrix, Z) / payoff_matrix.shape[0]
        return model_price  # TODO: un-discounted

    def get_girsanov_matrix(self, maturity_idx, vol_path, dW):
        vol_path = vol_path.reshape(vol_path.shape[0], vol_path.shape[1], -1)
        integral_1 = torch.cumsum(self.phi_1(vol_path) * dW[:, :, 0, None], dim=1)
        integral_2 = torch.cumsum(self.phi_2(vol_path) * dW[:, :, 1, None], dim=1)
        integral_12 = torch.cumsum(self.phi_1(vol_path) * self.phi_2(vol_path) * self.dt, dim=1)
        integral_1122 = torch.cumsum((torch.pow(self.phi_1(vol_path), 2)
                                     + torch.pow(self.phi_2(vol_path), 2)) * self.dt, dim=1)
        return torch.exp(- integral_1 - integral_2
                         - self.rho * integral_12 - 0.5 * integral_1122)[:, maturity_idx].squeeze()

    def get_girsanov_matrix_diff(self, maturity_idx, vol_path, dW, milstein=True):
        """Compute the Radon-Nikodym density Z at each requested maturity.

        Multiplicative discretisation of dZ = -Z φ₂ dW - ½ Z φ₂² dt:
            factor_i = 1 - φ₂(V_i) ΔW_i + ½ φ₂(V_i)² (ΔW_i² - dt)   [Milstein]

        Z_k = ∏_{i<k} factor_i,  Z_0 = 1.

        Milstein factors are computed before the Z_0=1 shift so that
        the correction for step i is applied to factor i, not factor i-1.
        A Milstein factor can be negative when φ₂ > 1/√dt (which occurs
        near V=0 when the Feller condition is violated); values are clamped
        to zero to keep Z a valid density.

        Args:
            maturity_idx: 1-D integer tensor of time-step indices.
            vol_path: variance paths, shape (N, n_steps+1).
            dW: Brownian increments, shape (N, n_steps+1, 2).
            milstein: include the Milstein correction term.

        Returns:
            Z at requested maturities, shape (N, len(maturity_idx)).
        """
        vol_path = vol_path.reshape(vol_path.shape[0], vol_path.shape[1], -1)
        phi_2_eval = self.phi_2(vol_path)

        factors = 1.0 - phi_2_eval * dW[:, :, 1, None]
        if milstein:
            factors = factors + 0.5 * torch.square(phi_2_eval) * (
                torch.square(dW[:, :, 1, None]) - self.dt
            )

        delta_Z = torch.cat(
            [torch.ones((factors.shape[0], 1, factors.shape[2]), device=device), factors],
            dim=1
        )[:, :-1, :]

        Z = torch.cumprod(delta_Z, dim=1)
        Z = torch.clamp(Z, min=0.0)
        return Z[:, maturity_idx].squeeze()

    def get_payoff_matrix(self, strikes, maturity_idx, price_path):
        strikes = strikes.reshape(-1, strikes.shape[0])
        # torch.cartesian_prod(strikes, maturities)
        # price_mat = price_path[:, :, None].repeat(1, strikes.shape[0], maturity_idx.shape[0])
        payoff_list = []
        for idx in maturity_idx:
            price = price_path[:, idx, None].repeat(1, strikes.shape[0])
            # CALL OPTION
            payoff_list.append(torch_max_func(price - strikes))
        return torch.stack(payoff_list, dim=2)

    def pre_train(self, n_epoch=100):
        loss_func = nn.MSELoss()
        optimizer_nsde = torch.optim.Adam(self.nsde_params, lr=1e-2)  # 1e-3
        early_stopping = EarlyStopping(tolerance=10, min_delta=0.)
        x = torch.rand((1000, 1), device=device)  # x = torch.linspace(0, 1, 1000, device=device)

        # calculate payoffs of the sample model paths
        with tqdm(range(n_epoch), unit='epoch') as t_epoch:
            for i in t_epoch:
                t_epoch.set_description(f'Pre-train: {i}')
                self.nsde_kappa.zero_grad()
                loss = loss_func(self.nsde_kappa(x), self.kappa*(self.theta - x))
                loss.backward()
                # nn.utils.clip_grad_norm_(self.nsde_params, 1.)
                optimizer_nsde.step()

                t_epoch.set_postfix(loss=loss.item())
                if early_stopping.early_stop or (loss.item() < 5e-8):
                    break

    def _loss_function(self, market_prices, model_prices, Z, lambd=0.1, lambd_var=0.01, weights=None):
        """Compute the training loss.

        Args:
            market_prices: target option prices, shape (n_strikes, n_maturities).
            model_prices: model option prices, same shape.
            Z: Radon-Nikodym weights, shape (N_paths, n_maturities).
            lambd: weight on the E[Z]=1 martingale penalty.
            lambd_var: weight on the Var(Z) penalty.  Penalising variance
                prevents a small number of paths from dominating the
                importance-weighted estimate.
            weights: denominator for the relative pricing error; defaults
                to market_prices + 1e-6.

        Returns:
            Scalar loss tensor.
        """
        if weights is None:
            weights = market_prices + 1e-6

        mse_loss_func = nn.MSELoss()

        def loss_func(target, output, weight):
            return torch.mean(torch.square((target - output)) / weight)

        price_loss = loss_func(market_prices, model_prices, weight=weights)
        girsanov_mean_loss = mse_loss_func(Z.mean(dim=0), torch.ones(Z.shape[1], device=device))
        return price_loss + lambd * girsanov_mean_loss

    def train(self, market_prices, strikes, maturity_idx, price_path, vol_path, dW,
              market_vix_prices=None, vix_strikes=None, vix_path=None,
              n_epoch=5000, weights=None):
        optimizer_nsde = torch.optim.Adam(self.nsde_params, lr=5e-4)  # 5e-5 5e-4
        # optimizer_nsde = torch.optim.RMSprop(self.nsde_params, lr=1e-4)
        scheduler_nsde = torch.optim.lr_scheduler.StepLR(optimizer_nsde, 50, gamma=0.99)
        # scheduler_nsde = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_nsde, 'min',
        #                                                             factor=0.1, patience=50,
        #                                                             threshold=5e-8, threshold_mode='abs',
        #                                                             cooldown=20, verbose=True)
        early_stopping = EarlyStopping(tolerance=200, min_delta=0.)
        # calculate payoffs of the sample model paths
        payoff_matrix = self.get_payoff_matrix(strikes, maturity_idx, price_path)
        if market_vix_prices is not None:
            vix_payoff_matrix = self.get_payoff_matrix(vix_strikes, maturity_idx, vix_path)

        old_loss = 1000.
        with tqdm(range(n_epoch), unit='ep') as t_epoch:
            for i in t_epoch:
                t_epoch.set_description(f'Epoch {i}')

                self.nsde_kappa.zero_grad()
                Z = self.get_girsanov_matrix_diff(maturity_idx, vol_path, dW)
                model_prices = self.get_model_price(payoff_matrix, Z)
                if market_vix_prices is not None:
                    model_vix_prices = self.get_model_price(vix_payoff_matrix, Z)

                loss = self._loss_function(market_prices, model_prices, Z, lambd=0.1, weights=weights)
                if market_vix_prices is not None:
                    loss += self._loss_function(market_vix_prices, model_vix_prices, Z, lambd=0., weights=None)

                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.nsde_params, 1.)
                optimizer_nsde.step()
                scheduler_nsde.step()

                t_epoch.set_postfix(
                    lr=f'{scheduler_nsde.get_last_lr()[0]:.2e}',
                    loss=f'{loss.item():.2e}',
                    grad=f'{grad_norm:.2e}',
                    Z_mean=f'{Z.mean().item():.3f}',
                    Z_std=f'{Z.std().item():.3f}',
                )

                early_stopping(new_loss=loss.item(), old_loss=old_loss)
                if early_stopping.early_stop:
                    print(f' Final loss: {loss.item():.4e}', end=' ')
                    print(f'Final MSE: {torch.mean(torch.square((model_prices - market_prices))).item():.4e}')
                    break

                old_loss = loss.item()

        with torch.no_grad():
            self.price_matrix = self.get_model_price(payoff_matrix, Z).detach()

    def get_price_matrix(self, strikes, maturity_idx, price_path, vol_path, dW):
        with torch.no_grad():
            Z = self.get_girsanov_matrix_diff(maturity_idx, vol_path, dW)
            payoff_matrix = self.get_payoff_matrix(strikes, maturity_idx, price_path)
            price_matrix = self.get_model_price(payoff_matrix, Z).detach()
        return price_matrix

    def generate_nsde_paths(self, T, n_steps, N_paths, scheme='full-truncation', no_grad=True):
        if no_grad:
            with torch.no_grad():
                dt = T / n_steps
                self.dt = dt

                price_path, vol_path = torch.zeros((N_paths, n_steps + 1), device=device), \
                                       torch.zeros((N_paths, n_steps + 1), device=device)
                price_path[:, 0], vol_path[:, 0] = np.log(self.S0), self.V0

                def discretize_price_func(v, f, dB):
                    return (self.r - 0.5 * f(v)) * dt + torch.sqrt(f(v)) * dB

                def discretize_vol_func(v, f1, f2, dB):
                    return v + self.nsde_kappa(f1(v[:, None])).detach().squeeze() * dt + self.s(f2(v)) * dB

                # TODO: Milstein correction:
                # + 0.5 * self.nsde_kappa(f1(v)) * self.grad_nsde_kappa(f1(v)) * (dB ** 2 - self.dt)

                cov = np.array([[1, self.rho], [self.rho, 1]])
                dW = np.sqrt(dt) * np.random.multivariate_normal(np.zeros(2), cov=cov, size=(N_paths, n_steps + 1))
                dW = torch.tensor(dW.astype(np.float32), device=device)

                for i in range(n_steps):
                    price_path[:, i + 1] = discretize_price_func(vol_path[:, i], torch_max_func, dW[:, i, 0])
                    # price_path[:, i + 1] = price_path[:, i+1]*np.exp(price_path[:, i + 1])

                    if scheme == 'full-truncation':
                        vol_path[:, i + 1] = discretize_vol_func(vol_path[:, i],
                                                                 torch_max_func, torch_max_func, dW[:, i, 1])
                    elif scheme == 'partial-truncation':
                        vol_path[:, i + 1] = discretize_vol_func(vol_path[:, i],
                                                                 lambda x: x, torch_max_func, dW[:, i, 1])
                    elif scheme == 'reflection':
                        vol_path[:, i + 1] = discretize_vol_func(vol_path[:, i],
                                                                 np.abs, np.abs, dW[:, i, 1])
                    else:
                        raise ValueError('Wrong scheme.')

                price_path = self.S0 * torch.exp(torch.cumsum(price_path, dim=1))
                vol_path = torch_max_func(vol_path)
        return price_path, vol_path, dW


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    market_params = dict(V0=0.02, kappa=1.5, theta=0.04, nu=0.5, rho=-0.7)
    np_data = np.load('./data/heston_{}={V0}_{}={kappa}_{}={theta}_{}={nu}_{}={rho}.npz'.format(
        *market_params, **market_params))
    strikes = np_data['K'].astype(np.float32)
    maturities = np_data['T'].astype(np.float32)
    market_prices = np_data['prices'].astype(np.float32)

    vix_data_path = './data/heston_VIX_{}={V0}_{}={kappa}_{}={theta}_{}={nu}_{}={rho}.npz'.format(
        *market_params, **market_params)
    if os.path.exists(vix_data_path):
        np_vix_data = np.load(vix_data_path)
        vix_strikes = np_vix_data['K'].astype(np.float32)
        vix_maturities = np_vix_data['T'].astype(np.float32)
        market_vix_prices = np_vix_data['prices'].astype(np.float32)
    else:
        vix_strikes = vix_maturities = market_vix_prices = None

    n_steps = 200
    N_paths = 100000

    model = SampleModel(V0=0.02, kappa=2.5, theta=0.04, nu=0.5, rho=-0.7)
    np_paths = model.generate_paths(T=maturities[-1], n_steps=n_steps, N_paths=N_paths,
                                    scheme='full-truncation', antithetic=True)
    np_vix_path = model.get_VIX(np_paths[1])

    time_grid = np.linspace(0, maturities[-1], n_steps + 1)
    S, V, dW = tuple(torch.from_numpy(path.astype(np.float32)) for path in np_paths)
    VIX = torch.from_numpy(np_vix_path.astype(np.float32))

    # problems with Radon-Nikodym
    V = torch.clamp(V, min=5e-4)  # 1e-4 5e-5

    maturity_index = searchsorted(time_grid, maturities)

    # training weights
    weights = get_vega(np.ones_like(strikes)*model.S0, market_prices, strikes, maturities)
    weights = np.nan_to_num(weights, nan=np.nanmin(weights), copy=True) + 1e-3

    nsde = NSDE(model, pre_train=True)

    # nsde.get_control_variate(strikes=torch.from_numpy(strikes).to(device),
    #                          maturities=torch.from_numpy(maturity_index).to(device),
    #                          price_path=S.to(device), vol_path=V.to(device), dW=dW.to(device))

    # before training
    tx = torch.linspace(0, 0.5, 100, device=device)
    npx = tx.cpu().numpy()
    nn_kappa = nsde.nsde_kappa(tx[:, None]).detach().cpu().numpy()

    sample_model_price = nsde.get_price_matrix(torch.from_numpy(strikes).to(device),
                                               torch.from_numpy(maturity_index).to(device),
                                               S.to(device),
                                               V.to(device),
                                               dW.to(device)).cpu().numpy()

    fig, ax = plt.subplots(3, figsize=(6, 12))
    ax[0].plot(npx, nn_kappa, '--', label=r'$\kappa^\mathcal{NN}_0$')
    ax[0].plot(npx, market_params['kappa'] * (market_params['theta'] - npx), 'r-', label='true')

    plot_IV_slices(market_prices, strikes, maturities,
                   S0=np.ones_like(strikes) * nsde.S0, r=0., typ='c', ax=ax[1], linestyle='--')
    plot_IV_slices(sample_model_price, strikes, maturities,
                   S0=np.ones_like(strikes) * nsde.S0, r=0., typ='c', ax=ax[1])

    nsde.train(market_prices=torch.from_numpy(market_prices).to(device),
               strikes=torch.from_numpy(strikes).to(device),
               maturity_idx=torch.from_numpy(maturity_index).to(device),
               price_path=S.to(device), vol_path=V.to(device), dW=dW.to(device),
               # market_vix_prices=torch.from_numpy(market_vix_prices).to(device),
               # vix_strikes=torch.from_numpy(vix_strikes).to(device), vix_path=VIX.to(device),
               weights=torch.from_numpy(weights).to(device),
               n_epoch=10000)

    # nsde_price_path, nsde_vol_path, _ = nsde.generate_nsde_paths(maturities[-1],
    #                                                              n_steps=n_steps, N_paths=N_paths)
    #
    # nsde_price_path, nsde_vol_path = nsde_price_path.cpu().numpy(), nsde_vol_path.cpu().numpy()

    # plt.plot(time_grid, nsde_price_path.T)
    # plt.show()

    # plt.plot(time_grid, nsde_vol_path.T)
    # plt.show()

    nn_kappa = nsde.nsde_kappa(tx[:, None]).detach().cpu().numpy()
    ax[0].plot(npx, nn_kappa, ':', label=r'$\kappa^\mathcal{NN}_n$')
    ax[0].minorticks_on()
    ax[0].legend(loc='best')

    nsde_option_price = nsde.get_price_matrix(torch.from_numpy(strikes).to(device),
                                              torch.from_numpy(maturity_index).to(device),
                                              S.to(device),
                                              V.to(device),
                                              dW.to(device)).cpu().numpy()
    # ax[1].plot(strikes, nsde_option_price)
    # ax[1].plot(strikes, market_prices, '--')

    plot_IV_slices(market_prices, strikes, maturities,
                   S0=np.ones_like(strikes)*nsde.S0, r=0., typ='c', ax=ax[2], linestyle='--')
    plot_IV_slices(nsde_option_price, strikes, maturities,
                   S0=np.ones_like(strikes)*nsde.S0, r=0., typ='c', ax=ax[2])

    fig.show()

    print('done')




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
