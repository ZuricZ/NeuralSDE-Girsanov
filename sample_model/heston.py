import numpy as np
import pandas as pd
import os
from datetime import datetime as dt
import time
import itertools
import matplotlib.pyplot as plt
from lib.utils import max_func, dstack_product, simpson, timing, shuffle_along_axis, searchsorted
import lib.compute_iv as iv
from lib.plot import plot_IV_slices
from scipy.optimize import fsolve
# from lmfit import Parameters, minimize
from scipy.optimize import dual_annealing
from scipy.stats import norm
from scipy.integrate import quad, quad_vec  # , simpson


# import quadpy

# Parallel computation using numba
# from numba import jit, njit, prange, float64, complex64
# from numba import cuda


class Heston:
    def __init__(self, S0=1, V0=0.04, r=0.0, kappa=1.5, theta=0.05, nu=.5, rho=-0.9):
        self.S0 = S0
        self.V0 = V0
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.nu = nu
        self.rho = rho

        self.dt = None

    @staticmethod
    def _char_func(u, tau, x, j, *params):
        v0, kappa, theta, nu, rho = params

        alpha = -0.5 * u ** 2 - 0.5 * 1j * u + 1j * j * u
        beta = kappa - rho * nu * j - rho * nu * 1j * u
        gamma = 0.5 * nu ** 2

        d = np.sqrt(beta ** 2 - 4 * alpha * gamma)
        r_p = (beta + d) / (nu ** 2)
        r_m = (beta - d) / (nu ** 2)

        g = r_m / r_p

        D = r_m * (1 - np.exp(-d * tau)) / (1 - g * np.exp(-d * tau))
        C = kappa * (r_m * tau - 2 / (nu ** 2) * np.log((1 - g * np.exp(-d * tau)) / (1 - g)))

        phi = np.real(np.exp(C * theta + D * v0 + 1j * u * x) / (1j * u))
        return phi

    def _P(self, tau, x, j, *params):
        # integral = simpson(self._char_func, a=1e-12, b=100., args=(tau, x, j, *params))
        integral = quad_vec(self._char_func, a=0., b=np.inf, workers=1, args=(tau, x, j, *params))[0]
        return 0.5 + integral / np.pi

    @timing
    def get_price(self, S, K, r, T, V0, kappa, theta, nu, rho):
        # Implemented from J. Gatheral Practitioner's Guide (circumventing Little Heston Trap)
        params = (V0, kappa, theta, nu, rho)
        F = S * np.exp(-r * T)
        x = np.log(F / K)
        return np.clip(K * (F / K * self._P(T, x, 1, *params) - self._P(T, x, 0, *params)), a_min=1e-11, a_max=None)

    def calibration_loss(self, model_params, option_params, market_price):
        # model_params = [V0, kappa, theta, nu, rho]
        # option_params = [S, K, r, T]
        return (market_price - self.get_price(*option_params, *model_params)) / market_price

    def calibrate(self, option_params, market_price, initial_val=None, lower_bounds=None, upper_bounds=None):
        """
        Levenberg Marquardt algorithm given volatility surface.
        option_params = [S, K, r, T]
        Function to be minimized: calibration_loss
        """
        if upper_bounds is None:
            upper_bounds = [10, 10, 10, 10, 0]
        if lower_bounds is None:
            lower_bounds = [1e-2, 1e-2, 1e-2, 1e-2, -1]
        if initial_val is None:
            initial_val = [0.5, 0.5, 0.5, 0.5, -0.5]

        bounds = list(zip(lower_bounds, upper_bounds))

        # Get a rough estimate
        results = dual_annealing(self.calibration_loss, bounds=bounds, args=[option_params, market_price],
                                 x0=initial_val)

        # Minimize the function
        # minimize(self.calibration_loss, args=[option_params, market_price])

    def generate_paths(self, T, n_steps, N_paths, scheme='full-truncation', antithetic=False):
        dt = T / n_steps
        self.dt = dt

        price_path, vol_path = np.zeros((N_paths, n_steps + 1)), np.zeros((N_paths, n_steps + 1))
        price_path[:, 0] = np.log(self.S0)
        vol_path[:, 0] = np.sqrt(self.V0) if scheme == 'BEM' else self.V0

        cov = np.array([[1, self.rho], [self.rho, 1]])

        discretize_price_func = lambda v, f, dB: (self.r - 0.5 * f(v)) * dt + np.sqrt(f(v)) * dB

        discretize_vol_func = lambda v, f1, f2, dB: v + self.kappa * (self.theta - f1(v)) * dt + \
                                                    self.nu * np.sqrt(f2(v)) * dB + \
                                                    0.25 * self.nu ** 2 * (dB ** 2 - self.dt)  # Milstein correction

        def discretize_vol_BEM_func(x, dB):
            theta_v = self.theta - 0.25 * self.nu ** 2 / self.kappa
            return (x + 0.5 * self.nu * dB + np.sqrt((x + 0.5 * self.nu * dB) ** 2 + self.kappa * theta_v * dt)) / (
                        2 + self.kappa * dt)

        if antithetic:
            dW = np.sqrt(dt) * np.random.multivariate_normal(np.zeros(2), cov=cov, size=(N_paths // 2, n_steps + 1))
            dW = np.concatenate([dW, -dW], axis=0)
            # dW = shuffle_along_axis(dW, axis=0)
        else:
            dW = np.sqrt(dt) * np.random.multivariate_normal(np.zeros(2), cov=cov, size=(N_paths, n_steps + 1))

        for i in range(n_steps):
            # price_path[:, i + 1] = price_path[:, i+1]*np.exp(price_path[:, i + 1])
            if scheme == 'full-truncation':
                price_path[:, i + 1] = discretize_price_func(vol_path[:, i], max_func, dW[:, i, 0])
                vol_path[:, i + 1] = discretize_vol_func(vol_path[:, i], max_func, max_func, dW[:, i, 1])
            elif scheme == 'partial-truncation':
                price_path[:, i + 1] = discretize_price_func(vol_path[:, i], max_func, dW[:, i, 0])
                vol_path[:, i + 1] = discretize_vol_func(vol_path[:, i], lambda x: x, max_func, dW[:, i, 1])
            elif scheme == 'reflection':
                price_path[:, i + 1] = discretize_price_func(vol_path[:, i], max_func, dW[:, i, 0])
                vol_path[:, i + 1] = discretize_vol_func(vol_path[:, i], np.abs, np.abs, dW[:, i, 1])
            elif scheme == 'BEM':  # TODO n_step+1
                price_path[:, i + 1] = discretize_price_func(vol_path[:, i]**2, max_func, dW[:, i, 0])
                vol_path[:, i + 1] = discretize_vol_BEM_func(vol_path[:, i], dW[:, i + 1, 1])
            else:
                raise ValueError('Wrong scheme.')

        price_path = self.S0 * np.exp(np.cumsum(price_path, axis=1))
        if scheme == 'BEM':
            vol_path = vol_path**2
        else:
            vol_path = max_func(vol_path)
        return price_path, vol_path, dW

    def get_VIX(self, vol_path, scaled=False):
        delta = 30 / 365
        a = (1 - np.exp(-self.kappa * delta)) / (self.kappa * delta)
        b = self.theta * (1 - a)

        if scaled:
            return 100 * np.sqrt((a * vol_path + b))
        else:
            return np.sqrt(a * vol_path + b)

    @timing
    def get_VIX_price(self, vix_strikes, vix_maturities, n_steps=100, N_paths=10 ** 6):
        _, vol_path, _ = heston.generate_paths(1, n_steps, N_paths, scheme='full-truncation', antithetic=True)
        vix_path = self.get_VIX(vol_path)
        time_grid = np.linspace(0, maturities[-1], n_steps + 1)
        maturity_index = searchsorted(time_grid, vix_maturities).astype(int)

        vix_path = vix_path[:, :, None].repeat((vix_strikes.shape[0]), axis=2)

        vix_prices = np.zeros((vix_strikes.shape[0], vix_maturities.shape[0]))
        for i, idx in enumerate(maturity_index):
            vix_prices[:, i] = max_func(vix_path[:, idx] - vix_strikes).mean(axis=0)

        return vix_prices


if __name__ == '__main__':
    heston = Heston()
    S, V, _ = heston.generate_paths(1, 100, 10000, scheme='full-truncation', antithetic=True)
    plt.plot(V[:1000, :].T)
    plt.show()
    strikes = np.linspace(0.75, 1.25, 25)
    maturities = np.arange(1, 7) * 1 / 12  # np.linspace(0.25, .5, 5)
    prod_mat = dstack_product(strikes, maturities)

    params = dict(V0=0.02, kappa=1.5, theta=0.06, nu=.5, rho=-0.7)

    prices = heston.get_price(S=1., K=prod_mat[:, 0], r=0., T=prod_mat[:, 1],
                              **params)
    # prices = np.concatenate(np.split(prices[:, 1, None], maturities.shape[0]), axis=1)
    prices = prices[:, None].reshape(strikes.shape[0], maturities.shape[0], order='F')
    # plt.plot(prices)
    # plt.show()

    np.savez('../data/heston_{}={V0}_{}={kappa}_{}={theta}_{}={nu}_{}={rho}.npz'.format(*params, **params),
             **{'K': strikes, 'T': maturities, 'prices': prices})
    plot_IV_slices(prices, strikes, maturities, S0=np.ones_like(strikes), r=0.)
    plt.show()

    VIX0 = heston.get_VIX(params['V0'])
    vix_strikes = strikes * VIX0
    vix_prices = heston.get_VIX_price(vix_strikes, maturities)

    plt.plot(vix_prices)
    plt.show()

    plot_IV_slices(vix_prices, vix_strikes, maturities,
                   S0=np.ones_like(strikes) * VIX0, r=0.)
    plt.show()

    np.savez('../data/heston_VIX_{}={V0}_{}={kappa}_{}={theta}_{}={nu}_{}={rho}.npz'.format(*params, **params),
             **{'K': vix_strikes, 'T': maturities, 'prices': vix_prices})

    print('done')

# @jit
# def _pricing_helpr(s, St, K, r, T, sigma, kappa, theta, nu, rho):
#     # https://github.com/CalebMigosi/code-more/blob/8eec3677944b018f21307f5de0636663d51250f9/EquityOptionsPricing/py/Heston%20Pricing%202.py
#     prod = rho * sigma * I * s
#
#
#     # Calculate d
#     d1 = (prod - kappa) ** 2
#     d2 = (sigma ** 2) * (I * s + s ** 2)
#     d = np.sqrt(d1 + d2)
#
#     # Calculate g
#     g1 = kappa - prod - d
#     g2 = kappa - prod + d
#     g = g1 / g2
#
#     # Calculate first exponential
#     exp1 = np.exp(np.log(St) * I * s) * np.exp(I * s * r * T)
#     exp2 = 1 - g * np.exp(-d * T)
#     exp3 = 1 - g
#     term1 = exp1 * np.power(exp2 / exp3, -2 * theta * kappa / (sigma ** 2))
#
#     # Calculate second exponential
#     exp4 = theta * kappa * T / (sigma ** 2)
#     exp5 = nu / (sigma ** 2)
#     exp6 = (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
#     term2 = np.exp((exp4 * g1) + (exp5 * g1 * exp6))
#
#     return term1 * term2
#
# @jit(forceobj=True)  # (allow for parallel processing with numba)
# def get_price(self, S, K, r, T, sigma, kappa, theta, nu, rho):
#     P, iterations, max_number = 0, 1000, 100
#     ds = max_number / iterations
#
#     element1 = 0.5 * (S - K * np.exp(-r * T))
#
#     # Calculate the complex integral
#     # Using j instead of i to avoid confusion
#     for j in prange(1, iterations):
#         s1 = ds * (2 * j + 1) / 2
#         s2 = s1 - I
#
#         numerator1 = self._pricing_helpr(s2, S, K, r, T, sigma, kappa, theta, nu, rho)
#         numerator2 = K * self._pricing_helpr(s1, S, K, r, T, sigma, kappa, theta, nu, rho)
#         denominator = np.exp(np.log(K) * I * s1) * I * s1
#
#         P = P + ds * (numerator1 - numerator2) / denominator
#
#     element2 = P / np.pi
#
#     return np.real((element1 + element2))
