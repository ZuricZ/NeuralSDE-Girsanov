import numpy as np
import torch
import scipy.stats
import py_vollib
# from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black.implied_volatility import implied_volatility as iv
from py_vollib.black.greeks.analytical import vega as vega
# import pandas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.cm as cm
# from heston_VIX import heston_VIX2
import pickle


class IV_numpy:

    def __init__(self, S, K, T, r, q=0.0):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.n = scipy.stats.norm.pdf
        self.N = scipy.stats.norm.cdf
        self.MAX_ITERATIONS = 10
        self.PRECISION = 1.0e-5

    def bs_price(self, cp_flag, v):
        d1 = (np.log(self.S / self.K) + (self.r + v * v / 2.) * self.T) / (v * np.sqrt(self.T))
        d2 = d1 - v * np.sqrt(self.T)
        if cp_flag == 'c':
            price = self.S * np.exp(-self.q * self.T) * self.N(d1) - self.K * np.exp(-self.r * self.T) * self.N(d2)
        else:
            price = self.K * np.exp(-self.r * self.T) * self.N(-d2) - self.S * np.exp(-self.q * self.T) * self.N(-d1)
        return price

    def bs_vega(self, v):
        d1 = (np.log(self.S / self.K) + (self.r + v * v / 2.) * self.T) / (v * np.sqrt(self.T))
        return self.S * np.sqrt(self.T) * self.n(d1)

    def find_vol(self, target_value, flag):
        sigma = 1.0 * np.ones_like(target_value)
        for i in range(0, self.MAX_ITERATIONS):
            price = self.bs_price(flag, sigma)
            vega = self.bs_vega(sigma) + abs(np.random.randn(*sigma.shape))*1e-7  # to keep vega from zero

            price = price
            diff = target_value - price  # our root

            if (abs(diff[~np.isnan(diff)]) < self.PRECISION).all():
                return sigma
            # value_below_intrinsic = sigma <= 0
            # sigma = sigma * ~value_below_intrinsic
            sigma[sigma < 0] = np.nan
            sigma[abs(diff) >= self.PRECISION] = sigma[abs(diff) >= self.PRECISION] \
                                                 + (diff / vega)[abs(diff) >= self.PRECISION]  # f(x) / f'(x)

        # value wasn't found, return best guess so far
        return sigma


def IV_newton(S, target, strikes, mat_times, r, flag='c'):
    ivs = []
    for i, mat in enumerate(mat_times):
        iv = IV_numpy(S, K=strikes, T=mat, r=r, q=0.0)
        ivs.append(iv.find_vol(target[:, i], flag))
    return np.stack(ivs, axis=1)


def IV_lib(F, target, strikes, mat_times, r, flag='c'):
    assert target.shape == (strikes.shape[0], mat_times.shape[0])
    mat = np.array(np.meshgrid(mat_times, strikes)).T.reshape(-1, 2)
    mat = np.c_[F.repeat(mat.shape[0] // F.shape[0]), mat]
    if isinstance(r, (list, tuple, np.ndarray)):
        mat = np.c_[mat, np.array(r).repeat(mat.shape[0] // F.shape[0])]
    mat = np.c_[target.T.reshape(-1), mat]

    def iv_with_exception_handling(price, F, K, r, T, flag):
        from py_lets_be_rational.exceptions import BelowIntrinsicException
        try:
            return iv(price, F, K, r, T, flag)  # py_vollib.black and py_vollib.black_scholes DIFFERENT ORDER PARAMS!!!
        except BelowIntrinsicException:
            return np.nan

    if isinstance(r, (list, tuple, np.ndarray)):
        IVs = np.apply_along_axis(lambda row: iv_with_exception_handling(row[0], row[1], row[3], row[4], row[2], flag),
                                  1, mat).reshape(target.shape[0], -1, order='F')
    else:
        IVs = np.apply_along_axis(lambda row: iv_with_exception_handling(row[0], row[1], row[3], r, row[2], flag),
                                  1, mat).reshape(target.shape[0], -1, order='F')

    # IVs = np.array([[iv_with_exception_handling(target[i, j], F[j], strikes[i], r, mat_times[j], flag)
    #                  for j in range(len(mat_times))] for i in range(len(strikes))])

    return IVs


def vega_lib(F, IVs, strikes, mat_times, r, flag='c'):
    assert IVs.shape == (strikes.shape[0], mat_times.shape[0])
    mat = np.array(np.meshgrid(mat_times, strikes)).T.reshape(-1, 2)
    mat = np.c_[F.repeat(mat.shape[0] // F.shape[0]), mat]
    mat = np.c_[IVs.T.reshape(-1), mat]

    def vega_with_exception_handling(iv, F, K, r, T, flag):
        from py_lets_be_rational.exceptions import BelowIntrinsicException
        try:
            return vega(flag, F, K, T, r, iv)
        except BelowIntrinsicException:
            return np.nan

    vegas = np.apply_along_axis(lambda row: vega_with_exception_handling(row[0], row[1], row[3], r, row[2], flag),
                                1, mat).reshape(IVs.shape[0], -1, order='F')

    return vegas


def get_vega(S, market_prices, strikes, maturities, flag='c'):
    ivs = IV_lib(S, target=market_prices, strikes=strikes, mat_times=maturities, r=0., flag=flag)
    return vega_lib(S, ivs, strikes, maturities, 0., flag='c')  # TODO: interest rate!!



if __name__ == '__main__':
    np_data = np.load('../data/heston_sigma=0.8_kappa=1.5_theta=0.04_nu=0.5_rho=-0.6.npz')
    strikes = np_data['K'].astype(np.float32)
    maturities = np_data['T'].astype(np.float32)
    market_prices = np_data['prices'].astype(np.float32)
    S = np.ones_like(strikes)

    iv_array = IV_lib(S, target=market_prices, strikes=strikes, mat_times=maturities, r=0., flag='c')

    plt.plot(strikes, iv_array)
    plt.show()

    print('done.')
