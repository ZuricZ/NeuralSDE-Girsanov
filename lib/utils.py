import numpy as np
import torch
import torch.nn as nn
from numba import jit, njit, prange, float64, complex64
from numba import cuda
from functools import wraps
from time import time
from lib.compute_iv import IV_lib
from bottleneck import push


def dstack_product(x, y):
    return np.dstack(np.meshgrid(x, y)).reshape(-1, 2)


def max_func(x):
    return np.maximum(x, 0)


def torch_max_func(x):
    return nn.functional.relu(x)


def searchsorted(known_array, test_array):
    index_sorted = np.argsort(known_array)
    known_array_sorted = known_array[index_sorted]
    known_array_middles = known_array_sorted[1:] - np.diff(known_array_sorted.astype('f'))/2
    idx1 = np.searchsorted(known_array_middles, test_array)
    indices = index_sorted[idx1]
    return indices


# @jit(parallel=True, forceobj=True)
def simpson(f, a, b, args, n=5000):
    """Approximates the definite integral of f from a to b by the
    composite Simpson's rule, using n subintervals (with n even)"""

    if n % 2:
        raise ValueError(f'n must be even (received n={n})')

    h = (b - a) / n
    s = f(a, *args) + f(b, *args)

    for i in range(1, n, 2):
        s += 4 * f(a + i * h, *args)
    for i in range(2, n-1, 2):
        s += 2 * f(a + i * h, *args)

    return s * h / 3


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        # print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te-ts))
        print('func:%r took: %2.4f sec' % (f.__name__, te - ts))
        return result
    return wrap


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            m.bias.data.fill_(1e-4)
            # nn.init.zeros_(m.bias)


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


def calc_weights(S0, strikes, maturities, market_prices, model_prices, weights, i):
    if (i % 500 == 0) and (i > 0):
        np_strikes = strikes.cpu().numpy()
        np_maturities = maturities.cpu().numpy()

        with torch.no_grad():
            market_ivs = IV_lib(S0 * np.ones_like(np_strikes),
                                market_prices.cpu().numpy(),
                                np_strikes,
                                np_maturities, r=0, flag='c')
            # make sure there is no NaNs
            market_ivs = push(market_ivs, axis=0)
            market_ivs = np.nan_to_num(market_ivs, nan=np.nanmean(market_ivs), copy=True)

            model_ivs = IV_lib(S0 * np.ones_like(np_strikes),
                               model_prices.detach().cpu().numpy(),
                               np_strikes,
                               np_maturities, r=0, flag='c')
            # make sure there is no NaNs
            model_ivs = push(model_ivs, axis=0)
            model_ivs = np.nan_to_num(model_ivs, nan=np.nanmean(model_ivs), copy=True)

        return 1 / torch.abs(torch.from_numpy(market_ivs - model_ivs)*5e3).to(weights.get_device())
    else:
        return weights