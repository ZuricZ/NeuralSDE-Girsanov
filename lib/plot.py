import numpy as np
import torch
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.cm as cm
import matplotlib
from cycler import cycler

from lib.compute_iv import IV_lib


def surf_plot_IVs(IVs, maturity_times, strikes, type='c', show=True):
    fig = plt.figure(np.random.randint(100) + 100)
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(maturity_times, strikes)

    if type == 'c':
        surf = ax.plot_surface(X, Y, IVs[:len(maturity_times), :].T, rstride=1, cstride=1, cmap=cm.coolwarm,
                               linewidth=0.1)
    else:
        surf = ax.plot_surface(X, Y, IVs[len(maturity_times):, :].T, rstride=1, cstride=1, cmap=cm.coolwarm,
                               linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    if show:
        plt.show()
    else:
        pickle.dump(fig, open(f'IV_{type}_surf.fig.pickle', 'wb'))
        # figx = pickle.load(open('FigureObject.fig.pickle', 'rb'))
        # figx.show()
        plt.savefig(f'IV_{type}_surf.png')


def plot_IV_slices(option_prices, strikes, maturities, S0=None, r=0., typ='c', ax=None, ignore_na=False, **kwargs):
    if S0 is None:
        S0 = np.ones_like(strikes)
    ivs = IV_lib(S0, target=option_prices, strikes=strikes, mat_times=maturities, r=r, flag=typ)
    ivs[ivs <= 0] = np.nan

    if ignore_na:
        mask = np.all(np.isfinite(ivs), axis=1)
        ivs = ivs[mask]
        strikes = strikes[mask]

    cm = plt.get_cmap('Dark2', len(maturities))
    color_list = [matplotlib.colors.rgb2hex(cm(i)) for i in range(cm.N)]
    if ax is None:
        plt.plot(strikes, ivs, **kwargs)
    else:
        ax.set_prop_cycle(cycler('color', color_list))
        ax.plot(strikes, ivs, **kwargs)
        # ax.set_ylim([np.nanmin(ivs), np.nanmax(ivs)])
