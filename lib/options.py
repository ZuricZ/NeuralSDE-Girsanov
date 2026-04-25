import torch
from dataclasses import dataclass
from abc import abstractmethod
from typing import List


@dataclass
class BaseOption:
    pass

    @abstractmethod
    def payoff(self, x: torch.Tensor, **kwargs):
        ...


class Lookback(BaseOption):

    def __init__(self, idx_traded: List[int] = None):
        self.idx_traded = idx_traded  # indices of traded assets. If None, then all assets are traded

    def payoff(self, x, **kwargs):
        """
        Parameters
        ----------
        x: torch.Tensor
            Path history. Tensor of shape (batch_size, N, d) where N is path length
        Returns
        -------
        payoff: torch.Tensor
            lookback option payoff. Tensor of shape (batch_size,1)
        """
        if self.idx_traded:
            basket = torch.sum(x[..., self.idx_traded], 2)  # (batch_size, N)
        else:
            if x.dim() < 3:
                basket = x
            else:
                basket = torch.sum(x, 2)  # (batch_size, N)
        payoff = torch.max(basket, 1)[0] - basket[:, -1]  # (batch_size)
        return payoff.unsqueeze(1)  # (batch_size, 1)


def vix2(ts, sigma):
    """
    Parameters
    ----------
    ts: torch.Tensor
        timegrid. Tensor of shape (N)
    sigma: torch.Tensor
        Vol history. Tensor of shape (batch_size, N-1, 1) where N is path length
    Returns
    -------
    payoff: torch.Tensor
        VIX futures. Tensor of shape (batch_size, 1)
    """
    batch_size = sigma.shape[0]
    increments = (ts[1:] - ts[:-1]).reshape(1, -1, 1).repeat(batch_size, 1, 1)  # (batch_size, N-1, 1)
    payoff = (increments * sigma ** 2).sum(1)  # (batch_size, 1)
    return 1 / (ts[-1] - ts[0]) * payoff


def vix2_log(ts, x):
    batch_size = x.shape[0]
    return -2 / (ts[-1] - ts[0]) * torch.log(x[:, -1, :] / x[:, 0, :])  # (batch_size, 1)


class Vanilla_Call(BaseOption):
    """
    Vectorized call option

    Attributes
    ----------
    self.K: torch.Tensor
        Tensor of size (1, N_strikes, 1)
    """

    def __init__(self, K: torch.tensor):
        """
        Parameters
        ----------
        K: torch.Tensor
            Strikes. Tensor of size (N_strikes)
        """
        self.K = K.reshape(1, -1, 1)

    def payoff(self, x, **kwargs):
        """
        Parameters
        ----------
        x: torch.Tensor
            Asset price at terminal time. Tensor of shape (batch_size, 1)

        Returns
        -------
        payoff: torch.Tensor
            Vanilla call option payoff. Tensor of shape (batch_size, N_strikes)
        """
        payoff = torch.clamp(x.unsqueeze(1) - self.K, 0)  # (batch_size, N_strikes, 1)
        return payoff.squeeze(2)


class Vanilla_Put(BaseOption):
    """
    Vectorized call option

    Attributes
    ----------
    self.K: torch.Tensor
        Tensor of size (1, N_strikes, 1)
    """

    def __init__(self, K: torch.tensor):
        """
        Parameters
        ----------
        K: torch.Tensor
            Strikes. Tensor of size (N_strikes)
        """
        self.K = K.reshape(1, -1, 1)

    def payoff(self, x, **kwargs):
        """
        Parameters
        ----------
        x: torch.Tensor
            Asset price at terminal time. Tensor of shape (batch_size, 1)

        Returns
        -------
        payoff: torch.Tensor
            Vanilla call option payoff. Tensor of shape (batch_size, N_strikes)
        """
        payoff = torch.clamp(self.K - x.unsqueeze(1), 0)  # (batch_size, N_strikes, 1)
        return payoff.squeeze(2)
