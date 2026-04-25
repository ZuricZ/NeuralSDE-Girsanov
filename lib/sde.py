import torch
import torch.nn as nn
from lib.networks import FFNN, ResFFNN
from torchsde import BrownianInterval, sdeint
import numpy as np
import matplotlib.pyplot as plt


class SDE(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.phi = FFNN(sizes=[2, 15, 15, 1], activation=nn.ReLU, output_activation=nn.Identity).to(device)
        self.sde_type = 'ito'
        self.noise_type = 'diagonal'

    def f(self, t, y):
        return torch.zeros_like(y, device=device)

    def g(self, t, y):
        return - y * self.phi(torch.cat([t*torch.zeros_like(y, device=device), y], dim=1))


if __name__ == '__main__':
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

    batch_size, state_size, t_size = 3, 1, 100
    ts = torch.linspace(0, 1, t_size, device=device)
    y0 = torch.full(size=(batch_size, state_size), fill_value=0.1, device=device)

    W = torch.from_numpy(np.sqrt(1/t_size)*np.random.rand(batch_size, state_size, t_size).astype(np.float32)).to(device)

    sde = SDE(device=device)
    bm = BrownianInterval(t0=ts[0],
                          t1=ts[-1],
                          W=W,
                          dt=1/t_size,
                          device=device)
    ys = sdeint(sde=sde, y0=y0, ts=ts,
                # bm=bm,
                method='srk'
                )

    plt.plot(ts.detach().cpu().numpy(), ys.detach().cpu().numpy()[:, :, 0])
    plt.show()

    print('done.')
