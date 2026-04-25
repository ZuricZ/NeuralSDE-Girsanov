import torch
import torch.nn as nn
from collections import namedtuple
from typing import Tuple, List, Union


class FFNN(nn.Module):
    def __init__(self, sizes: Union[Tuple[int], List[int]], activation=nn.ReLU, output_activation: nn = nn.Identity,
                 bias: bool = True):
        super().__init__()

        layers = []
        for j in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[j], sizes[j + 1], bias=bias))
            if j < (len(sizes) - 2):
                layers.append(activation())
            else:
                layers.append(output_activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, activation=nn.ReLU, bias: bool = True):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.activation = activation()
        self.create_residual_connection = True if input_dim == output_dim else False

    def forward(self, x):
        y = self.activation(self.linear(x))
        if self.create_residual_connection:
            y = x + y
        return y


class ResFFNN(nn.Module):
    def __init__(self, sizes: Union[Tuple[int], List[int]], flatten: bool = False,
                 activation=nn.ReLU, output_activation=nn.Identity, bias: bool = True):
        """
        Feedforward neural network with residual connection.
        Args:
            size: list of integers, specifies the hidden dimensions of each layer.
        """
        super(ResFFNN, self).__init__()
        blocks = list()
        self.input_dim = sizes[0]
        self.flatten = flatten
        input_dim_block = sizes[0]
        for hidden_dim in sizes[1:-1]:
            blocks.append(ResidualBlock(input_dim_block, hidden_dim, bias=bias, activation=activation))
            input_dim_block = hidden_dim
        blocks.append(nn.Linear(input_dim_block, sizes[-1], bias=bias))
        self.network = nn.Sequential(*blocks)
        self.blocks = blocks
        self.output_activation = output_activation()

    def forward(self, x):
        if self.flatten:
            x = x.reshape(x.shape[0], -1)
        out = self.network(x)
        return self.output_activation(out)
