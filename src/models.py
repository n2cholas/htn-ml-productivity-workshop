from typing import Callable, Sequence, Tuple

from flax import linen as nn
from shapecheck import check_shapes


class FullyConnected(nn.Module):
    n_classes: int
    layer_sizes: Sequence[int]
    activation: Callable = nn.relu

    @nn.compact
    @check_shapes(None, 'N,W,W,1', out_='N,-1')
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        for h_size in self.layer_sizes:
            x = nn.Dense(h_size)(x)
            x = self.activation(x)
        return nn.Dense(self.n_classes)(x)


class Convolutional(nn.Module):
    n_classes: int
    layer_sizes: Sequence[int]
    kernel_size: Tuple[int, int] = (3, 3)
    activation: Callable = nn.relu

    @nn.compact
    @check_shapes(None, 'N,W,W,1', out_='N,-1')
    def __call__(self, x):
        for h_size in self.layer_sizes:
            x = nn.Conv(h_size, self.kernel_size)(x)
            x = self.activation(x)
        return nn.Dense(self.n_classes)(x.mean((1, 2)))
