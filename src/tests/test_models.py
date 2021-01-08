import jax
import jax.numpy as jnp

from models import Convolutional, FullyConnected


def test_conv():
    model = Convolutional(10, [5, 5])
    imgs = jnp.ones((1, 28, 28, 1), jnp.float32)
    rng = jax.random.PRNGKey(1)
    variables = model.init(rng, imgs)
    out = model.apply(variables, imgs)
    assert out.shape == (1, 10)


def test_fc():
    model = FullyConnected(10, [5, 5])
    imgs = jnp.ones((1, 28, 28, 1), jnp.float32)
    rng = jax.random.PRNGKey(1)
    variables = model.init(rng, imgs)
    out = model.apply(variables, imgs)
    assert out.shape == (1, 10)
