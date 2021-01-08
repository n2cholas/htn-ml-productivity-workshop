import jax.numpy as jnp
import pytest
from hydra.experimental import compose, initialize

from main import accuracy, get_cross_entropy_loss, main


def test_loss():
    labels = jnp.array([0, 2])
    logits = jnp.array([[-1., 2., 1.], [5., -6., 2.]])
    loss_fn = get_cross_entropy_loss(from_logits=True)

    assert loss_fn(logits, labels) == pytest.approx(3.1988, 0.01)


# def test_label_smooth():
#     labels = jnp.array([0, 2])
#     logits = jnp.array([[-1., 2., 1.], [5., -6., 2.]])

#     assert cross_entropy_loss(logits, labels,
#                               label_smooth_amt=0.4) == pytest.approx(4.47833, 0.01)


def test_accuracy():
    labels = jnp.array([0, 2, 1])
    logits = jnp.array([[-1., 2., 1.], [5., -6., 2.], [-1., 0., -2.]])
    assert accuracy(logits, labels) == pytest.approx(0.333, 0.01)


@pytest.mark.slow
def test_main():
    # Ensure our training script runs top to bottom without errors.
    initialize(config_path='../conf')
    overrides = dict(num_steps=5,
                     train_bs=2,
                     val_bs=2,
                     test_mode=True,
                     eval_freq=2,
                     report_freq=2)
    cfg = compose("config.yaml", overrides=[f'{k}={v}' for k, v in overrides.items()])
    main(cfg)
