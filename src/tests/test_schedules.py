import jax
import numpy as np

from schedules import (cos_onecycle_lr, cos_onecycle_momentum, exponential_decay,
                       linear_onecycle_lr, linear_onecycle_momentum)


def test_linear_onecycle_lr():
    x = np.arange(10)
    fn = jax.vmap(
        linear_onecycle_lr(6,
                           max_lr=0.01,
                           pct_start=0.3,
                           pct_final_stage=0.85,
                           div_factor=25.0,
                           final_div_factor=1e4))

    np.testing.assert_allclose(fn(x), [
        3.9999999e-04, 5.7333331e-03, 9.4181821e-03, 6.5090908e-03, 3.5999995e-03,
        6.9090910e-04, 4.0046871e-08, 4.0000000e-08, 4.0000000e-08, 4.0000000e-08
    ])


def test_linear_onecycle_mom():
    x = np.arange(10)
    fn = jax.vmap(
        linear_onecycle_momentum(6,
                                 base_momentum=0.85,
                                 max_momentum=0.95,
                                 pct_start=0.3,
                                 pct_final_stage=0.85))

    np.testing.assert_allclose(fn(x), [
        0.95, 0.89444447, 0.8560606, 0.8863637, 0.9166667, 0.9469697, 0.95, 0.95, 0.95,
        0.95
    ])


def test_cos_onecycle_lr():
    x = np.arange(10)
    fn = jax.vmap(
        cos_onecycle_lr(6,
                        max_lr=0.01,
                        pct_start=0.3,
                        div_factor=25.0,
                        final_div_factor=1e4))

    np.testing.assert_allclose(fn(x), [
        3.9999932e-04, 6.0335104e-03, 9.9441539e-03, 8.1174569e-03, 4.6263705e-03,
        1.3347758e-03, 4.0000000e-08, 4.0000000e-08, 4.0000000e-08, 4.0000000e-08
    ])


def test_cos_onecycle_mom():
    x = np.arange(10)
    fn = jax.vmap(
        cos_onecycle_momentum(6, base_momentum=0.85, max_momentum=0.95, pct_start=0.3))

    np.testing.assert_allclose(fn(x), [
        0.95,
        0.8913176,
        0.85055846,
        0.8688255,
        0.9037365,
        0.9366526,
        0.95,
        0.95,
        0.95,
        0.95,
    ])


def test_exp_decay_rate():
    init_value = 0.01
    decay_rate = 0.3
    num_decay_steps = 4
    num_delay_steps = 3
    num_steps = 20
    fn = exponential_decay(init_value=init_value,
                           decay_rate=decay_rate,
                           num_decay_steps=num_decay_steps,
                           num_delay_steps=num_delay_steps)
    expected_list = [init_value] * num_delay_steps + [
        init_value * decay_rate**min(i, num_decay_steps)
        for i in range(num_steps - num_delay_steps)
    ]
    np.testing.assert_allclose(jax.vmap(fn)(np.arange(num_steps)),
                               expected_list,
                               rtol=1e-6)


def test_exp_decay_final_val():
    init_value = 0.01
    final_value = 0.0001
    num_decay_steps = 4
    num_delay_steps = 3
    num_steps = 20
    fn = exponential_decay(init_value=init_value,
                           final_value=final_value,
                           num_decay_steps=num_decay_steps,
                           num_delay_steps=num_delay_steps)
    expected_list = [init_value] * num_delay_steps + [
        init_value *
        (final_value / init_value)**(min(i, num_decay_steps) * 1. / num_decay_steps)
        for i in range(num_steps - num_delay_steps)
    ]
    np.testing.assert_allclose(jax.vmap(fn)(np.arange(num_steps)),
                               expected_list,
                               rtol=1e-6)


def test_exp_decay_final_val_and_rate():
    init_value = 0.01
    final_value = 0.0001
    decay_rate = 0.6
    num_delay_steps = 3
    num_steps = 20
    fn = exponential_decay(init_value=init_value,
                           final_value=final_value,
                           decay_rate=decay_rate,
                           num_delay_steps=num_delay_steps)
    expected_list = [init_value] * num_delay_steps + [
        max((init_value * decay_rate**i), final_value)
        for i in range(num_steps - num_delay_steps)
    ]
    np.testing.assert_allclose(jax.vmap(fn)(np.arange(num_steps)),
                               expected_list,
                               rtol=1e-6)
