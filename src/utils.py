from math import log10
from typing import Callable, Iterator, List, Tuple, TypeVar

import flax
import jax
import jax.numpy as jnp

MiniBatch = TypeVar('MiniBatch')
State = TypeVar('State')
LRSchedule = Callable[[int], float]
TrainStep = Callable[[MiniBatch, State], State]


@jax.tree_util.register_pytree_node_class
class MetricsGroup:
    def __init__(self, *names):
        self._m = {n: (0., 0) for n in names}

    def update(self, **names_and_values):
        upd = {
            n: (self._m[n][0] + v, self._m[n][1] + 1)
            for n, v in names_and_values.items()
        }
        # allowed to update only a subset of the metrics
        return self.from_dict({**self._m, **upd})

    def __getitem__(self, name):
        v, c = self._m[name]
        return v / c

    def reset(self):
        return self.__class__(*self._m.keys())

    def items(self):
        for k, (v, c) in self._m.items():
            yield k, v / c

    @classmethod
    def from_dict(cls, m):
        metrics = cls()
        metrics._m = m
        return metrics

    def tree_flatten(self):
        return self._m.values(), self._m.keys()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls.from_dict(dict(zip(aux_data, children)))


flax.serialization.register_serialization_state(
    ty=MetricsGroup,
    ty_to_state_dict=lambda x: {n: list(v) for n, v in x._m.items()},
    ty_from_state_dict=MetricsGroup.from_dict,
    override=True,
)


def find_lr(get_train_step: Callable[[LRSchedule], TrainStep],
            state: State,
            train_iter: Iterator[MiniBatch],
            init_value: float = 1e-8,
            final_value: float = 10.,
            beta: float = 0.98,
            num_steps: int = 100) -> Tuple[List[float], List[float]]:
    """Learning rate finding method by Smith 2018 (arxiv.org/pdf/1803.09820.pdf)

    train_step return updated state and the loss
    Modified version of Sylvain Gugger's: sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    """
    mult = (final_value / init_value)**(1 / num_steps)
    lrs = init_value * mult**jnp.arange(num_steps)
    sched = lambda s: {'learning_rate': lrs[s]}
    train_step = get_train_step(sched)

    avg_loss, best_loss = 0., 0.
    losses, log_lrs = [], []
    for batch_num, batch in enumerate(train_iter, start=1):
        state = train_step(batch, state)
        loss = state.metrics['loss']
        state = state.replace(metrics=state.metrics.reset())
        lr = state.optimizer_hparams['learning_rate']

        # Compute the smoothed loss
        avg_loss = beta*avg_loss + (1-beta) * loss
        smoothed_loss = avg_loss / (1 - beta**batch_num)

        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses

        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss

        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(log10(lr))

    return log_lrs, losses
