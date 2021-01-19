import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import time
from typing import Any, Dict

import flax
import hydra
import jax
import jax.numpy as jnp
from flax import optim
from flax.training import checkpoints
from shapecheck import check_shapes

import data
import utils


def get_cross_entropy_loss(from_logits=True):
    @check_shapes('N,-1', 'N', out_='')
    def cross_entropy_loss(logits, labels):
        ls = jax.nn.log_softmax(logits) if from_logits else logits
        return -(jax.nn.one_hot(labels, ls.shape[-1]) * ls).sum(-1).mean()

    return cross_entropy_loss


@check_shapes('N,-1', 'N', out_='')
def accuracy(logits, labels):
    return (logits.argmax(-1) == labels).mean()


def get_train_step(model, criterion, optim_sched):
    def train_step(batch, state):
        def loss_fn(params):
            logits = model.apply(params, batch['images'])
            loss = criterion(logits, batch['labels'])
            return loss, logits

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grad = grad_fn(state.optimizer.target)

        optimizer_hparams = optim_sched(state.step)
        optimizer = state.optimizer.apply_gradient(grad, **optimizer_hparams)
        metrics = state.metrics.update(loss=loss,
                                       accuracy=accuracy(logits, batch['labels']))

        return state.replace(
            optimizer=optimizer,
            metrics=metrics,
            optimizer_hparams=optimizer_hparams,
            step=state.step + 1,
        )

    return train_step


def get_eval_step(model, criterion):
    def eval_step(batch, params, metrics):
        logits = model.apply(params, batch['images'])
        return metrics.update(loss=criterion(logits, batch['labels']),
                              accuracy=accuracy(logits, batch['labels']))

    return eval_step


@flax.struct.dataclass
class TrainState:
    optimizer: optim.Optimizer
    metrics: utils.MetricsGroup
    optimizer_hparams: Dict[str, Any]
    step: int


@hydra.main(config_name='config', config_path='conf')
def main(cfg):
    rng = jax.random.PRNGKey(cfg.seed)
    rng, init_rng = jax.random.split(rng)

    model = hydra.utils.instantiate(cfg.model, n_classes=10)
    init_params = model.init(init_rng, jnp.ones((1, 28, 28, 1), jnp.float32))

    state = TrainState(
        optimizer=optim.Adam(learning_rate=0.01, weight_decay=0.,
                             eps=1e-4).create(init_params),
        metrics=utils.MetricsGroup('loss', 'accuracy'),
        optimizer_hparams={},  # will be populated during step
        step=1)

    eps_fn = hydra.utils.call(cfg.epsilon_sched)
    mom_fn = hydra.utils.call(cfg.momentum_sched)
    lr_fn = hydra.utils.call(cfg.lr_sched)

    def schedule(step):
        return {
            'learning_rate': lr_fn(step),
            'beta1': mom_fn(step),
            'eps': eps_fn(step)
        }

    criterion = hydra.utils.call(cfg.loss)
    train_step = jax.jit(get_train_step(model, criterion, schedule))
    eval_step = jax.jit(get_eval_step(model, criterion))
    train_ds, val_ds = data.get_data(seed=cfg.seed,
                                     train_bs=cfg.train_bs,
                                     val_bs=cfg.val_bs,
                                     subset=cfg.test_mode,
                                     image_generator=hydra.utils.call(
                                         cfg.image_generator))

    start_time = time.perf_counter()
    for i, batch in enumerate(data.as_numpy(train_ds), start=int(state.step)):
        state = train_step(batch, state)

        if i % cfg.report_freq == 0:
            time_per_step = (time.perf_counter() - start_time) / cfg.report_freq
            optim_hps = state.optimizer_hparams
            print(f"Iter {i:5d}  "
                  f"Acc: {state.metrics['accuracy']:.4f}  "
                  f"Loss: {state.metrics['loss']:.4f}  "
                  f"LR: {optim_hps['learning_rate']:.4f}  "
                  f"B1: {optim_hps['beta1']:.4f}  "
                  f"Eps: {optim_hps['eps']:.4f}  "
                  f"Time/Step: {time_per_step:.4f}")
            state = state.replace(metrics=state.metrics.reset())
            start_time = time.perf_counter()

        if i % cfg.eval_freq == 0:
            start_time = time.perf_counter()
            params = state.optimizer.target
            val_metrics = utils.MetricsGroup('loss', 'accuracy')
            for img, label in data.as_numpy(val_ds):
                val_metrics = eval_step(batch, params, val_metrics)
            print(f"Iter {i:5d}  "
                  f"Val Acc: {val_metrics['accuracy']:.4f}  "
                  f"Val Loss: {val_metrics['loss']:.4f}  "
                  f"Val Time: {time.perf_counter() - start_time:.4f}")
            start_time = time.perf_counter()
            if not cfg.test_mode:
                checkpoints.save_checkpoint(f'{os.getcwd()}/ckpts',
                                            jax.device_get(state),
                                            int(state.step),
                                            keep=5)

        if i > cfg.num_steps:
            break

    if not cfg.test_mode:
        checkpoints.save_checkpoint(f'{os.getcwd()}/ckpts',
                                    jax.device_get(state),
                                    int(state.step),
                                    keep=5)
    print('Done!')


if __name__ == '__main__':
    main()
