import jax
import jax.numpy as jnp


def exponential_decay(init_value,
                      final_value=None,
                      decay_rate=None,
                      num_decay_steps=None,
                      num_delay_steps=0):
    assert not decay_rate or decay_rate <= 1.

    if final_value and decay_rate and num_decay_steps:
        raise ValueError('You must specify exactly two of final_value, decay_rate,'
                         ' and num_decay_steps.')
    if final_value and decay_rate:
        decay_fn = lambda step: jnp.maximum((init_value *
                                             (decay_rate**step)), final_value)
    elif decay_rate and num_decay_steps:
        decay_fn = lambda step: (init_value * decay_rate**step)
    elif final_value and num_decay_steps:
        decay_fn = lambda step: (init_value *
                                 (final_value / init_value)**(step / num_decay_steps))
    else:
        raise ValueError('You must specify exactly two of final_value, decay_rate'
                         ', and num_decay_steps.')

    return lambda step: decay_fn(jnp.clip(step - num_delay_steps, 0, num_decay_steps))


def linear_interp(start, end, pct):
    return (end-start) * pct + start


def cos_interp(start, end, pct):
    return end + (start-end) / 2.0 * (jnp.cos(jnp.pi * pct) + 1)


def _get_branch_fn(anneal_fn, init_val, val, stage_pct, prev_pct):
    """Needs to be a seperate function to prevent python variable look-up madness."""
    return lambda p: anneal_fn(init_val, val, (p-prev_pct) / (stage_pct-prev_pct))


def piecewise_anneal(num_steps, anneal_fn, init_val, *vals_and_pcts):
    def schedule(step):
        nonlocal init_val  # need this to prevent UnboundLocalError: local variable 'init_val' referenced before assignment
        pct = step / num_steps
        ind, branches, prev_pct = 0, [], 0.
        for val, stage_pct in vals_and_pcts:
            ind = ind + jnp.uint8(pct > stage_pct)
            branches.append(
                _get_branch_fn(anneal_fn, init_val, val, stage_pct, prev_pct))
            init_val, prev_pct = val, stage_pct
        branches.append(lambda p: val)
        return jax.lax.switch(ind, branches, pct)

    return schedule


def linear_onecycle_lr(num_steps,
                       max_lr=0.01,
                       pct_start=0.3,
                       pct_final_stage=0.85,
                       div_factor=25.0,
                       final_div_factor=1e4):
    """ Based on Leslie Smith (2017): https://arxiv.org/abs/1708.07120"""
    init_lr = max_lr / div_factor
    final_lr = init_lr / final_div_factor
    return piecewise_anneal(num_steps, linear_interp, init_lr, (max_lr, pct_start),
                            (init_lr, pct_final_stage), (final_lr, 1.))


def linear_onecycle_momentum(num_steps,
                             base_momentum=0.85,
                             max_momentum=0.95,
                             pct_start=0.3,
                             pct_final_stage=0.85):
    """ Based on Leslie Smith (2017): https://arxiv.org/abs/1708.07120"""
    return piecewise_anneal(num_steps, linear_interp, max_momentum,
                            (base_momentum, pct_start), (max_momentum, pct_final_stage))


def cos_onecycle_lr(num_steps,
                    max_lr=0.01,
                    pct_start=0.3,
                    div_factor=25.0,
                    final_div_factor=1e4):
    """ Based on fastai and PyTorch 
    
        - https://fastai1.fast.ai/callbacks.one_cycle.html
        - https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.OneCycleLR
    """
    init_lr = max_lr / div_factor
    final_lr = init_lr / final_div_factor
    return piecewise_anneal(num_steps, cos_interp, init_lr, (max_lr, pct_start),
                            (final_lr, 1.))


def cos_onecycle_momentum(num_steps,
                          base_momentum=0.85,
                          max_momentum=0.95,
                          pct_start=0.3):
    """ Based on fastai and PyTorch 
    
        - https://fastai1.fast.ai/callbacks.one_cycle.html
        - https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.OneCycleLR
    """
    return piecewise_anneal(num_steps, cos_interp, max_momentum,
                            (base_momentum, pct_start), (max_momentum, 1.))
