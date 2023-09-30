"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

optimizer learning rate scheduling helpers
"""
from math import ceil


def noam_schedule(step, warmup_step=4000):
    """ original Transformer schedule"""
    if step <= warmup_step:
        return step / warmup_step
    return (warmup_step ** 0.5) * (step ** -0.5)


def warmup_linear(step, warmup_step, tot_step):
    """ BERT schedule """
    if step < warmup_step:
        return step / warmup_step
    return max(0, (tot_step-step)/(tot_step-warmup_step))

def warmup_inverse_sqrt(step, warmup_step, tot_step):
    """Decay the LR based on the inverse square root of the update number.
    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    learning rate (``--lr``). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

    During warmup::

      lrs = torch.linspace(cfg.warmup_init_lr, cfg.lr, cfg.warmup_updates)
      lr = lrs[update_num]

    After warmup::

      decay_factor = cfg.lr * sqrt(cfg.warmup_updates)
      lr = decay_factor / sqrt(update_num)
    """
    if step < warmup_step:
        return step / warmup_step
    else:
        return warmup_step**0.5 * step**-0.5


def get_lr_sched(global_step, opts):
    # learning rate scheduling
    if opts.lr_sched == 'linear':
        func = warmup_linear
    elif opts.lr_sched == 'inverse_sqrt':
        func = warmup_inverse_sqrt
    else:
        raise NotImplementedError(f'invalid lr scheduler {opts.lr_sched}')

    lr_this_step = opts.learning_rate * func(
        global_step, opts.warmup_steps, opts.num_train_steps
    )
    if lr_this_step <= 0:
        lr_this_step = 1e-8
    return lr_this_step
