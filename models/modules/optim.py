import torch
import torch_optimizer as optim

from .kar import Kar3, Kar2

from omegaconf import OmegaConf


def construct_optimizer(
    model,
    optim_cfg: OmegaConf,
):
    """
    Constructs an optimizer for a given model
    :param model: a list of parameters to be trained
    :param optim_cfg:
    :return: optimizer
    """
    
    # Unpack values from optim_cfg
    optimizer_type = optim_cfg.name
    lr = optim_cfg.lr

    # Construct optimizer
    if optimizer_type == "Kar3":
        optimizer = Kar3(
            params=model.parameters(),
            lr=lr,
            betas=(0.5, 0.95),
        )
    elif optimizer_type == "Kar2":
        optimizer = Kar2(
            params=model.parameters(),
            lr=lr,
            betas=(0.95),
        )
    
    return optimizer


def construct_scheduler(
    optimizer,
    scheduler_cfg: OmegaConf,
):
    """
    Creates a learning rate scheduler for a given model
    :param optimizer: the optimizer to be used
    :return: scheduler
    """

    # Unpack values from cfg.train.scheduler_params
    scheduler_type = scheduler_cfg.name
    mode = scheduler_cfg.mode

    # Get total iterations (used for CosineScheduler)
    total_iterations = scheduler_cfg.total_train_iters
    
    # Get iterations for warmup
    warmup_epochs = scheduler_cfg.warmup_epochs
    warmup_iterations = 0
    if warmup_epochs != -1:
        warmup_iterations = (
            scheduler_cfg.warmup_epochs * scheduler_cfg.iters_per_train_epoch
        )
        scheduler_cfg.warmup_iterations = warmup_iterations
        scheduler_cfg.remain_iterations = total_iterations - warmup_iterations
    else:
        scheduler_cfg.warmup_iterations = 0
        scheduler_cfg.remain_iterations = total_iterations

    # Create warm_up scheduler
    if warmup_epochs != -1:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_iterations,
        )
    else:
        warmup_scheduler = None

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=total_iterations - warmup_iterations,
            last_epoch=-warmup_iterations,
        )

    # Concatenate schedulers if required
    if warmup_scheduler is not None:
        # If both schedulers are defined, concatenate them
        if lr_scheduler is not None:
            lr_scheduler = ChainedScheduler(
                [
                    warmup_scheduler,
                    lr_scheduler,
                ]
            )
        # Otherwise, return only the warmup scheduler
        else:
            lr_scheduler = lr_scheduler

    return lr_scheduler, warmup_iterations


class ChainedScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Chains list of learning rate schedulers. It takes a list of chainable learning
    rate schedulers and performs consecutive step() functions belong to them by just
    one call.

    Args:
        schedulers (list): List of chained schedulers.

    Example:
        # >>> # Assuming optimizer uses lr = 1. for all groups
        # >>> # lr = 0.09     if epoch == 0
        # >>> # lr = 0.081    if epoch == 1
        # >>> # lr = 0.729    if epoch == 2
        # >>> # lr = 0.6561   if epoch == 3
        # >>> # lr = 0.59049  if epoch >= 4
        # >>> scheduler1 = ConstantLR(self.opt, factor=0.1, total_iters=2)
        # >>> scheduler2 = ExponentialLR(self.opt, gamma=0.9)
        # >>> scheduler = ChainedScheduler([scheduler1, scheduler2])
        # >>> for epoch in range(100):
        # >>>     train(...)
        # >>>     validate(...)
        # >>>     scheduler.step()
    """

    def __init__(self, schedulers):
        for scheduler_idx in range(1, len(schedulers)):
            if schedulers[scheduler_idx].optimizer != schedulers[0].optimizer:
                raise ValueError(
                    "ChainedScheduler expects all schedulers to belong to the same optimizer, but "
                    "got schedulers at index {} and {} to be different".format(
                        0, scheduler_idx
                    )
                )
        self._schedulers = list(schedulers)
        self.optimizer = self._schedulers[0].optimizer

    def step(self):
        for scheduler in self._schedulers:
            scheduler.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The wrapped scheduler states will also be saved.
        """
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "_schedulers")
        }
        state_dict["_schedulers"] = [None] * len(self._schedulers)

        for idx, s in enumerate(self._schedulers):
            state_dict["_schedulers"][idx] = s.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        _schedulers = state_dict.pop("_schedulers")
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict["_schedulers"] = _schedulers

        for idx, s in enumerate(_schedulers):
            self._schedulers[idx].load_state_dict(s)
