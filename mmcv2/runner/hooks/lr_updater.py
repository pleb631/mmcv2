import numbers
from collections.abc import Callable
from math import cos, pi
from typing import Any, TypeAlias, cast

from torch.optim import Optimizer, lr_scheduler

from mmcv2.utils import is_list_of

from .. import base_runner as runner

from .hook import HOOKS, Hook
from .optimizer import AmpOptimizerHook, OptimizerHook

LrGroups: TypeAlias = list[float] | dict[str, list[float]]
NativeSchedulers: TypeAlias = lr_scheduler.LRScheduler | dict[str, lr_scheduler.LRScheduler]


class LrUpdaterHook(Hook):
    """LR Scheduler in MMCV2.

    Args:
        by_epoch (bool): LR changes epoch by epoch
        warmup (string): Type of warmup used. It can be None(use no warmup),
            'constant', 'linear' or 'exp'
        warmup_iters (int): The number of iterations or epochs that warmup
            lasts
        warmup_ratio (float): LR used at the beginning of warmup equals to
            warmup_ratio * initial_lr
        warmup_by_epoch (bool): When warmup_by_epoch == True, warmup_iters
            means the number of epochs that warmup lasts, otherwise means the
            number of iteration that warmup lasts
    """

    def __init__(
        self,
        by_epoch: bool = True,
        warmup: str | None = None,
        warmup_iters: int = 0,
        warmup_ratio: float = 0.1,
        warmup_by_epoch: bool = False,
    ) -> None:
        # validate the "warmup" argument
        if warmup is not None and warmup not in ["constant", "linear", "exp"]:
            raise ValueError(
                f'"{warmup}" is not a supported type for warming up, valid types are "constant", "linear" and "exp"'
            )
        if warmup is not None:
            assert warmup_iters > 0, '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, '"warmup_ratio" must be in range (0,1]'

        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters: int | None = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.warmup_by_epoch = warmup_by_epoch

        if self.warmup_by_epoch:
            self.warmup_epochs: int | None = self.warmup_iters
            self.warmup_iters = None
        else:
            self.warmup_epochs = None

        self.base_lr: LrGroups = []  # initial lr for all param groups
        self.regular_lr: LrGroups = []  # expected lr if no warming up is performed
        self._native_schedulers: NativeSchedulers | None = None

    def _make_native_scheduler(self, optimizer: Optimizer) -> lr_scheduler.LRScheduler | None:
        if isinstance(self, FixedLrUpdaterHook):
            return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        if isinstance(self, StepLrUpdaterHook) and self.min_lr is None:
            if isinstance(self.step, int):
                return lr_scheduler.StepLR(optimizer, self.step, self.gamma)
            return lr_scheduler.MultiStepLR(optimizer, self.step, self.gamma)
        if isinstance(self, ExpLrUpdaterHook):
            return lr_scheduler.ExponentialLR(optimizer, self.gamma)
        return None

    def _advance_native(self, runner: "runner.BaseRunner") -> None:
        schedulers = cast(NativeSchedulers, self._native_schedulers)
        if isinstance(schedulers, dict):
            for scheduler in schedulers.values():
                scheduler.step()
        else:
            schedulers.step()
        if runner.meta is None:
            runner.meta = {}
        if isinstance(schedulers, dict):
            state = {key: scheduler.state_dict() for key, scheduler in schedulers.items()}
        else:
            state = schedulers.state_dict()
        cast(dict[str, Any], runner.meta)["lr_scheduler"] = state

    def _set_lr(self, runner: "runner.BaseRunner", lr_groups: LrGroups) -> None:
        if isinstance(runner.optimizer, dict):
            lr_groups = cast(dict[str, list[float]], lr_groups)
            for k, optim in runner.optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group["lr"] = lr
        else:
            optimizer = cast(Optimizer, runner.optimizer)
            lr_groups = cast(list[float], lr_groups)
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group["lr"] = lr

    def get_lr(self, runner: "runner.BaseRunner", base_lr: float) -> float:
        raise NotImplementedError

    def get_regular_lr(self, runner: "runner.BaseRunner") -> LrGroups:
        if isinstance(runner.optimizer, dict):
            lr_groups: dict[str, list[float]] = {}
            base_lr = cast(dict[str, list[float]], self.base_lr)
            for k in runner.optimizer:
                _lr_group = [self.get_lr(runner, _base_lr) for _base_lr in base_lr[k]]
                lr_groups.update({k: _lr_group})

            return lr_groups
        else:
            return [self.get_lr(runner, _base_lr) for _base_lr in cast(list[float], self.base_lr)]

    def get_warmup_lr(self, cur_iters: int) -> LrGroups:

        def _get_warmup_lr(cur_iters: int, regular_lr: list[float]) -> list[float]:
            warmup_iters = cast(int, self.warmup_iters)
            if self.warmup == "constant":
                warmup_lr = [_lr * self.warmup_ratio for _lr in regular_lr]
            elif self.warmup == "linear":
                k = (1 - cur_iters / warmup_iters) * (1 - self.warmup_ratio)
                warmup_lr = [_lr * (1 - k) for _lr in regular_lr]
            elif self.warmup == "exp":
                k = self.warmup_ratio ** (1 - cur_iters / warmup_iters)
                warmup_lr = [_lr * k for _lr in regular_lr]
            return warmup_lr

        if isinstance(self.regular_lr, dict):
            lr_groups: dict[str, list[float]] = {}
            for key, regular_lr in self.regular_lr.items():
                lr_groups[key] = _get_warmup_lr(cur_iters, regular_lr)
            return lr_groups
        else:
            return _get_warmup_lr(cur_iters, self.regular_lr)

    def before_run(self, runner: "runner.BaseRunner") -> None:
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        if isinstance(runner.optimizer, dict):
            self.base_lr = {}
            for k, optim in runner.optimizer.items():
                for group in optim.param_groups:
                    group.setdefault("initial_lr", float(group["lr"]))
                _base_lr = [float(group["initial_lr"]) for group in optim.param_groups]
                self.base_lr.update({k: _base_lr})
        else:
            optimizer = cast(Optimizer, runner.optimizer)
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", float(group["lr"]))
            self.base_lr = [float(group["initial_lr"]) for group in optimizer.param_groups]

        # Native schedulers step before the next training unit. This keeps
        # MMCV2's existing before_train_iter/epoch LR boundary exactly intact.
        optim_hooks = [hook for hook in runner.hooks if isinstance(hook, OptimizerHook)]
        can_map = (
            self.warmup is None
            and len(optim_hooks) == 1
            and optim_hooks[0].cumulative_iters == 1
            and not (isinstance(optim_hooks[0], AmpOptimizerHook) and optim_hooks[0].dtype == "float16")
        )
        meta = cast(dict[str, Any], runner.meta or {})
        if not can_map or ("lr_scheduler" not in meta and (runner.iter > 0 or runner.epoch > 0)):
            return
        if isinstance(runner.optimizer, dict):
            schedulers: dict[str, lr_scheduler.LRScheduler] = {}
            for key, optim in runner.optimizer.items():
                scheduler = self._make_native_scheduler(optim)
                if scheduler is None:
                    return
                schedulers[key] = scheduler
            self._native_schedulers = schedulers
        else:
            self._native_schedulers = self._make_native_scheduler(cast(Optimizer, runner.optimizer))
        if self._native_schedulers is None:
            return
        if "lr_scheduler" in meta:
            if isinstance(self._native_schedulers, dict):
                for key, scheduler in self._native_schedulers.items():
                    scheduler.load_state_dict(cast(dict[str, Any], meta["lr_scheduler"])[key])
            else:
                self._native_schedulers.load_state_dict(cast(dict[str, Any], meta["lr_scheduler"]))
        else:
            if runner.meta is None:
                runner.meta = {}
            if isinstance(self._native_schedulers, dict):
                cast(dict[str, Any], runner.meta)["lr_scheduler"] = {
                    key: scheduler.state_dict() for key, scheduler in self._native_schedulers.items()
                }
            else:
                cast(dict[str, Any], runner.meta)["lr_scheduler"] = self._native_schedulers.state_dict()

    def before_train_epoch(self, runner: "runner.BaseRunner") -> None:
        if self.warmup_iters is None:
            epoch_len = len(cast(Any, runner).data_loader)
            self.warmup_iters = cast(int, self.warmup_epochs) * epoch_len

        if not self.by_epoch:
            return

        if self._native_schedulers is not None:
            if runner.epoch > 0:
                self._advance_native(runner)
            return

        self.regular_lr = self.get_regular_lr(runner)
        self._set_lr(runner, self.regular_lr)

    def before_train_iter(self, runner: "runner.BaseRunner") -> None:
        cur_iter = runner.iter

        if self._native_schedulers is not None:
            if not self.by_epoch and cur_iter > 0:
                self._advance_native(runner)
            return
        warmup_iters = cast(int, self.warmup_iters)
        if not self.by_epoch:
            self.regular_lr = self.get_regular_lr(runner)
            if self.warmup is None or cur_iter >= warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(runner, warmup_lr)
        elif self.by_epoch:
            if self.warmup is None or cur_iter > warmup_iters:
                return
            elif cur_iter == warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(runner, warmup_lr)


@HOOKS.register_module()
class FixedLrUpdaterHook(LrUpdaterHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_lr(self, runner: "runner.BaseRunner", base_lr: float) -> float:
        return base_lr


@HOOKS.register_module()
class StepLrUpdaterHook(LrUpdaterHook):
    """Step LR scheduler with min_lr clipping.

    Args:
        step (int | list[int]): Step to decay the LR. If an int value is given,
            regard it as the decay interval. If a list is given, decay LR at
            these steps.
        gamma (float): Decay LR ratio. Defaults to 0.1.
        min_lr (float, optional): Minimum LR value to keep. If LR after decay
            is lower than `min_lr`, it will be clipped to this value. If None
            is given, we don't perform lr clipping. Default: None.
    """

    def __init__(
        self,
        step: int | list[int],
        gamma: float = 0.1,
        min_lr: float | None = None,
        **kwargs,
    ) -> None:
        if isinstance(step, list):
            assert is_list_of(step, int)
            assert all(s > 0 for s in step)
        elif isinstance(step, int):
            assert step > 0
        self.step = step
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(**kwargs)

    def get_lr(self, runner: "runner.BaseRunner", base_lr: float) -> float:
        progress = runner.epoch if self.by_epoch else runner.iter

        # calculate exponential term
        if isinstance(self.step, int):
            exp = progress // self.step
        else:
            exp = len(self.step)
            for i, s in enumerate(self.step):
                if progress < s:
                    exp = i
                    break

        lr = base_lr * (self.gamma**exp)
        if self.min_lr is not None:
            # clip to a minimum value
            lr = max(lr, self.min_lr)
        return lr


@HOOKS.register_module()
class ExpLrUpdaterHook(LrUpdaterHook):
    def __init__(self, gamma: float, **kwargs) -> None:
        self.gamma = gamma
        super().__init__(**kwargs)

    def get_lr(self, runner: "runner.BaseRunner", base_lr: float) -> float:
        progress = runner.epoch if self.by_epoch else runner.iter
        return base_lr * self.gamma**progress


@HOOKS.register_module()
class PolyLrUpdaterHook(LrUpdaterHook):
    def __init__(self, power: float = 1.0, min_lr: float = 0.0, **kwargs) -> None:
        self.power = power
        self.min_lr = min_lr
        super().__init__(**kwargs)

    def get_lr(self, runner: "runner.BaseRunner", base_lr: float) -> float:
        if self.by_epoch:
            progress = runner.epoch
            max_progress = cast(int, runner.max_epochs)
        else:
            progress = runner.iter
            max_progress = cast(int, runner.max_iters)
        coeff = (1 - progress / max_progress) ** self.power
        return (base_lr - self.min_lr) * coeff + self.min_lr


@HOOKS.register_module()
class InvLrUpdaterHook(LrUpdaterHook):
    def __init__(self, gamma: float, power: float = 1.0, **kwargs) -> None:
        self.gamma = gamma
        self.power = power
        super().__init__(**kwargs)

    def get_lr(self, runner: "runner.BaseRunner", base_lr: float) -> float:
        progress = runner.epoch if self.by_epoch else runner.iter
        return base_lr * (1 + self.gamma * progress) ** (-self.power)


@HOOKS.register_module()
class CosineAnnealingLrUpdaterHook(LrUpdaterHook):
    """CosineAnnealing LR scheduler.

    Args:
        min_lr (float, optional): The minimum lr. Default: None.
        min_lr_ratio (float, optional): The ratio of minimum lr to the base lr.
            Either `min_lr` or `min_lr_ratio` should be specified.
            Default: None.
    """

    def __init__(
        self,
        min_lr: float | None = None,
        min_lr_ratio: float | None = None,
        **kwargs,
    ) -> None:
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        super().__init__(**kwargs)

    def get_lr(self, runner: "runner.BaseRunner", base_lr: float):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = cast(int, runner.max_epochs)
        else:
            progress = runner.iter
            max_progress = cast(int, runner.max_iters)

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = cast(float, self.min_lr)
        return annealing_cos(base_lr, target_lr, progress / max_progress)


@HOOKS.register_module()
class FlatCosineAnnealingLrUpdaterHook(LrUpdaterHook):
    """Flat + Cosine lr schedule.

    Modified from https://github.com/fastai/fastai/blob/master/fastai/callback/schedule.py#L128 # noqa: E501

    Args:
        start_percent (float): When to start annealing the learning rate
            after the percentage of the total training steps.
            The value should be in range [0, 1).
            Default: 0.75
        min_lr (float, optional): The minimum lr. Default: None.
        min_lr_ratio (float, optional): The ratio of minimum lr to the base lr.
            Either `min_lr` or `min_lr_ratio` should be specified.
            Default: None.
    """

    def __init__(
        self,
        start_percent: float = 0.75,
        min_lr: float | None = None,
        min_lr_ratio: float | None = None,
        **kwargs,
    ) -> None:
        assert (min_lr is None) ^ (min_lr_ratio is None)
        if start_percent < 0 or start_percent > 1 or not isinstance(start_percent, float):
            raise ValueError(f"expected float between 0 and 1 start_percent, but got {start_percent}")
        self.start_percent = start_percent
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        super().__init__(**kwargs)

    def get_lr(self, runner: "runner.BaseRunner", base_lr: float) -> float:
        if self.by_epoch:
            start = round(cast(int, runner.max_epochs) * self.start_percent)
            progress = runner.epoch - start
            max_progress = cast(int, runner.max_epochs) - start
        else:
            start = round(cast(int, runner.max_iters) * self.start_percent)
            progress = runner.iter - start
            max_progress = cast(int, runner.max_iters) - start

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = cast(float, self.min_lr)

        if progress < 0:
            return base_lr
        else:
            return annealing_cos(base_lr, target_lr, progress / max_progress)


@HOOKS.register_module()
class CosineRestartLrUpdaterHook(LrUpdaterHook):
    """Cosine annealing with restarts learning rate scheme.

    Args:
        periods (list[int]): Periods for each cosine anneling cycle.
        restart_weights (list[float]): Restart weights at each
            restart iteration. Defaults to [1].
        min_lr (float, optional): The minimum lr. Default: None.
        min_lr_ratio (float, optional): The ratio of minimum lr to the base lr.
            Either `min_lr` or `min_lr_ratio` should be specified.
            Default: None.
    """

    def __init__(
        self,
        periods: list[int],
        restart_weights: list[float] | None = None,
        min_lr: float | None = None,
        min_lr_ratio: float | None = None,
        **kwargs,
    ) -> None:
        if restart_weights is None:
            restart_weights = [1]
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.periods = periods
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.restart_weights = restart_weights
        assert len(self.periods) == len(self.restart_weights), (
            "periods and restart_weights should have the same length."
        )
        super().__init__(**kwargs)

        self.cumulative_periods = [sum(self.periods[0 : i + 1]) for i in range(len(self.periods))]

    def get_lr(self, runner: "runner.BaseRunner", base_lr: float) -> float:
        if self.by_epoch:
            progress = runner.epoch
        else:
            progress = runner.iter

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = cast(float, self.min_lr)

        idx = get_position_from_periods(progress, self.cumulative_periods)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_periods[idx - 1]
        current_periods = self.periods[idx]

        alpha = min((progress - nearest_restart) / current_periods, 1)
        return annealing_cos(base_lr, target_lr, alpha, current_weight)


def get_position_from_periods(iteration: int, cumulative_periods: list[int]):
    """Get the position from a period list.

    It will return the index of the right-closest number in the period list.
    For example, the cumulative_periods = [100, 200, 300, 400],
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 3.

    Args:
        iteration (int): Current iteration.
        cumulative_periods (list[int]): Cumulative period list.

    Returns:
        int: The position of the right-closest number in the period list.
    """
    for i, period in enumerate(cumulative_periods):
        if iteration < period:
            return i
    raise ValueError(f"Current iteration {iteration} exceeds cumulative_periods {cumulative_periods}")


@HOOKS.register_module()
class CyclicLrUpdaterHook(LrUpdaterHook):
    """Cyclic LR Scheduler.

    Implement the cyclical learning rate policy (CLR) described in
    https://arxiv.org/pdf/1506.01186.pdf

    Different from the original paper, we use cosine annealing rather than
    triangular policy inside a cycle. This improves the performance in the
    3D detection area.

    Args:
        by_epoch (bool, optional): Whether to update LR by epoch.
        target_ratio (tuple[float], optional): Relative ratio of the highest LR
            and the lowest LR to the initial LR.
        cyclic_times (int, optional): Number of cycles during training
        step_ratio_up (float, optional): The ratio of the increasing process of
            LR in the total cycle.
        anneal_strategy (str, optional): {'cos', 'linear'}
            Specifies the annealing strategy: 'cos' for cosine annealing,
            'linear' for linear annealing. Default: 'cos'.
        gamma (float, optional): Cycle decay ratio. Default: 1.
            It takes values in the range (0, 1]. The difference between the
            maximum learning rate and the minimum learning rate decreases
            periodically when it is less than 1. `New in version 1.4.4.`
    """

    def __init__(
        self,
        by_epoch: bool = False,
        target_ratio: float | tuple[float, ...] = (10, 1e-4),
        cyclic_times: int = 1,
        step_ratio_up: float = 0.4,
        anneal_strategy: str = "cos",
        gamma: float = 1,
        **kwargs,
    ) -> None:
        if isinstance(target_ratio, float):
            target_ratio = (target_ratio, target_ratio / 1e5)
        elif isinstance(target_ratio, tuple):
            target_ratio = (target_ratio[0], target_ratio[0] / 1e5) if len(target_ratio) == 1 else target_ratio
        else:
            raise ValueError(f"target_ratio should be either float or tuple, got {type(target_ratio)}")

        assert len(target_ratio) == 2, '"target_ratio" must be list or tuple of two floats'
        assert 0 <= step_ratio_up < 1.0, '"step_ratio_up" must be in range [0,1)'
        assert 0 < gamma <= 1, '"gamma" must be in range (0, 1]'

        self.target_ratio: tuple[float, float] = cast(tuple[float, float], target_ratio)
        self.cyclic_times = cyclic_times
        self.step_ratio_up = step_ratio_up
        self.gamma = gamma
        self.max_iter_per_phase: int | None = None
        self.lr_phases: list[tuple[int, int, float, float]] = []
        # validate anneal_strategy
        if anneal_strategy not in ["cos", "linear"]:
            raise ValueError(f'anneal_strategy must be one of "cos" or "linear", instead got {anneal_strategy}')
        elif anneal_strategy == "cos":
            self.anneal_func: Callable[[float, float, float], float] = annealing_cos
        elif anneal_strategy == "linear":
            self.anneal_func = annealing_linear

        assert not by_epoch, 'currently only support "by_epoch" = False'
        super().__init__(by_epoch, **kwargs)

    def before_run(self, runner: "runner.BaseRunner") -> None:
        super().before_run(runner)
        # initiate lr_phases
        # total lr_phases are separated as up and down
        self.max_iter_per_phase = cast(int, runner.max_iters) // self.cyclic_times
        iter_up_phase = int(self.step_ratio_up * self.max_iter_per_phase)
        self.lr_phases.append((0, iter_up_phase, 1, self.target_ratio[0]))
        self.lr_phases.append(
            (
                iter_up_phase,
                self.max_iter_per_phase,
                self.target_ratio[0],
                self.target_ratio[1],
            )
        )

    def get_lr(self, runner: "runner.BaseRunner", base_lr: float) -> float:
        max_iter_per_phase = cast(int, self.max_iter_per_phase)
        curr_iter = runner.iter % max_iter_per_phase
        curr_cycle = runner.iter // max_iter_per_phase
        # Update weight decay
        scale = self.gamma**curr_cycle

        for start_iter, end_iter, start_ratio, end_ratio in self.lr_phases:
            if start_iter <= curr_iter < end_iter:
                # Apply cycle scaling to gradually reduce the difference
                # between max_lr and base lr. The target end_ratio can be
                # expressed as:
                # end_ratio = (base_lr + scale * (max_lr - base_lr)) / base_lr
                # iteration: 0-iter_up_phase:
                if start_iter == 0:
                    end_ratio = 1 - scale + end_ratio * scale
                # iteration: iter_up_phase-self.max_iter_per_phase
                else:
                    start_ratio = 1 - scale + start_ratio * scale
                progress = curr_iter - start_iter
                return self.anneal_func(
                    base_lr * start_ratio,
                    base_lr * end_ratio,
                    progress / (end_iter - start_iter),
                )
        return base_lr


@HOOKS.register_module()
class OneCycleLrUpdaterHook(LrUpdaterHook):
    """One Cycle LR Scheduler.

    The 1cycle learning rate policy changes the learning rate after every
    batch. The one cycle learning rate policy is described in
    https://arxiv.org/pdf/1708.07120.pdf

    Args:
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group.
        total_steps (int, optional): The total number of steps in the cycle.
            Note that if a value is not provided here, it will be the max_iter
            of runner. Default: None.
        pct_start (float): The percentage of the cycle (in number of steps)
            spent increasing the learning rate.
            Default: 0.3
        anneal_strategy (str): {'cos', 'linear'}
            Specifies the annealing strategy: 'cos' for cosine annealing,
            'linear' for linear annealing.
            Default: 'cos'
        div_factor (float): Determines the initial learning rate via
            initial_lr = max_lr/div_factor
            Default: 25
        final_div_factor (float): Determines the minimum learning rate via
            min_lr = initial_lr/final_div_factor
            Default: 1e4
        three_phase (bool): If three_phase is True, use a third phase of the
            schedule to annihilate the learning rate according to
            final_div_factor instead of modifying the second phase (the first
            two phases will be symmetrical about the step indicated by
            pct_start).
            Default: False
    """

    def __init__(
        self,
        max_lr: float | list[float] | dict[str, float | list[float]],
        total_steps: int | None = None,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        div_factor: float = 25,
        final_div_factor: float = 1e4,
        three_phase: bool = False,
        **kwargs,
    ) -> None:
        # validate by_epoch, currently only support by_epoch = False
        if "by_epoch" not in kwargs:
            kwargs["by_epoch"] = False
        else:
            assert not kwargs["by_epoch"], 'currently only support "by_epoch" = False'
        if not isinstance(max_lr, (numbers.Number, list, dict)):
            raise ValueError(f"the type of max_lr must be the one of list or dict, but got {type(max_lr)}")
        self._max_lr: float | list[float] | dict[str, float | list[float]] = max_lr
        if total_steps is not None:
            self.total_steps: int | None = total_steps
        else:
            self.total_steps = None
        # validate pct_start
        if pct_start < 0 or pct_start > 1 or not isinstance(pct_start, float):
            raise ValueError(f"expected float between 0 and 1 pct_start, but got {pct_start}")
        self.pct_start = pct_start
        # validate anneal_strategy
        if anneal_strategy not in ["cos", "linear"]:
            raise ValueError(f'anneal_strategy must be one of "cos" or "linear", instead got {anneal_strategy}')
        elif anneal_strategy == "cos":
            self.anneal_func: Callable[[float, float, float], float] = annealing_cos
        elif anneal_strategy == "linear":
            self.anneal_func = annealing_linear
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.three_phase = three_phase
        self.lr_phases: list[tuple[float, float, float]] = []
        super().__init__(**kwargs)

    def before_run(self, runner: "runner.BaseRunner") -> None:
        max_iters = cast(int, runner.max_iters)
        total_steps = self.total_steps if self.total_steps is not None else max_iters
        if total_steps < max_iters:
            raise ValueError(
                f"The total steps must be greater than or equal to max iterations {max_iters} of runner, but total steps is {total_steps}."
            )

        if isinstance(runner.optimizer, dict):
            self.base_lr = {}
            for k, optim in runner.optimizer.items():
                _max_lr = format_param(k, optim, self._max_lr)
                self.base_lr[k] = [lr / self.div_factor for lr in _max_lr]
                for group, lr in zip(optim.param_groups, self.base_lr[k]):
                    group.setdefault("initial_lr", lr)
        else:
            k = type(runner.optimizer).__name__
            _max_lr = format_param(k, cast(Optimizer, runner.optimizer), self._max_lr)
            self.base_lr = [lr / self.div_factor for lr in _max_lr]
            optim_param_groups = cast(Optimizer, runner.optimizer).param_groups
            for group, lr in zip(optim_param_groups, self.base_lr):
                group.setdefault("initial_lr", lr)

        if self.three_phase:
            self.lr_phases.append((float(self.pct_start * total_steps) - 1, 1, self.div_factor))
            self.lr_phases.append((float(2 * self.pct_start * total_steps) - 2, self.div_factor, 1))
            self.lr_phases.append((total_steps - 1, 1, 1 / self.final_div_factor))
        else:
            self.lr_phases.append((float(self.pct_start * total_steps) - 1, 1, self.div_factor))
            self.lr_phases.append((total_steps - 1, self.div_factor, 1 / self.final_div_factor))

    def get_lr(self, runner: "runner.BaseRunner", base_lr: float) -> float:
        curr_iter = runner.iter
        start_iter = 0
        for i, (end_iter, start_lr, end_lr) in enumerate(self.lr_phases):
            if curr_iter <= end_iter:
                pct = (curr_iter - start_iter) / (end_iter - start_iter)
                lr = self.anneal_func(base_lr * start_lr, base_lr * end_lr, pct)
                break
            start_iter = end_iter
        return lr if "lr" in locals() else base_lr


@HOOKS.register_module()
class LinearAnnealingLrUpdaterHook(LrUpdaterHook):
    """Linear annealing LR Scheduler decays the learning rate of each parameter
    group linearly.

    Args:
        min_lr (float, optional): The minimum lr. Default: None.
        min_lr_ratio (float, optional): The ratio of minimum lr to the base lr.
            Either `min_lr` or `min_lr_ratio` should be specified.
            Default: None.
    """

    def __init__(
        self,
        min_lr: float | None = None,
        min_lr_ratio: float | None = None,
        **kwargs,
    ):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        super().__init__(**kwargs)

    def get_lr(self, runner: "runner.BaseRunner", base_lr: float) -> float:
        if self.by_epoch:
            progress = runner.epoch
            max_progress = cast(int, runner.max_epochs)
        else:
            progress = runner.iter
            max_progress = cast(int, runner.max_iters)
        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = cast(float, self.min_lr)
        return annealing_linear(base_lr, target_lr, progress / max_progress)


def annealing_cos(start: float, end: float, factor: float, weight: float = 1.0) -> float:
    """Calculate annealing cos learning rate.

    Cosine anneal from `weight * start + (1 - weight) * end` to `end` as
    percentage goes from 0.0 to 1.0.

    Args:
        start (float): The starting learning rate of the cosine annealing.
        end (float): The ending learing rate of the cosine annealing.
        factor (float): The coefficient of `pi` when calculating the current
            percentage. Range from 0.0 to 1.0.
        weight (float, optional): The combination factor of `start` and `end`
            when calculating the actual starting learning rate. Default to 1.
    """
    cos_out = cos(pi * factor) + 1
    return end + 0.5 * weight * (start - end) * cos_out


def annealing_linear(start: float, end: float, factor: float) -> float:
    """Calculate annealing linear learning rate.

    Linear anneal from `start` to `end` as percentage goes from 0.0 to 1.0.

    Args:
        start (float): The starting learning rate of the linear annealing.
        end (float): The ending learing rate of the linear annealing.
        factor (float): The coefficient of `pi` when calculating the current
            percentage. Range from 0.0 to 1.0.
    """
    return start + (end - start) * factor


def format_param(name: str, optim: Optimizer, param: object) -> list[float] | tuple[float, ...]:
    if isinstance(param, numbers.Number):
        return [float(cast(Any, param))] * len(optim.param_groups)
    elif isinstance(param, (list, tuple)):  # multi param groups
        if len(param) != len(optim.param_groups):
            raise ValueError(f"expected {len(optim.param_groups)} values for {name}, got {len(param)}")
        return cast(list[float] | tuple[float, ...], param)
    else:  # multi optimizers
        params = cast(dict[str, list[float] | tuple[float, ...]], param)
        if name not in params:
            raise KeyError(f"{name} is not found in {params.keys()}")
        return params[name]
