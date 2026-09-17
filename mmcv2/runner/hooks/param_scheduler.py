from __future__ import annotations

from collections.abc import Iterable

from torch.optim.lr_scheduler import LRScheduler

from .hook import HOOKS, Hook


@HOOKS.register_module()
class ParamSchedulerHook(Hook):
    """Step PyTorch schedulers after optimizer updates.

    Args:
        schedulers: A scheduler or iterable of schedulers. Schedulers may set
            ``by_epoch`` themselves; otherwise ``by_epoch`` is used.
        by_epoch: Whether schedulers step after each training epoch. Set false
            for iteration schedules.
    """

    def __init__(self, schedulers: LRScheduler | Iterable[LRScheduler], by_epoch: bool = True):
        if isinstance(schedulers, LRScheduler):
            schedulers = [schedulers]
        self.schedulers = list(schedulers)
        if not self.schedulers or not all(isinstance(item, LRScheduler) for item in self.schedulers):
            raise TypeError("schedulers must contain torch LRScheduler instances")
        self.by_epoch = by_epoch

    def after_train_iter(self, runner) -> None:
        if self.by_epoch or not getattr(runner, "_optimizer_step_boundary", True):
            return
        for scheduler in self.schedulers:
            scheduler.step()
        self._store_state(runner)

    def after_train_epoch(self, runner) -> None:
        if not self.by_epoch:
            return
        for scheduler in self.schedulers:
            scheduler.step()
        self._store_state(runner)

    def before_run(self, runner) -> None:
        """Restore scheduler progress saved in a runner checkpoint, if any."""
        state = (runner.meta or {}).get("param_schedulers")
        if state is not None:
            self.load_state_dict(state)

    def _store_state(self, runner) -> None:
        if runner.meta is None:
            runner.meta = {}
        runner.meta["param_schedulers"] = self.state_dict()

    def state_dict(self) -> list[dict]:
        return [scheduler.state_dict() for scheduler in self.schedulers]

    def load_state_dict(self, state_dict: list[dict]) -> None:
        if len(state_dict) != len(self.schedulers):
            raise ValueError("scheduler state count differs from configured schedulers")
        for scheduler, state in zip(self.schedulers, state_dict):
            scheduler.load_state_dict(state)
