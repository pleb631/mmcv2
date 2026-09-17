from mmcv2.protocols import RunnerLike
from mmcv2.utils import Registry, is_method_overridden

HOOKS = Registry("hook")


class Hook:
    """Base class for callbacks invoked at runner lifecycle stages."""

    stages = (
        "before_run",
        "before_train_epoch",
        "before_train_iter",
        "after_train_iter",
        "after_train_epoch",
        "before_val_epoch",
        "before_val_iter",
        "after_val_iter",
        "after_val_epoch",
        "after_run",
    )

    def before_run(self, runner: RunnerLike) -> None:
        """Handle the start of a run."""
        pass

    def after_run(self, runner: RunnerLike) -> None:
        """Handle the end of a run."""
        pass

    def before_epoch(self, runner: RunnerLike) -> None:
        """Handle the start of an epoch."""
        pass

    def after_epoch(self, runner: RunnerLike) -> None:
        """Handle the end of an epoch."""
        pass

    def before_iter(self, runner: RunnerLike) -> None:
        """Handle the start of an iteration."""
        pass

    def after_iter(self, runner: RunnerLike) -> None:
        """Handle the end of an iteration."""
        pass

    def before_train_epoch(self, runner: RunnerLike) -> None:
        self.before_epoch(runner)

    def before_val_epoch(self, runner: RunnerLike) -> None:
        self.before_epoch(runner)

    def after_train_epoch(self, runner: RunnerLike) -> None:
        self.after_epoch(runner)

    def after_val_epoch(self, runner: RunnerLike) -> None:
        self.after_epoch(runner)

    def before_train_iter(self, runner: RunnerLike) -> None:
        self.before_iter(runner)

    def before_val_iter(self, runner: RunnerLike) -> None:
        self.before_iter(runner)

    def after_train_iter(self, runner: RunnerLike) -> None:
        self.after_iter(runner)

    def after_val_iter(self, runner: RunnerLike) -> None:
        self.after_iter(runner)

    def every_n_epochs(self, runner: RunnerLike, n: int) -> bool:
        """Return whether the current epoch is an ``n``-epoch boundary."""
        return (runner.epoch + 1) % n == 0 if n > 0 else False

    def every_n_inner_iters(self, runner: RunnerLike, n: int) -> bool:
        """Return whether the current inner iteration is an ``n``-step boundary."""
        return (runner.inner_iter + 1) % n == 0 if n > 0 else False

    def every_n_iters(self, runner: RunnerLike, n: int) -> bool:
        """Return whether the current global iteration is an ``n``-step boundary."""
        return (runner.iter + 1) % n == 0 if n > 0 else False

    def end_of_epoch(self, runner: RunnerLike) -> bool:
        """Return whether the current iteration is the last in its epoch."""
        return runner.inner_iter + 1 == len(runner.data_loader)

    def is_last_epoch(self, runner: RunnerLike) -> bool:
        """Return whether the current epoch is the final configured epoch."""
        return runner.epoch + 1 == runner._max_epochs

    def is_last_iter(self, runner: RunnerLike) -> bool:
        """Return whether the current iteration is the final configured step."""
        return runner.iter + 1 == runner._max_iters

    def get_triggered_stages(self) -> list[str]:
        """Return lifecycle stages overridden by this hook."""
        trigger_stages = set()
        for stage in Hook.stages:
            if is_method_overridden(stage, Hook, self):
                trigger_stages.add(stage)

        # some methods will be triggered in multi stages
        # use this dict to map method to stages.
        method_stages_map = {
            "before_epoch": ["before_train_epoch", "before_val_epoch"],
            "after_epoch": ["after_train_epoch", "after_val_epoch"],
            "before_iter": ["before_train_iter", "before_val_iter"],
            "after_iter": ["after_train_iter", "after_val_iter"],
        }

        for method, map_stages in method_stages_map.items():
            if is_method_overridden(method, Hook, self):
                trigger_stages.update(map_stages)

        return [stage for stage in Hook.stages if stage in trigger_stages]
