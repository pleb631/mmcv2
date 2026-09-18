from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar, cast
from weakref import ReferenceType, ref

import torch

if TYPE_CHECKING:
    from .base_runner import BaseRunner


_ACTIVE_RUNNER_CONTEXT: ContextVar[RunnerContext | None] = ContextVar("mmcv2_runner_context", default=None)
_ReturnT = TypeVar("_ReturnT")


class RunnerContext:
    """A non-owning, read-only view of a runner's live training state.

    The context stores only a weak reference to its runner. Properties resolve
    values dynamically, so there is one authoritative copy of mutable training
    state and retaining a context does not retain the runner or its resources.
    """

    def __init__(self, runner: BaseRunner) -> None:
        self._runner_ref: ReferenceType[BaseRunner] = ref(runner)

    @property
    def runner(self) -> BaseRunner:
        """Return the owning runner while it is alive."""
        runner = self._runner_ref()
        if runner is None:
            raise RuntimeError("The runner associated with this context no longer exists")
        return runner

    @classmethod
    def current(cls) -> RunnerContext:
        """Return the context active in the current execution context."""
        context = _ACTIVE_RUNNER_CONTEXT.get()
        if context is None:
            raise RuntimeError("No RunnerContext is active")
        # Fail immediately when a retained context outlives its runner.
        context.runner
        return context

    @contextmanager
    def activate(self) -> Iterator[RunnerContext]:
        """Make this context discoverable for the duration of a runner call."""
        token = _ACTIVE_RUNNER_CONTEXT.set(self)
        try:
            yield self
        finally:
            _ACTIVE_RUNNER_CONTEXT.reset(token)

    @property
    def model(self) -> torch.nn.Module:
        return self.runner.model

    @property
    def step_model(self) -> torch.nn.Module:
        return self.runner.step_model

    @property
    def optimizer(self) -> Any:
        return self.runner.optimizer

    @property
    def data_loader(self) -> Any | None:
        return getattr(self.runner, "data_loader", None)

    @property
    def dataset(self) -> Any | None:
        loader = self.data_loader
        source = getattr(loader, "_dataloader", loader)
        return getattr(source, "dataset", None)

    @property
    def data_batch(self) -> Any | None:
        return getattr(self.runner, "data_batch", None)

    @property
    def outputs(self) -> dict[str, Any] | None:
        return getattr(self.runner, "outputs", None)

    @property
    def mode(self) -> str | None:
        return self.runner.mode

    @property
    def epoch(self) -> int:
        return self.runner.epoch

    @property
    def iter(self) -> int:
        return self.runner.iter

    @property
    def inner_iter(self) -> int:
        return self.runner.inner_iter

    @property
    def max_epochs(self) -> int | None:
        return self.runner._max_epochs

    @property
    def max_iters(self) -> int | None:
        return self.runner._max_iters

    @property
    def rank(self) -> int:
        return self.runner.rank

    @property
    def world_size(self) -> int:
        return self.runner.world_size

    @property
    def device(self) -> torch.device:
        parameter = next(self.step_model.parameters(), None)
        return parameter.device if parameter is not None else torch.device("cpu")

    @property
    def work_dir(self) -> str | None:
        return self.runner.work_dir

    @property
    def logger(self) -> Any:
        return self.runner.logger

    @property
    def meta(self) -> dict[str, Any] | None:
        return cast(dict[str, Any] | None, self.runner.meta)

    @property
    def message_hub(self) -> Any:
        return self.runner.message_hub

    @property
    def log_buffer(self) -> Any:
        return self.runner.log_buffer

    @property
    def hooks(self) -> tuple[Any, ...]:
        return tuple(self.runner.hooks)

    def __repr__(self) -> str:
        runner = self._runner_ref()
        if runner is None:
            return "RunnerContext(expired=True)"
        return (
            f"RunnerContext(runner={runner.__class__.__name__}, mode={runner.mode!r}, "
            f"epoch={runner.epoch}, iter={runner.iter})"
        )


def activate_runner_context(method: Callable[..., _ReturnT]) -> Callable[..., _ReturnT]:
    """Run a runner method with its context available through ``current()``."""

    @wraps(method)
    def wrapped(runner: BaseRunner, *args: Any, **kwargs: Any) -> _ReturnT:
        with runner.ctx.activate():
            return method(runner, *args, **kwargs)

    return wrapped


__all__ = ["RunnerContext"]
