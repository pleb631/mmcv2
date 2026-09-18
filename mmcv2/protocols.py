from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

PathLike = str | Path


class RunnerLike(Protocol):
    """Runner surface consumed by hooks.

    This deliberately records the existing hook boundary without requiring a
    hook to depend on the concrete ``BaseRunner`` implementation.
    """

    model: Any
    # Hooks operate after runner setup, where these conditional constructor
    # values have been resolved for the active training workflow.
    optimizer: Any
    logger: logging.Logger
    meta: dict[str, Any] | None
    mode: str | None
    outputs: dict[str, Any]
    data_loader: Any
    work_dir: str
    timestamp: str
    log_buffer: Any
    message_hub: Any
    ctx: Any
    amp_dtype: Any
    _hooks: list[Any]
    _max_epochs: int
    _max_iters: int
    _optimizer_step_boundary: bool

    @property
    def epoch(self) -> int:
        """Return the current epoch index."""
        ...

    @property
    def iter(self) -> int:
        """Return the current global iteration index."""
        ...

    @property
    def inner_iter(self) -> int:
        """Return the current iteration index within the epoch."""
        ...

    @property
    def max_epochs(self) -> int:
        """Return the configured maximum number of epochs."""
        ...

    @property
    def max_iters(self) -> int:
        """Return the configured maximum number of iterations."""
        ...

    @property
    def rank(self) -> int:
        """Return the current distributed-process rank."""
        ...

    @property
    def world_size(self) -> int:
        """Return the number of distributed processes."""
        ...

    @property
    def hooks(self) -> list[Any]:
        """Return the registered runner hooks in execution order."""
        ...

    def current_lr(self) -> list[float] | dict[str, list[float]]:
        """Return current learning rates for the configured optimizer(s)."""
        ...

    def current_momentum(self) -> list[float] | dict[str, list[float]]:
        """Return current momentum values for the configured optimizer(s)."""
        ...

    def capture_rng_state(self) -> dict[str, Any]:
        """Capture random-number-generator state for checkpointing."""
        ...

    def save_checkpoint(
        self,
        out_dir: str,
        filename_tmpl: str = "epoch_{}.pth",
        save_optimizer: bool = True,
        meta: dict[str, Any] | None = None,
        create_symlink: bool = True,
    ) -> None:
        """Save model and runner state to a checkpoint.

        Args:
            out_dir: Directory receiving the checkpoint.
            filename_tmpl: Format string used to build the filename.
            save_optimizer: Whether to include optimizer state.
            meta: Additional metadata to store.
            create_symlink: Whether to update the latest-checkpoint link.
        """
        ...

    def resume(self, checkpoint: str, **kwargs: Any) -> None:
        """Resume the runner from a checkpoint."""
        ...


@runtime_checkable
class HookLike(Protocol):
    """Lifecycle callbacks understood by a runner."""

    def before_run(self, runner: RunnerLike) -> None:
        """Handle the start of a runner execution."""
        ...

    def after_run(self, runner: RunnerLike) -> None:
        """Handle the end of a runner execution."""
        ...

    def before_train_epoch(self, runner: RunnerLike) -> None:
        """Handle the start of a training epoch."""
        ...

    def after_train_epoch(self, runner: RunnerLike) -> None:
        """Handle the end of a training epoch."""
        ...

    def before_train_iter(self, runner: RunnerLike) -> None:
        """Handle the start of a training iteration."""
        ...

    def after_train_iter(self, runner: RunnerLike) -> None:
        """Handle the end of a training iteration."""
        ...

    def before_val_epoch(self, runner: RunnerLike) -> None:
        """Handle the start of a validation epoch."""
        ...

    def after_val_epoch(self, runner: RunnerLike) -> None:
        """Handle the end of a validation epoch."""
        ...

    def before_val_iter(self, runner: RunnerLike) -> None:
        """Handle the start of a validation iteration."""
        ...

    def after_val_iter(self, runner: RunnerLike) -> None:
        """Handle the end of a validation iteration."""
        ...

    def get_triggered_stages(self) -> list[str]:
        """Return lifecycle stages implemented by the hook."""
        ...


@runtime_checkable
class MetricLike(Protocol):
    """Streaming metric surface consumed by ``Evaluator``."""

    dataset_meta: dict[str, Any] | None

    def process(self, data_batch: Any, data_samples: Sequence[Any]) -> None:
        """Accumulate metric inputs from one processed batch."""
        ...

    def evaluate(self, size: int) -> dict[str, Any]:
        """Return metric values for the collected dataset samples."""
        ...


@runtime_checkable
class StorageBackendLike(Protocol):
    """Storage operations consumed by the functional file-I/O API."""

    @property
    def name(self) -> str:
        """Return the backend's display name."""
        ...

    @property
    def allow_symlink(self) -> bool:
        """Return whether the backend supports symbolic links."""
        ...

    def get(self, filepath: PathLike) -> bytes | memoryview:
        """Read binary content from a path."""
        ...

    def get_text(self, filepath: PathLike, encoding: str = "utf-8") -> str:
        """Read text content from a path."""
        ...

    def put(self, obj: bytes, filepath: PathLike) -> None:
        """Write binary content to a path."""
        ...

    def put_text(self, obj: str, filepath: PathLike, encoding: str = "utf-8") -> None:
        """Write text content to a path."""
        ...

    def remove(self, filepath: PathLike) -> None:
        """Remove a file from the backend."""
        ...

    def exists(self, filepath: PathLike) -> bool:
        """Return whether a path exists."""
        ...

    def isdir(self, filepath: PathLike) -> bool:
        """Return whether a path identifies a directory."""
        ...

    def isfile(self, filepath: PathLike) -> bool:
        """Return whether a path identifies a regular file."""
        ...

    def join_path(self, filepath: PathLike, *filepaths: PathLike) -> str:
        """Join path components using backend-specific rules."""
        ...

    def get_local_path(self, filepath: PathLike) -> AbstractContextManager[PathLike]:
        """Provide a context manager yielding a local path."""
        ...

    def list_dir_or_file(
        self,
        dir_path: PathLike,
        list_dir: bool = True,
        list_file: bool = True,
        suffix: str | tuple[str, ...] | None = None,
        recursive: bool = False,
    ) -> Iterator[str]:
        """Iterate over files and directories below a directory."""
        ...


__all__ = ["HookLike", "MetricLike", "RunnerLike", "StorageBackendLike"]
