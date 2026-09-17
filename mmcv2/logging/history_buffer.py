from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import numpy as np


class HistoryBuffer:
    """Bounded scalar history with weighted, extensible statistics."""

    _statistics_methods: ClassVar[dict[str, Callable[..., Any]]] = {}

    def __init__(
        self,
        log_history: Sequence[float] = (),
        count_history: Sequence[float] = (),
        max_length: int = 1_000_000,
    ) -> None:
        if len(log_history) != len(count_history):
            raise ValueError("log_history and count_history must have equal length")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.max_length = max_length
        self._log_history = np.asarray(log_history[-max_length:], dtype=np.float64)
        self._count_history = np.asarray(count_history[-max_length:], dtype=np.float64)
        self._statistics_methods.setdefault("mean", HistoryBuffer.mean)
        self._statistics_methods.setdefault("min", HistoryBuffer.min)
        self._statistics_methods.setdefault("max", HistoryBuffer.max)
        self._statistics_methods.setdefault("current", HistoryBuffer.current)

    def update(self, log_val: Any, count: Any = 1) -> None:
        if not isinstance(log_val, (int, float)) or not isinstance(count, (int, float)):
            raise TypeError("log_val and count must be numeric")
        if count <= 0:
            raise ValueError("count must be positive")
        self._log_history = np.append(self._log_history, float(log_val))[-self.max_length :]
        self._count_history = np.append(self._count_history, float(count))[-self.max_length :]

    @property
    def data(self) -> tuple[np.ndarray, np.ndarray]:
        return self._log_history, self._count_history

    @classmethod
    def register_statistics(cls, method: Callable[..., Any]) -> Callable[..., Any]:
        if method.__name__ in cls._statistics_methods:
            raise KeyError(f"Statistic {method.__name__!r} is already registered")
        cls._statistics_methods[method.__name__] = method
        return method

    def statistics(self, method_name: str, *args, **kwargs) -> Any:
        try:
            method = self._statistics_methods[method_name]
        except KeyError as exc:
            raise KeyError(f"Unknown statistic {method_name!r}") from exc
        return method(self, *args, **kwargs)

    def _window(self, window_size: int | None) -> tuple[np.ndarray, np.ndarray]:
        if not len(self._log_history):
            raise ValueError("HistoryBuffer is empty")
        if window_size is not None and (not isinstance(window_size, int) or window_size <= 0):
            raise ValueError("window_size must be a positive integer or None")
        size = len(self._log_history) if window_size is None else window_size
        return self._log_history[-size:], self._count_history[-size:]

    def mean(self, window_size: int | None = None) -> float:
        values, counts = self._window(window_size)
        return float(np.sum(values * counts) / np.sum(counts))

    def min(self, window_size: int | None = None) -> float:
        return float(self._window(window_size)[0].min())

    def max(self, window_size: int | None = None) -> float:
        return float(self._window(window_size)[0].max())

    def current(self) -> float:
        return float(self._window(1)[0][-1])

    def state_dict(self) -> dict[str, Any]:
        return {
            # Plain scalar lists remain compatible with torch.load's safe
            # ``weights_only=True`` mode when embedded in runner checkpoints.
            "log_history": self._log_history.tolist(),
            "count_history": self._count_history.tolist(),
            "max_length": self.max_length,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.max_length = int(state_dict["max_length"])
        self._log_history = np.asarray(state_dict["log_history"], dtype=np.float64)
        self._count_history = np.asarray(state_dict["count_history"], dtype=np.float64)
