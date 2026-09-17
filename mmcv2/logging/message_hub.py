from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any, ClassVar, cast

import numpy as np
import torch

from .history_buffer import HistoryBuffer


class MessageHub:
    """Named runtime state shared by runner components.

    Scalars are retained as :class:`HistoryBuffer` objects. Runtime values can
    opt into checkpoint persistence through the ``resumed`` flag.
    """

    _instances: ClassVar[dict[str, MessageHub]] = {}
    _current_name: ClassVar[str | None] = None

    def __init__(self, name: str) -> None:
        if not name:
            raise ValueError("name must be non-empty")
        self.name = name
        self._log_scalars: OrderedDict[str, HistoryBuffer] = OrderedDict()
        self._runtime_info: OrderedDict[str, Any] = OrderedDict()
        self._resumed_keys: dict[str, bool] = {}

    @classmethod
    def get_instance(cls, name: str, **kwargs) -> MessageHub:
        if name not in cls._instances:
            cls._instances[name] = cls(name=name, **kwargs)
        cls._current_name = name
        return cls._instances[name]

    @classmethod
    def get_current_instance(cls) -> MessageHub:
        if cls._current_name is None:
            return cls.get_instance("default")
        return cls._instances[cls._current_name]

    @classmethod
    def check_instance_created(cls, name: str) -> bool:
        return name in cls._instances

    def update_scalar(
        self,
        key: str,
        value: int | float | np.number | torch.Tensor,
        count: int = 1,
        *,
        resumed: bool = True,
    ) -> None:
        scalar_value: Any = value
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("Only scalar tensors can be logged")
            scalar_value = value.detach().item()
        elif isinstance(value, np.number):
            scalar_value = value.item()
        if not isinstance(scalar_value, (int, float)):
            raise TypeError("value must be a scalar")
        scalar = cast(int | float, scalar_value)
        self._log_scalars.setdefault(key, HistoryBuffer()).update(scalar, count)
        self._set_resumed(key, resumed)

    def update_scalars(self, log_dict: dict[str, Any], *, resumed: bool = True) -> None:
        for key, value in log_dict.items():
            if isinstance(value, dict):
                self.update_scalar(
                    key,
                    value["value"],
                    value.get("count", 1),
                    resumed=value.get("resumed", resumed),
                )
            else:
                self.update_scalar(key, value, resumed=resumed)

    def update_info(self, key: str, value: Any, *, resumed: bool = True) -> None:
        self._runtime_info[key] = value
        self._set_resumed(key, resumed)

    def update_info_dict(self, info_dict: dict[str, Any], *, resumed: bool = True) -> None:
        for key, value in info_dict.items():
            self.update_info(key, value, resumed=resumed)

    def _set_resumed(self, key: str, resumed: bool) -> None:
        if key in self._resumed_keys and self._resumed_keys[key] != resumed:
            raise ValueError(f"resumed for {key!r} cannot be changed")
        self._resumed_keys[key] = resumed

    @property
    def log_scalars(self) -> OrderedDict[str, HistoryBuffer]:
        return self._log_scalars

    @property
    def runtime_info(self) -> OrderedDict[str, Any]:
        return self._runtime_info

    def get_scalar(self, key: str) -> HistoryBuffer:
        return self._log_scalars[key]

    def get_info(self, key: str, default: Any = None) -> Any:
        return self._runtime_info.get(key, default)

    def pop_info(self, key: str, default: Any = None) -> Any:
        self._resumed_keys.pop(key, None)
        return self._runtime_info.pop(key, default)

    def state_dict(self) -> dict[str, Any]:
        scalars = {
            key: value.state_dict() for key, value in self._log_scalars.items() if self._resumed_keys.get(key, True)
        }
        runtime = {
            key: copy.deepcopy(value) for key, value in self._runtime_info.items() if self._resumed_keys.get(key, True)
        }
        return {
            "log_scalars": scalars,
            "runtime_info": runtime,
            "resumed_keys": copy.deepcopy(self._resumed_keys),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._log_scalars.clear()
        for key, value in state_dict.get("log_scalars", {}).items():
            buffer = HistoryBuffer()
            buffer.load_state_dict(value)
            self._log_scalars[key] = buffer
        self._runtime_info = OrderedDict(copy.deepcopy(state_dict.get("runtime_info", {})))
        self._resumed_keys.update(state_dict.get("resumed_keys", {}))
