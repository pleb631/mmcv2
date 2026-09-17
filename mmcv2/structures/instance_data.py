from __future__ import annotations

from collections.abc import Sized
from typing import Any

import numpy as np
import torch

from .base_data_element import BaseDataElement


class InstanceData(BaseDataElement):
    """Data container whose fields all describe the same number of instances."""

    def __setattr__(self, name: str, value: Any) -> None:
        if not name.startswith("_"):
            if not isinstance(value, Sized):
                raise AssertionError(f"{name} must implement __len__")
            if self._data_fields and len(value) != len(self):
                raise AssertionError(f"The length of {name} ({len(value)}) does not match {len(self)}")
        super().__setattr__(name, value)

    def __len__(self) -> int:
        """Return the number of instances represented by the fields."""
        if not self._data_fields:
            return 0
        return len(getattr(self, next(iter(self._data_fields))))

    def __getitem__(self, item: Any) -> InstanceData:
        """Select instances or fields and return a new container."""
        if isinstance(item, str):
            return super().__getitem__(item)
        if isinstance(item, int):
            if item >= len(self) or item < -len(self):
                raise IndexError("InstanceData index out of range")
            item = slice(item, None if item == -1 else item + 1)
        result = self.new()
        for key, value in self.items():
            if isinstance(value, list) and isinstance(item, torch.Tensor):
                indices = item.nonzero().flatten().tolist() if item.dtype == torch.bool else item.tolist()
                sliced = [value[i] for i in indices]
            elif isinstance(value, list) and isinstance(item, np.ndarray):
                indices = np.nonzero(item)[0].tolist() if item.dtype == bool else item.tolist()
                sliced = [value[i] for i in indices]
            else:
                sliced = value[item]
            setattr(result, key, sliced)
        return result

    @staticmethod
    def cat(instances_list: list[InstanceData]) -> InstanceData:
        """Concatenate instance containers along their instance dimension.

        Args:
            instances_list: Containers with matching field names.

        Returns:
            A new container containing all instances.
        """
        if not instances_list or not all(isinstance(x, InstanceData) for x in instances_list):
            raise AssertionError("instances_list must be a non-empty list of InstanceData")
        keys = set(instances_list[0].keys())
        if any(set(instance.keys()) != keys for instance in instances_list[1:]):
            raise AssertionError("All InstanceData objects must have identical fields")
        result = instances_list[0].new()
        for key in keys:
            values = [getattr(instance, key) for instance in instances_list]
            first = values[0]
            if isinstance(first, torch.Tensor):
                value = torch.cat(values, dim=0)
            elif isinstance(first, np.ndarray):
                value = np.concatenate(values, axis=0)
            elif isinstance(first, list):
                value = sum(values, [])
            elif hasattr(first.__class__, "cat"):
                value = first.__class__.cat(values)
            else:
                raise TypeError(f"Cannot concatenate field {key} of type {type(first)}")
            setattr(result, key, value)
        return result
