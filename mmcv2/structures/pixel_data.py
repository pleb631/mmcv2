from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .base_data_element import BaseDataElement


class PixelData(BaseDataElement):
    """Container for spatial tensors/arrays sharing the same H x W shape."""

    def __setattr__(self, name: str, value: Any) -> None:
        if not name.startswith("_"):
            if not isinstance(value, (torch.Tensor, np.ndarray)):
                raise AssertionError(f"{name} must be a tensor or numpy array")
            if value.ndim < 2:
                raise AssertionError(f"{name} must have at least two dimensions")
            if self._data_fields and tuple(value.shape[-2:]) != self.shape:
                raise AssertionError(f"The shape of {name} does not match {self.shape}")
        super().__setattr__(name, value)

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shared spatial ``(height, width)`` shape."""
        if not self._data_fields:
            return (0, 0)
        value = getattr(self, next(iter(self._data_fields)))
        return tuple(value.shape[-2:])

    def __getitem__(self, item: Any) -> PixelData:
        """Select fields or spatial values and return a new container."""
        if isinstance(item, str):
            return super().__getitem__(item)
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("PixelData spatial indexing requires a (height, width) tuple")
        item = tuple(slice(index, index + 1) if isinstance(index, int) else index for index in item)
        result = self.new()
        for key, value in self.items():
            sliced = value[(..., *item)]
            setattr(result, key, sliced)
        return result
