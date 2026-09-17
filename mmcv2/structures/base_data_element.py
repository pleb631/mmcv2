from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any, Self

import numpy as np
import torch


class BaseDataElement:
    """Container whose data fields and metainfo are tracked separately.

    Tensor-like operations are recursively applied to tensors, NumPy arrays
    and nested ``BaseDataElement`` objects while ordinary Python values are
    copied unchanged.
    """

    _metainfo_fields: set[str]
    _data_fields: set[str]

    def __init__(self, *, metainfo: dict | None = None, **kwargs) -> None:
        """Create an element with optional metadata and data fields.

        Args:
            metainfo: Metadata shared by all data fields.
            **kwargs: Initial data fields.
        """
        object.__setattr__(self, "_metainfo_fields", set())
        object.__setattr__(self, "_data_fields", set())
        if metainfo is not None:
            self.set_metainfo(metainfo)
        self.set_data(kwargs)

    def set_metainfo(self, metainfo: dict) -> None:
        """Update metadata fields in place."""
        for key, value in metainfo.items():
            self.set_field(value, key, field_type="metainfo")

    def set_data(self, data: dict) -> None:
        """Update data fields in place."""
        for key, value in data.items():
            setattr(self, key, value)

    def update(self, instance: BaseDataElement) -> None:
        """Merge metadata and data fields from another element."""
        self.set_metainfo(instance.metainfo)
        self.set_data(dict(instance.items()))

    def new(self, *, metainfo: dict | None = None, **kwargs) -> Self:
        """Create a new element of the same type."""
        result = self.__class__()
        result.set_metainfo(copy.deepcopy(self.metainfo))
        if metainfo:
            result.set_metainfo(metainfo)
        result.set_data(kwargs)
        return result

    def clone(self) -> Self:
        """Return a deep copy of this element."""
        return copy.deepcopy(self)

    def keys(self) -> list[str]:
        """Return names of data fields."""
        return [key for key in self._data_fields if hasattr(self, key)]

    def metainfo_keys(self) -> list[str]:
        """Return names of metadata fields."""
        return [key for key in self._metainfo_fields if hasattr(self, key)]

    def values(self) -> list[Any]:
        """Return values of data fields."""
        return [getattr(self, key) for key in self.keys()]

    def metainfo_values(self) -> list[Any]:
        """Return values of metadata fields."""
        return [getattr(self, key) for key in self.metainfo_keys()]

    def all_keys(self) -> list[str]:
        """Return names of metadata and data fields."""
        return self.metainfo_keys() + self.keys()

    def all_values(self) -> list[Any]:
        """Return values of metadata and data fields."""
        return self.metainfo_values() + self.values()

    def items(self) -> Iterator[tuple[str, Any]]:
        """Iterate over data-field name/value pairs."""
        for key in self.keys():
            yield key, getattr(self, key)

    def metainfo_items(self) -> Iterator[tuple[str, Any]]:
        """Iterate over metadata name/value pairs."""
        for key in self.metainfo_keys():
            yield key, getattr(self, key)

    def all_items(self) -> Iterator[tuple[str, Any]]:
        """Iterate over metadata and data-field pairs."""
        yield from self.metainfo_items()
        yield from self.items()

    @property
    def metainfo(self) -> dict[str, Any]:
        """Return a shallow copy of the metadata mapping."""
        return {key: copy.deepcopy(value) for key, value in self.metainfo_items()}

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self.set_field(value, name, field_type="data")

    def __delattr__(self, name: str) -> None:
        if name in self._metainfo_fields:
            self._metainfo_fields.remove(name)
        if name in self._data_fields:
            self._data_fields.remove(name)
        object.__delattr__(self, name)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return key in self._data_fields or key in self._metainfo_fields

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def get(self, key: str, default: Any = None) -> Any:
        """Return a data field or ``default`` when it is absent."""
        return getattr(self, key, default)

    def pop(self, key: str, *args) -> Any:
        """Remove and return a data field."""
        if key not in self:
            if args:
                return args[0]
            raise KeyError(key)
        value = getattr(self, key)
        delattr(self, key)
        return value

    def set_field(
        self,
        value: Any,
        name: str,
        dtype: type | tuple[type, ...] | None = None,
        field_type: str = "data",
    ) -> None:
        """Validate and register a metadata or data field.

        Args:
            value: Value assigned to the field.
            name: Field name.
            dtype: Optional accepted type or tuple of types.
            field_type: Either ``"data"`` or ``"metainfo"``.
        """
        if dtype is not None and not isinstance(value, dtype):
            raise AssertionError(f"{name} should be a {dtype}, but got {type(value)}")
        if field_type not in {"data", "metainfo"}:
            raise ValueError("field_type must be 'data' or 'metainfo'")
        target = self._data_fields if field_type == "data" else self._metainfo_fields
        other = self._metainfo_fields if field_type == "data" else self._data_fields
        if name in other:
            raise AttributeError(f"{name!r} is already used as another field type")
        target.add(name)
        object.__setattr__(self, name, value)

    @staticmethod
    def _apply(value: Any, method: str, *args, **kwargs) -> Any:
        if isinstance(value, BaseDataElement):
            return getattr(value, method)(*args, **kwargs)
        if isinstance(value, dict):
            return {k: BaseDataElement._apply(v, method, *args, **kwargs) for k, v in value.items()}
        if isinstance(value, list):
            return [BaseDataElement._apply(v, method, *args, **kwargs) for v in value]
        if isinstance(value, tuple):
            return tuple(BaseDataElement._apply(v, method, *args, **kwargs) for v in value)
        fn = getattr(value, method, None)
        return fn(*args, **kwargs) if callable(fn) else value

    def _transform(self, method: str, *args, **kwargs) -> Self:
        result = self.new()
        for key, value in self.items():
            setattr(result, key, self._apply(value, method, *args, **kwargs))
        return result

    def to(self, *args, **kwargs) -> Self:
        """Move tensor-like fields to a device or dtype."""
        return self._transform("to", *args, **kwargs)

    def cpu(self) -> Self:
        """Move tensor-like fields to CPU."""
        return self._transform("cpu")

    def cuda(self, *args, **kwargs) -> Self:
        """Move tensor-like fields to CUDA."""
        return self._transform("cuda", *args, **kwargs)

    def npu(self, *args, **kwargs) -> Self:
        return self._transform("npu", *args, **kwargs)

    def mlu(self, *args, **kwargs) -> Self:
        return self._transform("mlu", *args, **kwargs)

    def musa(self, *args, **kwargs) -> Self:
        return self._transform("musa", *args, **kwargs)

    def detach(self) -> Self:
        """Detach tensor-like fields from their computation graphs."""
        return self._transform("detach")

    def numpy(self) -> Self:
        """Convert tensor-like fields to NumPy arrays."""
        result = self.new()
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            else:
                value = self._apply(value, "numpy")
            setattr(result, key, value)
        return result

    def to_tensor(self) -> Self:
        """Convert NumPy fields to tensors."""
        result = self.new()
        for key, value in self.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            else:
                value = self._apply(value, "to_tensor")
            setattr(result, key, value)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Return data fields as a recursively converted dictionary."""

        def convert(value: Any) -> Any:
            if isinstance(value, BaseDataElement):
                return value.to_dict()
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(convert(v) for v in value)
            return value

        return {key: convert(value) for key, value in self.all_items()}

    def __repr__(self) -> str:
        fields = ", ".join(f"{key}={value!r}" for key, value in self.all_items())
        return f"{self.__class__.__name__}({fields})"
