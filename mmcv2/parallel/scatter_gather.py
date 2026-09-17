from typing import Any, cast

from torch import Tensor
from torch.nn.parallel._functions import Scatter as OrigScatter

from ._functions import Scatter
from .data_container import DataContainer

ScatterInputs = Tensor | DataContainer | tuple | list | dict


def scatter(inputs: ScatterInputs, target_gpus: list[int], dim: int = 0) -> list:
    """Scatter inputs to target gpus.

    The only difference from original :func:`scatter` is to add support for
    :type:`~mmcv2.parallel.DataContainer`.
    """

    def scatter_map(obj):
        if isinstance(obj, Tensor):
            if target_gpus != [-1]:
                return OrigScatter.apply(target_gpus, None, dim, obj)
            else:
                # for CPU inference we use self-implemented scatter
                return Scatter.forward(target_gpus, obj)
        if isinstance(obj, DataContainer):
            if obj.cpu_only:
                return obj.data
            else:
                return Scatter.forward(target_gpus, obj.data)
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            out = list(map(list, zip(*map(scatter_map, obj))))
            return out
        if isinstance(obj, dict) and len(obj) > 0:
            out = list(map(type(obj), zip(*map(scatter_map, obj.items()))))
            return out
        return [obj for _ in target_gpus]

    # ``scatter_map`` is recursive, so its closure contains a reference to
    # itself. Clear the closure cell after every call to avoid leaving a
    # reference cycle for the cyclic garbage collector.
    try:
        return cast(list[Any], scatter_map(inputs))
    finally:
        scatter_map = None  # type: ignore[assignment]


def scatter_kwargs(
    inputs: ScatterInputs, kwargs: ScatterInputs, target_gpus: list[int], dim: int = 0
) -> tuple[tuple, tuple]:
    """Scatter with support for kwargs dictionary."""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        length = len(kwargs) - len(inputs)
        inputs.extend([() for _ in range(length)])  # type: ignore
    elif len(kwargs) < len(inputs):
        length = len(inputs) - len(kwargs)
        kwargs.extend([{} for _ in range(length)])  # type: ignore
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs
