from collections.abc import Callable
from typing import Any, cast

from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .registry import MODULE_WRAPPERS
from .scatter_gather import scatter_kwargs


class _StepAdapter(nn.Module):
    """Route MMCV2 step methods through DDP.forward()."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, mode: str, *args, **kwargs) -> Any:
        if mode == "train":
            train_step = cast(Callable[..., Any], getattr(self.model, "train_step"))
            return train_step(*args, **kwargs)
        if mode == "val":
            val_step = cast(Callable[..., Any], getattr(self.model, "val_step"))
            return val_step(*args, **kwargs)
        if mode == "test":
            test_step = cast(Callable[..., Any], getattr(self.model, "test_step"))
            return test_step(*args, **kwargs)
        return self.model(*args, **kwargs)


@MODULE_WRAPPERS.register_module()
class MMDistributedDataParallel(nn.Module):
    """A thin MMCV2 step interface around native DistributedDataParallel.

    ``module`` exposes the original model so checkpoints retain parameter
    names and existing hooks continue to see the model they wrapped.
    """

    def __init__(self, module: nn.Module, *args, **kwargs):
        super().__init__()
        self.dim = kwargs.get("dim", 0)
        kwargs.setdefault("gradient_as_bucket_view", True)
        self._ddp = DistributedDataParallel(_StepAdapter(module), *args, **kwargs)

    @property
    def module(self) -> nn.Module:
        return self._ddp.module.model

    @property
    def device_ids(self):
        return self._ddp.device_ids

    def _call(self, mode: str, *args, **kwargs) -> Any:
        device_ids = self.device_ids
        target = [device_ids[0]] if device_ids else [-1]
        inputs, keyword_args = self.scatter(args, kwargs, target)
        return self._ddp(mode, *inputs[0], **keyword_args[0])

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def to_kwargs(self, inputs, kwargs, device_id):
        return self.scatter(inputs, kwargs, [device_id])

    def forward(self, *args, **kwargs) -> Any:
        return self._call("forward", *args, **kwargs)

    def train_step(self, *args, **kwargs) -> Any:
        return self._call("train", *args, **kwargs)

    def val_step(self, *args, **kwargs) -> Any:
        return self._call("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> Any:
        return self._call("test", *args, **kwargs)

    def no_sync(self):
        return self._ddp.no_sync()
