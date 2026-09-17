import torch
from torch import nn

from .registry import ACTIVATION_LAYERS


class HSwish(nn.Module):
    """Hard Swish Module.

    This module applies the hard swish function:

    .. math::
        Hswish(x) = x * ReLU6(x + 3) / 6

    Args:
        inplace (bool): can optionally do the operation in-place.
            Default: False.

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.act = nn.Hardswish(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the hard-swish activation."""
        return self.act(x)


ACTIVATION_LAYERS.register_module(module=nn.Hardswish, name="HSwish")
