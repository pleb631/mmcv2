import torch
import torch.nn.functional as F
from torch import nn

from .registry import ACTIVATION_LAYERS


@ACTIVATION_LAYERS.register_module()
class HSigmoid(nn.Module):
    """Hard Sigmoid Module. Apply the hard sigmoid function:
    Hsigmoid(x) = min(max((x + bias) / divisor, min_value), max_value)
    Default: Hsigmoid(x) = min(max((x + 3) / 6, 0), 1)

    Note:
        In MMCV2 v1.4.4, we modified the default value of args to align with
        PyTorch official.

    Args:
        bias (float): Bias of the input feature map. Default: 3.0.
        divisor (float): Divisor of the input feature map. Default: 6.0.
        min_value (float): Lower bound value. Default: 0.0.
        max_value (float): Upper bound value. Default: 1.0.

    Returns:
        Tensor: The output tensor.
    """

    def __init__(
        self,
        bias: float = 3.0,
        divisor: float = 6.0,
        min_value: float = 0.0,
        max_value: float = 1.0,
    ):
        super().__init__()
        self.bias = bias
        self.divisor = divisor
        assert self.divisor != 0
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the hard-sigmoid activation."""
        if self.bias == 3.0 and self.divisor == 6.0 and self.min_value == 0.0 and self.max_value == 1.0:
            return F.hardsigmoid(x)
        x = (x + self.bias) / self.divisor
        return x.clamp(self.min_value, self.max_value)
