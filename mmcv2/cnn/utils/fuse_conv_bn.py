from typing import cast

import torch
from torch import nn


def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.modules.batchnorm._BatchNorm) -> nn.Conv2d:
    """Fuse conv and bn into one module.

    Args:
        conv (nn.Module): Conv to be fused.
        bn (nn.Module): BN to be fused.

    Returns:
        nn.Module: Fused module.
    """
    running_mean = cast(torch.Tensor, bn.running_mean)
    running_var = cast(torch.Tensor, bn.running_var)
    weight = cast(torch.Tensor, bn.weight)
    bias = cast(torch.Tensor, bn.bias)
    conv_w = conv.weight
    conv_b = conv.bias if conv.bias is not None else torch.zeros_like(running_mean)

    factor = weight / torch.sqrt(running_var + bn.eps)
    conv.weight = nn.Parameter(conv_w * factor.reshape([conv.out_channels, 1, 1, 1]))
    conv.bias = nn.Parameter((conv_b - running_mean) * factor + bias)
    return conv


def fuse_conv_bn(module: nn.Module) -> nn.Module:
    """Recursively fuse conv and bn in a module.

    During inference, the functionary of batch norm layers is turned off
    but only the mean and var alone channels are used, which exposes the
    chance to fuse it with the preceding conv layers to save computations and
    simplify network structures.

    Args:
        module (nn.Module): Module to be fused.

    Returns:
        nn.Module: Fused module.
    """
    last_conv: nn.Conv2d | None = None
    last_conv_name: str | None = None

    for name, child in module.named_children():
        if isinstance(child, (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            fused_conv = _fuse_conv_bn(last_conv, child)
            module._modules[cast(str, last_conv_name)] = fused_conv
            # To reduce changes, set BN as Identity instead of deleting it.
            module._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, nn.Conv2d):
            last_conv = child
            last_conv_name = name
        else:
            fuse_conv_bn(child)
    return module
