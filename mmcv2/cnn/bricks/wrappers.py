import torch
from torch import nn

from .registry import CONV_LAYERS, UPSAMPLE_LAYERS


@CONV_LAYERS.register_module("Conv", force=True)
class Conv2d(nn.Conv2d):
    pass


@CONV_LAYERS.register_module("Conv3d", force=True)
class Conv3d(nn.Conv3d):
    pass


@CONV_LAYERS.register_module()
@CONV_LAYERS.register_module("deconv")
@UPSAMPLE_LAYERS.register_module("deconv", force=True)
class ConvTranspose2d(nn.ConvTranspose2d):
    pass


@CONV_LAYERS.register_module()
@CONV_LAYERS.register_module("deconv3d")
@UPSAMPLE_LAYERS.register_module("deconv3d", force=True)
class ConvTranspose3d(nn.ConvTranspose3d):
    pass


class MaxPool2d(nn.MaxPool2d):
    pass


class MaxPool3d(nn.MaxPool3d):
    pass


class Linear(torch.nn.Linear):
    pass
