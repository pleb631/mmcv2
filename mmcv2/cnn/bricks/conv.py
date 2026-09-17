from torch import nn

from .registry import CONV_LAYERS

CONV_LAYERS.register_module("Conv1d", module=nn.Conv1d)
CONV_LAYERS.register_module("Conv2d", module=nn.Conv2d)
CONV_LAYERS.register_module("Conv3d", module=nn.Conv3d)
CONV_LAYERS.register_module("Conv", module=nn.Conv2d)


def build_conv_layer(cfg: dict | None, *args, **kwargs) -> nn.Module:
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an conv layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = {"type": "Conv2d"}
    else:
        if "type" not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")
    if layer_type not in CONV_LAYERS:
        raise KeyError(f"Unrecognized layer type {layer_type}")
    conv_layer = CONV_LAYERS.get(layer_type)
    if conv_layer is None:
        raise KeyError(f"Unrecognized layer type {layer_type}")

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer
