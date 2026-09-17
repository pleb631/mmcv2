import io
import logging
import os.path as osp
import re
import time
from collections import OrderedDict
from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps
from typing import Any, cast

import torch
import torchvision
from torch import nn
from torch.hub import load_state_dict_from_url
from torch.optim import Optimizer

from .._version import __version__
from ..distributed import get_dist_info
from ..fileio import put
from ..parallel import is_module_wrapper
from ..utils import print_log

ENV_MMCV2_HOME = "MMCV2_HOME"
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
DEFAULT_CACHE_DIR = "~/.cache"
_LOAD_WEIGHTS_ONLY: ContextVar[bool] = ContextVar("mmcv2_checkpoint_weights_only", default=True)
_LOAD_MMAP: ContextVar[bool | None] = ContextVar("mmcv2_checkpoint_mmap", default=None)


def _weights_policy(loader):
    """Keep the registered two-argument loader contract intact."""

    @wraps(loader)
    def wrapped(*args, weights_only=None, mmap=None, **kwargs):
        if weights_only is None and mmap is None:
            return loader(*args, **kwargs)
        weights_token = _LOAD_WEIGHTS_ONLY.set(weights_only) if weights_only is not None else None
        mmap_token = _LOAD_MMAP.set(mmap) if mmap is not None else None
        try:
            return loader(*args, **kwargs)
        finally:
            if mmap_token is not None:
                _LOAD_MMAP.reset(mmap_token)
            if weights_token is not None:
                _LOAD_WEIGHTS_ONLY.reset(weights_token)

    return wrapped


def load_state_dict(
    module: nn.Module,
    state_dict: dict | OrderedDict,
    strict: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (dict or OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys: list[str] = []
    all_missing_keys: list[str] = []
    err_msg: list[str] = []

    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()  # type: ignore
    if metadata is not None:
        state_dict._metadata = metadata  # type: ignore

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=""):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        if is_module_wrapper(module):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            all_missing_keys,
            unexpected_keys,
            err_msg,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(module)
    # break load->load reference cycle
    load = None  # type: ignore

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [key for key in all_missing_keys if "num_batches_tracked" not in key]

    if unexpected_keys:
        err_msg.append(f"unexpected key in source state_dict: {', '.join(unexpected_keys)}\n")
    if missing_keys:
        err_msg.append(f"missing keys in source state_dict: {', '.join(missing_keys)}\n")

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(0, "The model and loaded state dict do not match exactly\n")
        err_msg = "\n".join(err_msg)  # type: ignore
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def get_torchvision_models():
    model_urls = {}
    weights_list = [
        torchvision.models.get_model_weights(model) for model in torchvision.models.list_models(torchvision.models)
    ]

    for cls in weights_list:
        default_weights = getattr(cls, "DEFAULT", None)
        if default_weights is None:
            continue
        cls_name = cls.__name__
        cls_key = cls_name.replace("_Weights", "").lower()
        model_urls[f"{cls_key}.default"] = default_weights.url
        for weight_enum in cls:
            cls_key = cls_name.replace("_Weights", "").lower()
            cls_key = f"{cls_key}.{weight_enum.name.lower()}"
            model_urls[cls_key] = weight_enum.url

    return model_urls


class CheckpointLoader:
    """A general checkpoint loader to manage all schemes."""

    _schemes: dict = {}

    @classmethod
    def _register_scheme(cls, prefixes: str | list | tuple, loader: Callable, force: bool = False) -> None:
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        else:
            assert isinstance(prefixes, (list, tuple))
        for prefix in prefixes:
            if (prefix not in cls._schemes) or force:
                cls._schemes[prefix] = loader
            else:
                raise KeyError(
                    f'{prefix} is already registered as a loader backend, add "force=True" if you want to override it'
                )
        # sort, longer prefixes take priority
        cls._schemes = OrderedDict(sorted(cls._schemes.items(), key=lambda t: t[0], reverse=True))

    @classmethod
    def register_scheme(
        cls,
        prefixes: str | list[str] | tuple[str, ...],
        loader: Callable | None = None,
        force: bool = False,
    ) -> Callable:
        """Register a loader to CheckpointLoader.

        This method can be used as a normal class method or a decorator.

        Args:
            prefixes (str or Sequence[str]):
            The prefix of the registered loader.
            loader (function, optional): The loader function to be registered.
                When this method is used as a decorator, loader is None.
                Defaults to None.
            force (bool, optional): Whether to override the loader
                if the prefix has already been registered. Defaults to False.
        """

        if loader is not None:
            cls._register_scheme(prefixes, loader, force=force)
            return  # type: ignore

        def _register(loader_cls):
            cls._register_scheme(prefixes, loader_cls, force=force)
            return loader_cls

        return _register

    @classmethod
    def _get_checkpoint_loader(cls, path: str):
        """Find a loader for the path, falling back to local files only.

        Args:
            path (str): checkpoint path

        Returns:
            callable: checkpoint loader
        """
        for p in cls._schemes:
            # use regular match to handle some cases that where the prefix of
            # loader has a prefix.
            if re.match(p, path) is not None:
                if p == "" and "://" in path:
                    break
                return cls._schemes[p]
        raise ValueError(
            f"Unsupported checkpoint URI: {path}. Supported sources: local files, HTTP(S), torchvision://."
        )

    @classmethod
    def load_checkpoint(
        cls,
        filename: str,
        map_location: str | Callable | None = None,
        logger: logging.Logger | None = None,
        weights_only: bool = True,
        mmap: bool | None = None,
    ) -> dict | OrderedDict:
        """load checkpoint through URL scheme path.

        Args:
            filename (str): checkpoint file name with given prefix
            map_location (str, optional): Same as :func:`torch.load`.
                Default: None
            logger (:mod:`logging.Logger`, optional): The logger for message.
                Default: None

        Returns:
            dict or OrderedDict: The loaded checkpoint.
        """

        checkpoint_loader = cls._get_checkpoint_loader(filename)
        class_name = checkpoint_loader.__name__  # type: ignore
        print_log(f"load checkpoint from {class_name[10:]} path: {filename}", logger)
        weights_token = _LOAD_WEIGHTS_ONLY.set(weights_only)
        mmap_token = _LOAD_MMAP.set(mmap)
        try:
            return checkpoint_loader(filename, map_location)  # type: ignore
        finally:
            _LOAD_MMAP.reset(mmap_token)
            _LOAD_WEIGHTS_ONLY.reset(weights_token)


@CheckpointLoader.register_scheme(prefixes="")
@_weights_policy
def load_from_local(
    filename: str,
    map_location: str | Callable | None = None,
) -> dict | OrderedDict:
    """load checkpoint by local file path.

    Args:
        filename (str): local checkpoint file path
        map_location (str, optional): Same as :func:`torch.load`.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    filename = osp.expanduser(filename)
    if not osp.isfile(filename):
        raise FileNotFoundError(f"{filename} can not be found.")
    load_options: dict[str, Any] = {
        "map_location": map_location,
        "weights_only": _LOAD_WEIGHTS_ONLY.get(),
    }
    if (mmap := _LOAD_MMAP.get()) is not None:
        load_options["mmap"] = mmap
    checkpoint = torch.load(filename, **load_options)
    return checkpoint


@CheckpointLoader.register_scheme(prefixes=("http://", "https://"))
@_weights_policy
def load_from_http(
    filename: str,
    map_location: str | Callable | None = None,
    model_dir: str | None = None,
) -> dict | OrderedDict:
    """load checkpoint through HTTP or HTTPS scheme path. In distributed
    setting, this function only download checkpoint at local rank 0.

    Args:
        filename (str): checkpoint file path with modelzoo or
            torchvision prefix
        map_location (str, optional): Same as :func:`torch.load`.
        model_dir (str, optional): directory in which to save the object,
            Default: None

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    rank, world_size = get_dist_info()
    if world_size <= 1:
        return load_state_dict_from_url(
            filename,
            model_dir=model_dir,
            map_location=map_location,
            weights_only=_LOAD_WEIGHTS_ONLY.get(),
        )

    if rank == 0:
        checkpoint = load_state_dict_from_url(
            filename,
            model_dir=model_dir,
            map_location=map_location,
            weights_only=_LOAD_WEIGHTS_ONLY.get(),
        )
        torch.distributed.barrier()
        return checkpoint

    torch.distributed.barrier()
    return load_state_dict_from_url(
        filename,
        model_dir=model_dir,
        map_location=map_location,
        weights_only=_LOAD_WEIGHTS_ONLY.get(),
    )


@CheckpointLoader.register_scheme(prefixes="torchvision://")
@_weights_policy
def load_from_torchvision(
    filename: str,
    map_location: str | Callable | None = None,
) -> dict | OrderedDict:
    """Load a checkpoint through a ``torchvision://`` URI.

    Args:
        filename (str): checkpoint file path with a torchvision prefix.
        map_location (str, optional): Same as :func:`torch.load`.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    model_urls = get_torchvision_models()
    model_name = filename[14:]

    # Support getting model urls in the same way as torchvision
    # `ResNet50_Weights.IMAGENET1K_V1` will be mapped to
    # resnet50.imagenet1k_v1.
    model_name = model_name.lower().replace("_weights", "")
    return load_from_http(model_urls[model_name], map_location=map_location)


def _load_checkpoint(
    filename: str,
    map_location: str | Callable | None = None,
    logger: logging.Logger | None = None,
    weights_only: bool = True,
    mmap: bool | None = None,
) -> dict | OrderedDict:
    """Load checkpoint from somewhere (modelzoo, file, url).

    Args:
        filename (str): Accept a local filepath, URL, or ``torchvision://`` URI.
        map_location (str, optional): Same as :func:`torch.load`.
           Default: None.
        logger (:mod:`logging.Logger`, optional): The logger for error message.
           Default: None

    Returns:
        dict or OrderedDict: The loaded checkpoint. It can be either an
           OrderedDict storing model weights or a dict containing other
           information, which depends on the checkpoint.
    """
    return CheckpointLoader.load_checkpoint(filename, map_location, logger, weights_only=weights_only, mmap=mmap)


def _load_checkpoint_with_prefix(
    prefix: str,
    filename: str,
    map_location: str | Callable | None = None,
    weights_only: bool = True,
    mmap: bool | None = None,
) -> dict | OrderedDict:
    """Load partial pretrained model with specific prefix.

    Args:
        prefix (str): The prefix of sub-module.
        filename (str): Accept a local filepath, URL, or ``torchvision://`` URI.
        map_location (str | None): Same as :func:`torch.load`. Default: None.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """

    checkpoint = _load_checkpoint(filename, map_location=map_location, weights_only=weights_only, mmap=mmap)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not prefix.endswith("."):
        prefix += "."
    prefix_len = len(prefix)

    state_dict = {k[prefix_len:]: v for k, v in state_dict.items() if k.startswith(prefix)}

    assert state_dict, f"{prefix} is not in the pretrained model"
    return state_dict


def load_checkpoint(
    model: torch.nn.Module,
    filename: str,
    map_location: str | Callable | None = None,
    strict: bool = False,
    logger: logging.Logger | None = None,
    revise_keys: list | None = None,
    weights_only: bool = True,
    mmap: bool | None = None,
) -> dict | OrderedDict:
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept a local filepath, URL, or ``torchvision://`` URI.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
        revise_keys (list): A list of customized keywords to modify the
            state_dict in checkpoint. Each item is a (pattern, replacement)
            pair of the regular expression operations. Default: strip
            the prefix 'module.' by [(r'^module\\.', '')].

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    if revise_keys is None:
        revise_keys = [(r"^module\.", "")]
    checkpoint = _load_checkpoint(filename, map_location, logger, weights_only=weights_only, mmap=mmap)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"No state_dict found in checkpoint file {filename}")
    # get state_dict from checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # strip prefix of state_dict
    metadata = getattr(state_dict, "_metadata", OrderedDict())
    for p, r in revise_keys:
        state_dict = OrderedDict({re.sub(p, r, k): v for k, v in state_dict.items()})
    # Keep metadata in state_dict
    object.__setattr__(state_dict, "_metadata", metadata)

    # load state_dict
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def weights_to_cpu(state_dict: OrderedDict) -> OrderedDict:
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    # Keep metadata in state_dict
    state_dict_cpu._metadata = getattr(  # type: ignore
        state_dict, "_metadata", OrderedDict()
    )
    return state_dict_cpu


def _save_to_state_dict(module: torch.nn.Module, destination: dict, prefix: str, keep_vars: bool) -> None:
    """Saves module state to `destination` dictionary.

    This method is modified from :meth:`torch.nn.Module._save_to_state_dict`.

    Args:
        module (nn.Module): The module to generate state_dict.
        destination (dict): A dict where state will be stored.
        prefix (str): The prefix for parameters and buffers used in this
            module.
    """
    for name, param in module._parameters.items():
        if param is not None:
            destination[prefix + name] = param if keep_vars else param.detach()
    for name, buf in module._buffers.items():
        # remove check of _non_persistent_buffers_set to allow nn.BatchNorm2d
        if buf is not None:
            destination[prefix + name] = buf if keep_vars else buf.detach()


def get_state_dict(
    module: torch.nn.Module,
    destination: OrderedDict | None = None,
    prefix: str = "",
    keep_vars: bool = False,
) -> OrderedDict:
    """Returns a dictionary containing a whole state of the module.

    Both parameters and persistent buffers (e.g. running averages) are
    included. Keys are corresponding parameter and buffer names.

    This method is modified from :meth:`torch.nn.Module.state_dict` to
    recursively check parallel module in case that the model has a complicated
    structure, e.g., nn.Module(nn.Module(DDP)).

    Args:
        module (nn.Module): The module to generate state_dict.
        destination (OrderedDict): Returned dict for the state of the
            module.
        prefix (str): Prefix of the key.
        keep_vars (bool): Whether to keep the variable property of the
            parameters. Default: False.

    Returns:
        dict: A dictionary containing a whole state of the module.
    """
    # recursively check parallel module in case that the model has a
    # complicated structure, e.g., nn.Module(nn.Module(DDP))
    if is_module_wrapper(module):
        module = cast(torch.nn.Module, module.module)

    # below is the same as torch.nn.Module.state_dict()
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()  # type: ignore
    destination._metadata[prefix[:-1]] = local_metadata = {  # type: ignore
        "version": module._version
    }
    _save_to_state_dict(module, destination, prefix, keep_vars)  # type: ignore
    for name, child in module._modules.items():
        if child is not None:
            get_state_dict(child, destination, prefix + name + ".", keep_vars=keep_vars)
    for hook in module._state_dict_hooks.values():
        hook_result = hook(module, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result
    return destination  # type: ignore


def save_checkpoint(
    model: torch.nn.Module,
    filename: str,
    optimizer: Optimizer | dict[str, Optimizer] | None = None,
    meta: dict | None = None,
    backend_args: dict | None = None,
) -> None:
    """Save checkpoint to file.

    The checkpoint will have 3 fields: ``meta``, ``state_dict`` and
    ``optimizer``. By default ``meta`` will contain version and time info.

    Args:
        model (Module): Module whose params are to be saved.
        filename (str): Checkpoint filename.
        optimizer (:obj:`Optimizer`, optional): Optimizer to be saved.
        meta (dict, optional): Metadata to be saved in checkpoint.
        backend_args (dict, optional): Storage backend selection arguments.
            Default: None.
            `New in version 1.3.16.`
    """
    if meta is None:
        meta = {}
    meta.update(mmcv2_version=__version__, time=time.asctime())

    if is_module_wrapper(model):
        model = cast(torch.nn.Module, model.module)

    if hasattr(model, "CLASSES") and model.CLASSES is not None:
        # save class name to the meta
        meta.update(CLASSES=model.CLASSES)

    checkpoint = {
        "meta": meta,
        "state_dict": weights_to_cpu(get_state_dict(model)),  # type: ignore
    }
    # save optimizer state dict in the checkpoint
    if isinstance(optimizer, Optimizer):
        checkpoint["optimizer"] = optimizer.state_dict()
    elif isinstance(optimizer, dict):
        checkpoint["optimizer"] = {}
        for name, optim in optimizer.items():
            checkpoint["optimizer"][name] = optim.state_dict()

    with io.BytesIO() as f:
        torch.save(checkpoint, f)
        put(f.getvalue(), filename, backend_args=backend_args)
