from torch import nn

from .registry import MODULE_WRAPPERS


def is_module_wrapper(module: nn.Module) -> bool:
    """Check if a module is a module wrapper.

    Native DataParallel and DistributedDataParallel, plus registered MMCV2
    wrappers such as MMDistributedDataParallel, are regarded as wrappers.
    Downstream projects may register their own wrapper classes in
    ``mmcv2.parallel.MODULE_WRAPPERS`` or one of its child registries.

    Args:
        module (nn.Module): The module to be checked.

    Returns:
        bool: True if the input module is a module wrapper.
    """

    def is_module_in_wrapper(module, module_wrapper):
        module_wrappers = tuple(module_wrapper.module_dict.values())
        if isinstance(module, module_wrappers):
            return True
        for child in module_wrapper.children.values():
            if is_module_in_wrapper(module, child):
                return True
        return False

    return is_module_in_wrapper(module, MODULE_WRAPPERS)
