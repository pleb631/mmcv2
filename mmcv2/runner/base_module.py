import copy
from abc import ABCMeta
from collections import defaultdict
from collections.abc import Iterable
from logging import FileHandler

import torch
from torch import nn

from mmcv2.distributed import master_only
from mmcv2.utils.logging import get_logger, logger_initialized, print_log

from ..cnn import initialize
from ..cnn.utils.weight_init import update_init_info


class BaseModule(nn.Module, metaclass=ABCMeta):
    """Base module with configurable parameter initialization.

    ``BaseModule`` is a wrapper of ``torch.nn.Module`` with additional
    functionality of parameter initialization. Compared with
    ``torch.nn.Module``, ``BaseModule`` mainly adds three attributes.

    - ``init_cfg``: the config to control the initialization.
    - ``init_weights``: The function of parameter initialization and recording
      initialization information.
    - ``_params_init_info``: Used to track the parameter initialization
      information. This attribute only exists during executing the
      ``init_weights``.

    Args:
        init_cfg (dict, optional): Initialization config dict.
    """

    _params_init_info: defaultdict[nn.Parameter, dict[str, str | torch.Tensor]]

    def __init__(self, init_cfg: dict | None = None):
        """Initialize BaseModule, inherited from `torch.nn.Module`"""

        # NOTE init_cfg can be defined in different levels, but init_cfg
        # in low levels has a higher priority.

        super().__init__()
        # define default value of init_cfg instead of hard code
        # in init_weights() function
        self._is_init = False

        self.init_cfg = copy.deepcopy(init_cfg)

    @property
    def is_init(self) -> bool:
        return self._is_init

    def init_weights(self) -> None:
        """Initialize the weights."""

        is_top_level_module = False
        # check if it is top-level module
        if not hasattr(self, "_params_init_info"):
            # The `_params_init_info` is used to record the initialization
            # information of the parameters
            # the key should be the obj:`nn.Parameter` of model and the value
            # should be a dict containing
            # - init_info (str): The string that describes the initialization.
            # - tmp_mean_value (FloatTensor): The mean of the parameter,
            #       which indicates whether the parameter has been modified.
            # this attribute would be deleted after all parameters
            # is initialized.
            params_init_info: defaultdict[nn.Parameter, dict[str, str | torch.Tensor]] = defaultdict(dict)
            object.__setattr__(self, "_params_init_info", params_init_info)
            is_top_level_module = True

            # Initialize the `_params_init_info`,
            # When detecting the `tmp_mean_value` of
            # the corresponding parameter is changed, update related
            # initialization information
            for name, param in self.named_parameters():
                self._params_init_info[param]["init_info"] = (
                    f"The value is the same before and after calling `init_weights` of {self.__class__.__name__} "
                )
                self._params_init_info[param]["tmp_mean_value"] = param.detach().mean()

            # pass `params_init_info` to all submodules
            # All submodules share the same `params_init_info`,
            # so it will be updated when parameters are
            # modified at any level of the model.
            for sub_module in self.modules():
                object.__setattr__(sub_module, "_params_init_info", self._params_init_info)

        # Get the initialized logger, if not exist,
        # create a logger named `mmcv2`
        logger_names = list(logger_initialized.keys())
        logger_name = logger_names[0] if logger_names else "mmcv2"

        module_name = self.__class__.__name__
        if not self._is_init:
            if self.init_cfg:
                print_log(
                    f"initialize {module_name} with init_cfg {self.init_cfg}",
                    logger=logger_name,
                )
                initialize(self, self.init_cfg)
                if isinstance(self.init_cfg, dict):
                    # prevent the parameters of
                    # the pre-trained model
                    # from being overwritten by
                    # the `init_weights`
                    if self.init_cfg["type"] == "Pretrained":
                        return

            for m in self.children():
                init_weights = getattr(m, "init_weights", None)
                if callable(init_weights):
                    init_weights()
                    # users may overload the `init_weights`
                    update_init_info(
                        m,
                        init_info=f"Initialized by user-defined `init_weights` in {m.__class__.__name__} ",
                    )

            self._is_init = True
        if is_top_level_module:
            self._dump_init_info(logger_name)

            for sub_module in self.modules():
                del sub_module._params_init_info

    @master_only
    def _dump_init_info(self, logger_name: str) -> None:
        """Dump the initialization information to a file named
        `initialization.log.json` in workdir.

        Args:
            logger_name (str): The name of logger.
        """

        logger = get_logger(logger_name)

        with_file_handler = False
        # dump the information to the logger file if there is a `FileHandler`
        for handler in logger.handlers:
            if isinstance(handler, FileHandler):
                stream = handler.stream
                if stream is None:
                    continue
                stream.write("Name of parameter - Initialization information\n")
                for name, param in self.named_parameters():
                    stream.write(f"\n{name} - {param.shape}: \n{self._params_init_info[param]['init_info']} \n")
                stream.flush()
                with_file_handler = True
        if not with_file_handler:
            for name, param in self.named_parameters():
                print_log(
                    f"\n{name} - {param.shape}: \n{self._params_init_info[param]['init_info']} \n ",
                    logger=logger_name,
                )

    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
            s += f"\ninit_cfg={self.init_cfg}"
        return s


class Sequential(BaseModule, nn.Sequential):
    """Sequential module with configurable parameter initialization.

    Args:
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, *args, init_cfg: dict | None = None):
        BaseModule.__init__(self, init_cfg)
        nn.Sequential.__init__(self, *args)


class ModuleList(BaseModule, nn.ModuleList):
    """ModuleList with configurable parameter initialization.

    Args:
        modules (iterable, optional): an iterable of modules to add.
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, modules: Iterable | None = None, init_cfg: dict | None = None):
        BaseModule.__init__(self, init_cfg)
        nn.ModuleList.__init__(self, modules)


class ModuleDict(BaseModule, nn.ModuleDict):
    """ModuleDict with configurable parameter initialization.

    Args:
        modules (dict, optional): a mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module).
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, modules: dict | None = None, init_cfg: dict | None = None):
        BaseModule.__init__(self, init_cfg)
        nn.ModuleDict.__init__(self, modules)
