from collections import OrderedDict

import numpy as np
import torch

from mmcv2.logging import MessageHub


class LogBuffer:
    """Compatibility view over legacy log output and modern scalar history."""

    def __init__(self, message_hub: MessageHub | None = None):
        self.message_hub = message_hub
        self.val_history = OrderedDict()
        self.n_history = OrderedDict()
        self.output = OrderedDict()
        self.ready = False

    def clear(self) -> None:
        self.val_history.clear()
        self.n_history.clear()
        self.clear_output()

    def clear_output(self) -> None:
        self.output.clear()
        self.ready = False

    def update(self, vars: dict, count: int = 1) -> None:
        assert isinstance(vars, dict)
        for key, var in vars.items():
            if isinstance(var, torch.Tensor):
                if var.numel() != 1:
                    raise ValueError("Only scalar tensors can be logged")
                var = var.detach().item()
            elif isinstance(var, np.number):
                var = var.item()
            if key not in self.val_history:
                self.val_history[key] = []
                self.n_history[key] = []
            self.val_history[key].append(var)
            self.n_history[key].append(count)
            if self.message_hub is not None:
                self.message_hub.update_scalar(key, var, count)

    def average(self, n: int = 0) -> None:
        """Average latest n values or all values."""
        assert n >= 0
        for key in self.val_history:
            values = np.array(self.val_history[key][-n:])
            nums = np.array(self.n_history[key][-n:])
            avg = np.sum(values * nums) / np.sum(nums)
            self.output[key] = avg
        self.ready = True
