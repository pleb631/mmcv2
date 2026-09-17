import os.path as osp
import platform
import shutil
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, cast

import torch
from torch.utils.data import DataLoader

from mmcv2.utils import is_list_of, symlink

from .base_runner import BaseRunner
from .builder import RUNNERS
from .checkpoint import save_checkpoint
from .utils import get_host_info


@RUNNERS.register_module()
class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def run_iter(self, data_batch: Any, train_mode: bool, **kwargs) -> None:
        """Run one training or validation batch."""
        with self.autocast_context() if train_mode else nullcontext():
            if train_mode:
                train_step = cast(Callable[..., dict[str, Any]], getattr(self.model, "train_step"))
                outputs = train_step(data_batch, self.optimizer, **kwargs)
            else:
                val_step = cast(Callable[..., dict[str, Any]], getattr(self.model, "val_step"))
                outputs = val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"model.train_step()" and "model.val_step()" must return a dict')
        if "log_vars" in outputs:
            self.log_buffer.update(outputs["log_vars"], outputs["num_samples"])
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        """Run one complete training epoch."""
        self.model.train()
        self.mode = "train"
        self.data_loader = data_loader
        if self._max_epochs is None:
            raise RuntimeError("max_epochs must be set before training")
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook("before_train_epoch")
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook("before_train_iter")
            with self.optim_context():
                self.run_iter(data_batch, train_mode=True, **kwargs)
                self.call_hook("after_train_iter")
            del self.data_batch
            self._iter += 1

        self.call_hook("after_train_epoch")
        self._epoch += 1

    @torch.inference_mode()
    def val(self, data_loader, **kwargs):
        """Run one complete validation epoch."""
        self.model.eval()
        self.mode = "val"
        self.data_loader = data_loader
        self.call_hook("before_val_epoch")
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook("before_val_iter")
            self.run_iter(data_batch, train_mode=False)
            self.call_hook("after_val_iter")
            del self.data_batch
        self.call_hook("after_val_epoch")

    def run(
        self,
        data_loaders: list[DataLoader],
        workflow: list[tuple[str, int]],
        **kwargs,
    ) -> None:
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        self._validate_workflow(workflow)
        assert self._max_epochs is not None, "max_epochs must be specified during instantiation"

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == "train":
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else "NONE"
        self.logger.info("Start running, host: %s, work_dir: %s", get_host_info(), work_dir)
        self.logger.info("Hooks will be executed in the following order:\n%s", self.get_hook_info())
        self.logger.info("workflow: %s, max: %d epochs", workflow, self._max_epochs)
        self.call_hook("before_run")
        self.restore_rng_state()

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if not hasattr(self, mode):
                    raise ValueError(f'runner has no method named "{mode}" to run an epoch')
                epoch_runner = getattr(self, mode)

                for _ in range(epochs):
                    if mode == "train" and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

        self.call_hook("after_run")

    def save_checkpoint(
        self,
        out_dir: str,
        filename_tmpl: str = "epoch_{}.pth",
        save_optimizer: bool = True,
        meta: dict | None = None,
        create_symlink: bool = True,
    ) -> None:
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = {}
        if self.meta is not None:
            meta.update(cast(dict[str, Any], self.meta))
        meta["optimizer_step_boundary"] = self._optimizer_step_boundary
        meta["rng_states"] = getattr(self, "_checkpoint_rng_states", None) or [self.capture_rng_state()]
        meta["message_hub"] = self.message_hub.state_dict()
        self._checkpoint_rng_states = None
        # Keep the new progress counters after meta.update(self.meta).
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, "latest.pth")
            if platform.system() != "Windows":
                symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)
