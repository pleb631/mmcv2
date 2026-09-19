import os.path as osp
import platform
import shutil
from collections.abc import Callable
from typing import Any, cast, no_type_check

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from mmcv2.utils import is_list_of, symlink

from .base_runner import BaseRunner
from .builder import RUNNERS
from .checkpoint import save_checkpoint
from .context import activate_runner_context
from .hooks import IterTimerHook
from .utils import get_host_info


class IterLoader:
    """Wrap a data loader as an indefinitely cycling iterator."""

    def __init__(self, dataloader: DataLoader):
        """Create an iterator wrapper for ``dataloader``."""
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._epoch = 0
        self._position = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def __next__(self) -> Any:
        try:
            data = next(self.iter_loader)
            self._position += 1
        except StopIteration:
            self._epoch += 1
            set_epoch = getattr(self._dataloader.sampler, "set_epoch", None)
            if callable(set_epoch):
                cast(Callable[[int], None], set_epoch)(self._epoch)
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)
            self._position = 1

        return data

    def state_dict(self) -> dict[str, int]:
        """Return the loader epoch and position for checkpointing."""
        return {"epoch": self._epoch, "position": self._position}

    def load_state_dict(self, state: dict[str, int]) -> None:
        """Restore the loader position from checkpoint state."""
        self._epoch = state["epoch"]
        self._position = 0
        set_epoch = getattr(self._dataloader.sampler, "set_epoch", None)
        if callable(set_epoch):
            cast(Callable[[int], None], set_epoch)(self._epoch)
        self.iter_loader = iter(self._dataloader)
        for _ in range(state["position"]):
            next(self.iter_loader)
            self._position += 1

    def __len__(self) -> int:
        return len(self._dataloader)


@RUNNERS.register_module()
class IterBasedRunner(BaseRunner):
    """Iteration-based Runner.

    This runner train models iteration by iteration.
    """

    def train(self, data_loader: IterLoader, **kwargs: Any) -> None:
        """Run the configured number of training iterations."""
        self.model.train()
        self.mode = "train"
        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        data_batch = next(data_loader)
        self.data_batch = data_batch
        self.call_hook("before_train_iter")
        with self.optim_context():
            data_batch = self.prepare_data_batch(data_batch, training=True)
            with self.autocast_context():
                training_step = getattr(self.model, "training_step")
                step_output = training_step(data_batch, self.inner_iter)
            outputs = self.format_step_output(step_output, data_batch, training=True)
            if "log_vars" in outputs:
                self.log_buffer.update(outputs["log_vars"], outputs["num_samples"])
            self.outputs = outputs
            self.call_hook("after_train_iter")
        del self.data_batch
        self._inner_iter += 1
        self._iter += 1

    @torch.inference_mode()
    def val(self, data_loader: IterLoader, **kwargs: Any) -> None:
        """Run validation iterations without tracking gradients."""
        self.model.eval()
        self.mode = "val"
        self.data_loader = data_loader
        data_batch = next(data_loader)
        self.data_batch = data_batch
        self.call_hook("before_val_iter")
        data_batch = self.prepare_data_batch(data_batch, training=False)
        validation_step = getattr(self.model, "validation_step")
        step_output = validation_step(data_batch, self.inner_iter)
        outputs = self.format_step_output(step_output, data_batch, training=False)
        if "log_vars" in outputs:
            self.log_buffer.update(outputs["log_vars"], outputs["num_samples"])
        self.outputs = outputs
        self.call_hook("after_val_iter")
        del self.data_batch
        self._inner_iter += 1

    @activate_runner_context
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
            workflow (list[tuple]): A list of (phase, iters) to specify the
                running order and iterations. E.g, [('train', 10000),
                ('val', 1000)] means running 10000 iterations for training and
                1000 iterations for validation, iteratively.
        """
        assert isinstance(data_loaders, list)
        assert is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        self._validate_workflow(workflow)
        assert self._max_iters is not None, "max_iters must be specified during instantiation"

        work_dir = self.work_dir if self.work_dir is not None else "NONE"
        self.logger.info("Start running, host: %s, work_dir: %s", get_host_info(), work_dir)
        self.logger.info("Hooks will be executed in the following order:\n%s", self.get_hook_info())
        self.logger.info("workflow: %s, max: %d iters", workflow, self._max_iters)
        self.call_hook("before_run")

        iter_loaders = [IterLoader(x) for x in data_loaders]
        self._iter_loaders = iter_loaders
        loader_states = getattr(self, "_resume_loader_states", None)
        if loader_states is not None:
            if len(loader_states) != len(iter_loaders):
                raise ValueError("Checkpoint dataloader count differs from the current workflow")
            for loader, state in zip(iter_loaders, loader_states):
                loader.load_state_dict(state)
            self._resume_loader_states = None
        elif self.iter > 0 and not self._rng_restored:
            self.logger.warning("Checkpoint has no dataloader cursor; exact iteration-based resume is not available.")
        self.restore_rng_state()

        self.call_hook("before_epoch")

        resume_cursor = getattr(self, "_resume_workflow_cursor", None)
        if resume_cursor is not None:
            flow_index, flow_position = resume_cursor
            if (
                flow_index < 0
                or flow_index >= len(workflow)
                or flow_position < 0
                or flow_position > workflow[flow_index][1]
            ):
                raise ValueError("Checkpoint workflow cursor is invalid for the current workflow")
            self._resume_workflow_cursor = None
        else:
            flow_index, flow_position = 0, 0
            if self.iter > 0 and len(workflow) > 1:
                self.logger.warning("Checkpoint has no workflow cursor; validation scheduling may change.")

        while self.iter < self._max_iters:
            for i, flow in enumerate(workflow):
                if i < flow_index:
                    continue
                self._inner_iter = 0
                mode, iters = flow
                if not isinstance(mode, str) or not hasattr(self, mode):
                    raise ValueError(f'runner has no method named "{mode}" to run a workflow')
                iter_runner = getattr(self, mode)
                for position in range(flow_position if i == flow_index else 0, iters):
                    if mode == "train" and self.iter >= self._max_iters:
                        break
                    # The checkpoint hook runs inside iter_runner, before
                    # its progress counter changes. Save the next flow slot.
                    self._workflow_cursor = (i, position + 1)
                    iter_runner(iter_loaders[i], **kwargs)
            flow_index, flow_position = 0, 0

        self.call_hook("after_epoch")
        self.call_hook("after_run")

    @no_type_check
    def resume(
        self,
        checkpoint: str,
        resume_optimizer: bool = True,
        map_location: str | Callable = "default",
        weights_only: bool = True,
        mmap: bool | None = None,
    ) -> None:
        """Resume model from checkpoint.

        Args:
            checkpoint (str): Checkpoint to resume from.
            resume_optimizer (bool, optional): Whether resume the optimizer(s)
                if the checkpoint file includes optimizer(s). Default to True.
            map_location (str, optional): Same as :func:`torch.load`.
                Default to 'default'.
        """
        if map_location == "default":
            accelerator = torch.accelerator.current_accelerator(check_available=True)
            if accelerator is not None:
                device = torch.device(accelerator.type, torch.accelerator.current_device_index())
                checkpoint = self.load_checkpoint(
                    checkpoint,
                    map_location=device,
                    weights_only=weights_only,
                    mmap=mmap,
                )
            else:
                checkpoint = self.load_checkpoint(checkpoint, weights_only=weights_only, mmap=mmap)
        else:
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=map_location,
                weights_only=weights_only,
                mmap=mmap,
            )

        self._epoch = checkpoint["meta"]["epoch"]
        self._iter = checkpoint["meta"]["iter"]
        self._inner_iter = checkpoint["meta"]["iter"]
        self.meta = checkpoint["meta"]
        if "message_hub" in self.meta:
            self.message_hub.load_state_dict(self.meta["message_hub"])
        self._resume_loader_states = self.meta.get("iter_loaders")
        self._resume_workflow_cursor = self.meta.get("workflow_cursor")
        if not self.meta.get("optimizer_step_boundary", True):
            self.logger.warning(
                "Checkpoint was saved inside a gradient accumulation window; resume is not numerically exact."
            )
        if "optimizer" in checkpoint and resume_optimizer:
            if isinstance(self.optimizer, Optimizer):
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            elif isinstance(self.optimizer, dict):
                for k in self.optimizer:
                    self.optimizer[k].load_state_dict(checkpoint["optimizer"][k])
            else:
                raise TypeError(f"Optimizer should be dict or torch.optim.Optimizer but got {type(self.optimizer)}")

        self.logger.info(f"resumed from epoch: {self.epoch}, iter {self.iter}")

    def save_checkpoint(  # type: ignore
        self,
        out_dir: str,
        filename_tmpl: str = "iter_{}.pth",
        meta: dict | None = None,
        save_optimizer: bool = True,
        create_symlink: bool = True,
    ) -> None:
        """Save checkpoint to file.

        Args:
            out_dir (str): Directory to save checkpoint files.
            filename_tmpl (str, optional): Checkpoint file template.
                Defaults to 'iter_{}.pth'.
            meta (dict, optional): Metadata to be saved in checkpoint.
                Defaults to None.
            save_optimizer (bool, optional): Whether save optimizer.
                Defaults to True.
            create_symlink (bool, optional): Whether create symlink to the
                latest checkpoint file. Defaults to True.
        """
        if meta is None:
            meta = {}
        if self.meta is not None:
            meta.update(cast(dict[str, Any], self.meta))
        meta["optimizer_step_boundary"] = self._optimizer_step_boundary
        meta["rng_states"] = getattr(self, "_checkpoint_rng_states", None) or [self.capture_rng_state()]
        meta["message_hub"] = self.message_hub.state_dict()
        self._checkpoint_rng_states = None
        if hasattr(self, "_iter_loaders"):
            meta["iter_loaders"] = [loader.state_dict() for loader in self._iter_loaders]
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
        if hasattr(self, "_workflow_cursor"):
            meta["workflow_cursor"] = self._workflow_cursor
        # CheckpointHook runs before the iteration counter is incremented.
        # Resume at the next microbatch, matching the iter_N filename.
        meta.update(epoch=self.epoch + 1, iter=self.iter + 1)

        filename = filename_tmpl.format(self.iter + 1)
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

    def register_training_hooks(
        self,
        lr_config,
        optimizer_config=None,
        checkpoint_config=None,
        log_config=None,
        momentum_config=None,
        custom_hooks_config=None,
    ):
        """Register default hooks for iter-based training.

        Checkpoint hook, optimizer stepper hook and logger hooks will be set to
        `by_epoch=False` by default.

        Default hooks include:

        +----------------------+-------------------------+
        | Hooks                | Priority                |
        +======================+=========================+
        | LrUpdaterHook        | VERY_HIGH (10)          |
        +----------------------+-------------------------+
        | MomentumUpdaterHook  | HIGH (30)               |
        +----------------------+-------------------------+
        | OptimizerStepperHook | ABOVE_NORMAL (40)       |
        +----------------------+-------------------------+
        | CheckpointSaverHook  | NORMAL (50)             |
        +----------------------+-------------------------+
        | IterTimerHook        | LOW (70)                |
        +----------------------+-------------------------+
        | LoggerHook(s)        | VERY_LOW (90)           |
        +----------------------+-------------------------+
        | CustomHook(s)        | defaults to NORMAL (50) |
        +----------------------+-------------------------+

        If custom hooks have same priority with default hooks, custom hooks
        will be triggered after default hooks.
        """
        if checkpoint_config is not None:
            checkpoint_config.setdefault("by_epoch", False)  # type: ignore
        if lr_config is not None:
            lr_config.setdefault("by_epoch", False)  # type: ignore
        if log_config is not None:
            for info in log_config["hooks"]:
                info.setdefault("by_epoch", False)
        super().register_training_hooks(
            lr_config=lr_config,
            momentum_config=momentum_config,
            optimizer_config=optimizer_config,
            checkpoint_config=checkpoint_config,
            log_config=log_config,
            timer_config=IterTimerHook(),
            custom_hooks_config=custom_hooks_config,
        )
