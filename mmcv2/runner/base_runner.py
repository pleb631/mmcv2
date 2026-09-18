import copy
import logging
import os.path as osp
import random
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from typing import Any, cast, no_type_check

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..distributed import get_dist_info
from ..logging import MessageHub
from ..parallel import is_module_wrapper
from ..utils import build_from_cfg, mkdir_or_exist
from .checkpoint import load_checkpoint
from .context import RunnerContext
from .hooks import HOOKS, Hook, ParamSchedulerHook
from .log_buffer import LogBuffer
from .priority import Priority, get_priority
from .utils import get_time_str


class BaseRunner(metaclass=ABCMeta):
    """The base class of Runner, a training helper for PyTorch.

    All subclasses should implement the following APIs:

    - ``run()``
    - ``train()``
    - ``val()``
    - ``save_checkpoint()``

    Args:
        model (:obj:`torch.nn.Module`): The model to be run.
        optimizer (dict or :obj:`torch.optim.Optimizer`): It can be either an
            optimizer (in most cases) or a dict of optimizers (in models that
            requires more than one optimizer, e.g., GAN).
        work_dir (str, optional): The working directory to save checkpoints
            and logs. Defaults to None.
        logger (:obj:`logging.Logger`): Logger used during training.
             Defaults to None. (The default value is just for backward
             compatibility)
        meta (dict | None): A dict records some import information such as
            environment info and seed, which will be logged in logger hook.
            Defaults to None.
        max_epochs (int, optional): Total training epochs.
        max_iters (int, optional): Total training iterations.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: dict | torch.optim.Optimizer | None = None,
        work_dir: str | None = None,
        logger: logging.Logger | None = None,
        meta: dict | None = None,
        max_iters: int | None = None,
        max_epochs: int | None = None,
        compile_cfg: bool | dict | None = None,
    ) -> None:
        step_model = model.module if is_module_wrapper(model) else model
        if not hasattr(step_model, "training_step"):
            raise TypeError("model must implement training_step()")

        # check the type of `optimizer`
        if isinstance(optimizer, dict):
            for name, optim in optimizer.items():
                if not isinstance(optim, Optimizer):
                    raise TypeError(
                        f'optimizer must be a dict of torch.optim.Optimizers, but optimizer["{name}"] is a {type(optim)}'
                    )
        # check the type of `logger`
        if not isinstance(logger, logging.Logger):
            raise TypeError(f"logger must be a logging.Logger object, but got {type(logger)}")

        # check the type of `meta`
        self.model = model
        self.compile_cfg = compile_cfg
        if compile_cfg is not None and compile_cfg is not False:
            compile_options = {} if compile_cfg is True else compile_cfg.copy()
            compile_target = model.module if is_module_wrapper(model) else model
            getattr(compile_target, "compile")(**compile_options)
        self.optimizer = optimizer
        self.logger = logger
        self.meta = meta
        # create work_dir
        if isinstance(work_dir, str):
            self.work_dir: str | None = osp.abspath(work_dir)
            mkdir_or_exist(self.work_dir)
        elif work_dir is None:
            self.work_dir = None

        # get model name from the model class
        if hasattr(self.model, "module"):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        rank, world_size = get_dist_info()
        self._rank: int = int(rank)
        self._world_size: int = int(world_size)
        self.timestamp = get_time_str()
        self.mode: str | None = None
        self._hooks: list[Hook] = []
        self._epoch: int = 0
        self._iter: int = 0
        self._inner_iter: int = 0
        self.amp_dtype = None
        self._optimizer_step_boundary = True
        self._rng_restored = False
        self.message_hub = MessageHub.get_instance(f"{self.__class__.__name__}-{id(self)}")

        if max_epochs is not None and max_iters is not None:
            raise ValueError("Only one of `max_epochs` or `max_iters` can be set.")

        self._max_epochs = max_epochs
        self._max_iters = max_iters
        # MessageHub owns runtime state; LogBuffer is used only by legacy hooks.
        self.log_buffer = LogBuffer(self.message_hub)
        self.ctx = RunnerContext(self)

    @property
    def model_name(self) -> str:
        """str: Name of the model, usually the module class name."""
        return self._model_name

    def autocast_context(self):
        """Autocast the training forward when an AMP optimizer hook opts in."""
        if self.amp_dtype is None:
            return nullcontext()
        device_type = next(self.model.parameters(), torch.empty(0)).device.type
        return torch.amp.autocast(device_type=device_type, dtype=self.amp_dtype)

    @contextmanager
    def optim_context(self):
        """Enter optimizer-provided contexts around forward and backward."""
        with ExitStack() as stack:
            for hook in self._hooks:
                context_factory = getattr(hook, "optim_context", None)
                if callable(context_factory):
                    stack.enter_context(cast(Any, context_factory(self)))
            yield

    @property
    def step_model(self) -> torch.nn.Module:
        """Return the task model below an optional parallel wrapper."""

        return cast(torch.nn.Module, self.model.module if is_module_wrapper(self.model) else self.model)

    def prepare_data_batch(self, data_batch: Any, dataloader_idx: int = 0) -> Any:
        """Move a batch through the model's Lightning-style transfer hook."""

        if is_module_wrapper(self.model):
            return data_batch
        transfer = getattr(self.step_model, "transfer_batch_to_device", None)
        if not callable(transfer):
            return data_batch
        device = next(self.step_model.parameters(), torch.empty(0)).device
        return transfer(data_batch, device, dataloader_idx)

    @staticmethod
    def _infer_batch_size(value: Any) -> int:
        if isinstance(value, torch.Tensor) and value.ndim:
            return int(value.shape[0])
        if isinstance(value, Mapping):
            for item in value.values():
                try:
                    return BaseRunner._infer_batch_size(item)
                except ValueError:
                    continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                try:
                    return BaseRunner._infer_batch_size(item)
                except ValueError:
                    continue
        raise ValueError("batch must contain a tensor with a batch dimension")

    def format_step_output(self, output: Any, data_batch: Any, *, training: bool) -> dict[str, Any]:
        """Normalize Lightning-style step results for MMCV optimizer hooks."""

        tensor_output = isinstance(output, torch.Tensor)
        if tensor_output:
            result: dict[str, Any] = {"loss": output}
        elif isinstance(output, Mapping):
            result = dict(output)
        elif training:
            raise TypeError("training_step must return a tensor or mapping")
        else:
            result = {"predictions": output}
        if training and "loss" not in result:
            raise KeyError("training_step mapping must contain 'loss'")
        if "num_samples" not in result:
            result["num_samples"] = self._infer_batch_size(data_batch)
        if "loss" in result:
            loss = result["loss"]
            if not isinstance(loss, torch.Tensor):
                raise TypeError("step output 'loss' must be a tensor")
            if training and tensor_output:
                log_vars: dict[str, Any] = {}
                log_vars.setdefault("loss", float(loss.detach()))
                result["log_vars"] = log_vars
        return result

    @staticmethod
    def _validate_workflow(workflow: list[tuple[str, int]]) -> None:
        """Reject workflows that cannot advance the training limit."""
        if not workflow or not any(phase == "train" for phase, _ in workflow):
            raise ValueError("workflow must include a train phase")
        for phase, count in workflow:
            if not isinstance(phase, str) or not isinstance(count, int) or count < 1:
                raise ValueError("workflow phases require a name and positive count")

    def capture_rng_state(self):
        """Capture Python, NumPy, and accelerator RNG states in metadata."""
        np_state = np.random.get_state(legacy=True)
        if not isinstance(np_state, tuple):
            raise RuntimeError("Expected NumPy legacy RNG state to be a tuple")
        accelerator = torch.accelerator.current_accelerator(check_available=True)
        accelerator_state = None
        if accelerator is not None:
            backend = getattr(torch, accelerator.type, None)
            get_all = getattr(backend, "get_rng_state_all", None)
            if not callable(get_all):
                raise RuntimeError(f"{accelerator.type} does not expose get_rng_state_all()")
            states = get_all()
            accelerator_state = {
                "type": accelerator.type,
                "states": states,
            }
        return {
            "python": random.getstate(),
            "numpy": (
                np_state[0],
                torch.as_tensor(np_state[1].astype(np.int64)),
                *np_state[2:],
            ),
            "torch": torch.get_rng_state(),
            "accelerator": accelerator_state,
        }

    def restore_rng_state(self):
        """Restore RNG states captured in checkpoint metadata once."""
        if self._rng_restored:
            return
        meta = self.meta
        if not isinstance(meta, dict):
            return
        states = meta.get("rng_states")
        if not states:
            return
        if len(states) != self.world_size:
            self.logger.warning(
                "Checkpoint RNG world size differs from the current world size; exact resume is not available."
            )
            return
        state = states[self.rank]
        if not isinstance(state, dict):
            raise TypeError("RNG state entries must be dictionaries")
        random.setstate(state["python"])
        name, values, position, has_gauss, cached = state["numpy"]
        np.random.set_state(
            (
                name,
                np.asarray(values.tolist(), dtype=np.uint32),
                position,
                has_gauss,
                cached,
            )
        )
        torch.set_rng_state(state["torch"].cpu())
        accelerator = torch.accelerator.current_accelerator(check_available=True)
        accelerator_state = state.get("accelerator")
        if accelerator is not None and accelerator_state and accelerator_state["type"] == accelerator.type:
            backend = getattr(torch, accelerator.type, None)
            states = [value.cpu() for value in accelerator_state["states"]]
            set_all = getattr(backend, "set_rng_state_all", None)
            set_one = getattr(backend, "set_rng_state", None)
            if callable(set_all):
                set_all(states)
            elif callable(set_one) and len(states) == 1:
                set_one(states[0])
        self._rng_restored = True

    @property
    def rank(self) -> int:
        """int: Rank of current process. (distributed training)"""
        return int(self._rank)

    @property
    def world_size(self) -> int:
        """int: Number of processes participating in the job.
        (distributed training)"""
        return int(self._world_size)

    @property
    def hooks(self) -> list[Hook]:
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self) -> int:
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self) -> int:
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self) -> int:
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters

    @abstractmethod
    def train(self):
        """Run one training stage."""
        pass

    @abstractmethod
    def val(self):
        """Run one validation stage."""
        pass

    @abstractmethod
    def run(self, data_loaders: list[DataLoader], workflow: list[tuple[str, int]], **kwargs) -> Any:
        """Execute a workflow consisting of named stages and repeat counts."""
        pass

    @abstractmethod
    def save_checkpoint(
        self,
        out_dir: str,
        filename_tmpl: str,
        save_optimizer: bool = True,
        meta: dict | None = None,
        create_symlink: bool = True,
    ) -> None:
        pass

    def current_lr(self) -> list[float] | dict[str, list[float]]:
        """Get current learning rates.

        Returns:
            list[float] | dict[str, list[float]]: Current learning rates of all
            param groups. If the runner has a dict of optimizers, this method
            will return a dict.
        """
        lr: list[float] | dict[str, list[float]]
        if isinstance(self.optimizer, torch.optim.Optimizer):
            lr = [float(group["lr"]) for group in self.optimizer.param_groups]
        elif isinstance(self.optimizer, dict):
            lr = {}
            for name, optim in self.optimizer.items():
                lr[name] = [float(group["lr"]) for group in optim.param_groups]
        else:
            raise RuntimeError("lr is not applicable because optimizer does not exist.")
        return lr

    def current_momentum(self) -> list[float] | dict[str, list[float]]:
        """Get current momentums.

        Returns:
            list[float] | dict[str, list[float]]: Current momentums of all
            param groups. If the runner has a dict of optimizers, this method
            will return a dict.
        """

        def _get_momentum(optimizer):
            momentums = []
            for group in optimizer.param_groups:
                if "momentum" in group:
                    momentums.append(group["momentum"])
                elif "betas" in group:
                    momentums.append(group["betas"][0])
                else:
                    momentums.append(0)
            return momentums

        if self.optimizer is None:
            raise RuntimeError("momentum is not applicable because optimizer does not exist.")
        elif isinstance(self.optimizer, torch.optim.Optimizer):
            momentums = _get_momentum(self.optimizer)
        elif isinstance(self.optimizer, dict):
            momentums = {}
            for name, optim in self.optimizer.items():
                momentums[name] = _get_momentum(optim)
        return momentums

    def register_hook(self, hook: Hook, priority: int | str | Priority = "NORMAL") -> None:
        """Register a hook into the hook list.

        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, "priority"):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority  # type: ignore
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:  # type: ignore
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def register_hook_from_cfg(self, hook_cfg: dict) -> None:
        """Register a hook from its cfg.

        Args:
            hook_cfg (dict): Hook config. It should have at least keys 'type'
              and 'priority' indicating its type and priority.

        Note:
            The specific hook class to register should not use 'type' and
            'priority' arguments during initialization.
        """
        hook_cfg = hook_cfg.copy()
        priority = hook_cfg.pop("priority", "NORMAL")
        hook = build_from_cfg(hook_cfg, HOOKS)
        self.register_hook(hook, priority=priority)

    def call_hook(self, fn_name: str) -> None:
        """Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
        """
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def get_hook_info(self) -> str:
        """Return a human-readable summary of registered hooks."""
        # Get hooks info in each stage
        stage_hook_map: dict[str, list] = {stage: [] for stage in Hook.stages}
        for hook in self.hooks:
            try:
                priority = Priority(hook.priority).name  # type: ignore
            except ValueError:
                priority = hook.priority  # type: ignore
            classname = hook.__class__.__name__
            hook_info = f"({priority:<12}) {classname:<35}"
            for trigger_stage in hook.get_triggered_stages():
                stage_hook_map[trigger_stage].append(hook_info)

        stage_hook_infos = []
        for stage in Hook.stages:
            hook_infos = stage_hook_map[stage]
            if len(hook_infos) > 0:
                info = f"{stage}:\n"
                info += "\n".join(hook_infos)
                info += "\n -------------------- "
                stage_hook_infos.append(info)
        return "\n".join(stage_hook_infos)

    def load_checkpoint(
        self,
        filename: str,
        map_location: str | Callable = "cpu",
        strict: bool = False,
        revise_keys: list | None = None,
        weights_only: bool = True,
        mmap: bool | None = None,
    ) -> dict | OrderedDict:
        """Load model parameters from a checkpoint.

        Args:
            filename: Checkpoint path or URI.
            map_location: Device mapping used during loading.
            strict: Whether to require an exact state-dict match.
            revise_keys: Regex substitutions applied to state-dict keys.
            weights_only: Whether to restrict deserialization to weights.
            mmap: Whether to memory-map the checkpoint when supported.

        Returns:
            The loaded checkpoint mapping.
        """
        if revise_keys is None:
            revise_keys = [(r"^module.", "")]
        return load_checkpoint(
            self.model,
            filename,
            map_location,
            strict,
            self.logger,
            revise_keys=revise_keys,
            weights_only=weights_only,
            mmap=mmap,
        )

    @no_type_check
    def resume(
        self,
        checkpoint: str,
        resume_optimizer: bool = True,
        map_location: str | Callable = "default",
        weights_only: bool = True,
        mmap: bool | None = None,
    ) -> None:
        """Restore model, runner, optimizer, and hook state.

        Args:
            checkpoint: Checkpoint path or URI.
            resume_optimizer: Whether to restore optimizer state.
            map_location: Device mapping, or ``"default"`` for the active
                accelerator.
            weights_only: Whether to restrict deserialization to weights.
            mmap: Whether to memory-map the checkpoint when supported.
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

        self._epoch = int(checkpoint["meta"]["epoch"])
        self._iter = int(checkpoint["meta"]["iter"])
        if self.meta is None:
            self.meta = {}
        self.meta.setdefault("hook_msgs", {})
        # load `last_ckpt`, `best_score`, `best_ckpt`, etc. for hook messages
        self.meta["hook_msgs"].update(checkpoint["meta"].get("hook_msgs", {}))

        # resume meta information meta
        self.meta = checkpoint["meta"]
        if "message_hub" in self.meta:
            self.message_hub.load_state_dict(self.meta["message_hub"])
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

        self.logger.info("resumed epoch %d, iter %d", self.epoch, self.iter)

    def register_lr_hook(self, lr_config: dict | Hook | None) -> None:
        """Create and register a learning-rate hook from its configuration."""
        if lr_config is None:
            return
        elif isinstance(lr_config, dict):
            assert "policy" in lr_config
            policy_type = lr_config.pop("policy")
            # If the type of policy is all in lower case, e.g., 'cyclic',
            # then its first letter will be capitalized, e.g., to be 'Cyclic'.
            # This is for the convenient usage of Lr updater.
            # Since this is not applicable for `
            # CosineAnnealingLrUpdater`,
            # the string will not be changed if it contains capital letters.
            if policy_type == policy_type.lower():
                policy_type = policy_type.title()
            hook_type = policy_type + "LrUpdaterHook"
            lr_config["type"] = hook_type
            hook = build_from_cfg(lr_config, HOOKS)
        else:
            hook = lr_config
        self.register_hook(hook, priority="VERY_HIGH")

    def register_momentum_hook(self, momentum_config: dict | Hook | None) -> None:
        """Create and register a momentum hook from its configuration."""
        if momentum_config is None:
            return
        if isinstance(momentum_config, dict):
            assert "policy" in momentum_config
            policy_type = momentum_config.pop("policy")
            # If the type of policy is all in lower case, e.g., 'cyclic',
            # then its first letter will be capitalized, e.g., to be 'Cyclic'.
            # This is for the convenient usage of momentum updater.
            # Since this is not applicable for
            # `CosineAnnealingMomentumUpdater`,
            # the string will not be changed if it contains capital letters.
            if policy_type == policy_type.lower():
                policy_type = policy_type.title()
            hook_type = policy_type + "MomentumUpdaterHook"
            momentum_config["type"] = hook_type
            hook = build_from_cfg(momentum_config, HOOKS)
        else:
            hook = momentum_config
        self.register_hook(hook, priority="HIGH")

    def register_optimizer_hook(self, optimizer_config: dict | Hook | None) -> None:
        """Create and register an optimizer hook from its configuration."""
        if optimizer_config is None:
            return
        if isinstance(optimizer_config, dict):
            optimizer_config.setdefault("type", "OptimizerHook")
            hook = build_from_cfg(optimizer_config, HOOKS)
        else:
            hook = optimizer_config
        self.register_hook(hook, priority="ABOVE_NORMAL")

    def register_param_scheduler_hook(self, schedulers, by_epoch: bool = True) -> None:
        """Register native PyTorch schedulers after optimizer updates.

        This is additive to the legacy ``lr_config`` updater API.  Callers
        should use one scheduling mechanism for a given optimizer.
        """
        if schedulers is None:
            return
        hook = (
            schedulers
            if isinstance(schedulers, ParamSchedulerHook)
            else ParamSchedulerHook(schedulers, by_epoch=by_epoch)
        )
        self.register_hook(hook, priority="LOW")

    def register_checkpoint_hook(self, checkpoint_config: dict | Hook | None) -> None:
        """Create and register a checkpoint hook from its configuration."""
        if checkpoint_config is None:
            return
        if isinstance(checkpoint_config, dict):
            checkpoint_config.setdefault("type", "CheckpointHook")
            hook = build_from_cfg(checkpoint_config, HOOKS)
        else:
            hook = checkpoint_config
        self.register_hook(hook, priority="NORMAL")

    def register_logger_hooks(self, log_config: dict | None) -> None:
        """Create and register configured logger hooks."""
        if log_config is None:
            return
        log_interval = log_config["interval"]
        for info in log_config["hooks"]:
            logger_hook = build_from_cfg(info, HOOKS, default_args={"interval": log_interval})
            self.register_hook(logger_hook, priority="VERY_LOW")

    def register_timer_hook(
        self,
        timer_config: dict | Hook | None,
    ) -> None:
        if timer_config is None:
            return
        if isinstance(timer_config, dict):
            timer_config_ = copy.deepcopy(timer_config)
            hook = build_from_cfg(timer_config_, HOOKS)
        else:
            hook = timer_config
        self.register_hook(hook, priority="LOW")

    def register_custom_hooks(self, custom_config: list | dict | Hook | None) -> None:
        """Create and register user-defined hooks."""
        if custom_config is None:
            return

        if not isinstance(custom_config, list):
            custom_config = [custom_config]

        for item in custom_config:
            if isinstance(item, dict):
                self.register_hook_from_cfg(item)
            else:
                self.register_hook(item, priority="NORMAL")

    def register_profiler_hook(
        self,
        profiler_config: dict | Hook | None,
    ) -> None:
        if profiler_config is None:
            return
        if isinstance(profiler_config, dict):
            profiler_config.setdefault("type", "ProfilerHook")
            hook = build_from_cfg(profiler_config, HOOKS)
        else:
            hook = profiler_config
        self.register_hook(hook)

    def register_training_hooks(
        self,
        lr_config: dict | Hook | None,
        optimizer_config: dict | Hook | None = None,
        checkpoint_config: dict | Hook | None = None,
        log_config: dict | None = None,
        momentum_config: dict | Hook | None = None,
        timer_config: dict | Hook | None = {"type": "IterTimerHook"},
        custom_hooks_config: list | dict | Hook | None = None,
        param_schedulers=None,
        param_scheduler_by_epoch: bool = True,
    ) -> None:
        """Register default and custom hooks for training.

        Default and custom hooks include:

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
        self.register_lr_hook(lr_config)
        self.register_momentum_hook(momentum_config)
        self.register_optimizer_hook(optimizer_config)
        self.register_param_scheduler_hook(param_schedulers, by_epoch=param_scheduler_by_epoch)
        self.register_checkpoint_hook(checkpoint_config)
        self.register_timer_hook(timer_config)
        self.register_logger_hooks(log_config)
        self.register_custom_hooks(custom_hooks_config)
