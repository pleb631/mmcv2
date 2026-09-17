import logging
from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_

from .hook import HOOKS, Hook


@HOOKS.register_module()
class OptimizerHook(Hook):
    def __init__(
        self,
        grad_clip: dict | None = None,
        detect_anomalous_params: bool = False,
        cumulative_iters: int = 1,
    ):
        assert isinstance(cumulative_iters, int) and cumulative_iters > 0, "cumulative_iters must be a positive integer"
        self.grad_clip = grad_clip
        self.detect_anomalous_params = detect_anomalous_params
        self.cumulative_iters = cumulative_iters
        self.initialized = False
        self.divisible_iters = 0
        self.remainder_iters = 0

    def before_run(self, runner) -> None:
        # Partial gradients are deliberately not part of the checkpoint format.
        runner.optimizer.zero_grad(set_to_none=True)
        runner._optimizer_step_boundary = True

    def clip_grads(self, params: Iterable[nn.Parameter]) -> Tensor | None:
        params = [p for p in params if p.requires_grad and p.grad is not None]
        if params:
            return clip_grad_norm_(params, **cast(dict[str, Any], self.grad_clip))
        return None

    def _loss_factor(self, runner) -> int:
        if self.cumulative_iters == 1:
            return 1
        remaining = runner.max_iters % self.cumulative_iters
        if remaining and runner.iter >= runner.max_iters - remaining:
            return remaining
        return self.cumulative_iters

    def _get_loss_factor(self, runner) -> int:
        return self._loss_factor(runner)

    def has_batch_norm(self, module: nn.Module) -> bool:
        return any(isinstance(child, nn.modules.batchnorm._BatchNorm) for child in module.modules())

    def _init(self, runner) -> None:
        if self.cumulative_iters == 1:
            self.initialized = True
            return
        self.divisible_iters = runner.max_iters // self.cumulative_iters * self.cumulative_iters
        self.remainder_iters = runner.max_iters - self.divisible_iters
        if runner.iter % self.cumulative_iters:
            runner.logger.warning("Resume inside a gradient accumulation window loses partial gradients.")
        if self.cumulative_iters > 1 and self.has_batch_norm(runner.model):
            runner.logger.warning("Accumulation with BatchNorm can differ from one larger batch.")
        self.initialized = True

    def _step_due(self, runner) -> bool:
        if self.cumulative_iters == 1:
            return True
        return (runner.iter + 1) % self.cumulative_iters == 0 or (runner.iter + 1 == runner.max_iters)

    def optim_context(self, runner):
        """Skip DDP synchronization on non-boundary accumulation steps."""
        no_sync = getattr(runner.model, "no_sync", None)
        if self.cumulative_iters > 1 and not self._step_due(runner) and callable(no_sync):
            return no_sync()
        return nullcontext()

    def _backward(self, loss: Tensor) -> None:
        loss.backward()

    def _before_step(self, runner) -> None:
        pass

    def _step(self, runner) -> None:
        runner.optimizer.step()

    def _after_step(self, runner) -> None:
        pass

    def after_train_iter(self, runner) -> None:
        if not self.initialized:
            self._init(runner)
        loss = runner.outputs["loss"] / self._loss_factor(runner)
        if self.detect_anomalous_params:
            self.detect_anomalous_parameters(loss, runner)
        self._backward(loss)
        due = self._step_due(runner)
        runner._optimizer_step_boundary = due
        if not due:
            return
        self._before_step(runner)
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                runner.log_buffer.update({"grad_norm": float(grad_norm)}, runner.outputs["num_samples"])
        self._step(runner)
        self._after_step(runner)
        runner.optimizer.zero_grad(set_to_none=True)

    def detect_anomalous_parameters(self, loss: Tensor, runner) -> None:
        parameters_in_graph = set()
        visited = set()

        def traverse(grad_fn):
            if grad_fn is None or grad_fn in visited:
                return
            visited.add(grad_fn)
            if hasattr(grad_fn, "variable"):
                parameters_in_graph.add(grad_fn.variable)
            for parent, _ in grad_fn.next_functions:
                traverse(parent)

        traverse(loss.grad_fn)
        for name, param in runner.model.named_parameters():
            if param.requires_grad and param not in parameters_in_graph:
                runner.logger.log(
                    level=logging.ERROR,
                    msg=f"{name} with shape {param.size()} is not in the graph",
                )


@HOOKS.register_module()
class GradientCumulativeOptimizerHook(OptimizerHook):
    """Compatibility name for the unified accumulation implementation."""

    def __init__(self, cumulative_iters: int = 1, **kwargs):
        super().__init__(cumulative_iters=cumulative_iters, **kwargs)


@HOOKS.register_module()
class AmpOptimizerHook(OptimizerHook):
    """Native autocast; loss scaling is used only with explicit float16."""

    def __init__(
        self,
        dtype: str = "bfloat16",
        loss_scale: float | str | dict = "dynamic",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if dtype not in ("bfloat16", "float16"):
            raise ValueError("dtype must be bfloat16 or float16")
        if dtype == "bfloat16" and loss_scale != "dynamic":
            raise ValueError("BF16 does not use loss scaling")
        self.dtype = dtype
        self.loss_scale = loss_scale
        self.loss_scaler = None
        self._fixed_scale = None

    def before_run(self, runner) -> None:
        super().before_run(runner)
        runner.amp_dtype = torch.bfloat16 if self.dtype == "bfloat16" else torch.float16
        if self.dtype == "float16":
            device_type = next(runner.model.parameters(), torch.empty(0)).device.type
            if isinstance(self.loss_scale, dict):
                self.loss_scaler = torch.amp.GradScaler(device_type, **self.loss_scale)
            else:
                init_scale = float(self.loss_scale) if self.loss_scale != "dynamic" else 65536.0
                self.loss_scaler = torch.amp.GradScaler(device_type, init_scale=init_scale)
                if self.loss_scale != "dynamic":
                    self._fixed_scale = init_scale
            meta = runner.meta or {}
            scaler_state = meta.get("fp16", {}).get("loss_scaler")
            if scaler_state:
                self.loss_scaler.load_state_dict(scaler_state)

    def _backward(self, loss: Tensor) -> None:
        if self.loss_scaler is None:
            loss.backward()
        else:
            self.loss_scaler.scale(loss).backward()

    def _before_step(self, runner) -> None:
        if self.loss_scaler is not None:
            self.loss_scaler.unscale_(runner.optimizer)

    def _step(self, runner) -> None:
        if self.loss_scaler is None:
            runner.optimizer.step()
        else:
            self.loss_scaler.step(runner.optimizer)

    def _after_step(self, runner) -> None:
        if self.loss_scaler is not None:
            self.loss_scaler.update(self._fixed_scale)
            if runner.meta is None:
                runner.meta = {}
            runner.meta.setdefault("fp16", {})["loss_scaler"] = self.loss_scaler.state_dict()
