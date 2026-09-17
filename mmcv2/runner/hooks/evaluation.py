from __future__ import annotations

import os.path as osp
import warnings
from collections.abc import Sequence, Sized
from math import inf
from typing import Any, cast

import torch
import torch.distributed as dist
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader

from mmcv2.evaluator import Evaluator, get_metric_value
from mmcv2.fileio import get_file_backend

from .hook import Hook
from .logger import LoggerHook


class EvalHook(Hook):
    """Run metric-based validation at a configured interval.

    Models must implement ``test_step(data_batch)`` and return a sequence of
    predictions/data samples. Metrics receive each batch incrementally through
    :class:`mmcv2.evaluator.Evaluator`; datasets do not own evaluation logic.
    """

    rule_map = {"greater": lambda x, y: x > y, "less": lambda x, y: x < y}
    init_value_map = {"greater": -inf, "less": inf}
    _default_greater_keys = [
        "acc",
        "top",
        "AR@",
        "auc",
        "precision",
        "mAP",
        "mDice",
        "mIoU",
        "mAcc",
        "aAcc",
    ]
    _default_less_keys = ["loss"]

    def __init__(
        self,
        dataloader: DataLoader,
        evaluator: Evaluator,
        start: int | None = None,
        interval: int = 1,
        by_epoch: bool = True,
        save_best: str | None = None,
        rule: str | None = None,
        greater_keys: list[str] | None = None,
        less_keys: list[str] | None = None,
        out_dir: str | None = None,
        backend_args: dict | None = None,
    ) -> None:
        if interval <= 0:
            raise ValueError(f"interval must be positive, but got {interval}")
        if start is not None and start < 0:
            raise ValueError(f"start must be non-negative, but got {start}")

        self.dataloader = dataloader
        self.evaluator = evaluator
        dataset_meta = getattr(dataloader.dataset, "metainfo", None)
        if dataset_meta is None:
            dataset_meta = getattr(dataloader.dataset, "METAINFO", None)
        self.evaluator.dataset_meta = dataset_meta
        self.interval = interval
        self.start = start
        self.by_epoch = by_epoch
        self.save_best = save_best
        self.initial_flag = True
        self.greater_keys = greater_keys or self._default_greater_keys
        self.less_keys = less_keys or self._default_less_keys
        self.out_dir = out_dir
        self.backend_args = backend_args

        if save_best is not None:
            self.best_ckpt_path: str | None = None
            self._init_rule(rule, save_best)

    def _init_rule(self, rule: str | None, key_indicator: str) -> None:
        if rule not in self.rule_map and rule is not None:
            raise KeyError(f"rule must be greater, less or None, but got {rule}")
        if rule is None and key_indicator != "auto":
            indicator = key_indicator.lower()
            greater = [key.lower() for key in self.greater_keys]
            less = [key.lower() for key in self.less_keys]
            if indicator in greater or any(key in indicator for key in greater):
                rule = "greater"
            elif indicator in less or any(key in indicator for key in less):
                rule = "less"
            else:
                raise ValueError(f"Cannot infer the comparison rule for {key_indicator!r}")
        self.rule = rule
        self.key_indicator = key_indicator
        if rule is not None:
            self.compare_func = self.rule_map[rule]

    def before_run(self, runner) -> None:
        if not self.out_dir:
            self.out_dir = runner.work_dir
        self.backend = get_file_backend(self.out_dir, backend_args=self.backend_args)
        if self.out_dir != runner.work_dir:
            basename = osp.basename(runner.work_dir.rstrip(osp.sep))
            self.out_dir = self.backend.join_path(self.out_dir, basename)
            runner.logger.info(f"Best checkpoints will be saved to {self.out_dir}")
        if self.save_best is not None:
            if runner.meta is None:
                runner.meta = {}
            runner.meta.setdefault("hook_msgs", {})
            self.best_ckpt_path = runner.meta["hook_msgs"].get("best_ckpt")

    def before_train_iter(self, runner) -> None:
        if not self.by_epoch and self.initial_flag:
            if self.start is not None and runner.iter >= self.start:
                self.after_train_iter(runner)
            self.initial_flag = False

    def before_train_epoch(self, runner) -> None:
        if self.by_epoch and self.initial_flag:
            if self.start is not None and runner.epoch >= self.start:
                self.after_train_epoch(runner)
            self.initial_flag = False

    def after_train_iter(self, runner) -> None:
        if not self.by_epoch and self._should_evaluate(runner):
            for hook in runner._hooks:
                if isinstance(hook, LoggerHook):
                    hook.after_train_iter(runner)
            runner.log_buffer.clear()
            self._do_evaluate(runner)

    def after_train_epoch(self, runner) -> None:
        if self.by_epoch and self._should_evaluate(runner):
            self._do_evaluate(runner)

    def _do_evaluate(self, runner) -> None:
        model = runner.model
        model.eval()
        test_step = getattr(model, "test_step", None)
        if not callable(test_step):
            raise TypeError("model must implement test_step(data_batch)")
        with torch.inference_mode():
            for data_batch in self.dataloader:
                data_samples = test_step(data_batch)
                if not isinstance(data_samples, Sequence) or isinstance(data_samples, (str, bytes, dict)):
                    raise TypeError("test_step must return a sequence of data samples")
                self.evaluator.process(data_samples, data_batch)
        dataset = cast(Sized, self.dataloader.dataset)
        metrics = self.evaluator.evaluate(len(dataset))
        runner.log_buffer.output["eval_iter_num"] = len(self.dataloader)
        key_score = self._record_metrics(runner, metrics)
        if self.save_best and key_score is not None:
            self._save_ckpt(runner, key_score)

    def _record_metrics(self, runner, metrics: dict[str, Any]) -> Any:
        runner.log_buffer.output.update(metrics)
        runner.message_hub.update_scalars(metrics)
        runner.log_buffer.ready = True
        if self.save_best is None:
            return None
        if not metrics:
            warnings.warn("Metrics are empty; best-checkpoint saving is skipped")
            return None
        if self.key_indicator == "auto":
            self._init_rule(self.rule, next(iter(metrics)))
        return get_metric_value(cast(str, self.key_indicator), metrics)

    def _should_evaluate(self, runner) -> bool:
        current = runner.epoch if self.by_epoch else runner.iter
        check_time = self.every_n_epochs if self.by_epoch else self.every_n_iters
        if self.start is None:
            return check_time(runner, self.interval)
        return current + 1 >= self.start and (current + 1 - self.start) % self.interval == 0

    def _save_ckpt(self, runner, key_score: float) -> None:
        if self.by_epoch:
            current = f"epoch_{runner.epoch + 1}"
            progress_name, progress = "epoch", runner.epoch + 1
        else:
            current = f"iter_{runner.iter + 1}"
            progress_name, progress = "iter", runner.iter + 1
        hook_msgs = cast(dict[str, Any], runner.meta["hook_msgs"])
        best_score = hook_msgs.get("best_score", self.init_value_map[cast(str, self.rule)])
        if not self.compare_func(key_score, best_score):
            return
        hook_msgs["best_score"] = key_score
        if self.best_ckpt_path and self.backend.isfile(self.best_ckpt_path):
            self.backend.remove(self.best_ckpt_path)
        indicator = str(self.key_indicator).replace("/", "_")
        filename = f"best_{indicator}_{current}.pth"
        self.best_ckpt_path = self.backend.join_path(cast(str, self.out_dir), filename)
        hook_msgs["best_ckpt"] = self.best_ckpt_path
        runner.save_checkpoint(cast(str, self.out_dir), filename_tmpl=filename, create_symlink=False)
        runner.logger.info(f"Best {self.key_indicator} is {key_score:.4f} at {progress_name} {progress}")


class DistEvalHook(EvalHook):
    """Metric-based distributed evaluation with synchronized BN buffers."""

    def __init__(self, *args, broadcast_bn_buffer: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.broadcast_bn_buffer = broadcast_bn_buffer

    def _do_evaluate(self, runner) -> None:
        if self.broadcast_bn_buffer and dist.is_available() and dist.is_initialized():
            for module in runner.model.modules():
                if isinstance(module, _BatchNorm) and module.track_running_stats:
                    dist.broadcast(cast(torch.Tensor, module.running_var), 0)
                    dist.broadcast(cast(torch.Tensor, module.running_mean), 0)
        super()._do_evaluate(runner)
