import logging
import random
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from mmcv2.runner import (
    AmpOptimizerHook,
    CheckpointLoader,
    EpochBasedRunner,
    IterBasedRunner,
    OptimizerHook,
    StepLrUpdaterHook,
    save_checkpoint,
)
from mmcv2.runner.iter_based_runner import IterLoader
from torch import nn
from torch.utils.data import DataLoader


class StepModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, x, batch_idx):
        output = self(x)
        return {
            "loss": output.square().mean(),
            "num_samples": x.size(0),
            "output_dtype": output.dtype,
        }

    def validation_step(self, x, batch_idx):
        return self.training_step(x, batch_idx)


def test_bf16_forward_and_fp16_scaler_selection(tmp_path):
    model = StepModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=str(tmp_path),
        logger=logging.getLogger("modern-amp"),
        max_epochs=1,
        meta={},
    )
    bf16 = AmpOptimizerHook(dtype="bfloat16")
    bf16.before_run(runner)
    runner.run_iter(torch.ones(2, 2), train_mode=True)
    assert runner.outputs["output_dtype"] == torch.bfloat16
    assert bf16.loss_scaler is None

    fp16 = AmpOptimizerHook(dtype="float16")
    fp16.before_run(runner)
    assert runner.amp_dtype == torch.float16
    assert fp16.loss_scaler is not None


def test_accumulation_step_boundaries_and_set_to_none():
    model = StepModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.step = Mock(wraps=optimizer.step)
    runner = SimpleNamespace(
        model=model,
        optimizer=optimizer,
        max_iters=5,
        iter=0,
        outputs=None,
        logger=logging.getLogger("modern-optimizer"),
        log_buffer=Mock(),
        meta={},
    )
    hook = OptimizerHook(cumulative_iters=2)
    hook.before_run(runner)
    for index in range(5):
        runner.iter = index
        runner.outputs = model.training_step(torch.ones(2, 2), index)
        hook.after_train_iter(runner)
        assert runner._optimizer_step_boundary == (index in (1, 3, 4))
    assert optimizer.step.call_count == 3
    assert all(param.grad is None for param in model.parameters())


def test_runner_wraps_accumulation_forward_and_backward_in_no_sync(tmp_path):
    class SyncTrackingModel(StepModel):
        def __init__(self):
            super().__init__()
            self.sync_disabled = False
            self.forward_sync_states = []
            self.backward_sync_states = []
            self.linear.weight.register_hook(lambda grad: self.backward_sync_states.append(self.sync_disabled) or grad)

        @contextmanager
        def no_sync(self):
            self.sync_disabled = True
            try:
                yield
            finally:
                self.sync_disabled = False

        def training_step(self, x, batch_idx):
            self.forward_sync_states.append(self.sync_disabled)
            return super().training_step(x, batch_idx)

    model = SyncTrackingModel()
    runner = EpochBasedRunner(
        model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        work_dir=str(tmp_path),
        logger=logging.getLogger("modern-no-sync"),
        max_epochs=1,
        meta={},
    )
    runner.register_hook(OptimizerHook(cumulative_iters=2))
    runner.run([DataLoader(torch.ones(4, 2), batch_size=1)], [("train", 1)])
    assert model.forward_sync_states == [True, False, True, False]
    assert model.backward_sync_states == [True, False, True, False]


def test_runner_compile_config_targets_underlying_module(tmp_path):
    model = StepModel()
    model.compile = Mock()
    EpochBasedRunner(
        model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        work_dir=str(tmp_path),
        logger=logging.getLogger("modern-compile"),
        max_epochs=1,
        meta={},
        compile_cfg={"backend": "eager", "dynamic": True},
    )
    model.compile.assert_called_once_with(backend="eager", dynamic=True)


def test_runner_compiled_model_executes_training_step(tmp_path):
    model = StepModel()
    runner = EpochBasedRunner(
        model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        work_dir=str(tmp_path),
        logger=logging.getLogger("modern-compile-runtime"),
        max_epochs=1,
        meta={},
        compile_cfg={"backend": "eager"},
    )
    runner.run_iter(torch.ones(2, 2), train_mode=True)
    assert runner.outputs["loss"].isfinite()


def test_checkpoint_restricted_by_default_with_explicit_trusted_opt_out(tmp_path):
    model = StepModel()
    normal = tmp_path / "normal.pth"
    save_checkpoint(
        model,
        str(normal),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        meta={"epoch": 1, "iter": 2},
    )
    checkpoint = CheckpointLoader.load_checkpoint(str(normal))
    assert "state_dict" in checkpoint and "optimizer" in checkpoint
    mmap_checkpoint = CheckpointLoader.load_checkpoint(str(normal), mmap=True)
    assert "state_dict" in mmap_checkpoint

    custom = tmp_path / "custom.pth"
    torch.save({"state_dict": model.state_dict(), "custom": StepModel()}, custom)
    with pytest.raises(Exception):
        CheckpointLoader.load_checkpoint(str(custom))
    trusted = CheckpointLoader.load_checkpoint(str(custom), weights_only=False)
    assert isinstance(trusted["custom"], StepModel)


def test_native_step_lr_matches_old_curve_and_resume():
    model = StepModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    runner = SimpleNamespace(
        optimizer=optimizer,
        hooks=[OptimizerHook()],
        model=model,
        meta={},
        iter=0,
        epoch=0,
    )
    hook = StepLrUpdaterHook(step=2, gamma=0.1, by_epoch=False)
    hook.before_run(runner)
    observed = []
    for index in range(5):
        runner.iter = index
        hook.before_train_iter(runner)
        observed.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
    assert observed == pytest.approx([0.1, 0.1, 0.01, 0.01, 0.001])

    resumed_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    resumed_optimizer.load_state_dict(optimizer.state_dict())
    resumed = SimpleNamespace(
        optimizer=resumed_optimizer,
        hooks=[OptimizerHook()],
        model=model,
        meta=dict(runner.meta),
        iter=5,
        epoch=0,
    )
    resumed_hook = StepLrUpdaterHook(step=2, gamma=0.1, by_epoch=False)
    resumed_hook.before_run(resumed)
    resumed_hook.before_train_iter(resumed)
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(0.001)


def test_iter_checkpoint_marks_partial_accumulation_and_next_iter(tmp_path, caplog):
    model = StepModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    logger = logging.getLogger("modern-resume")
    runner = IterBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=str(tmp_path),
        logger=logger,
        max_iters=4,
        meta={},
    )
    runner._iter = 1
    runner._optimizer_step_boundary = False
    runner.save_checkpoint(str(tmp_path), create_symlink=False)
    path = tmp_path / "iter_2.pth"
    checkpoint = CheckpointLoader.load_checkpoint(str(path))
    assert checkpoint["meta"]["iter"] == 2
    assert checkpoint["meta"]["optimizer_step_boundary"] is False

    resumed_model = StepModel()
    resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=0.1)
    resumed = IterBasedRunner(
        resumed_model,
        optimizer=resumed_optimizer,
        work_dir=str(tmp_path),
        logger=logger,
        max_iters=4,
        meta={},
    )
    with caplog.at_level(logging.WARNING):
        resumed.resume(str(path), map_location="cpu")
    assert resumed.iter == 2
    assert "not numerically exact" in caplog.text


def test_runner_restores_python_numpy_and_torch_rng(tmp_path):
    model = StepModel()
    runner = EpochBasedRunner(
        model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        work_dir=str(tmp_path),
        logger=logging.getLogger("modern-rng"),
        max_epochs=1,
        meta={},
    )
    runner.meta["rng_states"] = [runner.capture_rng_state()]
    expected = (random.random(), np.random.rand(), torch.rand(1))
    runner.restore_rng_state()
    observed = (random.random(), np.random.rand(), torch.rand(1))
    assert observed[0] == expected[0]
    assert observed[1] == expected[1]
    assert torch.equal(observed[2], expected[2])


def test_iter_loader_restores_next_batch():
    loader = DataLoader(torch.arange(10), batch_size=2, shuffle=False)
    original = IterLoader(loader)
    next(original)
    next(original)
    state = original.state_dict()
    expected = next(original)

    restored = IterLoader(loader)
    restored.load_state_dict(state)
    assert torch.equal(next(restored), expected)
