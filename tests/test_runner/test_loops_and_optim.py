from types import SimpleNamespace

import pytest
import torch
from mmcv2.runner import ParamSchedulerHook
from torch import nn
from torch.optim.lr_scheduler import StepLR


def test_param_scheduler_steps_only_after_optimizer_update():
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    hook = ParamSchedulerHook(scheduler, by_epoch=False)
    runner = SimpleNamespace(_optimizer_step_boundary=False, meta={})

    hook.after_train_iter(runner)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)
    runner._optimizer_step_boundary = True
    optimizer.step()
    hook.after_train_iter(runner)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


def test_param_scheduler_restores_its_progress_into_runner_meta():
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    hook = ParamSchedulerHook(scheduler, by_epoch=False)
    runner = SimpleNamespace(_optimizer_step_boundary=True, meta={})
    optimizer.step()
    hook.after_train_iter(runner)

    restored_optimizer = torch.optim.SGD([nn.Parameter(torch.tensor(1.0))], lr=1.0)
    restored_scheduler = StepLR(restored_optimizer, step_size=1, gamma=0.1)
    restored_hook = ParamSchedulerHook(restored_scheduler, by_epoch=False)
    restored_hook.before_run(SimpleNamespace(meta=runner.meta))

    assert restored_scheduler.last_epoch == scheduler.last_epoch
