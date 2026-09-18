import gc
import logging
import weakref

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mmcv2.runner import EpochBasedRunner, Hook, RunnerContext


class ContextAwareModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.seen_contexts: list[RunnerContext] = []

    def training_step(self, data_batch, batch_idx):
        del batch_idx
        self.seen_contexts.append(RunnerContext.current())
        features, _ = data_batch
        return (features * self.weight).mean()

    def validation_step(self, data_batch, batch_idx):
        del batch_idx
        return data_batch


class ContextHook(Hook):
    def __init__(self) -> None:
        self.before_context = None
        self.batch = None

    def before_run(self, runner) -> None:
        self.before_context = RunnerContext.current()

    def before_train_iter(self, runner) -> None:
        self.batch = runner.ctx.data_batch


def build_runner():
    model = ContextAwareModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    runner = EpochBasedRunner(
        model=model,
        optimizer=optimizer,
        logger=logging.getLogger("runner-context-test"),
        max_epochs=1,
    )
    return runner, model, optimizer


def test_runner_context_is_a_live_read_only_view():
    runner, model, optimizer = build_runner()
    dataset = TensorDataset(torch.ones(2, 1), torch.zeros(2))
    loader = DataLoader(dataset, batch_size=1)
    runner.data_loader = loader
    runner.outputs = {"loss": torch.tensor(1.0)}

    assert runner.ctx.runner is runner
    assert runner.ctx.model is model
    assert runner.ctx.step_model is model
    assert runner.ctx.optimizer is optimizer
    assert runner.ctx.data_loader is loader
    assert runner.ctx.dataset is dataset
    assert runner.ctx.outputs is runner.outputs
    assert runner.ctx.epoch == 0
    assert runner.ctx.iter == 0
    assert runner.ctx.device == torch.device("cpu")

    runner._epoch = 2
    runner._iter = 7
    assert runner.ctx.epoch == 2
    assert runner.ctx.iter == 7


def test_runner_context_is_active_throughout_run():
    runner, model, _ = build_runner()
    loader = DataLoader(TensorDataset(torch.ones(1, 1), torch.zeros(1)), batch_size=1)
    hook = ContextHook()
    runner.register_hook(hook)

    with pytest.raises(RuntimeError, match="No RunnerContext"):
        RunnerContext.current()

    runner.run([loader], [("train", 1)])

    assert hook.before_context is runner.ctx
    assert hook.batch is not None
    assert model.seen_contexts == [runner.ctx]
    with pytest.raises(RuntimeError, match="No RunnerContext"):
        RunnerContext.current()


def test_runner_context_does_not_retain_runner():
    runner, _, _ = build_runner()
    context = runner.ctx
    runner_ref = weakref.ref(runner)

    del runner
    gc.collect()

    assert runner_ref() is None
    assert repr(context) == "RunnerContext(expired=True)"
    with pytest.raises(RuntimeError, match="no longer exists"):
        _ = context.model
