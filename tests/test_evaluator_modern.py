from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from mmcv2.evaluator import BaseMetric, Evaluator
from mmcv2.logging import MessageHub
from mmcv2.runner import EvalHook
from mmcv2.runner.log_buffer import LogBuffer
from mmcv2.structures import BaseDataElement
from torch.utils.data import DataLoader


class Accuracy(BaseMetric):
    default_prefix = "classification"

    def process(self, data_batch: Any, data_samples: Sequence[Any]) -> None:
        self.results.extend(sample["pred"] == sample["target"] for sample in data_samples)

    def compute_metrics(self, results: list[Any]) -> dict[str, Any]:
        return {"accuracy": sum(results) / len(results)}


def test_evaluator_offline_and_structured_samples():
    evaluator = Evaluator(Accuracy())
    samples = [
        BaseDataElement(pred=1, target=1),
        BaseDataElement(pred=0, target=1),
    ]
    assert evaluator.offline_evaluate(samples) == {"classification/accuracy": 0.5}
    assert evaluator.metrics[0].results == []


def test_eval_hook_streams_test_step_outputs_into_evaluator():
    class Dataset:
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return index

    class Model:
        def eval(self):
            return self

        def test_step(self, data_batch):
            return [{"pred": int(index == 0), "target": 1} for index in data_batch.tolist()]

    evaluator = Evaluator(Accuracy())
    hook = EvalHook(DataLoader(Dataset(), batch_size=2), evaluator)
    runner = SimpleNamespace(
        model=Model(),
        log_buffer=LogBuffer(),
        message_hub=MessageHub("evaluation-test"),
    )
    hook._do_evaluate(runner)
    assert runner.log_buffer.output["classification/accuracy"] == 0.5
