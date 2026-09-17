from collections.abc import Sequence
from typing import Any

from mmcv2.evaluator import Evaluator, MetricLike
from mmcv2.fileio import StorageBackendLike, get_file_backend
from mmcv2.runner import Hook, HookLike


class StructuralMetric:
    def __init__(self) -> None:
        self.dataset_meta: dict[str, Any] | None = None
        self.results: list[Any] = []

    def process(self, data_batch: Any, data_samples: Sequence[Any]) -> None:
        self.results.extend(data_samples)

    def evaluate(self, size: int) -> dict[str, Any]:
        result = {"count": len(self.results), "size": size}
        self.results.clear()
        return result


def test_public_protocols_match_builtin_extensions(tmp_path):
    assert isinstance(Hook(), HookLike)
    assert isinstance(get_file_backend(tmp_path), StorageBackendLike)


def test_evaluator_accepts_structural_metric():
    metric = StructuralMetric()
    assert isinstance(metric, MetricLike)
    evaluator = Evaluator(metric)
    assert evaluator.offline_evaluate([1, 2]) == {"count": 2, "size": 2}
