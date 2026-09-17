from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import torch

from mmcv2.distributed import broadcast_object_list, collect_results, is_main_process
from mmcv2.fileio import dump


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


class BaseMetric(ABC):
    """Base class for metrics that aggregate results across batches.

    Subclasses append per-sample values in ``results`` and implement
    ``compute_metrics`` to produce the final mapping.

    Args:
        prefix: Optional prefix added to every metric name.
    """

    default_prefix: str | None = None

    def __init__(self, *, prefix: str | None = None) -> None:
        """Initialize an empty result buffer."""
        self.prefix = prefix if prefix is not None else self.default_prefix
        self.results: list[Any] = []
        self.dataset_meta: dict[str, Any] | None = None

    @abstractmethod
    def process(self, data_batch: Any, data_samples: Sequence[Any]) -> None:
        """Extract metric inputs from one processed batch.

        Args:
            data_batch: Original input batch.
            data_samples: Model outputs for the batch.
        """

    @abstractmethod
    def compute_metrics(self, results: list[Any]) -> dict[str, Any]:
        """Compute metrics from all gathered results.

        Args:
            results: Per-sample values collected by ``process``.

        Returns:
            A mapping of metric names to values.
        """

    def evaluate(self, size: int) -> dict[str, Any]:
        """Gather buffered results and compute the final metrics.

        Args:
            size: Number of samples in the evaluated dataset.

        Returns:
            Metric values on the main process; an empty mapping elsewhere.
        """
        collected = collect_results(self.results, size)
        metrics: dict[str, Any] | None = None
        if is_main_process():
            metrics = self.compute_metrics(collected or [])
            if self.prefix:
                metrics = {f"{self.prefix}/{key}": value for key, value in metrics.items()}
        objects: list[Any] = [metrics]
        broadcast_object_list(objects)
        self.results.clear()
        return objects[0] or {}


class DumpResults(BaseMetric):
    """Metric that serializes all processed samples to a file.

    Args:
        out_file_path: Destination path accepted by ``mmcv2.fileio.dump``.
        prefix: Optional metric name prefix.
    """

    def __init__(self, out_file_path: str, *, prefix: str | None = None) -> None:
        """Create a result-dumping metric."""
        super().__init__(prefix=prefix)
        self.out_file_path = out_file_path

    def process(self, data_batch: Any, data_samples: Sequence[Any]) -> None:
        """Append CPU-safe copies of the batch samples."""
        self.results.extend(_to_cpu(list(data_samples)))

    def compute_metrics(self, results: list[Any]) -> dict[str, Any]:
        """Write results and return an empty metric mapping.

        Args:
            results: Gathered prediction samples.

        Returns:
            An empty dictionary because this metric only writes a file.
        """
        dump(results, self.out_file_path)
        return {}
