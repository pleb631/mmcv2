from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from mmcv2.structures import BaseDataElement
from mmcv2.protocols import MetricLike


class Evaluator:
    """Compose one or more streaming metrics.

    Args:
        metrics: Metric instance or non-empty sequence of metric instances.
    """

    def __init__(self, metrics: MetricLike | Sequence[MetricLike]) -> None:
        """Initialize the evaluator and validate its metrics."""
        if isinstance(metrics, MetricLike):
            metrics = [metrics]
        if not metrics or not all(isinstance(metric, MetricLike) for metric in metrics):
            raise TypeError("metrics must implement the MetricLike protocol")
        self.metrics = list(metrics)
        self._dataset_meta: dict[str, Any] | None = None

    @property
    def dataset_meta(self) -> dict[str, Any] | None:
        """Return metadata shared with each metric."""
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, value: dict[str, Any] | None) -> None:
        """Set metadata shared with each metric.

        Args:
            value: Dataset metadata, or ``None`` to clear it.
        """
        self._dataset_meta = value
        for metric in self.metrics:
            metric.dataset_meta = value

    def process(self, data_samples: Sequence[Any], data_batch: Any = None) -> None:
        """Pass a batch of predictions to every metric.

        Args:
            data_samples: Predictions or processed samples for the batch.
            data_batch: Optional input batch needed by metrics.
        """
        samples = [sample.to_dict() if isinstance(sample, BaseDataElement) else sample for sample in data_samples]
        for metric in self.metrics:
            metric.process(data_batch, samples)

    def evaluate(self, size: int) -> dict[str, Any]:
        """Compute and merge metric results.

        Args:
            size: Number of samples in the evaluated dataset.

        Returns:
            A mapping of metric names to computed values.
        """
        output: dict[str, Any] = {}
        for metric in self.metrics:
            values = metric.evaluate(size)
            duplicate = output.keys() & values.keys()
            if duplicate:
                raise ValueError(f"Duplicate metric names: {sorted(duplicate)}")
            output.update(values)
        return output

    def offline_evaluate(
        self,
        data_samples: Iterable[Any],
        data: Iterable[Any] | None = None,
        *,
        chunk_size: int = 1,
    ) -> dict[str, Any]:
        """Evaluate stored predictions without rerunning model inference.

        Args:
            data_samples: Iterable of stored prediction samples.
            data: Optional iterable of input batches consumed alongside samples.
            chunk_size: Number of samples passed to ``process`` at a time.

        Returns:
            The metric result mapping.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        samples_iter = iter(data_samples)
        data_iter = iter(data) if data is not None else None
        size = 0
        while True:
            chunk: list[Any] = []
            batch: list[Any] = []
            for _ in range(chunk_size):
                try:
                    chunk.append(next(samples_iter))
                    if data_iter is not None:
                        batch.append(next(data_iter))
                except StopIteration:
                    break
            if not chunk:
                break
            size += len(chunk)
            self.process(chunk, batch if data_iter is not None else None)
        return self.evaluate(size)
