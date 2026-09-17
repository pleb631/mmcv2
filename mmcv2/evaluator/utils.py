from typing import Any


def get_metric_value(indicator: str, metrics: dict[str, Any]) -> Any:
    """Look up a metric by its full name or unprefixed suffix.

    Args:
        indicator: Exact metric name or the final slash-separated component.
        metrics: Mapping returned by an evaluator.

    Returns:
        The matching metric value.

    Raises:
        KeyError: If no metric matches or more than one metric is ambiguous.
    """
    if indicator in metrics:
        return metrics[indicator]
    matches = [value for key, value in metrics.items() if key.split("/")[-1] == indicator]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"Metric {indicator!r} was not found")
    raise KeyError(f"Metric {indicator!r} is ambiguous")
