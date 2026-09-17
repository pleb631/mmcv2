from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from typing import Any, cast

import torch
from torch import nn


def _numel(output: Any) -> int:
    if isinstance(output, torch.Tensor):
        return output.numel()
    if isinstance(output, (tuple, list)):
        return sum(_numel(item) for item in output)
    if isinstance(output, dict):
        return sum(_numel(item) for item in output.values())
    return 0


def _kernel_product(value: Any) -> int:
    if isinstance(value, int):
        return value
    return math.prod(value)


def _flops_for_module(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> int | None:
    output_elements = _numel(output)
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return output_elements * _kernel_product(module.kernel_size) * module.in_channels // module.groups
    if isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        return output_elements * _kernel_product(module.kernel_size) * module.in_channels // module.groups
    if isinstance(module, nn.Linear):
        return output_elements * module.in_features
    if isinstance(
        module,
        (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
            nn.LayerNorm,
        ),
    ):
        return output_elements
    if isinstance(
        module,
        (
            nn.ReLU,
            nn.ReLU6,
            nn.LeakyReLU,
            nn.ELU,
            nn.GELU,
            nn.SiLU,
            nn.PReLU,
        ),
    ):
        return output_elements
    if isinstance(
        module,
        (
            nn.MaxPool1d,
            nn.MaxPool2d,
            nn.MaxPool3d,
            nn.AvgPool1d,
            nn.AvgPool2d,
            nn.AvgPool3d,
        ),
    ):
        return output_elements * _kernel_product(module.kernel_size)
    if isinstance(module, (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)):
        if inputs and isinstance(inputs[0], torch.Tensor) and output_elements:
            return inputs[0].numel()
    if isinstance(module, (nn.Identity, nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.Flatten)):
        return 0
    return None


class _HookAnalyzer:
    """Base implementation for module-hook complexity analyzers."""

    def __init__(self, model: nn.Module, inputs: torch.Tensor | tuple[Any, ...]) -> None:
        """Create an analyzer for a model and its example inputs."""
        self.model = model
        self.inputs = inputs if isinstance(inputs, tuple) else (inputs,)
        self._by_module: Counter[str] | None = None
        self._by_operator: Counter[str] | None = None
        self._unsupported: Counter[str] | None = None
        self._custom_handlers: dict[type[nn.Module], Callable[[nn.Module, tuple[Any, ...], Any], int]] = {}

    def _measure(self, module: nn.Module, inputs: tuple[Any, ...], output: Any) -> int | None:
        raise NotImplementedError

    def _analyze(self) -> None:
        if self._by_module is not None:
            return
        by_module: Counter[str] = Counter()
        by_operator: Counter[str] = Counter()
        unsupported: Counter[str] = Counter()
        handles = []
        names = {module: name for name, module in self.model.named_modules()}

        def hook(module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
            if any(True for _ in module.children()):
                return
            handler = self._custom_handlers.get(type(module))
            amount = handler(module, args, output) if handler else self._measure(module, args, output)
            op_name = module.__class__.__name__.lower()
            if amount is None:
                unsupported[op_name] += 1
                return
            name = names[module]
            by_module[name] += int(amount)
            by_operator[op_name] += int(amount)
            parts = name.split(".") if name else []
            for depth in range(len(parts)):
                by_module[".".join(parts[:depth])] += int(amount)

        for module in self.model.modules():
            if not any(True for _ in module.children()):
                handles.append(module.register_forward_hook(hook))
        training = self.model.training
        try:
            self.model.eval()
            with torch.inference_mode():
                self.model(*self.inputs)
        finally:
            self.model.train(training)
            for handle in handles:
                handle.remove()
        by_module.setdefault("", sum(value for key, value in by_module.items() if key and "." not in key))
        if not by_module[""]:
            by_module[""] = sum(by_operator.values())
        self._by_module = by_module
        self._by_operator = by_operator
        self._unsupported = unsupported

    def total(self, module_name: str = "") -> int:
        """Return the total measured operations for a module.

        Args:
            module_name: Module name. An empty string selects the whole model.
        """
        self._analyze()
        assert self._by_module is not None
        return int(self._by_module[module_name])

    def by_module(self) -> Counter[str]:
        """Return measured operations grouped by module name."""
        self._analyze()
        assert self._by_module is not None
        return self._by_module.copy()

    def by_operator(self) -> Counter[str]:
        """Return measured operations grouped by module type."""
        self._analyze()
        assert self._by_operator is not None
        return self._by_operator.copy()

    def unsupported_ops(self, module_name: str = "") -> Counter[str]:
        """Return unsupported operator counts for a module."""
        self._analyze()
        assert self._unsupported is not None
        return self._unsupported.copy()

    def set_op_handle(
        self,
        handlers: Mapping[type[nn.Module], Callable[..., int] | None] | None = None,
        **named_handlers: Callable[..., int] | None,
    ) -> _HookAnalyzer:
        """Register or remove custom operation handlers.

        Args:
            handlers: Mapping from module classes to measurement callables.
                A value of ``None`` removes a handler.
            **named_handlers: Handlers addressed by names in ``torch.nn``.

        Returns:
            This analyzer, for fluent configuration.
        """
        handlers = {} if handlers is None else dict(handlers)
        for name, handler in named_handlers.items():
            module_type = getattr(nn, name, None)
            if isinstance(module_type, type) and issubclass(module_type, nn.Module):
                handlers[module_type] = handler
        for key, handler in handlers.items():
            if isinstance(key, type) and issubclass(key, nn.Module):
                if handler is None:
                    self._custom_handlers.pop(key, None)
                else:
                    self._custom_handlers[key] = handler
        self._by_module = self._by_operator = self._unsupported = None
        return self

    def unsupported_ops_warnings(self, enabled: bool) -> _HookAnalyzer:
        """Set compatibility behavior for unsupported-operation warnings."""
        return self

    def uncalled_modules_warnings(self, enabled: bool) -> _HookAnalyzer:
        """Set compatibility behavior for uncalled-module warnings."""
        return self

    def tracer_warnings(self, mode: str) -> _HookAnalyzer:
        """Set compatibility behavior for tracer warnings."""
        return self


class FlopAnalyzer(_HookAnalyzer):
    """Module-hook FLOP analyzer using one multiply-add as one operation."""

    def _measure(self, module: nn.Module, inputs: tuple[Any, ...], output: Any) -> int | None:
        return _flops_for_module(module, inputs, output)


class ActivationAnalyzer(_HookAnalyzer):
    """Count outputs produced by parameterized layers."""

    def _measure(self, module: nn.Module, inputs: tuple[Any, ...], output: Any) -> int | None:
        if isinstance(
            module,
            (
                nn.Conv1d,
                nn.Conv2d,
                nn.Conv3d,
                nn.ConvTranspose1d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
                nn.Linear,
            ),
        ):
            return _numel(output)
        if isinstance(module, (nn.Identity, nn.Flatten, nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            return 0
        return None


def flop_count(
    model: nn.Module,
    inputs: tuple[Any, ...],
    supported_ops: Mapping[type[nn.Module], Callable[..., int] | None] | None = None,
) -> tuple[defaultdict[str, float], Counter[str]]:
    """Count model FLOPs by operator.

    Args:
        model: Model to analyze.
        inputs: Example positional inputs passed to the model.
        supported_ops: Optional custom module handlers.

    Returns:
        A mapping of operator names to GFLOPs and unsupported-operation counts.
    """
    analyzer = FlopAnalyzer(model, inputs)
    if supported_ops:
        analyzer.set_op_handle(supported_ops)
    values: defaultdict[str, float] = defaultdict(float)
    values.update({key: value / 1e9 for key, value in analyzer.by_operator().items()})
    return values, analyzer.unsupported_ops()


def activation_count(
    model: nn.Module,
    inputs: tuple[Any, ...],
    supported_ops: Mapping[type[nn.Module], Callable[..., int] | None] | None = None,
) -> tuple[defaultdict[str, float], Counter[str]]:
    """Count output activations by operator.

    Args:
        model: Model to analyze.
        inputs: Example positional inputs passed to the model.
        supported_ops: Optional custom module handlers.

    Returns:
        A mapping of operator names to millions of activations and unsupported
        operation counts.
    """
    analyzer = ActivationAnalyzer(model, inputs)
    if supported_ops:
        analyzer.set_op_handle(supported_ops)
    values: defaultdict[str, float] = defaultdict(float)
    values.update({key: value / 1e6 for key, value in analyzer.by_operator().items()})
    return values, analyzer.unsupported_ops()


def parameter_count(model: nn.Module) -> defaultdict[str, int]:
    """Count parameters for the model and every module prefix."""
    result: defaultdict[str, int] = defaultdict(int)
    for name, parameter in model.named_parameters():
        size = parameter.numel()
        parts = name.split(".")
        for depth in range(len(parts) + 1):
            result[".".join(parts[:depth])] += size
    return result


def _format_size(value: int) -> str:
    for unit, scale in (("G", 10**9), ("M", 10**6), ("K", 10**3)):
        if abs(value) >= scale:
            return f"{value / scale:.3g}{unit}"
    return str(value)


def parameter_count_table(model: nn.Module, max_depth: int = 3) -> str:
    """Format a markdown table containing model parameter counts.

    Args:
        model: Model whose parameters are listed.
        max_depth: Maximum dotted-name depth included in the table.

    Returns:
        A markdown-formatted parameter table.
    """
    if max_depth <= 0:
        raise ValueError("max_depth must be positive")
    counts = parameter_count(model)
    rows = ["| name | shape | parameters |", "|---|---:|---:|"]
    rows.append(f"| model |  | {_format_size(counts[''])} |")
    for name, parameter in model.named_parameters():
        if name.count(".") + 1 <= max_depth:
            rows.append(f"| {name} | {tuple(parameter.shape)} | {_format_size(parameter.numel())} |")
    return "\n".join(rows)


def get_model_complexity_info(
    model: nn.Module,
    input_shape: tuple[int, ...] | tuple[tuple[int, ...], ...] | None = None,
    inputs: torch.Tensor | tuple[Any, ...] | None = None,
    show_table: bool = True,
    show_arch: bool = True,
) -> dict[str, Any]:
    """Compute FLOPs, activations, and parameter counts for a model.

    Args:
        model: Model to analyze.
        input_shape: Shape or tuple of shapes used to create example inputs.
        inputs: Explicit example input tensor or tuple of inputs.
        show_table: Whether to include the parameter table in the result.
        show_arch: Whether to include the model representation in the result.

    Returns:
        A dictionary containing complexity statistics and optional displays.

    Raises:
        ValueError: If neither or both input specifications are provided.
    """
    if (input_shape is None) == (inputs is None):
        raise ValueError('Exactly one of "input_shape" and "inputs" must be set')
    if inputs is None:
        assert input_shape is not None
        try:
            parameter = next(model.parameters())
            device, dtype = parameter.device, parameter.dtype
        except StopIteration:
            device, dtype = torch.device("cpu"), torch.float32
        if input_shape and all(isinstance(value, int) for value in input_shape):
            shape = cast(tuple[int, ...], input_shape)
            inputs = (torch.randn((1, *shape), device=device, dtype=dtype),)
        elif input_shape and all(isinstance(value, tuple) for value in input_shape):
            shapes = cast(tuple[tuple[int, ...], ...], input_shape)
            inputs = tuple(torch.randn((1, *shape), device=device, dtype=dtype) for shape in shapes)
        else:
            raise ValueError("input_shape must contain integers or tuples of integers")
    assert inputs is not None
    normalized = inputs if isinstance(inputs, tuple) else (inputs,)
    flop_analyzer = FlopAnalyzer(model, normalized)
    activation_analyzer = ActivationAnalyzer(model, normalized)
    flops = flop_analyzer.total()
    activations = activation_analyzer.total()
    params = parameter_count(model)[""]
    table = parameter_count_table(model) if show_table else ""
    arch = repr(model) if show_arch else ""
    return {
        "flops": flops,
        "flops_str": _format_size(flops),
        "activations": activations,
        "activations_str": _format_size(activations),
        "params": params,
        "params_str": _format_size(params),
        "out_table": table,
        "out_arch": arch,
    }
