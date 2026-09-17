import torch
from mmcv2.analysis import (
    ActivationAnalyzer,
    FlopAnalyzer,
    get_model_complexity_info,
    parameter_count,
    parameter_count_table,
)
from torch import nn


def test_model_analysis_public_api():
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU(), nn.Flatten(), nn.Linear(144, 2))
    inputs = (torch.randn(1, 3, 8, 8),)
    assert FlopAnalyzer(model, inputs).total() == 4320
    assert ActivationAnalyzer(model, inputs).total() == 146
    assert parameter_count(model)[""] == 402
    assert "0.weight" in parameter_count_table(model)
    result = get_model_complexity_info(model, inputs=inputs, show_arch=False)
    assert result["flops"] == 4320
    assert result["params"] == 402
