import numpy as np
import pytest
import torch

from mmcv2.runner.log_buffer import LogBuffer


def test_log_buffer_converts_scalar_tensors_to_python_numbers():
    buffer = LogBuffer()

    buffer.update({"loss": torch.tensor(2.0), "accuracy": np.float32(75.0)}, count=2)
    buffer.update({"loss": torch.tensor(4.0), "accuracy": np.float32(100.0)}, count=1)
    buffer.average()

    assert buffer.val_history["loss"] == [2.0, 4.0]
    assert buffer.output["loss"] == pytest.approx(8 / 3)
    assert buffer.output["accuracy"] == pytest.approx(250 / 3)


def test_log_buffer_rejects_non_scalar_tensors():
    buffer = LogBuffer()

    with pytest.raises(ValueError, match="Only scalar tensors can be logged"):
        buffer.update({"loss": torch.ones(2)})
