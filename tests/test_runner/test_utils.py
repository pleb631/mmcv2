import os
import random

import numpy as np
import pytest
import torch
from mmcv2.runner import set_random_seed
from mmcv2.runner.log_buffer import LogBuffer


def test_set_random_seed():
    set_random_seed(0)
    a_random = random.randint(0, 10)
    a_np_random = np.random.rand(2, 2)
    a_torch_random = torch.rand(2, 2)
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is False
    assert os.environ["PYTHONHASHSEED"] == str(0)

    set_random_seed(0, True)
    b_random = random.randint(0, 10)
    b_np_random = np.random.rand(2, 2)
    b_torch_random = torch.rand(2, 2)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False

    assert a_random == b_random
    assert np.equal(a_np_random, b_np_random).all()
    assert torch.equal(a_torch_random, b_torch_random)


def test_log_buffer_converts_scalar_tensors_to_python_numbers():
    buffer = LogBuffer()
    buffer.update({"loss": torch.tensor(2.0), "accuracy": np.float32(75.0)}, count=2)
    buffer.update({"loss": torch.tensor(4.0), "accuracy": np.float32(100.0)}, count=1)
    buffer.average()
    assert buffer.val_history["loss"] == [2.0, 4.0]
    assert buffer.output["loss"] == pytest.approx(8 / 3)
    assert buffer.output["accuracy"] == pytest.approx(250 / 3)


def test_log_buffer_rejects_non_scalar_tensors():
    with pytest.raises(ValueError, match="Only scalar tensors can be logged"):
        LogBuffer().update({"loss": torch.ones(2)})
