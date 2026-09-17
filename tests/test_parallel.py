import socket

import pytest
import torch
import torch.distributed as dist
from mmcv2.parallel import (
    MODULE_WRAPPERS,
    MMDistributedDataParallel,
    is_module_wrapper,
)
from mmcv2.parallel._functions import Scatter, get_input_device, scatter
from mmcv2.utils import Registry
from torch import nn
from torch.nn.parallel import DataParallel, DistributedDataParallel


@pytest.mark.skipif(not dist.is_available(), reason="distributed unavailable")
def test_is_module_wrapper():

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 2, 1)

        def forward(self, x):
            return self.conv(x)

        def training_step(self, x, batch_idx):
            return {"mode": "train", "output": self(x)}

        def validation_step(self, x, batch_idx):
            return {"mode": "val", "output": self(x)}

        def test_step(self, x):
            return [{"mode": "test", "output": self(x)}]

    model = Model()
    assert not is_module_wrapper(model)

    dp = DataParallel(model)
    assert is_module_wrapper(dp)

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
    try:
        ddp = DistributedDataParallel(model)
        assert is_module_wrapper(ddp)
        mmddp = MMDistributedDataParallel(model)
        assert is_module_wrapper(mmddp)
        assert mmddp.module is model
        inputs = torch.ones(1, 2, 2, 2, device=next(model.parameters()).device)
        assert mmddp.training_step(inputs, 0)["mode"] == "train"
        assert mmddp.validation_step(inputs, 0)["mode"] == "val"
        assert mmddp.test_step(inputs)[0]["mode"] == "test"
    finally:
        dist.destroy_process_group()

    # test module wrapper registry
    @MODULE_WRAPPERS.register_module()
    class ModuleWrapper:
        def __init__(self, module):
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    module_wraper = ModuleWrapper(model)
    assert is_module_wrapper(module_wraper)

    # test module wrapper registry in downstream repo
    MMRAZOR_MODULE_WRAPPERS = Registry("mmrazor module wrapper", parent=MODULE_WRAPPERS, scope="mmrazor")
    MMPOSE_MODULE_WRAPPERS = Registry("mmpose module wrapper", parent=MODULE_WRAPPERS, scope="mmpose")

    @MMRAZOR_MODULE_WRAPPERS.register_module()
    class ModuleWrapperInRazor:
        def __init__(self, module):
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    @MMPOSE_MODULE_WRAPPERS.register_module()
    class ModuleWrapperInPose:
        def __init__(self, module):
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    wrapped_module = ModuleWrapperInRazor(model)
    assert is_module_wrapper(wrapped_module)

    wrapped_module = ModuleWrapperInPose(model)
    assert is_module_wrapper(wrapped_module)


def test_get_input_device():
    # if the device is CPU, return -1
    input = torch.zeros([1, 3, 3, 3])
    assert get_input_device(input) == -1
    inputs = [torch.zeros([1, 3, 3, 3]), torch.zeros([1, 4, 4, 4])]
    assert get_input_device(inputs) == -1

    # if the device is GPU, return the index of device
    if torch.cuda.is_available():
        input = torch.zeros([1, 3, 3, 3]).cuda()
        assert get_input_device(input) == 0
        inputs = [torch.zeros([1, 3, 3, 3]).cuda(), torch.zeros([1, 4, 4, 4]).cuda()]
        assert get_input_device(inputs) == 0


def test_scatter():
    # if the device is CPU, just return the input
    input = torch.zeros([1, 3, 3, 3])
    output = scatter(input=input, devices=[-1])
    assert torch.allclose(input, output)

    inputs = [torch.zeros([1, 3, 3, 3]), torch.zeros([1, 4, 4, 4])]
    outputs = scatter(input=inputs, devices=[-1])
    for input, output in zip(inputs, outputs):
        assert torch.allclose(input, output)

    # if the device is GPU, copy the input from CPU to GPU
    if torch.cuda.is_available():
        input = torch.zeros([1, 3, 3, 3])
        output = scatter(input=input, devices=[0])
        assert torch.allclose(input.cuda(), output)

        inputs = [torch.zeros([1, 3, 3, 3]), torch.zeros([1, 4, 4, 4])]
        outputs = scatter(input=inputs, devices=[0])
        for input, output in zip(inputs, outputs):
            assert torch.allclose(input.cuda(), output)


@pytest.mark.skipif(torch.__version__ == "parrots", reason="not supported in parrots now")
def test_Scatter():
    # if the device is CPU, just return the input
    target_gpus = [-1]
    input = torch.zeros([1, 3, 3, 3])
    outputs = Scatter.forward(target_gpus, input)
    assert isinstance(outputs, tuple)
    assert torch.allclose(input, outputs[0])

    target_gpus = [-1]
    inputs = [torch.zeros([1, 3, 3, 3]), torch.zeros([1, 4, 4, 4])]
    outputs = Scatter.forward(target_gpus, inputs)
    assert isinstance(outputs, tuple)
    for input, output in zip(inputs, outputs):
        assert torch.allclose(input, output)

    # if the device is GPU, copy the input from CPU to GPU
    if torch.cuda.is_available():
        target_gpus = [0]
        input = torch.zeros([1, 3, 3, 3])
        outputs = Scatter.forward(target_gpus, input)
        assert isinstance(outputs, tuple)
        assert torch.allclose(input.cuda(), outputs[0])

        target_gpus = [0]
        inputs = [torch.zeros([1, 3, 3, 3]), torch.zeros([1, 4, 4, 4])]
        outputs = Scatter.forward(target_gpus, inputs)
        assert isinstance(outputs, tuple)
        for input, output in zip(inputs, outputs):
            assert torch.allclose(input.cuda(), output[0])
