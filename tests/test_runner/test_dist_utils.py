from unittest.mock import patch

import pytest
import torch
from mmcv2.distributed import init_dist


@pytest.mark.parametrize("world_size,local_rank", [(1, 0), (4, 1)])
def test_distributed_initializes_torchrun_single_machine(monkeypatch, world_size, local_rank):
    monkeypatch.setenv("LOCAL_RANK", str(local_rank))
    monkeypatch.setenv("RANK", str(local_rank))
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    monkeypatch.setenv("LOCAL_WORLD_SIZE", str(world_size))
    with (
        patch(
            "torch.accelerator.current_accelerator",
            return_value=torch.device("cuda"),
        ),
        patch("torch.accelerator.set_device_index") as set_device,
        patch("torch.distributed.init_process_group") as init_group,
    ):
        init_dist("pytorch", backend="nccl")
    device = torch.device("cuda", local_rank)
    set_device.assert_called_once_with(device)
    init_group.assert_called_once_with(backend="nccl", device_id=device)


def test_torchrun_cpu_uses_gloo(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    with (
        patch("torch.accelerator.current_accelerator", return_value=None),
        patch("torch.accelerator.set_device_index") as set_device,
        patch("torch.distributed.get_default_backend_for_device", return_value="gloo") as get_backend,
        patch("torch.distributed.init_process_group") as init_group,
    ):
        init_dist("pytorch")
    set_device.assert_not_called()
    get_backend.assert_called_once_with("cpu")
    init_group.assert_called_once_with(backend="gloo")


def test_torchrun_multi_machine_supported(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    with (
        patch("torch.accelerator.current_accelerator", return_value=None),
        patch("torch.distributed.get_default_backend_for_device", return_value="gloo"),
        patch("torch.distributed.init_process_group") as init_group,
    ):
        init_dist("pytorch")
    init_group.assert_called_once_with(backend="gloo")


@pytest.mark.parametrize("launcher", ["mpi", "slurm", "invalid_launcher"])
def test_unsupported_launcher_rejected(launcher):
    with pytest.raises(ValueError, match="Invalid launcher type"):
        init_dist(launcher)
