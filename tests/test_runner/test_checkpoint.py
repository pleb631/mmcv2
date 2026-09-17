from collections import OrderedDict
from unittest.mock import patch

import pytest
import torch
from mmcv2.parallel.registry import MODULE_WRAPPERS
from mmcv2.runner.checkpoint import (
    _load_checkpoint,
    _load_checkpoint_with_prefix,
    get_state_dict,
    load_checkpoint,
    load_from_local,
    save_checkpoint,
)
from torch import nn, optim
from torch.nn.parallel import DataParallel


@MODULE_WRAPPERS.register_module()
class DDPWrapper:
    def __init__(self, module):
        self.module = module


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)
        self.norm = nn.BatchNorm2d(3)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()
        self.conv = nn.Conv2d(3, 3, 1)


def assert_tensor_equal(tensor_a, tensor_b):
    assert tensor_a.eq(tensor_b).all()


def test_get_state_dict():
    if torch.__version__ == "parrots":
        state_dict_keys = {
            "block.conv.weight",
            "block.conv.bias",
            "block.norm.weight",
            "block.norm.bias",
            "block.norm.running_mean",
            "block.norm.running_var",
            "conv.weight",
            "conv.bias",
        }
    else:
        state_dict_keys = {
            "block.conv.weight",
            "block.conv.bias",
            "block.norm.weight",
            "block.norm.bias",
            "block.norm.running_mean",
            "block.norm.running_var",
            "block.norm.num_batches_tracked",
            "conv.weight",
            "conv.bias",
        }

    model = Model()
    state_dict = get_state_dict(model)
    assert isinstance(state_dict, OrderedDict)
    assert set(state_dict.keys()) == state_dict_keys

    assert_tensor_equal(state_dict["block.conv.weight"], model.block.conv.weight)
    assert_tensor_equal(state_dict["block.conv.bias"], model.block.conv.bias)
    assert_tensor_equal(state_dict["block.norm.weight"], model.block.norm.weight)
    assert_tensor_equal(state_dict["block.norm.bias"], model.block.norm.bias)
    assert_tensor_equal(state_dict["block.norm.running_mean"], model.block.norm.running_mean)
    assert_tensor_equal(state_dict["block.norm.running_var"], model.block.norm.running_var)
    if torch.__version__ != "parrots":
        assert_tensor_equal(
            state_dict["block.norm.num_batches_tracked"],
            model.block.norm.num_batches_tracked,
        )
    assert_tensor_equal(state_dict["conv.weight"], model.conv.weight)
    assert_tensor_equal(state_dict["conv.bias"], model.conv.bias)

    wrapped_model = DDPWrapper(model)
    state_dict = get_state_dict(wrapped_model)
    assert isinstance(state_dict, OrderedDict)
    assert set(state_dict.keys()) == state_dict_keys
    assert_tensor_equal(state_dict["block.conv.weight"], wrapped_model.module.block.conv.weight)
    assert_tensor_equal(state_dict["block.conv.bias"], wrapped_model.module.block.conv.bias)
    assert_tensor_equal(state_dict["block.norm.weight"], wrapped_model.module.block.norm.weight)
    assert_tensor_equal(state_dict["block.norm.bias"], wrapped_model.module.block.norm.bias)
    assert_tensor_equal(
        state_dict["block.norm.running_mean"],
        wrapped_model.module.block.norm.running_mean,
    )
    assert_tensor_equal(
        state_dict["block.norm.running_var"],
        wrapped_model.module.block.norm.running_var,
    )
    if torch.__version__ != "parrots":
        assert_tensor_equal(
            state_dict["block.norm.num_batches_tracked"],
            wrapped_model.module.block.norm.num_batches_tracked,
        )
    assert_tensor_equal(state_dict["conv.weight"], wrapped_model.module.conv.weight)
    assert_tensor_equal(state_dict["conv.bias"], wrapped_model.module.conv.bias)

    # wrapped inner module
    for name, module in wrapped_model.module._modules.items():
        module = DataParallel(module)
        wrapped_model.module._modules[name] = module
    state_dict = get_state_dict(wrapped_model)
    assert isinstance(state_dict, OrderedDict)
    assert set(state_dict.keys()) == state_dict_keys
    assert_tensor_equal(state_dict["block.conv.weight"], wrapped_model.module.block.module.conv.weight)
    assert_tensor_equal(state_dict["block.conv.bias"], wrapped_model.module.block.module.conv.bias)
    assert_tensor_equal(state_dict["block.norm.weight"], wrapped_model.module.block.module.norm.weight)
    assert_tensor_equal(state_dict["block.norm.bias"], wrapped_model.module.block.module.norm.bias)
    assert_tensor_equal(
        state_dict["block.norm.running_mean"],
        wrapped_model.module.block.module.norm.running_mean,
    )
    assert_tensor_equal(
        state_dict["block.norm.running_var"],
        wrapped_model.module.block.module.norm.running_var,
    )
    if torch.__version__ != "parrots":
        assert_tensor_equal(
            state_dict["block.norm.num_batches_tracked"],
            wrapped_model.module.block.module.norm.num_batches_tracked,
        )
    assert_tensor_equal(state_dict["conv.weight"], wrapped_model.module.conv.module.weight)
    assert_tensor_equal(state_dict["conv.bias"], wrapped_model.module.conv.module.bias)


def test_load_checkpoint_with_prefix(tmp_path):

    class FooModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 2)
            self.conv2d = nn.Conv2d(3, 1, 3)
            self.conv2d_2 = nn.Conv2d(3, 2, 3)

    model = FooModule()
    nn.init.constant_(model.linear.weight, 1)
    nn.init.constant_(model.linear.bias, 2)
    nn.init.constant_(model.conv2d.weight, 3)
    nn.init.constant_(model.conv2d.bias, 4)
    nn.init.constant_(model.conv2d_2.weight, 5)
    nn.init.constant_(model.conv2d_2.bias, 6)

    checkpoint = tmp_path / "model.pth"
    torch.save(model.state_dict(), checkpoint)
    prefix = "conv2d"
    state_dict = _load_checkpoint_with_prefix(prefix, str(checkpoint))
    assert torch.equal(model.conv2d.state_dict()["weight"], state_dict["weight"])
    assert torch.equal(model.conv2d.state_dict()["bias"], state_dict["bias"])

    # test whether prefix is in pretrained model
    with pytest.raises(AssertionError):
        prefix = "back"
        _load_checkpoint_with_prefix(prefix, str(checkpoint))


def test_load_checkpoint():
    import os
    import re
    import tempfile

    class PrefixModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Model()

    pmodel = PrefixModel()
    model = Model()
    checkpoint_path = os.path.join(tempfile.gettempdir(), "checkpoint.pth")

    # add prefix
    torch.save(model.state_dict(), checkpoint_path)
    state_dict = load_checkpoint(pmodel, checkpoint_path, revise_keys=[(r"^", "backbone.")])
    for key in pmodel.backbone.state_dict():
        assert torch.equal(pmodel.backbone.state_dict()[key], state_dict[key])
    # strip prefix
    torch.save(pmodel.state_dict(), checkpoint_path)
    state_dict = load_checkpoint(model, checkpoint_path, revise_keys=[(r"^backbone\.", "")])

    for key in state_dict:
        key_stripped = re.sub(r"^backbone\.", "", key)
        assert torch.equal(model.state_dict()[key_stripped], state_dict[key])
    os.remove(checkpoint_path)


def test_load_checkpoint_metadata():
    import os
    import tempfile

    from mmcv2.runner import load_checkpoint, save_checkpoint

    class ModelV1(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = Block()
            self.conv1 = nn.Conv2d(3, 3, 1)
            self.conv2 = nn.Conv2d(3, 3, 1)
            nn.init.normal_(self.conv1.weight)
            nn.init.normal_(self.conv2.weight)

    class ModelV2(nn.Module):
        _version = 2

        def __init__(self):
            super().__init__()
            self.block = Block()
            self.conv0 = nn.Conv2d(3, 3, 1)
            self.conv1 = nn.Conv2d(3, 3, 1)
            nn.init.normal_(self.conv0.weight)
            nn.init.normal_(self.conv1.weight)

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, *args, **kwargs):
            """load checkpoints."""

            # Names of some parameters in has been changed.
            version = local_metadata.get("version", None)
            if version is None or version < 2:
                state_dict_keys = list(state_dict.keys())
                convert_map = {"conv1": "conv0", "conv2": "conv1"}
                for k in state_dict_keys:
                    for ori_str, new_str in convert_map.items():
                        if k.startswith(prefix + ori_str):
                            new_key = k.replace(ori_str, new_str)
                            state_dict[new_key] = state_dict[k]
                            del state_dict[k]

            super()._load_from_state_dict(state_dict, prefix, local_metadata, *args, **kwargs)

    model_v1 = ModelV1()
    model_v1_conv0_weight = model_v1.conv1.weight.detach()
    model_v1_conv1_weight = model_v1.conv2.weight.detach()
    model_v2 = ModelV2()
    model_v2_conv0_weight = model_v2.conv0.weight.detach()
    model_v2_conv1_weight = model_v2.conv1.weight.detach()
    ckpt_v1_path = os.path.join(tempfile.gettempdir(), "checkpoint_v1.pth")
    ckpt_v2_path = os.path.join(tempfile.gettempdir(), "checkpoint_v2.pth")

    # Save checkpoint
    save_checkpoint(model_v1, ckpt_v1_path)
    save_checkpoint(model_v2, ckpt_v2_path)

    # test load v1 model
    load_checkpoint(model_v2, ckpt_v1_path)
    assert torch.allclose(model_v2.conv0.weight, model_v1_conv0_weight)
    assert torch.allclose(model_v2.conv1.weight, model_v1_conv1_weight)

    # test load v2 model
    load_checkpoint(model_v2, ckpt_v2_path)
    assert torch.allclose(model_v2.conv0.weight, model_v2_conv0_weight)
    assert torch.allclose(model_v2.conv1.weight, model_v2_conv1_weight)


def test_load_classes_name():
    import os
    import tempfile

    from mmcv2.runner import load_checkpoint, save_checkpoint

    checkpoint_path = os.path.join(tempfile.gettempdir(), "checkpoint.pth")
    model = Model()
    save_checkpoint(model, checkpoint_path)
    checkpoint = load_checkpoint(model, checkpoint_path)
    assert "meta" in checkpoint and "CLASSES" not in checkpoint["meta"]

    model.CLASSES = ("class1", "class2")
    save_checkpoint(model, checkpoint_path)
    checkpoint = load_checkpoint(model, checkpoint_path)
    assert "meta" in checkpoint and "CLASSES" in checkpoint["meta"]
    assert checkpoint["meta"]["CLASSES"] == ("class1", "class2")

    model = Model()
    wrapped_model = DDPWrapper(model)
    save_checkpoint(wrapped_model, checkpoint_path)
    checkpoint = load_checkpoint(wrapped_model, checkpoint_path)
    assert "meta" in checkpoint and "CLASSES" not in checkpoint["meta"]

    wrapped_model.module.CLASSES = ("class1", "class2")
    save_checkpoint(wrapped_model, checkpoint_path)
    checkpoint = load_checkpoint(wrapped_model, checkpoint_path)
    assert "meta" in checkpoint and "CLASSES" in checkpoint["meta"]
    assert checkpoint["meta"]["CLASSES"] == ("class1", "class2")

    # remove the temp file
    os.remove(checkpoint_path)


def test_checkpoint_loader():
    import os
    import tempfile

    from mmcv2.runner import CheckpointLoader, _load_checkpoint, save_checkpoint

    checkpoint_path = os.path.join(tempfile.gettempdir(), "checkpoint.pth")
    model = Model()
    save_checkpoint(model, checkpoint_path)
    checkpoint = _load_checkpoint(checkpoint_path)
    assert "meta" in checkpoint and "CLASSES" not in checkpoint["meta"]
    # remove the temp file
    os.remove(checkpoint_path)

    filenames = [
        "http://xx.xx/xx.pth",
        "https://xx.xx/xx.pth",
        "torchvision://xx.xx/xx.pth",
    ]
    fn_names = [
        "load_from_http",
        "load_from_http",
        "load_from_torchvision",
    ]

    for filename, fn_name in zip(filenames, fn_names):
        loader = CheckpointLoader._get_checkpoint_loader(filename)
        assert loader.__name__ == fn_name

    for filename in (
        "mmcls://xx.xx/xx.pth",
        "s3://xx.xx/xx.pth",
    ):
        with pytest.raises(ValueError, match="Unsupported checkpoint URI"):
            CheckpointLoader._get_checkpoint_loader(filename)

    @CheckpointLoader.register_scheme(prefixes="ftp://")
    def load_from_ftp(filename, map_location):
        return {"filename": filename}

    # test register_loader
    filename = "ftp://xx.xx/xx.pth"
    loader = CheckpointLoader._get_checkpoint_loader(filename)
    assert loader.__name__ == "load_from_ftp"

    def load_from_ftp1(filename, map_location):
        return {"filename": filename}

    # test duplicate registered error
    with pytest.raises(KeyError):
        CheckpointLoader.register_scheme("ftp://", load_from_ftp1)

    # test force param
    CheckpointLoader.register_scheme("ftp://", load_from_ftp1, force=True)
    checkpoint = CheckpointLoader.load_checkpoint(filename)
    assert checkpoint["filename"] == filename

    # test print function name
    loader = CheckpointLoader._get_checkpoint_loader(filename)
    assert loader.__name__ == "load_from_ftp1"

    # test sort
    @CheckpointLoader.register_scheme(prefixes="a/b")
    def load_from_ab(filename, map_location):
        return {"filename": filename}

    @CheckpointLoader.register_scheme(prefixes="a/b/c")
    def load_from_abc(filename, map_location):
        return {"filename": filename}

    filename = "a/b/c/d"
    loader = CheckpointLoader._get_checkpoint_loader(filename)
    assert loader.__name__ == "load_from_abc"


def test_save_checkpoint(tmp_path):
    model = Model()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # 1. save to disk
    filename = str(tmp_path / "checkpoint1.pth")
    save_checkpoint(model, filename)

    filename = str(tmp_path / "checkpoint2.pth")
    save_checkpoint(model, filename, optimizer)

    filename = str(tmp_path / "checkpoint3.pth")
    save_checkpoint(model, filename, meta={"test": "test"})

    filename = str(tmp_path / "checkpoint4.pth")
    save_checkpoint(model, filename, backend_args={"backend": "disk"})


def test_load_from_local(tmp_path):
    checkpoint_path = tmp_path / "dummy_checkpoint_used_to_test_load_from_local.pth"
    model = Model()
    save_checkpoint(model, str(checkpoint_path))
    checkpoint = load_from_local(str(checkpoint_path), map_location=None)
    assert_tensor_equal(checkpoint["state_dict"]["block.conv.weight"], model.block.conv.weight)


def _load_checkpoint_from_http(url, map_location=None):
    return "url:" + url


def _load_checkpoint_from_url(url, map_location=None, model_dir=None, weights_only=True):
    return _load_checkpoint_from_http(url)


def _load_checkpoint_from_file(filepath, map_location=None, weights_only=True):
    return "local:" + filepath


@patch("mmcv2.runner.checkpoint.load_from_http", _load_checkpoint_from_http)
@patch("mmcv2.runner.checkpoint.load_state_dict_from_url", _load_checkpoint_from_url)
@patch(
    "mmcv2.runner.checkpoint.get_torchvision_models",
    new=lambda: {
        "resnet50.imagenet1k_v1": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
        "resnet50.default": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    },
)
@patch("torch.load", _load_checkpoint_from_file)
def test_load_external_checkpoint_urls(tmp_path):
    expected_url = "url:https://download.pytorch.org/models/resnet50-0676ba61.pth"
    assert _load_checkpoint("torchvision://resnet50.imagenet1k_v1") == expected_url
    assert _load_checkpoint("torchvision://ResNet50_Weights.IMAGENET1K_V1") == expected_url
    assert _load_checkpoint("torchvision://resnet50.default") == expected_url
    assert _load_checkpoint("http://localhost/train.pth") == "url:http://localhost/train.pth"

    with pytest.raises(FileNotFoundError, match="train.pth can not be found."):
        _load_checkpoint("train.pth")
    local_path = tmp_path / "test.pth"
    local_path.touch()
    assert _load_checkpoint(str(local_path)) == f"local:{local_path}"
