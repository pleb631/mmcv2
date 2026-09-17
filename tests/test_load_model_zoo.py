from unittest.mock import patch

import pytest
from mmcv2.runner.checkpoint import _load_checkpoint


def load_from_http(url, map_location=None):
    return "url:" + url


def load_state_dict_from_url(url, map_location=None, model_dir=None, weights_only=True):
    return load_from_http(url)


def load(filepath, map_location=None, weights_only=True):
    return "local:" + filepath


@patch("mmcv2.runner.checkpoint.load_from_http", load_from_http)
@patch("mmcv2.runner.checkpoint.load_state_dict_from_url", load_state_dict_from_url)
@patch(
    "mmcv2.runner.checkpoint.get_torchvision_models",
    new=lambda: {
        "resnet50.imagenet1k_v1": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
        "resnet50.default": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    },
)
@patch("torch.load", load)
def test_load_external_url(tmp_path):
    assert (
        _load_checkpoint("torchvision://resnet50.imagenet1k_v1")
        == "url:https://download.pytorch.org/models/resnet50-0676ba61.pth"
    )
    assert (
        _load_checkpoint("torchvision://ResNet50_Weights.IMAGENET1K_V1")
        == "url:https://download.pytorch.org/models/resnet50-0676ba61.pth"
    )
    _load_checkpoint("torchvision://resnet50.default")

    # test http:// https://
    url = _load_checkpoint("http://localhost/train.pth")
    assert url == "url:http://localhost/train.pth"

    # test local file
    with pytest.raises(FileNotFoundError, match="train.pth can not be found."):
        _load_checkpoint("train.pth")
    local_path = tmp_path / "test.pth"
    local_path.touch()
    assert _load_checkpoint(str(local_path)) == f"local:{local_path}"
