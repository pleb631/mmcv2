import pytest
import torch
from mmcv2.cnn.bricks import ConvModule
from mmcv2.cnn.utils import revert_sync_batchnorm


@pytest.mark.skipif(torch.__version__ == "parrots", reason="not supported in parrots now")
def test_revert_syncbn():
    conv = ConvModule(3, 8, 2, norm_cfg={"type": "SyncBN"})
    x = torch.randn(1, 3, 10, 10)
    y = conv(x)
    assert y.shape == (1, 8, 9, 9)
    conv = revert_sync_batchnorm(conv)
    y = conv(x)
    assert y.shape == (1, 8, 9, 9)
