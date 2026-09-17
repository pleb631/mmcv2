import numpy as np
import pytest
import torch
from mmcv2.structures import BaseDataElement, InstanceData, LabelData, PixelData


def test_base_data_element_transforms_and_fields():
    sample = BaseDataElement(metainfo={"image_id": 3}, tensor=torch.tensor([1.0]))
    sample.extra = BaseDataElement(value=torch.tensor([2.0]))
    assert sample.metainfo == {"image_id": 3}
    assert sample.numpy().tensor.tolist() == [1.0]
    assert sample.to_dict()["extra"]["value"].item() == 2
    assert sample.clone() is not sample
    with pytest.raises(AttributeError):
        sample.set_field(4, "image_id")


def test_instance_data_slice_and_cat():
    data = InstanceData(
        labels=torch.tensor([1, 2, 3]),
        boxes=np.arange(12).reshape(3, 4),
        names=["a", "b", "c"],
    )
    assert len(data[1]) == 1
    selected = data[torch.tensor([True, False, True])]
    assert selected.names == ["a", "c"]
    joined = InstanceData.cat([data[:1], data[1:]])
    assert len(joined) == 3
    with pytest.raises(AssertionError):
        data.scores = torch.ones(2)


def test_pixel_and_label_data():
    pixels = PixelData(image=torch.ones(3, 8, 9))
    assert pixels.shape == (8, 9)
    assert pixels[2, 3].shape == (1, 1)
    labels = torch.tensor([0, 2])
    onehot = LabelData.label_to_onehot(labels, 4)
    assert torch.equal(LabelData.onehot_to_label(onehot), labels)
