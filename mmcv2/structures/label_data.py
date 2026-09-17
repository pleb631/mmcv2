import torch

from .base_data_element import BaseDataElement


class LabelData(BaseDataElement):
    """Structured label data and label/one-hot conversion helpers."""

    @staticmethod
    def onehot_to_label(onehot: torch.Tensor) -> torch.Tensor:
        """Convert a one-hot tensor to the indices of active labels."""
        if onehot.ndim != 1:
            raise AssertionError("onehot must be a one-dimensional tensor")
        return onehot.nonzero().squeeze(1)

    @staticmethod
    def label_to_onehot(label: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Convert label indices to a one-hot tensor.

        Args:
            label: One-dimensional tensor of class indices.
            num_classes: Total number of classes.

        Returns:
            A one-dimensional one-hot tensor.
        """
        if label.ndim != 1:
            raise AssertionError("label must be a one-dimensional tensor")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        onehot = label.new_zeros(num_classes)
        onehot[label.long()] = 1
        return onehot
