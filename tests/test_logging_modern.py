import io

import pytest
import torch
from mmcv2.logging import HistoryBuffer, MessageHub


def test_history_buffer_weighted_statistics():
    history = HistoryBuffer(max_length=2)
    history.update(1, 1)
    history.update(3, 3)
    assert history.mean() == pytest.approx(2.5)
    history.update(5)
    assert history.current() == 5
    assert history.min() == 3


def test_message_hub_roundtrip():
    hub = MessageHub("source")
    hub.update_scalar("loss", torch.tensor(2.0), count=2)
    hub.update_info("epoch", 4)
    hub.update_info("temporary", object(), resumed=False)
    restored = MessageHub("target")
    restored.load_state_dict(hub.state_dict())
    assert restored.get_scalar("loss").mean() == 2
    assert restored.get_info("epoch") == 4
    assert restored.get_info("temporary") is None


def test_message_hub_state_is_safe_checkpoint_data():
    hub = MessageHub("safe-checkpoint")
    hub.update_scalar("loss", 2.0, count=2)
    buffer = io.BytesIO()
    torch.save({"message_hub": hub.state_dict()}, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    restored = MessageHub("safe-checkpoint-restored")
    restored.load_state_dict(checkpoint["message_hub"])
    assert restored.get_scalar("loss").mean() == 2
