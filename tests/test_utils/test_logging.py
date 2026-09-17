import io
import logging
import os
import re
import tempfile
from unittest.mock import patch

import pytest
import torch
from mmcv2 import get_logger, print_log
from mmcv2.logging import HistoryBuffer, MessageHub


@patch("torch.distributed.get_rank", lambda: 0)
@patch("torch.distributed.is_initialized", lambda: True)
@patch("torch.distributed.is_available", lambda: True)
def test_get_logger_rank0():
    logger = get_logger("rank0.pkg1")
    assert isinstance(logger, logging.Logger)
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert logger.handlers[0].level == logging.INFO

    logger = get_logger("rank0.pkg2", log_level=logging.DEBUG)
    assert isinstance(logger, logging.Logger)
    assert len(logger.handlers) == 1
    assert logger.handlers[0].level == logging.DEBUG

    # the name can not be used to open the file a second time in windows,
    # so `delete` should be set as `False` and we need to manually remove it
    with tempfile.NamedTemporaryFile(delete=False) as f:
        logger = get_logger("rank0.pkg3", log_file=f.name)
        assert isinstance(logger, logging.Logger)
        assert len(logger.handlers) == 2
        assert isinstance(logger.handlers[0], logging.StreamHandler)
        assert isinstance(logger.handlers[1], logging.FileHandler)
        logger_pkg3 = get_logger("rank0.pkg3")
        assert id(logger_pkg3) == id(logger)
        # flushing and closing all handlers in order to remove `f.name`
        logging.shutdown()

    os.remove(f.name)

    logger_pkg3 = get_logger("rank0.pkg3.subpkg")
    assert logger_pkg3.handlers == logger_pkg3.handlers


@patch("torch.distributed.get_rank", lambda: 1)
@patch("torch.distributed.is_initialized", lambda: True)
@patch("torch.distributed.is_available", lambda: True)
def test_get_logger_rank1():
    logger = get_logger("rank1.pkg1")
    assert isinstance(logger, logging.Logger)
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert logger.handlers[0].level == logging.INFO

    # the name can not be used to open the file a second time in windows,
    # so `delete` should be set as `False` and we need to manually remove it
    with tempfile.NamedTemporaryFile(delete=False) as f:
        logger = get_logger("rank1.pkg2", log_file=f.name)
        assert isinstance(logger, logging.Logger)
        assert len(logger.handlers) == 1
        assert logger.handlers[0].level == logging.INFO
        # flushing and closing all handlers in order to remove `f.name`
        logging.shutdown()

    os.remove(f.name)


def test_print_log_print(capsys):
    print_log("welcome", logger=None)
    out, _ = capsys.readouterr()
    assert out == "welcome\n"


def test_print_log_silent(capsys, caplog):
    print_log("welcome", logger="silent")
    out, _ = capsys.readouterr()
    assert out == ""
    assert len(caplog.records) == 0


def test_print_log_logger(caplog):
    print_log("welcome", logger="mmcv2")
    assert caplog.record_tuples[-1] == ("mmcv2", logging.INFO, "welcome")

    print_log("welcome", logger="mmcv2", level=logging.ERROR)
    assert caplog.record_tuples[-1] == ("mmcv2", logging.ERROR, "welcome")

    # the name can not be used to open the file a second time in windows,
    # so `delete` should be set as `False` and we need to manually remove it
    with tempfile.NamedTemporaryFile(delete=False) as f:
        logger = get_logger("abc", log_file=f.name)
        print_log("welcome", logger=logger)
        assert caplog.record_tuples[-1] == ("abc", logging.INFO, "welcome")
        with open(f.name) as fin:
            log_text = fin.read()
            regex_time = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}"
            match = re.fullmatch(regex_time + r" - abc - INFO - welcome\n", log_text)
            assert match is not None
        # flushing and closing all handlers in order to remove `f.name`
        logging.shutdown()

    os.remove(f.name)


def test_print_log_exception():
    with pytest.raises(TypeError):
        print_log("welcome", logger=0)


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
