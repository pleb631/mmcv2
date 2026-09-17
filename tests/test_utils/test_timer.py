from unittest.mock import patch

import mmcv2
import pytest


def test_timer_init():
    timer = mmcv2.Timer(start=False)
    assert not timer.is_running
    timer.start()
    assert timer.is_running
    timer = mmcv2.Timer()
    assert timer.is_running


def test_timer_run():
    with patch("mmcv2.utils.timer.time", side_effect=[0, 0, 1, 2, 2, 2]):
        timer = mmcv2.Timer()
        assert timer.since_start() == 1
        assert timer.since_last_check() == 1
        assert timer.since_start() == 2
    timer = mmcv2.Timer(False)
    with pytest.raises(mmcv2.TimerError):
        timer.since_start()
    with pytest.raises(mmcv2.TimerError):
        timer.since_last_check()


def test_timer_context(capsys):
    with patch("mmcv2.utils.timer.time", side_effect=[0, 0, 0, 1, 1]):
        with mmcv2.Timer():
            pass
    out, _ = capsys.readouterr()
    assert float(out) == 1
    with patch("mmcv2.utils.timer.time", side_effect=[0, 0, 0, 1, 1]):
        with mmcv2.Timer(print_tmpl="time: {:.1f}s"):
            pass
    out, _ = capsys.readouterr()
    assert out == "time: 1.0s\n"
