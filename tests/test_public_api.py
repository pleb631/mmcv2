import importlib
import subprocess
import sys

import mmcv2
import pytest


def test_domain_apis_are_explicit_opt_in():
    distributed = importlib.import_module("mmcv2.distributed")
    fileio = importlib.import_module("mmcv2.fileio")
    runner = importlib.import_module("mmcv2.runner")

    assert distributed.init_dist is not None
    assert fileio.get_text is not None
    assert runner.EpochBasedRunner is not None
    assert not hasattr(mmcv2, "FileClient")
    assert not hasattr(runner, "init_dist")


def test_root_does_not_eagerly_import_domain_apis():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import mmcv2; assert 'distributed' not in mmcv2.__dict__; assert 'runner' not in mmcv2.__dict__",
        ],
        check=True,
    )


def test_engine_namespace_was_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("mmcv2.engine")
