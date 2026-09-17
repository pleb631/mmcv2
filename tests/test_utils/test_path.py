import os.path as osp
from pathlib import Path

import mmcv2
import pytest


def test_is_filepath():
    assert mmcv2.is_filepath(__file__)
    assert mmcv2.is_filepath("abc")
    assert mmcv2.is_filepath(Path("/etc"))
    assert not mmcv2.is_filepath(0)


def test_check_file_exist():
    mmcv2.check_file_exist(__file__)
    with pytest.raises(FileNotFoundError):
        mmcv2.check_file_exist("no_such_file.txt")


def test_scandir():
    folder = osp.join(osp.dirname(osp.dirname(__file__)), "data/for_scan")
    filenames = ["a.bin", "1.txt", "2.txt", "1.json", "2.json", "3.TXT"]
    assert set(mmcv2.scandir(folder)) == set(filenames)
    assert set(mmcv2.scandir(Path(folder))) == set(filenames)
    assert set(mmcv2.scandir(folder, ".txt")) == {filename for filename in filenames if filename.endswith(".txt")}
    assert set(mmcv2.scandir(folder, (".json", ".txt"))) == {
        filename for filename in filenames if filename.endswith((".txt", ".json"))
    }
    assert set(mmcv2.scandir(folder, ".png")) == set()

    # path of sep is `\\` in windows but `/` in linux, so osp.join should be
    # used to join string for compatibility
    filenames_recursive = [
        "a.bin",
        "1.txt",
        "2.txt",
        "1.json",
        "2.json",
        "3.TXT",
        osp.join("sub", "1.json"),
        osp.join("sub", "1.txt"),
        ".file",
    ]
    # .file starts with '.' and is a file so it will not be scanned
    assert set(mmcv2.scandir(folder, recursive=True)) == {
        filename for filename in filenames_recursive if filename != ".file"
    }
    assert set(mmcv2.scandir(Path(folder), recursive=True)) == {
        filename for filename in filenames_recursive if filename != ".file"
    }
    assert set(mmcv2.scandir(folder, ".txt", recursive=True)) == {
        filename for filename in filenames_recursive if filename.endswith(".txt")
    }
    assert set(mmcv2.scandir(folder, ".TXT", recursive=True, case_sensitive=False)) == {
        filename for filename in filenames_recursive if filename.endswith((".txt", ".TXT"))
    }
    assert set(mmcv2.scandir(folder, (".TXT", ".JSON"), recursive=True, case_sensitive=False)) == {
        filename for filename in filenames_recursive if filename.endswith((".txt", ".json", ".TXT"))
    }
    with pytest.raises(TypeError):
        list(mmcv2.scandir(123))
    with pytest.raises(TypeError):
        list(mmcv2.scandir(folder, 111))
