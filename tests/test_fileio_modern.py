import os.path as osp

from mmcv2 import BaseStorageBackend
from mmcv2.fileio import (
    copyfile,
    exists,
    get,
    get_file_backend,
    get_local_path,
    get_text,
    isfile,
    join_path,
    list_dir_or_file,
    put,
    put_text,
    register_backend,
)


def test_functional_fileio(tmp_path):
    binary = tmp_path / "nested" / "data.bin"
    text = tmp_path / "note.txt"
    put(b"abc", binary)
    put_text("hello", text)
    assert get(binary) == b"abc"
    assert get_text(text) == "hello"
    assert exists(binary) and isfile(binary)
    assert get_file_backend(binary).name == "HardDiskBackend"
    assert join_path(tmp_path, "note.txt") == str(text)
    assert sorted(list_dir_or_file(tmp_path, list_dir=False, recursive=True)) == [
        osp.join("nested", "data.bin"),
        "note.txt",
    ]
    target = tmp_path / "copy.bin"
    copyfile(binary, target)
    with get_local_path(target) as local:
        assert get(local) == b"abc"


def test_functional_backend_registration(tmp_path):
    class InMemoryBackend(BaseStorageBackend):
        values: dict[str, bytes] = {}

        def get(self, filepath):
            return self.values[str(filepath)]

        def get_text(self, filepath, encoding="utf-8"):
            return self.get(filepath).decode(encoding)

        def put(self, obj, filepath):
            self.values[str(filepath)] = obj

        def put_text(self, obj, filepath, encoding="utf-8"):
            self.put(obj.encode(encoding), filepath)

    register_backend("memory", InMemoryBackend, force=True, prefixes="mem")
    path = "mem://result.txt"
    put_text("ready", path)
    assert get_text(path) == "ready"
