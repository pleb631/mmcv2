from __future__ import annotations

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from mmcv2.protocols import StorageBackendLike

from .file_client import BaseStorageBackend, FileClient

PathLike = str | Path


def register_backend(
    name: str,
    backend: type[BaseStorageBackend] | None = None,
    *,
    force: bool = False,
    prefixes: str | list[str] | tuple[str, ...] | None = None,
):
    """Register a storage backend for the functional file-I/O API.

    Args:
        name: Unique backend name.
        backend: Backend class to register. If omitted, the function returns
            a decorator.
        force: Whether to replace an existing backend with the same name.
        prefixes: URI prefixes routed to the backend.

    Returns:
        The backend class when used as a decorator, otherwise ``None``.
    """
    return FileClient.register_backend(name, backend=backend, force=force, prefixes=prefixes)


def get_file_backend(uri: PathLike | None = None, *, backend_args: dict[str, Any] | None = None) -> StorageBackendLike:
    """Return the storage backend selected for a URI.

    Args:
        uri: Path or URI used to select a backend.
        backend_args: Backend construction arguments.

    Returns:
        A storage backend instance.
    """
    client = (
        FileClient.infer_client(backend_args, uri)
        if backend_args
        else FileClient(prefix=FileClient.parse_uri_prefix(uri) if uri is not None else None)
    )
    return client.client


def _backend(uri: PathLike, backend_args: dict[str, Any] | None) -> StorageBackendLike:
    return get_file_backend(uri, backend_args=backend_args)


def get(filepath: PathLike, *, backend_args: dict[str, Any] | None = None) -> bytes | memoryview:
    """Read binary data from a local or remote path."""
    return _backend(filepath, backend_args).get(filepath)


def get_text(
    filepath: PathLike,
    *,
    encoding: str = "utf-8",
    backend_args: dict[str, Any] | None = None,
) -> str:
    """Read text from a local or remote path.

    Args:
        filepath: File path or URI.
        encoding: Text encoding used by the backend.
        backend_args: Backend construction arguments.
    """
    return _backend(filepath, backend_args).get_text(filepath, encoding)


def put(obj: bytes, filepath: PathLike, *, backend_args: dict[str, Any] | None = None) -> None:
    """Write binary data to a local or remote path."""
    _backend(filepath, backend_args).put(obj, filepath)


def put_text(
    obj: str,
    filepath: PathLike,
    *,
    encoding: str = "utf-8",
    backend_args: dict[str, Any] | None = None,
) -> None:
    """Write text to a local or remote path."""
    backend = _backend(filepath, backend_args)
    try:
        backend.put_text(obj, filepath, encoding=encoding)
    except TypeError:
        backend.put_text(obj, filepath)


def exists(filepath: PathLike, *, backend_args: dict[str, Any] | None = None) -> bool:
    """Return whether a path exists."""
    return _backend(filepath, backend_args).exists(filepath)


def isdir(filepath: PathLike, *, backend_args: dict[str, Any] | None = None) -> bool:
    """Return whether a path identifies a directory."""
    return _backend(filepath, backend_args).isdir(filepath)


def isfile(filepath: PathLike, *, backend_args: dict[str, Any] | None = None) -> bool:
    """Return whether a path identifies a regular file."""
    return _backend(filepath, backend_args).isfile(filepath)


def join_path(
    filepath: PathLike,
    *filepaths: PathLike,
    backend_args: dict[str, Any] | None = None,
) -> str:
    """Join path components using the selected backend's path rules."""
    return _backend(filepath, backend_args).join_path(filepath, *filepaths)


def remove(filepath: PathLike, *, backend_args: dict[str, Any] | None = None) -> None:
    """Remove a file from the selected backend."""
    _backend(filepath, backend_args).remove(filepath)


@contextmanager
def get_local_path(filepath: PathLike, *, backend_args: dict[str, Any] | None = None) -> Iterator[PathLike]:
    """Yield a local path, downloading remote content when necessary."""
    with _backend(filepath, backend_args).get_local_path(filepath) as local_path:
        yield local_path


def list_dir_or_file(
    dir_path: PathLike,
    *,
    list_dir: bool = True,
    list_file: bool = True,
    suffix: str | tuple[str, ...] | None = None,
    recursive: bool = False,
    backend_args: dict[str, Any] | None = None,
) -> Iterator[str]:
    """Iterate over files and directories below a directory.

    Args:
        dir_path: Directory to scan.
        list_dir: Whether to include directories.
        list_file: Whether to include files.
        suffix: Optional filename suffix filter.
        recursive: Whether to scan nested directories.
        backend_args: Backend construction arguments.
    """
    return _backend(dir_path, backend_args).list_dir_or_file(
        dir_path,
        list_dir=list_dir,
        list_file=list_file,
        suffix=suffix,
        recursive=recursive,
    )


def copyfile(
    src: PathLike,
    dst: PathLike,
    *,
    src_backend_args: dict[str, Any] | None = None,
    dst_backend_args: dict[str, Any] | None = None,
) -> None:
    """Copy one file within or between storage backends."""
    content = get(src, backend_args=src_backend_args)
    put(
        content.tobytes() if isinstance(content, memoryview) else content,
        dst,
        backend_args=dst_backend_args,
    )


def copytree(
    src: PathLike,
    dst: PathLike,
    *,
    backend_args: dict[str, Any] | None = None,
) -> None:
    """Recursively copy a local directory tree."""
    backend = _backend(src, backend_args)
    if backend.name not in {"HardDiskBackend", "LocalBackend"}:
        raise NotImplementedError("copytree currently supports local storage only")
    shutil.copytree(src, dst, dirs_exist_ok=True)
