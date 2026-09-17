from collections.abc import Callable
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, TextIO

from ..utils import is_list_of
from .backend import get, get_text, put, put_text
from .handlers import BaseFileHandler, JsonHandler, PickleHandler, YamlHandler

FileLikeObject = TextIO | StringIO | BytesIO

file_handlers = {
    "json": JsonHandler(),
    "yaml": YamlHandler(),
    "yml": YamlHandler(),
    "pickle": PickleHandler(),
    "pkl": PickleHandler(),
}


def load(
    file: str | Path | FileLikeObject,
    file_format: str | None = None,
    backend_args: dict | None = None,
    **kwargs,
):
    """Load data from json/yaml/pickle files.

    This method provides a unified api for loading data from serialized files.

    Note:
        In v1.3.16 and later, ``load`` supports loading data from serialized
        files those can be storaged in different backends.

    Args:
        file (str or :obj:`Path` or file-like object): Filename or a file-like
            object.
        file_format (str, optional): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include "json", "yaml/yml" and
            "pickle/pkl".
        backend_args (dict, optional): Arguments selecting a storage backend.
            Default: None.

    Examples:
        >>> load('/path/of/your/file')  # file is storaged in disk
        >>> load('https://path/of/your/file')  # file is storaged in Internet

    Returns:
        The content from the file.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None and isinstance(file, str):
        file_format = file.split(".")[-1]
    if file_format not in file_handlers:
        raise TypeError(f"Unsupported format: {file_format}")

    handler = file_handlers[file_format]
    f: FileLikeObject
    if isinstance(file, str):
        if handler.str_like:
            with StringIO(get_text(file, backend_args=backend_args)) as f:
                obj = handler.load_from_fileobj(f, **kwargs)
        else:
            with BytesIO(get(file, backend_args=backend_args)) as f:
                obj = handler.load_from_fileobj(f, **kwargs)
    elif hasattr(file, "read"):
        obj = handler.load_from_fileobj(file, **kwargs)
    else:
        raise TypeError('"file" must be a filepath str or a file-object')
    return obj


def dump(
    obj: Any,
    file: str | Path | FileLikeObject | None = None,
    file_format: str | None = None,
    backend_args: dict | None = None,
    **kwargs,
):
    """Dump data to json/yaml/pickle strings or files.

    This method provides a unified api for dumping data as strings or to files,
    and also supports custom arguments for each file format.

    Note:
        In v1.3.16 and later, ``dump`` supports dumping data as strings or to
        files which is saved to different backends.

    Args:
        obj (any): The python object to be dumped.
        file (str or :obj:`Path` or file-like object, optional): If not
            specified, then the object is dumped to a str, otherwise to a file
            specified by the filename or file-like object.
        file_format (str, optional): Same as :func:`load`.
        backend_args (dict, optional): Arguments selecting a storage backend.
            Default: None.

    Examples:
        >>> dump('hello world', '/path/of/your/file')  # disk

    Returns:
        bool: True for success, False otherwise.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None:
        if isinstance(file, str):
            file_format = file.split(".")[-1]
        elif file is None:
            raise ValueError("file_format must be specified since file is None")
    if file_format not in file_handlers:
        raise TypeError(f"Unsupported format: {file_format}")
    f: FileLikeObject
    handler = file_handlers[file_format]
    if file is None:
        return handler.dump_to_str(obj, **kwargs)
    elif isinstance(file, str):
        if handler.str_like:
            with StringIO() as f:
                handler.dump_to_fileobj(obj, f, **kwargs)
                put_text(f.getvalue(), file, backend_args=backend_args)
        else:
            with BytesIO() as f:
                handler.dump_to_fileobj(obj, f, **kwargs)
                put(f.getvalue(), file, backend_args=backend_args)
    elif hasattr(file, "write"):
        handler.dump_to_fileobj(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')


def _register_handler(handler: BaseFileHandler, file_formats: str | list[str]) -> None:
    """Register a handler for some file extensions.

    Args:
        handler (:obj:`BaseFileHandler`): Handler to be registered.
        file_formats (str or list[str]): File formats to be handled by this
            handler.
    """
    if isinstance(file_formats, str):
        file_formats = [file_formats]
    if not is_list_of(file_formats, str):
        raise TypeError("file_formats must be a str or a list of str")
    for ext in file_formats:
        file_handlers[ext] = handler


def register_handler(file_formats: str | list, **kwargs) -> Callable:

    def wrap(cls):
        _register_handler(cls(**kwargs), file_formats)
        return cls

    return wrap
