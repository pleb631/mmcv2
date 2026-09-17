from abc import ABCMeta, abstractmethod


class BaseFileHandler(metaclass=ABCMeta):
    """Define serialization operations for a file format."""

    # `str_like` is a flag to indicate whether the type of file object is
    # str-like object or bytes-like object. Pickle only processes bytes-like
    # objects but json only processes str-like object. If it is str-like
    # object, `StringIO` will be used to process the buffer.
    str_like = True

    @abstractmethod
    def load_from_fileobj(self, file, **kwargs):
        """Load an object from an open file object."""
        pass

    @abstractmethod
    def dump_to_fileobj(self, obj, file, **kwargs):
        """Serialize an object to an open file object."""
        pass

    @abstractmethod
    def dump_to_str(self, obj, **kwargs):
        """Serialize an object to bytes or text."""
        pass

    def load_from_path(self, filepath: str, mode: str = "r", **kwargs):
        """Load an object from a local path.

        Args:
            filepath: Local file path.
            mode: File-open mode.
            **kwargs: Format-specific loading options.
        """
        with open(filepath, mode) as f:
            return self.load_from_fileobj(f, **kwargs)

    def dump_to_path(self, obj, filepath: str, mode: str = "w", **kwargs):
        """Serialize an object to a local path."""
        with open(filepath, mode) as f:
            self.dump_to_fileobj(obj, f, **kwargs)
