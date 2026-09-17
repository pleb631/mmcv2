import pickle

from .base import BaseFileHandler


class PickleHandler(BaseFileHandler):
    """Serialize Python objects with :mod:`pickle`."""

    str_like = False

    def load_from_fileobj(self, file, **kwargs):
        """Load an object from an open binary file."""
        return pickle.load(file, **kwargs)

    def load_from_path(self, filepath, **kwargs):
        """Load an object from a local binary path."""
        return super().load_from_path(filepath, mode="rb", **kwargs)

    def dump_to_str(self, obj, **kwargs):
        """Return the pickle byte representation of ``obj``."""
        kwargs.setdefault("protocol", 2)
        return pickle.dumps(obj, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        """Write a pickle representation to an open binary file."""
        kwargs.setdefault("protocol", 2)
        pickle.dump(obj, file, **kwargs)

    def dump_to_path(self, obj, filepath, **kwargs):
        """Write a pickle representation to a local binary path."""
        super().dump_to_path(obj, filepath, mode="wb", **kwargs)
