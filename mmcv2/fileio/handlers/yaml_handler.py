import yaml

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader  # type: ignore

from .base import BaseFileHandler  # isort:skip


class YamlHandler(BaseFileHandler):
    """Serialize YAML values using the fastest available PyYAML loader."""

    def load_from_fileobj(self, file, **kwargs):
        """Load a YAML value from an open text file."""
        kwargs.setdefault("Loader", Loader)
        return yaml.load(file, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        """Write a YAML value to an open text file."""
        kwargs.setdefault("Dumper", Dumper)
        yaml.dump(obj, file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        """Return a YAML serialization of ``obj``."""
        kwargs.setdefault("Dumper", Dumper)
        return yaml.dump(obj, **kwargs)
