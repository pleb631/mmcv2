from ._version import __version__, parse_version
from .fileio import *
from .image import *
from .utils import *

version_info = tuple(int(x) for x in __version__.split(".")[:3])


def parse_version_info(version_str: str, length: int = 4) -> tuple[int | str, ...]:
    """Parse a version string into a padded tuple."""
    version, pre, pre_number, post = parse_version(version_str)
    release: list[int | str] = list(version[:length])
    release.extend([0] * (length - len(release)))
    if pre:
        release.extend([pre, pre_number])
    elif post is not None:
        release.extend(["post", post])
    else:
        release.extend([0, 0])
    return tuple(release)
