import re

__version__ = "0.1.0"

_VERSION_RE = re.compile(
    r"^v?(?P<release>\d+(?:\.\d+)*)"
    r"(?:(?P<pre>a|b|rc)(?P<pre_number>\d*)|"
    r"(?P<dev>dev)(?P<dev_number>\d*))?"
    r"(?:post(?P<post_number>\d*))?"
    r"(?:\+[0-9A-Za-z._-]+)?$"
)


def parse_version(
    version_str: str,
) -> tuple[tuple[int, ...], str | None, int, int | None]:
    """Parse the PEP 440 forms supported by MMCV2 version comparisons."""
    match = _VERSION_RE.fullmatch(version_str)
    if match is None:
        raise ValueError(f"Invalid version: {version_str!r}")

    release = tuple(int(part) for part in match["release"].split("."))
    if match["dev"]:
        pre, pre_number = "dev", int(match["dev_number"] or 0)
    elif match["pre"]:
        pre, pre_number = match["pre"], int(match["pre_number"] or 0)
    else:
        pre, pre_number = None, 0
    post = match["post_number"]
    return release, pre, pre_number, int(post or 0) if post is not None else None
