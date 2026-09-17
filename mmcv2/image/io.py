import os.path as osp
from pathlib import Path

import cv2
import numpy as np
from cv2 import (
    IMREAD_COLOR,
    IMREAD_GRAYSCALE,
    IMREAD_IGNORE_ORIENTATION,
    IMREAD_UNCHANGED,
)

from mmcv2.fileio import get, put
from mmcv2.utils import is_filepath

imread_flags: dict[str, int] = {
    "color": IMREAD_COLOR,
    "grayscale": IMREAD_GRAYSCALE,
    "unchanged": IMREAD_UNCHANGED,
    "color_ignore_orientation": IMREAD_IGNORE_ORIENTATION | IMREAD_COLOR,
    "grayscale_ignore_orientation": IMREAD_IGNORE_ORIENTATION | IMREAD_GRAYSCALE,
}


def imread(
    img_or_path: np.ndarray | str | Path,
    flag: str | int = "color",
    channel_order: str = "bgr",
    backend_args: dict | None = None,
) -> np.ndarray:
    """Read an image.

    Args:
        img_or_path (ndarray or str or Path): Either a numpy array or str or
            pathlib.Path. If it is a numpy array (loaded image), then
            it will be returned as is.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale`, `unchanged`,
            `color_ignore_orientation` and `grayscale_ignore_orientation`.
            By default, OpenCV rotates the image according to its EXIF info
            unless called with `unchanged` or `*_ignore_orientation` flags.
        channel_order (str): Order of channel, candidates are `bgr` and `rgb`.
        backend_args (dict | None): Arguments selecting a storage backend.
            Backend selection arguments for :func:`mmcv2.fileio.get`.
            Default: None.

    Returns:
        ndarray: Loaded image array.

    Examples:
        >>> import mmcv2
        >>> img_path = '/path/to/img.jpg'
        >>> img = mmcv2.imread(img_path)
        >>> img = mmcv2.imread(img_path, flag='color', channel_order='rgb')
        >>> http_img_path = 'http://path/to/img.jpg'
        >>> img = mmcv2.imread(http_img_path)
        >>> img = mmcv2.imread(http_img_path, backend_args={
        ...     'backend': 'http'})
    """

    if isinstance(img_or_path, Path):
        img_or_path = str(img_or_path)

    if isinstance(img_or_path, np.ndarray):
        return img_or_path
    img_bytes = get(img_or_path, backend_args=backend_args)
    if isinstance(img_bytes, np.ndarray):
        return img_bytes
    return imfrombytes(img_bytes, flag, channel_order)


def imfrombytes(
    content: bytes | bytearray | memoryview,
    flag: str | int = "color",
    channel_order: str = "bgr",
) -> np.ndarray:
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Same as :func:`imread`.
        channel_order (str): The channel order of the output, candidates
            are 'bgr' and 'rgb'. Default to 'bgr'.
    Returns:
        ndarray: Loaded image array.

    Examples:
        >>> img_path = '/path/to/img.jpg'
        >>> with open(img_path, 'rb') as f:
        >>>     img_buff = f.read()
        >>> img = mmcv2.imfrombytes(img_buff)
        >>> img = mmcv2.imfrombytes(img_buff, flag='color', channel_order='rgb')
    """
    if channel_order not in ("bgr", "rgb"):
        raise ValueError('channel_order must be either "bgr" or "rgb"')
    img_np = np.frombuffer(content, np.uint8)
    flag_value = imread_flags[flag] if isinstance(flag, str) else flag
    img = cv2.imdecode(img_np, flag_value)
    if not isinstance(img, np.ndarray):
        raise ValueError("Failed to decode image bytes")
    if flag_value == IMREAD_COLOR and channel_order == "rgb":
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    return img


def imwrite(
    img: np.ndarray,
    file_path: str,
    params: list[int] | None = None,
    backend_args: dict | None = None,
) -> bool:
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv :func:`imwrite` interface.
        backend_args (dict | None): Arguments selecting a storage backend.
            Backend selection arguments for :func:`mmcv2.fileio.put`.
            Default: None.

    Returns:
        bool: Successful or not.

    Examples:
        >>> # write to hard disk client
        >>> ret = mmcv2.imwrite(img, '/path/to/img.jpg')
    """
    assert is_filepath(file_path)
    file_path = str(file_path)
    img_ext = osp.splitext(file_path)[-1]
    # Encode image according to image suffix.
    # For example, if image path is '/path/your/img.jpg', the encode
    # format is '.jpg'.
    if params is None:
        flag, img_buff = cv2.imencode(img_ext, img)
    else:
        flag, img_buff = cv2.imencode(img_ext, img, params)
    put(img_buff.tobytes(), file_path, backend_args=backend_args)
    return flag
