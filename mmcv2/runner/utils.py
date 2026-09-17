import os
import random
import time
import warnings
from getpass import getuser
from socket import gethostname

import numpy as np
import torch

from ..distributed import get_dist_info


def get_host_info() -> str:
    """Get hostname and username.

    Return empty string if exception raised, e.g. ``getpass.getuser()`` will
    lead to error in docker container
    """
    host = ""
    try:
        host = f"{getuser()}@{gethostname()}"
    except Exception as e:
        warnings.warn(f"Host or user not found: {e!s}")
    finally:
        return host


def get_time_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def set_random_seed(seed: int, deterministic: bool = False, use_rank_shift: bool = False) -> None:
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
        rank_shift (bool): Whether to add rank number to the random seed to
            have different random seed in different threads. Default: False.
    """
    if use_rank_shift:
        rank, _ = get_dist_info()
        seed += rank
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed seeds the CPU and every available accelerator.
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
