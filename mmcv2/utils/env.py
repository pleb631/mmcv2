import os.path as osp
import platform
import subprocess
import sys
import sysconfig
from collections import defaultdict

import cv2
import torch
import torchvision
from torch.utils.cpp_extension import CUDA_HOME

from mmcv2._version import __version__


def collect_env():
    """Collect the information of the running environments.

    Returns:
        dict: The environment information. The following fields are contained.

            - sys.platform: The variable of ``sys.platform``.
            - Python: Python version.
            - CUDA available: Bool, indicating if CUDA is available.
            - GPU devices: Device type of each GPU.
            - CUDA_HOME (optional): The env var ``CUDA_HOME``.
            - NVCC (optional): NVCC version.
            - GCC: GCC version, "n/a" if GCC is not installed.
            - MSVC: Microsoft Virtual C++ Compiler version, Windows only.
            - PyTorch: PyTorch version.
            - PyTorch compiling details: The output of \
                ``torch.__config__.show()``.
            - TorchVision: TorchVision version.
            - OpenCV: OpenCV version.
            - MMCV2: MMCV2 version.
    """
    env_info = {}
    env_info["sys.platform"] = sys.platform
    env_info["Python"] = sys.version.replace("\n", "")

    cuda_available = torch.cuda.is_available()
    env_info["CUDA available"] = cuda_available

    if cuda_available:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, device_ids in devices.items():
            env_info["GPU " + ",".join(device_ids)] = name

        env_info["CUDA_HOME"] = CUDA_HOME

        if CUDA_HOME is not None and osp.isdir(CUDA_HOME):
            try:
                nvcc = osp.join(CUDA_HOME, "bin/nvcc")
                nvcc = subprocess.check_output(f'"{nvcc}" -V', shell=True)
                nvcc = nvcc.decode("utf-8").strip()
                release = nvcc.rfind("Cuda compilation tools")
                build = nvcc.rfind("Build ")
                nvcc = nvcc[release:build].strip()
            except subprocess.SubprocessError:
                nvcc = "Not Available"
            env_info["NVCC"] = nvcc

    try:
        # Check C++ Compiler.
        # For Unix-like, sysconfig has 'CC' variable like 'gcc -pthread ...',
        # indicating the compiler used, we use this to get the compiler name
        cc = sysconfig.get_config_var("CC")
        if cc:
            cc = osp.basename(cc.split()[0])
            cc_info = subprocess.check_output(f"{cc} --version", shell=True)
            env_info["GCC"] = cc_info.decode("utf-8").partition("\n")[0].strip()
        else:
            env_info["MSVC"] = platform.python_compiler()
            env_info["GCC"] = "n/a"
    except subprocess.CalledProcessError:
        env_info["GCC"] = "n/a"

    env_info["PyTorch"] = torch.__version__
    env_info["PyTorch compiling details"] = torch.__config__.show()

    env_info["TorchVision"] = torchvision.__version__

    env_info["OpenCV"] = cv2.__version__

    env_info["MMCV2"] = __version__

    return env_info
