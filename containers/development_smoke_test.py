import argparse
import json
import os
import platform

import lightning
import torch
import torchvision
import transformers
import wandb


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    gpu_available = torch.cuda.is_available()
    result = {
        "python": platform.python_version(),
        "container_cuda": os.environ.get("CUDA_VERSION"),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torchvision": torchvision.__version__,
        "lightning": lightning.__version__,
        "transformers": transformers.__version__,
        "wandb": wandb.__version__,
        "uid": os.getuid(),
        "gpu": {
            "available": gpu_available,
            "name": torch.cuda.get_device_name(0) if gpu_available else None,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if platform.python_version_tuple()[:2] != ("3", "13"):
        print("error: development image requires Python 3.13")
        return 1
    if os.getuid() == 0:
        print("error: development image must not run as root")
        return 1
    if os.environ.get("CUDA_VERSION") != "13.0.2":
        print("error: development image requires CUDA 13.0.2")
        return 1
    if torch.version.cuda != "13.0":
        print("error: PyTorch must use its CUDA 13.0 build")
        return 1
    if args.require_gpu and not gpu_available:
        print("error: no CUDA GPU is visible")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
