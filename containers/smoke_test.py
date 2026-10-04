import argparse
import json
import os
import platform
import shutil
import subprocess


def nvidia_status() -> dict[str, object]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "reason": "nvidia-smi not found"}

    result = subprocess.run(
        [
            executable,
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return {
            "available": False,
            "reason": result.stderr.strip() or f"nvidia-smi exited {result.returncode}",
        }

    devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {"available": bool(devices), "devices": devices}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the seg-opt container runtime"
    )
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    gpu = nvidia_status()
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "uid": os.getuid(),
        "gpu": gpu,
    }
    print(json.dumps(report, indent=2))

    if os.getuid() == 0:
        print("error: container must not run as root")
        return 1
    if args.require_gpu and not gpu["available"]:
        print("error: no NVIDIA GPU is visible")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
