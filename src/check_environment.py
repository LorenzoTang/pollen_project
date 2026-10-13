from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from typing import Any, Dict


REQUIRED_MODULES = ("yaml", "PIL", "numpy", "torch", "torchvision", "rfdetr")


def build_environment_report() -> Dict[str, Any]:
    modules = {name: importlib.util.find_spec(name) is not None for name in REQUIRED_MODULES}
    report: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "modules": modules,
        "torch": {},
        "status": "passed" if all(modules.values()) else "failed",
    }
    if modules["torch"]:
        import torch

        cuda_available = torch.cuda.is_available()
        report["torch"] = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": cuda_available,
            "device_count": torch.cuda.device_count(),
            "devices": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ],
        }
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Check the RF-DETR training environment.")
    parser.add_argument(
        "--require-cuda", action="store_true", help="Fail unless PyTorch can use an NVIDIA GPU."
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    report = build_environment_report()
    if args.require_cuda and not report.get("torch", {}).get("cuda_available", False):
        report["status"] = "failed"
        report["error"] = "CUDA-enabled PyTorch and an accessible NVIDIA GPU are required."
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())