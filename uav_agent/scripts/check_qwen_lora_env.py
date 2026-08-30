#!/usr/bin/env python3
"""Check the isolated, offline Qwen3-VL LoRA training environment."""

from __future__ import annotations

import argparse
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.lora.modeling import validate_local_model_directory  # noqa: E402


DEFAULT_MODEL = Path(
    "/home/amax/ry/vlm_drones/models/initial_model/Qwen3-VL-4B-Instruct"
)
_PACKAGES = (
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("transformers", "transformers"),
    ("peft", "peft"),
    ("accelerate", "accelerate"),
    ("safetensors", "safetensors"),
    ("Pillow", "PIL"),
    ("tensorboard", "tensorboard"),
    ("pytest", "pytest"),
)


def _package_report() -> tuple[dict[str, object], dict[str, object]]:
    modules: dict[str, object] = {}
    report: dict[str, object] = {}
    for distribution, module_name in _PACKAGES:
        item: dict[str, object] = {"installed": False, "importable": False}
        try:
            item["version"] = metadata.version(distribution)
            item["installed"] = True
        except metadata.PackageNotFoundError:
            item["version"] = None
        try:
            module = importlib.import_module(module_name)
            modules[module_name] = module
            item["importable"] = True
        except Exception as exc:  # environment checker must report binary/import failures
            item["import_error"] = f"{type(exc).__name__}: {exc}"
        report[distribution] = item
    return report, modules


def check_environment(
    model_path: str | Path = DEFAULT_MODEL, *, require_bf16: bool = False
) -> dict[str, object]:
    """Return a JSON-safe environment report; never load model weights."""

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    package_report, modules = _package_report()
    errors: list[str] = []
    if sys.version_info[:2] != (3, 11):
        errors.append(
            f"Python 3.11 is required, found {platform.python_version()}"
        )
    for name, item in package_report.items():
        if not item["installed"] or not item["importable"]:
            errors.append(f"required package is unavailable: {name}")

    qwen_class = False
    transformers = modules.get("transformers")
    if transformers is not None:
        try:
            getattr(transformers, "Qwen3VLForConditionalGeneration")
            getattr(transformers, "AutoProcessor")
            qwen_class = True
        except (AttributeError, ImportError, RuntimeError) as exc:
            errors.append(
                "Transformers lacks usable Qwen3VLForConditionalGeneration/AutoProcessor: "
                f"{exc}"
            )

    cuda_available = False
    gpu_names: list[str] = []
    bf16_supported = False
    torch = modules.get("torch")
    if torch is not None:
        try:
            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                gpu_names = [
                    str(torch.cuda.get_device_name(index))
                    for index in range(torch.cuda.device_count())
                ]
                bf16_supported = bool(torch.cuda.is_bf16_supported())
        except Exception as exc:
            errors.append(f"CUDA probe failed: {type(exc).__name__}: {exc}")
    if not cuda_available:
        errors.append("CUDA GPU is required for real Qwen3-VL LoRA training")
    if require_bf16 and not bf16_supported:
        errors.append("--require-bf16 was requested but the visible GPU lacks bf16")

    model = validate_local_model_directory(model_path)
    if not model["complete"]:
        errors.append("local Qwen3-VL model directory is incomplete")
    return {
        "ok": not errors,
        "offline": {
            "HF_HUB_OFFLINE": os.environ["HF_HUB_OFFLINE"],
            "TRANSFORMERS_OFFLINE": os.environ["TRANSFORMERS_OFFLINE"],
        },
        "python": {
            "version": platform.python_version(),
            "required_major_minor": "3.11",
        },
        "packages": package_report,
        "qwen3_vl_model_class_available": qwen_class,
        "cuda": {
            "available": cuda_available,
            "gpu_count": len(gpu_names),
            "gpu_names": gpu_names,
            "bf16_supported": bf16_supported,
        },
        "local_model": model,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--require-bf16",
        action="store_true",
        help="fail when the visible CUDA device cannot execute bf16",
    )
    args = parser.parse_args(argv)
    report = check_environment(args.model, require_bf16=args.require_bf16)
    print(
        json.dumps(
            report, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
        )
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
