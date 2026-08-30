"""Offline Qwen3-VL loading and fail-closed language-only PEFT wiring.

Heavy dependencies are imported only inside the loader functions so ordinary
unit tests can exercise target safety with tiny Python fakes.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Iterable

from training.lora.config import LoraScaffoldConfig, LoraScaffoldError


class LoraModelError(RuntimeError):
    """Raised when offline loading or LoRA isolation cannot be proven safe."""


LANGUAGE_ATTENTION = "language_attention"
LANGUAGE_MLP = "language_mlp"
VISION = "vision"
CONNECTOR = "connector"
OTHER = "other"

_SAFE_TARGET_PATTERN = re.compile(
    r"^model\.language_model\.[A-Za-z0-9_]+(?:\.(?:[A-Za-z0-9_]+|\*))*$"
)
_CONNECTOR_MARKERS = ("connector", "merger", "projector")
_VISION_PARTS = frozenset({"visual", "vision", "vision_tower", "vision_model", "vit"})


@dataclass(frozen=True, slots=True)
class ModuleInventory:
    language_attention_modules: tuple[str, ...]
    language_mlp_modules: tuple[str, ...]
    vision_modules: tuple[str, ...]
    connector_modules: tuple[str, ...]
    other_linear_modules: tuple[str, ...]

    @property
    def language_candidates(self) -> tuple[str, ...]:
        return tuple(
            sorted(self.language_attention_modules + self.language_mlp_modules)
        )

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "language_attention_modules": list(self.language_attention_modules),
            "language_mlp_modules": list(self.language_mlp_modules),
            "language_candidates": list(self.language_candidates),
            "vision_modules": list(self.vision_modules),
            "connector_modules": list(self.connector_modules),
            "other_linear_modules": list(self.other_linear_modules),
        }


@dataclass(frozen=True, slots=True)
class TargetModuleReport:
    requested_patterns: tuple[str, ...]
    matched_language_modules: tuple[str, ...]
    matched_vision_modules: tuple[str, ...]
    matched_connector_modules: tuple[str, ...]
    matched_unsupported_modules: tuple[str, ...]
    unmatched_patterns: tuple[str, ...]

    @property
    def target_module_count(self) -> int:
        return len(self.matched_language_modules)

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_patterns": list(self.requested_patterns),
            "matched_language_modules": list(self.matched_language_modules),
            "matched_vision_modules": list(self.matched_vision_modules),
            "matched_connector_modules": list(self.matched_connector_modules),
            "matched_unsupported_modules": list(self.matched_unsupported_modules),
            "unmatched_patterns": list(self.unmatched_patterns),
            "target_module_count": self.target_module_count,
        }


@dataclass(frozen=True, slots=True)
class ParameterStats:
    total_parameters: int
    trainable_parameters: int

    @property
    def trainable_percentage(self) -> float:
        if self.total_parameters == 0:
            return 0.0
        return 100.0 * self.trainable_parameters / self.total_parameters

    def to_dict(self) -> dict[str, int | float]:
        return {
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "trainable_percentage": self.trainable_percentage,
        }


@dataclass(frozen=True, slots=True)
class BaseModelBundle:
    model: Any
    processor: Any


@dataclass(frozen=True, slots=True)
class QwenLoraModelBundle:
    model: Any
    processor: Any
    target_report: TargetModuleReport
    parameter_stats: ParameterStats


def classify_module_name(name: str) -> str:
    """Classify a full module/parameter path, checking deny-listed areas first."""

    parts = tuple(part.lower() for part in name.split(".") if part)
    if any(
        marker in part for part in parts for marker in _CONNECTOR_MARKERS
    ):
        return CONNECTOR
    if any(part in _VISION_PARTS or part.startswith("vision_") for part in parts):
        return VISION
    if "language_model" in parts and any(
        part in {"self_attn", "attention", "attn"} for part in parts
    ):
        return LANGUAGE_ATTENTION
    if "language_model" in parts and any(
        part in {"mlp", "feed_forward"} for part in parts
    ):
        return LANGUAGE_MLP
    return OTHER


def _is_linear(module: object) -> bool:
    # This exact class-name check recognizes torch.nn.Linear without importing
    # torch at module import time, and makes tiny dependency-free fakes possible.
    return module.__class__.__name__ == "Linear"


def inspect_model_linear_modules(model: object) -> ModuleInventory:
    """Inventory actual Linear module names exposed by ``model.named_modules``."""

    try:
        named_modules = model.named_modules()
    except AttributeError as exc:
        raise LoraModelError("model must provide named_modules()") from exc
    buckets: dict[str, list[str]] = {
        LANGUAGE_ATTENTION: [],
        LANGUAGE_MLP: [],
        VISION: [],
        CONNECTOR: [],
        OTHER: [],
    }
    for name, module in named_modules:
        if _is_linear(module):
            buckets[classify_module_name(name)].append(name)
    return ModuleInventory(
        language_attention_modules=tuple(sorted(buckets[LANGUAGE_ATTENTION])),
        language_mlp_modules=tuple(sorted(buckets[LANGUAGE_MLP])),
        vision_modules=tuple(sorted(buckets[VISION])),
        connector_modules=tuple(sorted(buckets[CONNECTOR])),
        other_linear_modules=tuple(sorted(buckets[OTHER])),
    )


def _validate_target_pattern(pattern: object) -> str:
    if not isinstance(pattern, str) or not pattern.strip():
        raise LoraModelError("LoRA target pattern must be non-empty text")
    result = pattern.strip()
    if (
        not _SAFE_TARGET_PATTERN.fullmatch(result)
        or ".." in result
        or "**" in result
    ):
        raise LoraModelError(
            "unsafe LoRA target pattern; use a fully qualified language-only name "
            "or anchored glob beginning with 'model.language_model.': "
            f"{result!r}"
        )
    return result


def audit_lora_target_modules(
    model: object, target_patterns: Iterable[str]
) -> TargetModuleReport:
    """Expand safe patterns against actual module names without applying PEFT."""

    requested = tuple(_validate_target_pattern(item) for item in target_patterns)
    if not requested:
        raise LoraModelError("target_modules must not be empty")
    if len(requested) != len(set(requested)):
        raise LoraModelError("target_modules contains duplicates")

    inventory = inspect_model_linear_modules(model)
    category_by_name: dict[str, str] = {}
    for category, names in (
        (LANGUAGE_ATTENTION, inventory.language_attention_modules),
        (LANGUAGE_MLP, inventory.language_mlp_modules),
        (VISION, inventory.vision_modules),
        (CONNECTOR, inventory.connector_modules),
        (OTHER, inventory.other_linear_modules),
    ):
        category_by_name.update((name, category) for name in names)

    matchers = {
        pattern: re.compile(
            r"^" + re.escape(pattern).replace(r"\*", r"[^.]+") + r"$"
        )
        for pattern in requested
    }
    matches_by_pattern: dict[str, list[str]] = {
        pattern: [
            name for name in category_by_name if matcher.fullmatch(name) is not None
        ]
        for pattern, matcher in matchers.items()
    }
    matched_names = sorted(
        {name for names in matches_by_pattern.values() for name in names}
    )

    return TargetModuleReport(
        requested_patterns=requested,
        matched_language_modules=tuple(
            name
            for name in matched_names
            if category_by_name[name] in {LANGUAGE_ATTENTION, LANGUAGE_MLP}
        ),
        matched_vision_modules=tuple(
            name for name in matched_names if category_by_name[name] == VISION
        ),
        matched_connector_modules=tuple(
            name for name in matched_names if category_by_name[name] == CONNECTOR
        ),
        matched_unsupported_modules=tuple(
            name for name in matched_names if category_by_name[name] == OTHER
        ),
        unmatched_patterns=tuple(
            pattern for pattern, names in matches_by_pattern.items() if not names
        ),
    )


def validate_lora_target_modules(
    model: object, target_patterns: Iterable[str]
) -> TargetModuleReport:
    """Require non-empty, exact language-only expansion for every target pattern."""

    report = audit_lora_target_modules(model, target_patterns)
    errors: list[str] = []
    if not report.matched_language_modules:
        errors.append("matched language target modules must be greater than zero")
    if report.matched_vision_modules:
        errors.append(
            "vision target modules are forbidden: "
            + ", ".join(report.matched_vision_modules)
        )
    if report.matched_connector_modules:
        errors.append(
            "connector/merger target modules are forbidden: "
            + ", ".join(report.matched_connector_modules)
        )
    if report.matched_unsupported_modules:
        errors.append(
            "only language attention/MLP Linear modules may be targeted: "
            + ", ".join(report.matched_unsupported_modules)
        )
    if report.unmatched_patterns:
        errors.append(
            "target patterns matched zero Linear modules: "
            + ", ".join(report.unmatched_patterns)
        )
    if errors:
        raise LoraModelError("; ".join(errors))
    return report


def exact_target_regex(module_names: Iterable[str]) -> str:
    """Build a single anchored PEFT regex from already validated full names."""

    names = tuple(sorted(set(module_names)))
    if not names:
        raise LoraModelError("cannot build a PEFT target regex from zero modules")
    return r"^(?:" + "|".join(re.escape(name) for name in names) + r")$"


def _parameter_count(parameter: object) -> int:
    try:
        count = parameter.numel()
    except AttributeError as exc:
        raise LoraModelError("model parameter does not provide numel()") from exc
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise LoraModelError("model parameter returned an invalid numel()")
    return count


def freeze_all_parameters(model: object) -> None:
    """Freeze every base parameter before PEFT creates adapter parameters."""

    try:
        parameters = model.parameters()
    except AttributeError as exc:
        raise LoraModelError("model must provide parameters()") from exc
    for parameter in parameters:
        parameter.requires_grad = False


def verify_lora_trainable_parameters(model: object) -> ParameterStats:
    """Prove every trainable parameter is a language LoRA parameter."""

    try:
        named_parameters = model.named_parameters()
    except AttributeError as exc:
        raise LoraModelError("model must provide named_parameters()") from exc
    total = 0
    trainable = 0
    violations: list[str] = []
    for name, parameter in named_parameters:
        count = _parameter_count(parameter)
        total += count
        if not bool(getattr(parameter, "requires_grad", False)):
            continue
        trainable += count
        lowered_parts = tuple(part.lower() for part in name.split("."))
        is_lora = any(part.startswith("lora_") for part in lowered_parts)
        category = classify_module_name(name)
        if not is_lora:
            violations.append(f"non-LoRA base parameter is trainable: {name}")
        if category == VISION:
            violations.append(f"vision parameter is trainable: {name}")
        elif category == CONNECTOR:
            violations.append(f"connector/merger parameter is trainable: {name}")
        elif category not in {LANGUAGE_ATTENTION, LANGUAGE_MLP}:
            violations.append(f"LoRA parameter is outside the language backbone: {name}")
    if violations:
        raise LoraModelError("; ".join(violations))
    if trainable <= 0:
        raise LoraModelError("LoRA injection produced zero trainable parameters")
    return ParameterStats(total_parameters=total, trainable_parameters=trainable)


def verify_peft_targeted_modules(
    peft_model: object, expected_names: tuple[str, ...]
) -> tuple[str, ...]:
    """Verify PEFT's own target audit agrees with our pre-injection inventory."""

    raw = getattr(peft_model, "targeted_module_names", None)
    if raw is None:
        raise LoraModelError(
            "installed PEFT does not expose targeted_module_names; cannot prove safe injection"
        )
    actual = tuple(str(name) for name in raw)
    if not actual:
        raise LoraModelError("PEFT reported zero targeted modules")
    mapped: set[str] = set()
    unexpected: list[str] = []
    for name in actual:
        candidates = tuple(
            expected
            for expected in expected_names
            if name == expected or name.endswith("." + expected)
        )
        if len(candidates) != 1:
            unexpected.append(name)
            continue
        expected = candidates[0]
        if classify_module_name(name) not in {LANGUAGE_ATTENTION, LANGUAGE_MLP}:
            unexpected.append(name)
            continue
        mapped.add(expected)
    missing = sorted(set(expected_names) - mapped)
    if unexpected or missing:
        raise LoraModelError(
            "PEFT target audit disagrees with validated language modules; "
            f"unexpected={sorted(unexpected)}, missing={missing}"
        )
    return tuple(sorted(actual))


def validate_local_model_directory(model_path: str | Path) -> dict[str, object]:
    """Validate a local Qwen3-VL checkpoint without importing Transformers."""

    path = Path(model_path).expanduser().resolve()
    errors: list[str] = []
    if not path.is_dir():
        return {"path": str(path), "complete": False, "errors": ["directory does not exist"]}

    required = ("config.json", "tokenizer_config.json", "preprocessor_config.json")
    for filename in required:
        if not (path / filename).is_file():
            errors.append(f"missing {filename}")
    if not (path / "tokenizer.json").is_file() and not (
        (path / "vocab.json").is_file() and (path / "merges.txt").is_file()
    ):
        errors.append("missing tokenizer.json or vocab.json + merges.txt")

    architecture: str | None = None
    config_path = path / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            architectures = config.get("architectures", [])
            if isinstance(architectures, list) and architectures:
                architecture = str(architectures[0])
            if architecture != "Qwen3VLForConditionalGeneration":
                errors.append(
                    "config architecture must be Qwen3VLForConditionalGeneration"
                )
        except (OSError, UnicodeError, json.JSONDecodeError, AttributeError) as exc:
            errors.append(f"invalid config.json: {exc}")

    index_path = path / "model.safetensors.index.json"
    weight_files: set[str] = set()
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                errors.append("model.safetensors.index.json has no weight_map")
            else:
                invalid_filenames = [
                    filename
                    for filename in weight_map.values()
                    if not isinstance(filename, str)
                    or not filename
                    or Path(filename).name != filename
                ]
                if invalid_filenames:
                    errors.append("weight_map contains invalid or unsafe filenames")
                weight_files = {
                    filename
                    for filename in weight_map.values()
                    if isinstance(filename, str)
                    and filename
                    and Path(filename).name == filename
                }
                for filename in sorted(weight_files):
                    candidate = path / filename
                    if not candidate.is_file():
                        errors.append(f"missing weight shard {filename!r}")
        except (OSError, UnicodeError, json.JSONDecodeError, AttributeError) as exc:
            errors.append(f"invalid model.safetensors.index.json: {exc}")
    else:
        weight_files = {
            candidate.name
            for candidate in path.glob("*.safetensors")
            if candidate.is_file() and not candidate.name.startswith("adapter_model")
        }
        if not weight_files:
            errors.append("missing local model safetensors weights")
    return {
        "path": str(path),
        "complete": not errors,
        "architecture": architecture,
        "weight_files": sorted(weight_files),
        "errors": errors,
    }


def _offline_environment() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def load_qwen_base_model(
    base_model_path: str | Path,
    *,
    bf16: bool = False,
    gradient_checkpointing: bool = False,
) -> BaseModelBundle:
    """Load the official Qwen3-VL model and processor from local files only."""

    _offline_environment()
    validation = validate_local_model_directory(base_model_path)
    if not validation["complete"]:
        raise LoraModelError(
            "local Qwen3-VL model directory is incomplete: "
            + "; ".join(str(item) for item in validation["errors"])
        )
    try:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except (ImportError, RuntimeError) as exc:
        raise LoraModelError(
            "qwen_lora environment requires torch and a Transformers build with "
            "Qwen3VLForConditionalGeneration"
        ) from exc

    path = str(validation["path"])
    dtype = torch.bfloat16 if bf16 else torch.float32
    try:
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            path,
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    except Exception as exc:  # third-party loaders use several exception types
        raise LoraModelError(f"could not load local Qwen3-VL checkpoint {path}: {exc}") from exc
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "config"):
            model.config.use_cache = False
    return BaseModelBundle(model=model, processor=processor)


def load_qwen_lora_model(
    config: LoraScaffoldConfig,
    *,
    reporter: Callable[[str], object] | None = print,
) -> QwenLoraModelBundle:
    """Load local Qwen3-VL, inject exact language LoRA, and verify isolation."""

    config.require_active()
    if config.target_modules is None:
        raise LoraScaffoldError("active config requires target_modules")
    if config.rank is None or config.lora_alpha is None or config.lora_dropout is None:
        raise LoraScaffoldError("active config requires rank/alpha/dropout")

    base = load_qwen_base_model(
        config.base_model_path,
        bf16=bool(config.bf16),
        gradient_checkpointing=bool(config.gradient_checkpointing),
    )
    tokenizer = getattr(base.processor, "tokenizer", None)
    if tokenizer is not None and config.model_max_length is not None:
        tokenizer.model_max_length = config.model_max_length

    target_report = validate_lora_target_modules(base.model, config.target_modules)
    exact_regex = exact_target_regex(target_report.matched_language_modules)
    freeze_all_parameters(base.model)
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except (ImportError, RuntimeError) as exc:
        raise LoraModelError("peft is required in the separate qwen_lora environment") from exc
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=exact_regex,
    )
    try:
        model = get_peft_model(base.model, peft_config)
    except Exception as exc:
        raise LoraModelError(f"PEFT LoRA injection failed: {exc}") from exc

    verify_peft_targeted_modules(model, target_report.matched_language_modules)
    stats = verify_lora_trainable_parameters(model)
    if reporter is not None:
        reporter(f"total parameters: {stats.total_parameters}")
        reporter(f"trainable parameters: {stats.trainable_parameters}")
        reporter(f"trainable percentage: {stats.trainable_percentage:.6f}%")
        reporter(f"LoRA target module count: {target_report.target_module_count}")
    return QwenLoraModelBundle(
        model=model,
        processor=base.processor,
        target_report=target_report,
        parameter_stats=stats,
    )


__all__ = [
    "BaseModelBundle",
    "CONNECTOR",
    "LANGUAGE_ATTENTION",
    "LANGUAGE_MLP",
    "LoraModelError",
    "ModuleInventory",
    "OTHER",
    "ParameterStats",
    "QwenLoraModelBundle",
    "TargetModuleReport",
    "VISION",
    "audit_lora_target_modules",
    "classify_module_name",
    "exact_target_regex",
    "freeze_all_parameters",
    "inspect_model_linear_modules",
    "load_qwen_base_model",
    "load_qwen_lora_model",
    "validate_local_model_directory",
    "validate_lora_target_modules",
    "verify_peft_targeted_modules",
    "verify_lora_trainable_parameters",
]
