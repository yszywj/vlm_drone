"""Strict, immutable role-to-LoRA routing for Qwen model calls."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


DEFAULT_ADAPTER_CONFIG = Path(__file__).resolve().parents[1] / "configs/adapters.json"


class AdapterRegistryError(ValueError):
    """Raised when adapter configuration or selection is unsafe."""


class ModelCallRole(str, Enum):
    MISSION_INTERPRETATION = "MISSION_INTERPRETATION"
    FLEET_PLAN = "FLEET_PLAN"
    FLEET_REPLAN = "FLEET_REPLAN"
    AGENT_SPATIAL_PLAN = "AGENT_SPATIAL_PLAN"
    RUNTIME_VISUAL_REVIEW = "RUNTIME_VISUAL_REVIEW"
    RUNTIME_REPLAN = "RUNTIME_REPLAN"


class AdapterStatus(str, Enum):
    DISABLED = "disabled"
    PLACEHOLDER = "placeholder"
    ACTIVE = "active"


@dataclass(frozen=True, slots=True)
class AdapterSpec:
    name: str
    status: AdapterStatus
    served_model_name: str
    path: Path | None
    base_model_name: str
    rank: int | None


@dataclass(frozen=True, slots=True)
class AdapterSelection:
    call_role: ModelCallRole
    requested_adapter: str
    adapter_status: AdapterStatus
    effective_model: str
    fallback_used: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "call_role": self.call_role.value,
            "requested_adapter": self.requested_adapter,
            "adapter_status": self.adapter_status.value,
            "effective_model": self.effective_model,
            "fallback_used": self.fallback_used,
        }


def _strict_json(path: Path) -> Mapping[str, object]:
    def reject_constant(value: str) -> object:
        raise AdapterRegistryError(f"non-standard JSON constant {value!r} is forbidden")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise AdapterRegistryError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdapterRegistryError(f"could not load adapter config {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise AdapterRegistryError("adapter config root must be an object")
    return raw


def _exact_keys(value: Mapping[str, object], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise AdapterRegistryError(f"{context} keys are invalid: {'; '.join(details)}")


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdapterRegistryError(f"{field} must be a non-empty string")
    normalized = value.strip()
    if any(ord(character) < 0x20 for character in normalized):
        raise AdapterRegistryError(f"{field} must not contain control characters")
    return normalized


class AdapterRegistry:
    """Load adapter slots once and resolve trusted call roles deterministically."""

    def __init__(self, config_path: str | Path = DEFAULT_ADAPTER_CONFIG) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        raw = _strict_json(self.config_path)
        _exact_keys(
            raw,
            {"schema_version", "base_model", "adapters", "routing", "fallback_to_base"},
            "adapter config",
        )
        if raw["schema_version"] != 1:
            raise AdapterRegistryError("adapter config schema_version must be 1")
        base = raw["base_model"]
        if not isinstance(base, Mapping):
            raise AdapterRegistryError("base_model must be an object")
        _exact_keys(base, {"served_model_name"}, "base_model")
        self.base_model_name = _text(base["served_model_name"], "base_model.served_model_name")
        if not isinstance(raw["fallback_to_base"], bool):
            raise AdapterRegistryError("fallback_to_base must be a boolean")
        self.fallback_to_base = raw["fallback_to_base"]

        adapter_raw = raw["adapters"]
        if not isinstance(adapter_raw, Mapping) or not adapter_raw:
            raise AdapterRegistryError("adapters must be a non-empty object")
        adapters: dict[str, AdapterSpec] = {}
        for raw_name, value in adapter_raw.items():
            name = _text(raw_name, "adapter name")
            if not isinstance(value, Mapping):
                raise AdapterRegistryError(f"adapter {name!r} must be an object")
            _exact_keys(
                value,
                {"status", "served_model_name", "path", "base_model_name", "rank"},
                f"adapter {name}",
            )
            try:
                status = AdapterStatus(value["status"])
            except (TypeError, ValueError) as exc:
                raise AdapterRegistryError(
                    f"adapter {name} status must be disabled, placeholder, or active"
                ) from exc
            served = _text(value["served_model_name"], f"adapter {name}.served_model_name")
            lineage = _text(value["base_model_name"], f"adapter {name}.base_model_name")
            if lineage != self.base_model_name:
                raise AdapterRegistryError(
                    f"adapter {name} base model lineage {lineage!r} does not match "
                    f"configured base {self.base_model_name!r}"
                )
            raw_path = value["path"]
            path: Path | None = None
            if raw_path is not None:
                path_text = _text(raw_path, f"adapter {name}.path")
                candidate = Path(path_text).expanduser()
                path = (
                    candidate.resolve()
                    if candidate.is_absolute()
                    else (self.config_path.parent / candidate).resolve()
                )
            rank = value["rank"]
            if rank is not None and (
                isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0
            ):
                raise AdapterRegistryError(f"adapter {name}.rank must be null or positive")
            if status is AdapterStatus.ACTIVE:
                if path is None or not path.is_dir():
                    raise AdapterRegistryError(
                        f"active adapter {name} path does not exist or is not a "
                        f"directory: {path}"
                    )
                adapter_config = path / "adapter_config.json"
                if adapter_config.is_symlink() or not adapter_config.is_file():
                    raise AdapterRegistryError(
                        f"active adapter {name} requires a regular non-symlink "
                        f"adapter_config.json: {adapter_config}"
                    )
                weight_files = tuple(sorted(path.glob("*.safetensors")))
                if not weight_files or any(
                    weight.is_symlink() or not weight.is_file()
                    for weight in weight_files
                ):
                    raise AdapterRegistryError(
                        f"active adapter {name} requires at least one regular "
                        f"non-symlink .safetensors file in {path}"
                    )
                if rank is None:
                    raise AdapterRegistryError(f"active adapter {name} requires rank")
            else:
                if path is not None:
                    raise AdapterRegistryError(
                        f"{status.value} adapter {name} must use path=null"
                    )
                if rank is not None:
                    raise AdapterRegistryError(
                        f"{status.value} adapter {name} must use rank=null"
                    )
            adapters[name] = AdapterSpec(name, status, served, path, lineage, rank)
        served_names = [adapter.served_model_name for adapter in adapters.values()]
        duplicates = sorted(
            name for name in set(served_names) if served_names.count(name) > 1
        )
        if duplicates:
            raise AdapterRegistryError(
                "adapter served_model_name values must be unique: "
                + ", ".join(duplicates)
            )
        if self.base_model_name in served_names:
            raise AdapterRegistryError(
                "adapter served_model_name must differ from the base served model"
            )
        self.adapters: Mapping[str, AdapterSpec] = MappingProxyType(adapters)

        routing_raw = raw["routing"]
        if not isinstance(routing_raw, Mapping):
            raise AdapterRegistryError("routing must be an object")
        expected_roles = {role.value for role in ModelCallRole}
        _exact_keys(routing_raw, expected_roles, "routing")
        routing: dict[ModelCallRole, str] = {}
        for role in ModelCallRole:
            adapter_name = _text(routing_raw[role.value], f"routing.{role.value}")
            if adapter_name not in adapters:
                raise AdapterRegistryError(
                    f"routing.{role.value} references unknown adapter {adapter_name!r}"
                )
            routing[role] = adapter_name
        self.routing: Mapping[ModelCallRole, str] = MappingProxyType(routing)

    def resolve(self, call_role: ModelCallRole | str) -> AdapterSelection:
        try:
            role = call_role if isinstance(call_role, ModelCallRole) else ModelCallRole(call_role)
        except (TypeError, ValueError) as exc:
            raise AdapterRegistryError(f"unknown model call role: {call_role!r}") from exc
        adapter_name = self.routing[role]
        adapter = self.adapters[adapter_name]
        if adapter.status is AdapterStatus.DISABLED:
            raise AdapterRegistryError(
                f"adapter {adapter_name} is disabled and cannot serve {role.value}"
            )
        if adapter.status is AdapterStatus.ACTIVE:
            return AdapterSelection(
                role, adapter_name, adapter.status, adapter.served_model_name, False
            )
        if not self.fallback_to_base:
            raise AdapterRegistryError(
                f"adapter {adapter_name} is placeholder and base fallback is disabled"
            )
        return AdapterSelection(
            role, adapter_name, adapter.status, self.base_model_name, True
        )

    @property
    def active_adapters(self) -> tuple[AdapterSpec, ...]:
        return tuple(
            adapter
            for adapter in self.adapters.values()
            if adapter.status is AdapterStatus.ACTIVE
        )


__all__ = [
    "AdapterRegistry",
    "AdapterRegistryError",
    "AdapterSelection",
    "AdapterSpec",
    "AdapterStatus",
    "DEFAULT_ADAPTER_CONFIG",
    "ModelCallRole",
]
