#!/usr/bin/env python3
"""Generate a Fleet planning proposal without starting simulation or flight."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from enum import Enum
import json
from math import isfinite
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.loader import load_config  # noqa: E402
from fleet.planning_service import FleetPlanningService  # noqa: E402
from fleet.planning_tool import FleetPlanningTool  # noqa: E402
from models.adapter_registry import (  # noqa: E402
    AdapterRegistry,
    DEFAULT_ADAPTER_CONFIG,
    ModelCallRole,
)
from models.model_client_factory import ModelClientFactory  # noqa: E402


class _InputParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _InputParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=_PROJECT_ROOT / "configs/multi_uav_demo.yaml"
    )
    parser.add_argument("--adapter-config", type=Path, default=DEFAULT_ADAPTER_CONFIG)
    parser.add_argument("--base-url")
    instruction = parser.add_mutually_exclusive_group()
    instruction.add_argument("--instruction")
    instruction.add_argument("--instruction-file", type=Path)
    parser.add_argument("--interpreter-max-tokens", type=int, default=6144)
    parser.add_argument("--fleet-max-tokens", type=int, default=2048)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--output", type=Path, help="new JSON file; existing files are never overwritten")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true", help="validate local configuration with no model calls")
    mode.add_argument("--tool-schema", action="store_true", help="print the model-visible tool definition with no model calls")
    return parser


def _validate_options(args: argparse.Namespace) -> None:
    for name in ("interpreter_max_tokens", "fleet_max_tokens"):
        if not 256 <= getattr(args, name) <= 8192:
            raise ValueError(f"--{name.replace('_', '-')} must be within [256, 8192]")
    if not isfinite(args.timeout_s) or args.timeout_s <= 0.0:
        raise ValueError("--timeout-s must be finite and greater than zero")


def _read_instruction(args: argparse.Namespace) -> str | None:
    value = args.instruction
    if args.instruction_file is not None:
        with args.instruction_file.expanduser().open(encoding="utf-8") as stream:
            value = stream.read(8193)
    if value is None:
        if args.validate_only or args.tool_schema:
            return None
        raise ValueError("provide --instruction or --instruction-file")
    if not value.strip() or len(value) > 8192 or "\x00" in value:
        raise ValueError("instruction must contain non-whitespace text, at most 8192 characters, and no NUL")
    return value


def _prepare_output(path: Path | None) -> Path | None:
    if path is None:
        return None
    destination = path.expanduser().absolute()
    if os.path.lexists(destination):
        raise FileExistsError("--output already exists; choose a new file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Check that a new file can be created before invoking the model.  The
    # final hard-link publication below also rejects concurrent creation.
    with tempfile.TemporaryFile(dir=destination.parent):
        pass
    return destination


def _json_default(value: object) -> object:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported audit value type: {type(value).__name__}")


def _emit(payload: object, destination: Path | None) -> None:
    text = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, indent=2, default=_json_default
    ) + "\n"
    if destination is None:
        sys.stdout.write(text)
        return
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        # Linking a complete sibling file publishes atomically and fails if
        # the destination appeared after preflight; replace() would overwrite.
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        _validate_options(args)
        instruction = _read_instruction(args)
        destination = _prepare_output(args.output)
        if args.tool_schema:
            _emit(FleetPlanningTool.tool_definition(), destination)
            return 0

        config = load_config(args.config)
        registry = AdapterRegistry(args.adapter_config)
        selections: list[dict[str, object]] = []
        calls: list[dict[str, object]] = []
        audit: dict[str, object] = {}
        # Validate the two selected roles even when no inference is requested.
        for role in (ModelCallRole.MISSION_INTERPRETATION, ModelCallRole.FLEET_PLAN):
            registry.resolve(role)
        if args.validate_only:
            result: dict[str, object] = {
                "status": "validated",
                "execution_started": False,
                "executable": False,
            }
        else:
            factory = ModelClientFactory(
                registry, base_url=args.base_url, timeout_s=args.timeout_s,
                selection_logger=selections.append, call_logger=calls.append,
            )
            service = FleetPlanningService(
                config, factory,
                interpreter_max_tokens=args.interpreter_max_tokens,
                fleet_max_tokens=args.fleet_max_tokens,
            )
            result = FleetPlanningTool(service).invoke(
                {"instruction": instruction}, audit_context=audit
            )
        _emit({
            "result": result,
            "audit_context": audit,
            "adapter_selections": selections,
            "model_call_records": calls,
        }, destination)
        return 0 if result.get("status") in {"validated", "proposal_ready"} else 2
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        print(f"plan_fleet: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
