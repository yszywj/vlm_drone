#!/usr/bin/env python3
"""Build trusted, startup-only vLLM LoRA arguments from adapter config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.adapter_registry import (  # noqa: E402
    AdapterRegistry,
    AdapterRegistryError,
    DEFAULT_ADAPTER_CONFIG,
)


def build_vllm_lora_args(
    registry: AdapterRegistry,
    *,
    expected_base_model_name: str | None = None,
) -> tuple[str, ...]:
    if expected_base_model_name is not None:
        expected = expected_base_model_name.strip()
        if not expected or expected != registry.base_model_name:
            raise AdapterRegistryError(
                "served base model name does not match adapters.json base lineage"
            )
    active = registry.active_adapters
    if not active:
        return ()
    ranks = [adapter.rank for adapter in active]
    if any(rank is None for rank in ranks):
        raise AdapterRegistryError("every active adapter must declare rank")
    arguments = ["--enable-lora", "--lora-modules"]
    arguments.extend(
        f"{adapter.served_model_name}={adapter.path}" for adapter in active
    )
    arguments.extend(
        (
            "--max-loras",
            str(len(active)),
            "--max-cpu-loras",
            str(len(active)),
            "--max-lora-rank",
            str(max(int(rank) for rank in ranks if rank is not None)),
        )
    )
    return tuple(arguments)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_ADAPTER_CONFIG)
    parser.add_argument("--expected-base-model-name")
    parser.add_argument("--format", choices=("json", "lines"), default="json")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        registry = AdapterRegistry(arguments.config)
        result = build_vllm_lora_args(
            registry,
            expected_base_model_name=arguments.expected_base_model_name,
        )
    except (AdapterRegistryError, OSError, TypeError, ValueError) as exc:
        print(f"adapter configuration error: {exc}", file=sys.stderr)
        return 2
    if arguments.format == "json":
        print(json.dumps(list(result), ensure_ascii=False, allow_nan=False))
    elif result:
        print("\n".join(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
