#!/usr/bin/env python3
"""Check an OpenAI-compatible Qwen server without importing Isaac Sim."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback
from urllib import error as urllib_error
from urllib import request as urllib_request


# ``python scripts/check_qwen_server.py`` places ``scripts/`` on sys.path, not
# the project package root.  Add the latter explicitly so the documented direct
# invocation works without requiring callers to set PYTHONPATH.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.base import (  # noqa: E402  (path setup must precede local imports)
    ChatMessage,
    GenerationOptions,
    ModelClientError,
    ModelProtocolError,
)
from models.openai_compatible_client import OpenAICompatibleClient  # noqa: E402
from models.adapter_registry import (  # noqa: E402
    AdapterRegistry,
    AdapterRegistryError,
    DEFAULT_ADAPTER_CONFIG,
)


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check /v1/models, then send one minimal text request to a local "
            "OpenAI-compatible Qwen server."
        )
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("QWEN_API_BASE", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible API base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--adapter-config",
        type=Path,
        default=DEFAULT_ADAPTER_CONFIG,
        help="trusted role/Adapter registry (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("QWEN_MODEL", "Qwen3-VL-4B-Instruct"),
        help="served model name (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("QWEN_API_KEY", "EMPTY"),
        help="API key; defaults to QWEN_API_KEY (value is never printed)",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=os.environ.get("QWEN_REQUEST_TIMEOUT_S", "60"),
        help="request timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="include a traceback when a check fails",
    )
    return parser


def _redact(text: str, api_key: str | None) -> str:
    """Keep credentials out of both ordinary and debug error output."""

    if api_key:
        return text.replace(api_key, "<redacted>")
    return text


def _report_error(exc: Exception, *, api_key: str | None, debug: bool) -> None:
    error_type = type(exc).__name__
    message = _redact(str(exc), api_key)
    print(f"[Qwen server check] FAILED ({error_type}): {message}", file=sys.stderr)
    if debug:
        print(_redact(traceback.format_exc(), api_key), file=sys.stderr, end="")


def _check_response_content(content: str) -> str:
    """Validate required content and optionally recognize ``status=ok`` JSON."""

    if not isinstance(content, str):
        raise ModelProtocolError("chat completion content is not a string")
    stripped = content.strip()
    if not stripped:
        raise ModelProtocolError("chat completion returned empty content")

    try:
        decoded = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return "non-empty text returned (response was not strict JSON)"

    if isinstance(decoded, dict) and decoded.get("status") == "ok":
        return 'JSON response verified: {"status":"ok"}'
    return "non-empty JSON returned"


def _fetch_model_ids(
    *, base_url: str, api_key: str, timeout_s: float
) -> frozenset[str]:
    request = urllib_request.Request(
        base_url.rstrip("/") + "/models",
        method="GET",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    try:
        with urllib_request.urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, urllib_error.URLError) as exc:
        raise ModelProtocolError(f"could not read /models: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ModelProtocolError("/models response must contain a data list")
    model_ids: set[str] = set()
    for index, item in enumerate(payload["data"]):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ModelProtocolError(f"/models data[{index}] has no string id")
        model_ids.add(item["id"])
    return frozenset(model_ids)


def _validate_served_models(
    model_ids: frozenset[str],
    *,
    requested_base_model: str,
    registry: AdapterRegistry,
) -> None:
    if requested_base_model != registry.base_model_name:
        raise AdapterRegistryError(
            "--model does not match the base model lineage in adapters.json"
        )
    required = {requested_base_model}
    required.update(adapter.served_model_name for adapter in registry.active_adapters)
    missing = sorted(required - model_ids)
    if missing:
        raise ModelProtocolError(
            "/models is missing required base/active Adapter model(s): "
            + ", ".join(missing)
        )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        registry = AdapterRegistry(args.adapter_config)
        client = OpenAICompatibleClient(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            timeout_s=args.timeout,
        )

        print("[Qwen server check] checking model endpoint ...")
        model_ids = _fetch_model_ids(
            base_url=client.base_url,
            api_key=args.api_key,
            timeout_s=client.timeout_s,
        )
        _validate_served_models(
            model_ids,
            requested_base_model=args.model,
            registry=registry,
        )
        print("[Qwen server check] healthcheck OK")

        response = client.chat(
            [
                ChatMessage(
                    role="user",
                    content='Return exactly this JSON object: {"status":"ok"}',
                )
            ],
            options=GenerationOptions(
                temperature=0.0,
                max_tokens=32,
                top_p=1.0,
            ),
        )
        result_detail = _check_response_content(response.content)
        print(f"[Qwen server check] chat OK: {result_detail}")
        return 0
    except ModelClientError as exc:
        _report_error(exc, api_key=args.api_key, debug=args.debug)
        return 1
    except Exception as exc:  # Keep the CLI concise for configuration bugs too.
        _report_error(exc, api_key=args.api_key, debug=args.debug)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
