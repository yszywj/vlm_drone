#!/usr/bin/env python3
"""Generate strict diagnostic JSONL with base or PEFT Fleet Planner LoRA."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fleet_data.validator import (  # noqa: E402
    FLEET_DATASET_SPLITS,
    PATCH_OUTPUT_KIND,
    PLAN_OUTPUT_KIND,
    parse_input_request,
    validate_fleet_output,
    validate_fleet_plan_patch,
)
from training.lora.dataset import (  # noqa: E402
    FleetPlannerSFTDataset,
    canonical_json,
)


class FleetPlannerPredictionError(RuntimeError):
    """Raised for prediction setup, generation, or output publication errors."""


@dataclass(frozen=True, slots=True)
class PredictionRuntime:
    processor: object
    model: object
    model_kind: str
    base_model_path: Path
    adapter_path: Path | None


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _local_directory(path: str | Path, *, field: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FleetPlannerPredictionError(
            f"{field} must be an existing local directory: {resolved}"
        )
    return resolved


def load_prediction_runtime(
    *,
    base_model: str | Path,
    adapter: str | Path | None,
    base_only: bool,
) -> PredictionRuntime:
    """Load local Qwen3-VL and, unless base-only, one local PEFT adapter.

    Heavy training/inference dependencies are intentionally imported here,
    never when this module is imported by ordinary CPU tests.
    """

    if base_only == (adapter is not None):
        raise FleetPlannerPredictionError(
            "select exactly one prediction mode: --base-only or --adapter"
        )
    base_path = _local_directory(base_model, field="base_model")
    adapter_path = (
        None
        if adapter is None
        else _local_directory(adapter, field="adapter")
    )
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as exc:  # pragma: no cover - qwen_lora environment only
        raise FleetPlannerPredictionError(
            "qwen_lora inference requires torch and a Transformers build with "
            "Qwen3VLForConditionalGeneration"
        ) from exc

    try:
        processor = AutoProcessor.from_pretrained(
            str(base_path),
            local_files_only=True,
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(base_path),
            local_files_only=True,
            torch_dtype="auto",
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if adapter_path is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise FleetPlannerPredictionError(
                    "PEFT is required for --adapter prediction"
                ) from exc
            model = PeftModel.from_pretrained(
                model,
                str(adapter_path),
                is_trainable=False,
                local_files_only=True,
            )
        model.eval()
    except FleetPlannerPredictionError:
        raise
    except Exception as exc:  # pragma: no cover - real local model integration
        raise FleetPlannerPredictionError(
            f"could not load local prediction runtime: {type(exc).__name__}: {exc}"
        ) from exc
    return PredictionRuntime(
        processor=processor,
        model=model,
        model_kind="base" if base_only else "adapter",
        base_model_path=base_path,
        adapter_path=adapter_path,
    )


def _input_length(input_ids: object) -> int:
    shape = getattr(input_ids, "shape", None)
    if shape is not None and len(shape) >= 1:
        return int(shape[-1])
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()  # type: ignore[union-attr]
    if isinstance(input_ids, Sequence) and not isinstance(input_ids, (str, bytes)):
        if input_ids and isinstance(input_ids[0], Sequence):
            return len(input_ids[0])  # type: ignore[arg-type]
        return len(input_ids)
    raise FleetPlannerPredictionError("processor input_ids has an unsupported shape")


def _move_inputs(inputs: object, model: object) -> object:
    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device  # type: ignore[attr-defined]
        except (AttributeError, StopIteration, TypeError):
            return inputs
    if callable(getattr(inputs, "to", None)):
        return inputs.to(device)  # type: ignore[union-attr]
    if isinstance(inputs, Mapping):
        return {
            key: value.to(device) if callable(getattr(value, "to", None)) else value
            for key, value in inputs.items()
        }
    return inputs


def _first_sequence(value: object) -> object:
    sequences = getattr(value, "sequences", value)
    try:
        return sequences[0]  # type: ignore[index]
    except (IndexError, KeyError, TypeError) as exc:
        raise FleetPlannerPredictionError(
            "model.generate returned no token sequence"
        ) from exc


def _slice_tokens(tokens: object, start: int) -> object:
    try:
        return tokens[start:]  # type: ignore[index]
    except TypeError as exc:
        raise FleetPlannerPredictionError(
            "model.generate returned an unsupported token sequence"
        ) from exc


def generate_raw_output(
    runtime: PredictionRuntime,
    prompt_messages: Sequence[Mapping[str, str]],
    *,
    max_new_tokens: int,
) -> str:
    """Run one deterministic, text-only Qwen generation."""

    if max_new_tokens <= 0:
        raise FleetPlannerPredictionError("max_new_tokens must be positive")
    processor = runtime.processor
    model = runtime.model
    template = getattr(processor, "apply_chat_template", None)
    tokenizer = getattr(processor, "tokenizer", None)
    if not callable(template) and tokenizer is not None:
        template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(template):
        raise FleetPlannerPredictionError(
            "processor/tokenizer does not provide apply_chat_template"
        )
    try:
        prompt = template(
            list(prompt_messages),
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(prompt, str):
            raise TypeError("chat template did not return text")
        inputs = processor(
            text=[prompt],
            return_tensors="pt",
            padding=True,
        )
        if not isinstance(inputs, Mapping) or "input_ids" not in inputs:
            raise TypeError("processor result has no input_ids")
        # Qwen3-VL's official generation example drops this processor-only
        # field before calling the conditional-generation model.
        if "token_type_ids" in inputs:
            if not hasattr(inputs, "pop"):
                inputs = dict(inputs)
            inputs.pop("token_type_ids", None)
        prompt_length = _input_length(inputs["input_ids"])
        inputs = _move_inputs(inputs, model)
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
        continuation = _slice_tokens(_first_sequence(generated), prompt_length)
        decoder = getattr(processor, "batch_decode", None)
        if not callable(decoder) and tokenizer is not None:
            decoder = getattr(tokenizer, "batch_decode", None)
        if not callable(decoder):
            raise TypeError("processor/tokenizer has no batch_decode")
        try:
            decoded = decoder(
                [continuation],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            decoded = decoder([continuation], skip_special_tokens=True)
        if not isinstance(decoded, Sequence) or not decoded or not isinstance(
            decoded[0], str
        ):
            raise TypeError("batch_decode returned no text")
        return decoded[0]
    except FleetPlannerPredictionError:
        raise
    except Exception as exc:
        raise FleetPlannerPredictionError(
            f"deterministic generation failed: {type(exc).__name__}: {exc}"
        ) from exc


def strict_json_object(text: str) -> Mapping[str, object]:
    """Parse exactly one standards-compliant JSON object with unique keys."""

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite JSON number {value}")
        return parsed

    value = json.loads(
        text,
        parse_constant=lambda constant: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant {constant}")
        ),
        object_pairs_hook=reject_duplicates,
        parse_float=finite_float,
    )
    if not isinstance(value, Mapping):
        raise ValueError("model output JSON root must be an object")
    return value


def prediction_diagnostics(
    *,
    sample_id: str,
    output_kind: str,
    request: object,
    raw_model_output: str,
    model_kind: str,
    adapter: str | None,
) -> dict[str, object]:
    """Separate JSON parsing from production Fleet schema validation."""

    parsed_output: Mapping[str, object] | None = None
    parse_error: str | None = None
    schema_error: str | None = None
    try:
        parsed_output = strict_json_object(raw_model_output)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        parse_error = f"{type(exc).__name__}: {exc}"

    schema_valid = False
    if parsed_output is not None:
        try:
            parsed_request = parse_input_request(request)
            if output_kind == PLAN_OUTPUT_KIND:
                validate_fleet_output(parsed_output, request=parsed_request)
            elif output_kind == PATCH_OUTPUT_KIND:
                validate_fleet_plan_patch(parsed_output, request=parsed_request)
            else:
                raise ValueError(f"unsupported output_kind {output_kind!r}")
            schema_valid = True
        except (KeyError, OverflowError, TypeError, ValueError) as exc:
            schema_error = f"{type(exc).__name__}: {exc}"

    return {
        "sample_id": sample_id,
        "output_kind": output_kind,
        "raw_model_output": raw_model_output,
        "parsed_output": None if parsed_output is None else dict(parsed_output),
        "parse_success": parsed_output is not None,
        "parse_error": parse_error,
        "schema_valid": schema_valid,
        "schema_error": schema_error,
        "model_kind": model_kind,
        "adapter": adapter,
    }


def generate_predictions(
    *,
    dataset_root: str | Path,
    split: str,
    output: str | Path,
    runtime: PredictionRuntime,
    max_new_tokens: int = 2048,
    max_samples: int | None = None,
    raw_generator: Callable[
        [PredictionRuntime, Sequence[Mapping[str, str]], int], str
    ]
    | None = None,
) -> dict[str, object]:
    """Generate and atomically publish diagnostics JSONL in dataset order."""

    dataset = FleetPlannerSFTDataset(
        dataset_root,
        split=split,
        max_samples=max_samples,
    )
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    generator = raw_generator
    temporary_path: Path | None = None
    count = 0
    parse_success_count = 0
    schema_valid_count = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for feature in dataset:
                raw_output = (
                    generate_raw_output(
                        runtime,
                        feature["messages"][:-1],  # type: ignore[index]
                        max_new_tokens=max_new_tokens,
                    )
                    if generator is None
                    else generator(
                        runtime,
                        feature["messages"][:-1],  # type: ignore[index]
                        max_new_tokens,
                    )
                )
                if not isinstance(raw_output, str):
                    raise FleetPlannerPredictionError(
                        "prediction generator must return text"
                    )
                diagnostics = prediction_diagnostics(
                    sample_id=str(feature["sample_id"]),
                    output_kind=str(feature["output_kind"]),
                    request=feature["request"],
                    raw_model_output=raw_output,
                    model_kind=runtime.model_kind,
                    adapter=(
                        None
                        if runtime.adapter_path is None
                        else str(runtime.adapter_path)
                    ),
                )
                handle.write(canonical_json(diagnostics) + "\n")
                count += 1
                parse_success_count += int(bool(diagnostics["parse_success"]))
                schema_valid_count += int(bool(diagnostics["schema_valid"]))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return {
        "output": str(destination),
        "split": split,
        "sample_count": count,
        "parse_success_count": parse_success_count,
        "schema_valid_count": schema_valid_count,
        "model_kind": runtime.model_kind,
        "adapter": None if runtime.adapter_path is None else str(runtime.adapter_path),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--base-only", action="store_true")
    mode.add_argument("--adapter", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=FLEET_DATASET_SPLITS,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=_positive_int, default=2048)
    parser.add_argument("--max-samples", type=_positive_int)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    runtime_loader: Callable[..., PredictionRuntime] = load_prediction_runtime,
) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        runtime = runtime_loader(
            base_model=args.base_model,
            adapter=args.adapter,
            base_only=args.base_only,
        )
        summary = generate_predictions(
            dataset_root=args.dataset,
            split=args.split,
            output=args.output,
            runtime=runtime,
            max_new_tokens=args.max_new_tokens,
            max_samples=args.max_samples,
        )
        print(canonical_json(summary))
        return 0
    except (OSError, TypeError, ValueError, FleetPlannerPredictionError) as exc:
        print(f"Fleet Planner prediction error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FleetPlannerPredictionError",
    "PredictionRuntime",
    "build_argument_parser",
    "generate_predictions",
    "generate_raw_output",
    "load_prediction_runtime",
    "main",
    "prediction_diagnostics",
    "strict_json_object",
]
