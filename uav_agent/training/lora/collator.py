"""Assistant-only loss collation for text-only Qwen Fleet Planner SFT."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


IGNORE_INDEX = -100


class FleetPlannerCollatorError(ValueError):
    """Raised rather than silently training on an incorrectly aligned span."""


def _token_ids(value: object, *, context: str) -> list[int]:
    """Normalize common tokenizer/processor return shapes to one token list."""

    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise FleetPlannerCollatorError(
                f"{context} chat template result has no input_ids"
            )
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()  # type: ignore[union-attr]
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise FleetPlannerCollatorError(
            f"{context} chat template did not return token IDs"
        )
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise FleetPlannerCollatorError(
                f"{context} unexpectedly returned a token batch"
            )
        value = list(value[0])
    if any(isinstance(token, bool) or not isinstance(token, int) for token in value):
        raise FleetPlannerCollatorError(
            f"{context} contains a non-integer token ID"
        )
    return list(value)


def _messages(feature: Mapping[str, object]) -> list[dict[str, str]]:
    raw = feature.get("messages")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise FleetPlannerCollatorError("feature.messages must be an array")
    result: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping) or set(item) != {"role", "content"}:
            raise FleetPlannerCollatorError(
                f"feature.messages[{index}] must contain only role/content"
            )
        role = item["role"]
        content = item["content"]
        if not isinstance(role, str) or not isinstance(content, str):
            raise FleetPlannerCollatorError(
                f"feature.messages[{index}] role/content must be strings"
            )
        result.append({"role": role, "content": content})
    if [item["role"] for item in result] != ["system", "user", "assistant"]:
        raise FleetPlannerCollatorError(
            "Fleet SFT messages must be exactly system, user, assistant"
        )
    if not result[-1]["content"]:
        raise FleetPlannerCollatorError("assistant target must not be empty")
    return result


class AssistantOnlyDataCollator:
    """Apply the Qwen chat template and supervise only its assistant suffix.

    The system+user conversation is rendered with
    ``add_generation_prompt=True``.  The complete conversation is rendered
    separately.  A strict token-prefix check establishes the boundary, so a
    tokenizer/template change fails closed instead of leaking request tokens
    into the language-model loss.
    """

    def __init__(
        self,
        processor_or_tokenizer: object,
        *,
        model_max_length: int,
        pad_to_multiple_of: int | None = None,
    ) -> None:
        if (
            isinstance(model_max_length, bool)
            or not isinstance(model_max_length, int)
            or model_max_length <= 0
        ):
            raise FleetPlannerCollatorError(
                "model_max_length must be a positive integer"
            )
        if pad_to_multiple_of is not None and (
            isinstance(pad_to_multiple_of, bool)
            or not isinstance(pad_to_multiple_of, int)
            or pad_to_multiple_of <= 0
        ):
            raise FleetPlannerCollatorError(
                "pad_to_multiple_of must be a positive integer or null"
            )
        tokenizer = getattr(processor_or_tokenizer, "tokenizer", None)
        if tokenizer is None:
            tokenizer = processor_or_tokenizer
        template_owner = (
            processor_or_tokenizer
            if callable(getattr(processor_or_tokenizer, "apply_chat_template", None))
            else tokenizer
        )
        if not callable(getattr(template_owner, "apply_chat_template", None)):
            raise FleetPlannerCollatorError(
                "processor/tokenizer must provide apply_chat_template"
            )
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if (
            isinstance(pad_token_id, bool)
            or not isinstance(pad_token_id, int)
            or pad_token_id < 0
        ):
            raise FleetPlannerCollatorError(
                "tokenizer.pad_token_id must be a non-negative integer"
            )

        self.processor_or_tokenizer = processor_or_tokenizer
        self.tokenizer = tokenizer
        self.template_owner = template_owner
        self.model_max_length = model_max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_token_id = pad_token_id

    def _apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool,
    ) -> list[int]:
        try:
            encoded = self.template_owner.apply_chat_template(
                list(messages),
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception as exc:
            raise FleetPlannerCollatorError(
                f"Qwen apply_chat_template failed: {type(exc).__name__}: {exc}"
            ) from exc
        return _token_ids(encoded, context="Qwen")

    def _encode_feature(
        self,
        feature: Mapping[str, object],
    ) -> tuple[list[int], list[int]]:
        messages = _messages(feature)
        prompt_ids = self._apply_chat_template(
            messages[:-1],
            add_generation_prompt=True,
        )
        full_ids = self._apply_chat_template(
            messages,
            add_generation_prompt=False,
        )
        if not prompt_ids:
            raise FleetPlannerCollatorError("Qwen prompt template produced no tokens")
        if len(full_ids) <= len(prompt_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
            raise FleetPlannerCollatorError(
                "Qwen full conversation is not prefixed by the generation prompt; "
                "assistant loss boundary cannot be proven"
            )

        input_ids = full_ids[: self.model_max_length]
        supervised_start = len(prompt_ids)
        if len(input_ids) <= supervised_start:
            sample_id = feature.get("sample_id", "<unknown>")
            raise FleetPlannerCollatorError(
                f"sample {sample_id} has no assistant tokens within "
                f"model_max_length={self.model_max_length}"
            )
        labels = [IGNORE_INDEX] * supervised_start + input_ids[supervised_start:]
        return input_ids, labels

    def __call__(
        self,
        features: Sequence[Mapping[str, object]],
    ) -> Mapping[str, Any]:
        if not features:
            raise FleetPlannerCollatorError("cannot collate an empty batch")
        encoded = [self._encode_feature(feature) for feature in features]
        padded_length = max(len(input_ids) for input_ids, _ in encoded)
        if self.pad_to_multiple_of is not None:
            remainder = padded_length % self.pad_to_multiple_of
            if remainder:
                padded_length += self.pad_to_multiple_of - remainder

        batch_input_ids: list[list[int]] = []
        batch_attention_mask: list[list[int]] = []
        batch_labels: list[list[int]] = []
        for input_ids, labels in encoded:
            padding = padded_length - len(input_ids)
            batch_input_ids.append(input_ids + [self.pad_token_id] * padding)
            batch_attention_mask.append([1] * len(input_ids) + [0] * padding)
            batch_labels.append(labels + [IGNORE_INDEX] * padding)

        # Keep torch optional at import time so dataset/config validation and
        # ordinary unit-test discovery do not import the training stack.
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - qwen_lora env preflight
            raise RuntimeError(
                "torch is required when materializing an SFT batch"
            ) from exc
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                batch_attention_mask,
                dtype=torch.long,
            ),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


# Concise compatibility name for trainer call sites.
FleetPlannerDataCollator = AssistantOnlyDataCollator


__all__ = [
    "IGNORE_INDEX",
    "AssistantOnlyDataCollator",
    "FleetPlannerCollatorError",
    "FleetPlannerDataCollator",
]
