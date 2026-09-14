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

    The generation prompt is kept exactly as it will appear at inference.
    If BPE merges its trailing newline with the first answer character, a
    verified text prefix permits separately encoding the completion.  This
    supervises the whole answer without supervising any prompt characters.
    Overlong examples are rejected, never silently truncated.
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
        if full_ids[: len(prompt_ids)] != prompt_ids:
            full_ids = self._encode_across_bpe_boundary(messages, prompt_ids, full_ids)

        input_ids = full_ids
        supervised_start = len(prompt_ids)
        if len(input_ids) > self.model_max_length:
            sample_id = feature.get("sample_id", "<unknown>")
            raise FleetPlannerCollatorError(
                f"sample {sample_id} needs {len(input_ids)} tokens, exceeds "
                f"model_max_length={self.model_max_length}; refusing truncation"
            )
        if len(input_ids) <= supervised_start:
            sample_id = feature.get("sample_id", "<unknown>")
            raise FleetPlannerCollatorError(
                f"sample {sample_id} has no assistant tokens within "
                f"model_max_length={self.model_max_length}"
            )
        labels = [IGNORE_INDEX] * supervised_start + input_ids[supervised_start:]
        return input_ids, labels

    def _encode_across_bpe_boundary(
        self,
        messages: Sequence[Mapping[str, str]],
        prompt_ids: list[int],
        full_ids: list[int],
    ) -> list[int]:
        """Prove a text boundary when tokenization crosses that boundary.

        Encoding prompt and completion separately mirrors generation: the
        already-tokenized prompt cannot be retokenized by the first new token.
        Both template tokenizations are checked before using this fallback,
        so unrelated template corruption still fails closed.
        """

        error = (
            "Qwen full conversation is not prefixed by the generation prompt; "
            "assistant loss boundary cannot be proven"
        )
        encode = getattr(self.tokenizer, "encode", None)
        if not callable(encode):
            raise FleetPlannerCollatorError(error)
        try:
            prompt_text = self.template_owner.apply_chat_template(
                list(messages[:-1]), tokenize=False, add_generation_prompt=True,
            )
            full_text = self.template_owner.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=False,
            )
            if (
                not isinstance(prompt_text, str)
                or not isinstance(full_text, str)
                or not full_text.startswith(prompt_text)
                or len(full_text) <= len(prompt_text)
            ):
                raise FleetPlannerCollatorError(error)
            encoded_prompt = _token_ids(
                encode(prompt_text, add_special_tokens=False), context="prompt text",
            )
            encoded_full = _token_ids(
                encode(full_text, add_special_tokens=False), context="full text",
            )
            if encoded_prompt != prompt_ids or encoded_full != full_ids:
                raise FleetPlannerCollatorError(error)
            completion = _token_ids(
                encode(full_text[len(prompt_text):], add_special_tokens=False),
                context="assistant completion",
            )
            if not completion:
                raise FleetPlannerCollatorError("assistant completion has no tokens")
            return prompt_ids + completion
        except FleetPlannerCollatorError:
            raise
        except Exception as exc:
            raise FleetPlannerCollatorError(f"{error}: {exc}") from exc

    def encode_feature(self, feature: Mapping[str, object]) -> dict[str, list[int]]:
        """Tokenize once before Trainer iteration, using the same loss boundary."""

        input_ids, labels = self._encode_feature(feature)
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}

    def __call__(
        self,
        features: Sequence[Mapping[str, object]],
    ) -> Mapping[str, Any]:
        if not features:
            raise FleetPlannerCollatorError("cannot collate an empty batch")
        encoded = []
        for feature in features:
            if "messages" in feature:
                encoded.append(self._encode_feature(feature))
            else:
                input_ids = _token_ids(feature.get("input_ids"), context="cached input")
                labels = _token_ids(feature.get("labels"), context="cached labels")
                supervised_start = next(
                    (index for index, label in enumerate(labels) if label != IGNORE_INDEX),
                    len(labels),
                )
                if (
                    not input_ids or len(input_ids) != len(labels)
                    or len(input_ids) > self.model_max_length
                    or any(token < 0 for token in input_ids)
                    or not 0 < supervised_start < len(labels)
                    or labels[supervised_start:] != input_ids[supervised_start:]
                    or ("attention_mask" in feature and feature["attention_mask"] != [1] * len(input_ids))
                ):
                    raise FleetPlannerCollatorError("invalid cached assistant-only example")
                encoded.append((input_ids, labels))
        padded_length = max(len(input_ids) for input_ids, _ in encoded)
        if self.pad_to_multiple_of is not None:
            remainder = padded_length % self.pad_to_multiple_of
            if remainder:
                padded_length += self.pad_to_multiple_of - remainder
        if padded_length > self.model_max_length:
            raise FleetPlannerCollatorError("batch padding would exceed model_max_length")

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
