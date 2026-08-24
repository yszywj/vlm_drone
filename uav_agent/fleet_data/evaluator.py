"""Field-level offline metrics for production Fleet contract JSONL."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path

from fleet.types import FleetMissionRequest
from fleet_data.validator import (
    PATCH_OUTPUT_KIND,
    PLAN_OUTPUT_KIND,
    FleetDatasetValidationError,
    extract_explicit_target_aliases,
    load_split,
    parse_input_request,
    required_payloads_by_target,
    validate_fleet_output,
    validate_fleet_plan_patch,
    validate_sample,
)


Prediction = tuple[str, Mapping[str, object]]


def _payload(
    sample: Mapping[str, object],
) -> tuple[str, Mapping[str, object]]:
    kind = str(sample["output_kind"])
    key = "output" if kind == PLAN_OUTPUT_KIND else "fleet_plan_patch"
    value = sample.get(key)
    if not isinstance(value, Mapping):
        raise FleetDatasetValidationError(f"{key} must be an object")
    return kind, value


def _assignments(
    kind: str,
    output: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    key = "assignments" if kind == PLAN_OUTPUT_KIND else "replacement_assignments"
    raw = output.get(key, [])
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return tuple(item for item in raw if isinstance(item, Mapping))
    return ()


def _text_key(value: object) -> str | None:
    """Return a usable semantic key without coercing invalid model output."""

    if not isinstance(value, str) or not value:
        return None
    return value


def _routing_signature(
    assignments: Sequence[Mapping[str, object]],
) -> tuple[tuple[str | None, str | None], ...]:
    """Build a deterministic multiset-like routing signature.

    Invalid values remain ``None`` instead of being stringified into something
    that could accidentally compare equal to a legitimate routing ID.
    """

    return tuple(
        sorted(
            (
                (_text_key(item.get("uav_id")), _text_key(item.get("target_alias")))
                for item in assignments
            ),
            key=lambda item: (item[0] or "", item[1] or ""),
        )
    )


def _match_assignments_one_to_one(
    gold: Sequence[Mapping[str, object]],
    predicted: Sequence[Mapping[str, object]],
) -> dict[int, Mapping[str, object]]:
    """Match assignments by stable task semantics, never by generated IDs.

    ``assignment_id`` is a routing identifier chosen freely by the model, so it
    cannot be an evaluation alignment key.  A unique ``target_alias`` is the
    stable semantic identity of a Fleet task.  Duplicate predicted aliases are
    deliberately left unmatched: choosing whichever duplicate happens to score
    best would make duplicate-claim output look better than it is.

    For a future legitimate shared-target gold plan, an alias group can only be
    disambiguated when both sides contain the same number of assignments and
    each ``uav_id`` provides a unique, identical route key.  Any other duplicate
    or malformed group remains unmatched and therefore scores zero for its gold
    assignments.
    """

    gold_by_alias: dict[str, list[int]] = {}
    predicted_by_alias: dict[str, list[int]] = {}
    for index, item in enumerate(gold):
        alias = _text_key(item.get("target_alias"))
        if alias is not None:
            gold_by_alias.setdefault(alias, []).append(index)
    for index, item in enumerate(predicted):
        alias = _text_key(item.get("target_alias"))
        if alias is not None:
            predicted_by_alias.setdefault(alias, []).append(index)

    matches: dict[int, Mapping[str, object]] = {}
    for alias in sorted(set(gold_by_alias) & set(predicted_by_alias)):
        gold_indexes = gold_by_alias[alias]
        predicted_indexes = predicted_by_alias[alias]
        if len(gold_indexes) == len(predicted_indexes) == 1:
            matches[gold_indexes[0]] = predicted[predicted_indexes[0]]
            continue
        if len(gold_indexes) != len(predicted_indexes) or len(gold_indexes) < 2:
            continue

        gold_by_uav: dict[str, int] = {}
        predicted_by_uav: dict[str, int] = {}
        ambiguous = False
        for index in gold_indexes:
            uav_id = _text_key(gold[index].get("uav_id"))
            if uav_id is None or uav_id in gold_by_uav:
                ambiguous = True
                break
            gold_by_uav[uav_id] = index
        if ambiguous:
            continue
        for index in predicted_indexes:
            uav_id = _text_key(predicted[index].get("uav_id"))
            if uav_id is None or uav_id in predicted_by_uav:
                ambiguous = True
                break
            predicted_by_uav[uav_id] = index
        if ambiguous or set(gold_by_uav) != set(predicted_by_uav):
            continue
        for uav_id, gold_index in gold_by_uav.items():
            matches[gold_index] = predicted[predicted_by_uav[uav_id]]
    return matches


def _load_predictions(path: str | Path) -> dict[str, Prediction]:
    result: dict[str, Prediction] = {}

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise FleetDatasetValidationError(
                    f"duplicate prediction JSON key {key!r}"
                )
            value[key] = item
        return value

    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            raise FleetDatasetValidationError(
                f"prediction line {line_number} is blank"
            )
        try:
            row = json.loads(
                line,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    FleetDatasetValidationError(
                        f"non-standard constant {constant}"
                    )
                ),
                object_pairs_hook=reject_duplicates,
            )
        except json.JSONDecodeError as exc:
            raise FleetDatasetValidationError(
                f"prediction line {line_number}: {exc}"
            ) from exc
        if not isinstance(row, Mapping):
            raise FleetDatasetValidationError(
                "prediction rows must be JSON objects"
            )
        # Preserve the original {sample_id, output} prediction interface for
        # normal plans. Patch predictions must be explicit to prevent contract
        # ambiguity.
        if set(row) == {"sample_id", "output"}:
            kind = PLAN_OUTPUT_KIND
            payload = row["output"]
        elif row.get("output_kind") == PLAN_OUTPUT_KIND and set(row) == {
            "sample_id",
            "output_kind",
            "output",
        }:
            kind = PLAN_OUTPUT_KIND
            payload = row["output"]
        elif row.get("output_kind") == PATCH_OUTPUT_KIND and set(row) == {
            "sample_id",
            "output_kind",
            "fleet_plan_patch",
        }:
            kind = PATCH_OUTPUT_KIND
            payload = row["fleet_plan_patch"]
        else:
            raise FleetDatasetValidationError(
                "prediction rows must contain sample_id plus an explicit "
                "FleetMissionPlan or FleetPlanPatch payload"
            )
        sample_id = row["sample_id"]
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise FleetDatasetValidationError("prediction sample_id is invalid")
        if sample_id in result or not isinstance(payload, Mapping):
            raise FleetDatasetValidationError(
                f"invalid duplicate prediction {sample_id}"
            )
        result[sample_id] = (kind, payload)
    return result


def _covered_tasks(
    kind: str,
    output: Mapping[str, object],
    *,
    request: FleetMissionRequest,
) -> int:
    # The caller always supplies a production FleetMissionRequest.  Keeping the
    # runtime import local avoids adding another public schema dialect here.
    inventory = {item.uav_id: item for item in request.uav_inventory}
    targets = {item.target_alias: item for item in request.target_requests}
    payload_requirements = required_payloads_by_target(request)
    assigned_aliases: set[str] = set()
    for item in _assignments(kind, output):
        alias = _text_key(item.get("target_alias"))
        uav_id = _text_key(item.get("uav_id"))
        target = None if alias is None else targets.get(alias)
        uav = None if uav_id is None else inventory.get(uav_id)
        if target is None or uav is None or not uav.available:
            continue
        if target.requested_uav_id is not None and target.requested_uav_id != uav_id:
            continue
        if set(payload_requirements.get(alias, ())) - set(uav.payload_capabilities):
            continue
        assigned_aliases.add(alias)

    notes_key = "unassigned_requirements" if kind == PLAN_OUTPUT_KIND else "reason_codes"
    raw_notes = output.get(notes_key, [])
    notes = (
        raw_notes
        if isinstance(raw_notes, Sequence) and not isinstance(raw_notes, (str, bytes))
        else ()
    )
    explicitly_unassigned = extract_explicit_target_aliases(
        notes,
        request.target_aliases,
    )
    # Sets make duplicate assignments/reasons idempotent.  Noise that does not
    # exactly name a trusted target cannot improve missing-task coverage.
    return len(assigned_aliases | set(explicitly_unassigned))


def evaluate_predictions(
    gold_split: str | Path,
    predictions: str | Path | None = None,
) -> dict[str, float | int]:
    samples = load_split(gold_split)
    predicted: dict[str, Prediction] = (
        {
            str(sample["sample_id"]): _payload(sample)
            for sample in samples
        }
        if predictions is None
        else _load_predictions(predictions)
    )
    gold_sample_ids = {str(sample["sample_id"]) for sample in samples}
    extra_sample_ids = sorted(set(predicted) - gold_sample_ids)
    if extra_sample_ids:
        raise FleetDatasetValidationError(
            "predictions contain sample_id values outside the gold split: "
            + ", ".join(extra_sample_ids)
        )
    counters = {
        "schema": 0,
        "output_kind": 0,
        "routing": 0,
        "count": 0,
        "uav": 0,
        "target": 0,
        "region": 0,
        "duration": 0,
        "coordination": 0,
        "final": 0,
        "duplicate_claim": 0,
        "missing_tasks": 0,
        "invalid_uav": 0,
    }
    assignment_denominator = 0
    invalid_uav_denominator = 0
    total_tasks = 0
    for sample in samples:
        sample_id = str(sample["sample_id"])
        gold_kind, gold = _payload(sample)
        gold_items = _assignments(gold_kind, gold)
        invalid_uav_capacity = max(1, len(gold_items))
        invalid_uav_denominator += invalid_uav_capacity
        task_count = int(sample["metadata"]["task_count"])  # type: ignore[index]
        total_tasks += task_count
        prediction_record = predicted.get(sample_id)
        if prediction_record is None:
            counters["missing_tasks"] += task_count
            assignment_denominator += len(gold_items)
            continue
        prediction_kind, prediction = prediction_record
        counters["output_kind"] += int(prediction_kind == gold_kind)
        request = parse_input_request(sample["input"])
        try:
            if prediction_kind == PLAN_OUTPUT_KIND:
                validate_fleet_output(prediction, request=request)
            else:
                validate_fleet_plan_patch(prediction, request=request)
            counters["schema"] += 1
        except FleetDatasetValidationError:
            pass
        try:
            contextual_sample = dict(sample)
            contextual_sample.pop("output", None)
            contextual_sample.pop("fleet_plan_patch", None)
            contextual_sample["output_kind"] = prediction_kind
            contextual_sample[
                "output"
                if prediction_kind == PLAN_OUTPUT_KIND
                else "fleet_plan_patch"
            ] = prediction
            validate_sample(contextual_sample)
            counters["final"] += 1
        except FleetDatasetValidationError:
            pass

        predicted_items = _assignments(prediction_kind, prediction)
        matches = _match_assignments_one_to_one(gold_items, predicted_items)
        assignment_denominator += len(gold_items)
        counters["count"] += int(len(gold_items) == len(predicted_items))
        counters["routing"] += int(
            _routing_signature(gold_items) == _routing_signature(predicted_items)
        )
        for gold_index, gold_item in enumerate(gold_items):
            candidate = matches.get(gold_index, {})
            counters["uav"] += int(
                candidate.get("uav_id") == gold_item.get("uav_id")
            )
            counters["target"] += int(
                candidate.get("target_alias") == gold_item.get("target_alias")
            )
            counters["region"] += int(
                candidate.get("search_region")
                == gold_item.get("search_region")
            )
            counters["duration"] += int(
                candidate.get("track_duration_s")
                == gold_item.get("track_duration_s")
            )
        predicted_target_ids = [
            str(item.get("target_alias"))
            for item in predicted_items
        ]
        counters["duplicate_claim"] += int(
            len(predicted_target_ids) != len(set(predicted_target_ids))
        )
        available_uav_ids = set(request.available_uav_ids)
        invalid_uav_count = sum(
            _text_key(item.get("uav_id")) not in available_uav_ids
            for item in predicted_items
        )
        # The denominator is fixed by Gold, while saturation keeps the rate in
        # [0, 1].  Appending valid/invalid assignment noise can never dilute it.
        counters["invalid_uav"] += min(
            invalid_uav_count,
            invalid_uav_capacity,
        )
        counters["missing_tasks"] += max(
            0,
            task_count
            - _covered_tasks(
                prediction_kind,
                prediction,
                request=request,
            ),
        )
        counters["coordination"] += int(
            prediction.get("coordination_policy")
            == gold.get("coordination_policy")
        )
    count = len(samples)
    denominator = max(1, assignment_denominator)
    return {
        "sample_count": count,
        "schema_valid_rate": counters["schema"] / count,
        "output_kind_accuracy": counters["output_kind"] / count,
        "fleet_routing_exact_match": counters["routing"] / count,
        "assignment_count_accuracy": counters["count"] / count,
        "uav_assignment_accuracy": counters["uav"] / denominator,
        "target_assignment_accuracy": counters["target"] / denominator,
        "region_accuracy": counters["region"] / denominator,
        "track_duration_accuracy": counters["duration"] / denominator,
        "duplicate_target_claim_rate": counters["duplicate_claim"] / count,
        "missing_task_rate": counters["missing_tasks"] / max(1, total_tasks),
        "invalid_uav_rate": counters["invalid_uav"]
        / max(1, invalid_uav_denominator),
        "coordination_policy_accuracy": counters["coordination"] / count,
        "final_plan_valid_rate": counters["final"] / count,
    }


__all__ = ["evaluate_predictions"]
