"""Non-mutating review of legacy RGB-D labels, including entire temporal windows."""
from collections import Counter
from functools import lru_cache
from pathlib import Path

from datasets.target_state.sequence import build_sequences
from training.target_state.association_quality import AssociationPolicy, depth_support
from training.target_state.data import _foreground_cluster_anchor_depth, _load_depth, _target_depth


def scan_records(records, *, dataset_root, history_size, max_history_age_s,
                 policy=AssociationPolicy()):
    records = tuple(records)
    root = Path(dataset_root).resolve()

    @lru_cache(maxsize=4)
    def load_depth(relative):
        path = (root / relative).resolve(strict=True)
        if root not in path.parents:
            raise ValueError("depth path escapes dataset root")
        return _load_depth(path)

    counts = Counter(frame_records=len(records), physical_captures=len({
        (r.episode_id, r.uav_id, r.timestamp_s) for r in records}),
        episodes=len({r.episode_id for r in records}))
    findings, previous = [], {}
    for record in sorted(records, key=lambda r: (r.episode_id, r.timestamp_s, r.frame_id)):
        detector, label = record.detector_prediction, record.training_label
        key = (record.episode_id, record.assignment_id, record.uav_id, detector.candidate_id)
        prior = previous.get(key) if detector.candidate_id is not None else None
        if detector.candidate_id is not None:
            previous[key] = record
        reasons = ["collector_unresolved"] if record.association_review_required else []
        evidence = {}
        counts["detected_records"] += int(detector.detected)
        if label is not None and detector.detected:
            counts["screened_positive_detections"] += 1
            depth = load_depth(record.sensor_input.depth_path)
            width, height = record.sensor_input.camera.resolution_wh_px
            if depth.shape != (height, width):
                raise ValueError(f"depth/camera resolution mismatch: {record.frame_id}")
            target_depth = _target_depth(record)
            half_extent = max(policy.legacy_half_extent_m,
                              policy.legacy_half_extent_fraction * abs(target_depth))
            support = depth_support(depth, detector.bbox_xyxy_normalized,
                max(1e-6, target_depth - half_extent) if target_depth > 0 else target_depth,
                target_depth + half_extent, policy=policy)
            if not support["supported"]:
                reasons.append(support["status"])
            raw, anchor = _foreground_cluster_anchor_depth(depth, detector.bbox_xyxy_normalized,
                min_depth_m=policy.sampled_minimum_depth_m, max_depth_m=policy.sampled_maximum_depth_m)
            if raw > 0 and "depth_interval_m" in support:
                lower, upper = support["depth_interval_m"]
                if not lower <= raw <= upper:
                    reasons.append("sampled_ray_outside_legacy_interval")
            elif raw <= 0:
                reasons.append("no_sampled_depth")
            center = label.center_pixel_uv
            x1, y1, x2, y2 = detector.bbox_xyxy_normalized
            outside = center is not None and not (x1 * width <= center[0] <= x2 * width
                                                  and y1 * height <= center[1] <= y2 * height)
            transition = (prior is not None and prior.training_label is None
                          and record.timestamp_s - prior.timestamp_s <= max_history_age_s)
            # Centre outside a box / absent-to-present transitions can be valid
            # occlusion events. Diagnostic only, never an automatic veto.
            counts["projected_center_outside_bbox"] += int(outside)
            counts["null_to_positive_candidate_transitions"] += int(transition)
            evidence = dict(depth_support=support, target_center_depth_m=target_depth,
                sampled_depth_m=raw, anchor_uv_px=list(anchor),
                center_outside_bbox=outside, null_to_positive_transition=transition,
                occlusion_ratio=label.occlusion_ratio,
                instance_mask_available=record.sensor_input.instance_mask_path is not None)
        if reasons:
            findings.append(dict(frame_id=record.frame_id, episode_id=record.episode_id,
                timestamp_s=record.timestamp_s, candidate_id=detector.candidate_id,
                tracker_id=detector.tracker_id, instance_id=None if label is None else label.instance_id,
                bbox_xyxy_normalized=detector.bbox_xyxy_normalized,
                reasons=sorted(set(reasons)), evidence=evidence))
    load_depth.cache_clear()
    flagged = {r["frame_id"] for r in findings}
    sequences = build_sequences(records, history_size=history_size, max_history_age_s=max_history_age_s)
    affected = []
    for sequence in sequences:
        members = [r.frame_id for r in (*sequence.history, sequence.reference) if r.frame_id in flagged]
        if members:
            affected.append(dict(sequence_id=sequence.sequence_id,
                reference_frame_id=sequence.reference.frame_id, review_frame_ids=members))
    counts.update(review_required_records=len(findings), sequences=len(sequences),
                  review_affected_sequences=len(affected),
                  review_reference_sequences=sum(s.reference.frame_id in flagged for s in sequences))
    return dict(counts=dict(counts), reason_counts=dict(Counter(
        reason for finding in findings for reason in finding["reasons"])),
        findings=findings, affected_sequences=affected)
