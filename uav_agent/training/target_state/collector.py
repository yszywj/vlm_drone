"""Strict writer for externally captured Isaac target-state observations.

This module does not launch Isaac Sim and does not run a detector.  An external
collection adapter must synchronously capture RGB-D, execute the real deployed
detector, and construct the privileged label.  This writer keeps those sections
separate while materializing a reproducible offline dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from hashlib import sha256
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from datasets.target_state.dataset import (
    DatasetCheckReport,
    build_manifest,
    check_dataset,
    compute_dataset_sha256,
)
from datasets.target_state.schema import SensorInput, TargetStateFrameRecord
from datasets.target_state.sequence import build_sequences
from common.loopback_url import validate_loopback_http_url


class TargetStateCollectionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class VerifiedYoloDeployment:
    """A receipt created only after worker health/model-info preflight.

    The dataset writer intentionally cannot infer this fact from a user-supplied
    checksum.  Collection entrypoints must contact the deployed worker, validate
    its exact identity, and then pass this receipt.
    """

    worker_url: str
    model_family: str
    model_names: tuple[tuple[int, str], ...]
    model_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "worker_url",
            validate_loopback_http_url(self.worker_url, "worker_url"),
        )
        if self.model_family != "yolo":
            raise ValueError("target-state collection requires model_family='yolo'")
        names = tuple(self.model_names)
        if names != ((0, "cube"),):
            raise ValueError("target-state collection requires exactly model_names={0: 'cube'}")
        object.__setattr__(self, "model_names", names)
        digest = self.model_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in digest)
        ):
            raise ValueError("model_sha256 must be a 64-character hexadecimal digest")
        object.__setattr__(self, "model_sha256", digest.lower())

    def to_manifest_dict(self) -> dict[str, object]:
        return {
            "worker_url": self.worker_url,
            "model_family": self.model_family,
            "model_names": {str(index): name for index, name in self.model_names},
            "model_sha256": self.model_sha256,
            "preflight_verified": True,
        }


def require_privileged_collection_acknowledgements(
    *, oracle_label_generation: bool, acknowledge_privileged_oracle: bool
) -> None:
    if not isinstance(oracle_label_generation, bool) or not isinstance(acknowledge_privileged_oracle, bool):
        raise TypeError("collection acknowledgement flags must be bool")
    if not oracle_label_generation or not acknowledge_privileged_oracle:
        raise TargetStateCollectionError(
            "target-state collection writes privileged simulator labels; both "
            "oracle_label_generation and acknowledge_privileged_oracle are required"
        )


class TargetStateDatasetWriter:
    """Write synchronized external captures without synthesizing detector output."""

    def __init__(
        self,
        root: str | Path,
        *,
        yolo_model_sha256: str | None = None,
        verified_yolo_deployment: VerifiedYoloDeployment | None = None,
        split_seed: int = 42,
        history_size: int = 6,
        max_history_age_s: float = 2.0,
        generation_commit_sha: str | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if verified_yolo_deployment is not None and not isinstance(
            verified_yolo_deployment, VerifiedYoloDeployment
        ):
            raise TypeError("verified_yolo_deployment must be VerifiedYoloDeployment or None")
        resolved_sha = (
            verified_yolo_deployment.model_sha256
            if verified_yolo_deployment is not None
            else yolo_model_sha256
        )
        if (
            not isinstance(resolved_sha, str)
            or len(resolved_sha) != 64
            or any(character not in "0123456789abcdef" for character in resolved_sha.lower())
        ):
            raise ValueError("yolo_model_sha256 must be a 64-character hexadecimal digest")
        if not 4 <= history_size <= 8:
            raise ValueError("history_size must be within [4, 8]")
        if max_history_age_s <= 0.0:
            raise ValueError("max_history_age_s must be positive")
        if self.root.exists() and any(self.root.iterdir()):
            raise TargetStateCollectionError(f"output directory must be new or empty: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        self.yolo_model_sha256 = resolved_sha.lower()
        self.verified_yolo_deployment = verified_yolo_deployment
        self.split_seed = split_seed
        self.history_size = history_size
        self.max_history_age_s = max_history_age_s
        self.generation_commit_sha = generation_commit_sha or os.environ.get("UAV_AGENT_TRAINING_COMMIT_SHA", "nogit")
        self._records: list[TargetStateFrameRecord] = []
        self._frame_ids: set[str] = set()
        self._asset_signatures: dict[str, tuple[str, str]] = {}
        self._closed = False
        self._stream = (self.root / "frames.jsonl").open("x", encoding="utf-8")

    def append(
        self,
        record: TargetStateFrameRecord,
        *,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        instance_mask: np.ndarray | None = None,
        asset_id: str | None = None,
    ) -> TargetStateFrameRecord:
        if self._closed:
            raise TargetStateCollectionError("dataset writer is already finalized")
        if not isinstance(record, TargetStateFrameRecord):
            raise TypeError("record must be TargetStateFrameRecord")
        if record.frame_id in self._frame_ids:
            raise TargetStateCollectionError(f"duplicate frame_id: {record.frame_id}")
        width, height = record.sensor_input.camera.resolution_wh_px
        rgb_array = np.asarray(rgb)
        depth_array = np.asarray(depth_m, dtype=np.float32)
        if rgb_array.shape != (height, width, 3) or rgb_array.dtype != np.uint8:
            raise ValueError("rgb must be uint8 with synchronized camera shape [H,W,3]")
        if depth_array.shape != (height, width):
            raise ValueError("depth_m must have synchronized camera shape [H,W]")
        if not np.any(np.isfinite(depth_array) & (depth_array > 0.0)):
            raise ValueError("depth_m must contain at least one positive finite sample")
        normalized_asset_id = record.frame_id if asset_id is None else str(asset_id)
        if (
            not normalized_asset_id
            or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-" for character in normalized_asset_id)
            or len(normalized_asset_id) > 64
        ):
            raise ValueError("asset_id must be a bounded routing-safe identifier")
        prefix = Path(record.uav_id) / record.episode_id
        rgb_relative = (Path("rgb") / prefix / f"{normalized_asset_id}.jpg").as_posix()
        depth_relative = (Path("depth") / prefix / f"{normalized_asset_id}.npy").as_posix()
        mask_relative: str | None = None
        if instance_mask is not None:
            mask_array = np.asarray(instance_mask)
            if mask_array.shape != (height, width):
                raise ValueError("instance_mask must have synchronized camera shape [H,W]")
            # Instance masks are target-specific and therefore keep record IDs
            # even when RGB-D assets are shared by multiple targets.
            mask_relative = (Path("instance_masks") / prefix / f"{record.frame_id}.png").as_posix()
        for relative in (rgb_relative, depth_relative, mask_relative):
            if relative is not None:
                (self.root / relative).parent.mkdir(parents=True, exist_ok=True)
        signature = (
            sha256(memoryview(np.ascontiguousarray(rgb_array))).hexdigest(),
            sha256(memoryview(np.ascontiguousarray(depth_array))).hexdigest(),
        )
        asset_key = f"{record.uav_id}/{record.episode_id}/{normalized_asset_id}"
        prior_signature = self._asset_signatures.get(asset_key)
        if prior_signature is not None and prior_signature != signature:
            raise TargetStateCollectionError(
                f"asset_id {normalized_asset_id!r} was reused with different RGB-D"
            )
        if prior_signature is None:
            Image.fromarray(rgb_array).save(self.root / rgb_relative, quality=95, optimize=True)
            np.save(self.root / depth_relative, depth_array, allow_pickle=False)
            self._asset_signatures[asset_key] = signature
        if instance_mask is not None and mask_relative is not None:
            mask_uint8 = np.where(np.asarray(instance_mask) > 0, 255, 0).astype(np.uint8)
            Image.fromarray(mask_uint8).save(self.root / mask_relative)
        sensor = replace(
            record.sensor_input,
            rgb_path=rgb_relative,
            depth_path=depth_relative,
            instance_mask_path=mask_relative,
        )
        stored = replace(record, sensor_input=sensor)
        self._stream.write(json.dumps(stored.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")
        self._stream.flush()
        self._records.append(stored)
        self._frame_ids.add(stored.frame_id)
        return stored

    def finalize(self) -> tuple[Path, DatasetCheckReport]:
        if self._closed:
            raise TargetStateCollectionError("dataset writer is already finalized")
        self._stream.close()
        self._closed = True
        sequences = build_sequences(
            self._records,
            history_size=self.history_size,
            max_history_age_s=self.max_history_age_s,
        )
        report = check_dataset(
            self.root,
            sequences=sequences,
            history_size=self.history_size,
            max_history_age_s=self.max_history_age_s,
            split_seed=self.split_seed,
        )
        if not report.ok:
            raise TargetStateCollectionError(
                "written target-state dataset failed validation: " + "; ".join(report.errors[:5])
            )
        dataset_sha = compute_dataset_sha256(self.root, self._records)
        manifest = build_manifest(
            self._records,
            sequences,
            dataset_sha256=dataset_sha,
            split_seed=self.split_seed,
            generation_commit_sha=self.generation_commit_sha,
        )
        manifest.update(
            {
                "detector_prediction_source": (
                    "real_yolo_deployment_output"
                    if self.verified_yolo_deployment is not None
                    else "external_capture_spool_unverified"
                ),
                "yolo_model_sha256": self.yolo_model_sha256,
                "detector_deployment": (
                    None
                    if self.verified_yolo_deployment is None
                    else self.verified_yolo_deployment.to_manifest_dict()
                ),
                "candidate_id_source": (
                    "sensor_only_bbox_color_temporal_linker"
                    if self.verified_yolo_deployment is not None
                    else "external_capture_spool_unverified"
                ),
                "detector_truth_association": (
                    "offline_privileged_one_to_one_iou_after_worker_inference"
                    if self.verified_yolo_deployment is not None
                    else "external_capture_spool_unverified"
                ),
                "oracle_usage": "offline_training_labels_only",
                "history_size": self.history_size,
                "max_history_age_s": self.max_history_age_s,
            }
        )
        path = self.root / "dataset_manifest.json"
        path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return path, report

    def abort(self) -> None:
        """Close the stream but retain partial files for forensic recovery."""

        if not self._closed:
            self._stream.close()
            self._closed = True


__all__ = [
    "TargetStateCollectionError", "TargetStateDatasetWriter",
    "VerifiedYoloDeployment",
    "require_privileged_collection_acknowledgements",
]
