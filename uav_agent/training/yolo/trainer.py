"""Framework-neutral YOLO training API with a lazy Ultralytics backend."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from training.yolo.config import YoloTrainConfig, YoloTrainingConfigError
from training.yolo.dataset import DatasetValidationReport, YoloDatasetValidator
from training.yolo.registry import (
    ModelRegistryError,
    build_model_manifest,
    load_validation_gate,
    sha256_file,
    utc_now_iso,
    write_json,
    write_model_manifest,
)


class YoloTrainingError(RuntimeError):
    """Raised for actionable preflight, training, validation, or export failures."""


@dataclass(frozen=True, slots=True)
class TrainingPreflight:
    ok: bool
    diagnostics: tuple[str, ...]
    dataset_report: DatasetValidationReport | None
    requested_device: str | int
    output_directory: Path
    model_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "diagnostics": list(self.diagnostics),
            "requested_device": self.requested_device,
            "output_directory": str(self.output_directory),
            "model_sha256": self.model_sha256,
            "dataset": (
                None
                if self.dataset_report is None
                else self.dataset_report.to_statistics_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class TrainingResult:
    run_dir: Path
    best_model_path: Path
    last_model_path: Path
    model_manifest_path: Path
    elapsed_s: float
    metrics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "best_model_path": str(self.best_model_path),
            "last_model_path": str(self.last_model_path),
            "model_manifest_path": str(self.model_manifest_path),
            "elapsed_s": self.elapsed_s,
            "metrics": _plain(self.metrics),
        }


@dataclass(frozen=True, slots=True)
class ValidationResult:
    model_path: Path
    model_sha256: str
    dataset_yaml: Path
    model_family: str
    task: str
    passed: bool
    map50: float | None
    map50_95: float | None
    precision: float | None
    recall: float | None
    per_class: Mapping[str, Mapping[str, float | None]]
    small_target_metrics: Mapping[str, Any]
    latency_ms: Mapping[str, float | None]
    raw_metrics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "validated_at": utc_now_iso(),
            "model_path": str(self.model_path),
            "model_sha256": self.model_sha256,
            "dataset_yaml": str(self.dataset_yaml),
            "model_family": self.model_family,
            "task": self.task,
            "passed": self.passed,
            "mAP50": self.map50,
            "mAP50-95": self.map50_95,
            "precision": self.precision,
            "recall": self.recall,
            "per_class": _plain(self.per_class),
            "small_target_metrics": _plain(self.small_target_metrics),
            "latency_ms": _plain(self.latency_ms),
            "raw_metrics": _plain(self.raw_metrics),
        }


@dataclass(frozen=True, slots=True)
class PredictionResult:
    model_path: Path
    source: str
    image_count: int
    elapsed_s: float
    results: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class ExportResult:
    model_path: Path
    exported_path: Path
    format: str
    validation_report: Path
    export_manifest_path: Path
    dynamic_prompts_supported: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "exported_path": str(self.exported_path),
            "format": self.format,
            "validation_report": str(self.validation_report),
            "export_manifest_path": str(self.export_manifest_path),
            "dynamic_prompts_supported": self.dynamic_prompts_supported,
        }


@runtime_checkable
class YoloTrainingBackend(Protocol):
    def preflight(self, config: YoloTrainConfig) -> TrainingPreflight:
        ...

    def train(self, config: YoloTrainConfig) -> TrainingResult:
        ...

    def validate(
        self,
        *,
        model_path: str | Path,
        dataset_yaml: str | Path,
        device: str | int,
        imgsz: int = 960,
        task: str = "detect",
        model_family: str = "yolo",
    ) -> ValidationResult:
        ...

    def predict(
        self,
        *,
        model_path: str | Path,
        source: str | Path,
        device: str | int,
        confidence: float = 0.25,
        imgsz: int = 960,
        task: str = "detect",
        model_family: str = "yolo",
    ) -> PredictionResult:
        ...

    def export(
        self,
        *,
        model_path: str | Path,
        validation_report: str | Path,
        format: str,
        device: str | int,
        imgsz: int = 960,
        model_family: str = "yolo",
        freeze_yoloe_prompts: bool = False,
        half: bool = False,
    ) -> ExportResult:
        ...


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _plain(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _cuda_indices(device: str | int) -> tuple[int, ...] | None:
    if isinstance(device, int):
        return () if device < 0 else (device,)
    normalized = device.lower()
    if normalized in {"cpu", "mps"}:
        return None
    indices = []
    for item in normalized.split(","):
        value = item.removeprefix("cuda:")
        indices.append(int(value))
    return tuple(indices)


def _normalized_model_family(value: str) -> str:
    family = str(value).strip().lower()
    if family not in {"yolo", "yoloe"}:
        raise YoloTrainingError("model_family must be yolo or yoloe")
    return family


def _result_metrics(result: Any) -> dict[str, Any]:
    raw = getattr(result, "results_dict", None)
    if isinstance(raw, Mapping):
        return {str(key): _plain(value) for key, value in raw.items()}
    metrics = getattr(result, "metrics", None)
    if isinstance(metrics, Mapping):
        return {str(key): _plain(value) for key, value in metrics.items()}
    return {}


class UltralyticsTrainingBackend:
    """Ultralytics adapter. Importing this module never imports Ultralytics."""

    def __init__(
        self,
        *,
        model_factory: Callable[[Path, str, str], Any] | None = None,
    ) -> None:
        self._model_factory = model_factory

    def _versions(self) -> tuple[str, str]:
        if self._model_factory is not None:
            return "injected", "injected"
        try:
            ultralytics = importlib.import_module("ultralytics")
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise YoloTrainingError(
                "Ultralytics training dependencies are unavailable; use the isolated "
                "yolo_perception environment"
            ) from exc
        return str(getattr(ultralytics, "__version__", "unknown")), str(
            getattr(torch, "__version__", "unknown")
        )

    def _create_model(self, path: Path, model_family: str, task: str) -> Any:
        if not path.is_file():
            raise YoloTrainingError(
                f"model does not exist: {path}; automatic model downloads are disabled"
            )
        family = _normalized_model_family(model_family)
        if self._model_factory is not None:
            try:
                return self._model_factory(path, family, task)
            except Exception as exc:
                raise YoloTrainingError(
                    f"cannot load model checkpoint {path}: {exc}"
                ) from exc
        try:
            ultralytics = importlib.import_module("ultralytics")
        except ImportError as exc:
            raise YoloTrainingError(
                "Ultralytics is not installed in this Python environment"
            ) from exc
        class_name = "YOLOE" if family == "yoloe" else "YOLO"
        model_type = getattr(ultralytics, class_name, None)
        if model_type is None:
            raise YoloTrainingError(
                f"installed Ultralytics does not provide {class_name}; "
                "update the isolated environment"
            )
        try:
            return model_type(str(path), task=task)
        except TypeError:
            try:
                return model_type(str(path))
            except Exception as exc:
                raise YoloTrainingError(
                    f"cannot load model checkpoint {path}: {exc}"
                ) from exc
        except Exception as exc:
            raise YoloTrainingError(f"cannot load model checkpoint {path}: {exc}") from exc

    def preflight(self, config: YoloTrainConfig) -> TrainingPreflight:
        diagnostics: list[str] = []
        report: DatasetValidationReport | None = None
        model_hash: str | None = None
        try:
            config.require_runtime_paths()
            assert config.base_model_path is not None
            assert config.dataset_yaml is not None
            model_hash = sha256_file(config.base_model_path)
            report = YoloDatasetValidator(task=config.task).validate(config.dataset_yaml)
            if not report.ok:
                diagnostics.extend(
                    f"dataset:{issue.code}: {issue.message} ({issue.path or '-'})"
                    for issue in report.errors
                )
            model_source = config.resume or config.base_model_path
            self._create_model(model_source, config.model_family, config.task)
        except (
            YoloTrainingConfigError,
            YoloTrainingError,
            ModelRegistryError,
            OSError,
            ValueError,
        ) as exc:
            diagnostics.append(str(exc))

        output_parent = _nearest_existing_parent(config.run_dir)
        if config.run_dir.exists() and not config.run_dir.is_dir():
            diagnostics.append(f"run output path is not a directory: {config.run_dir}")
        elif config.run_dir.exists() and config.resume is None:
            diagnostics.append(
                f"run output directory already exists: {config.run_dir}; choose a new run_name"
            )
        if not output_parent.is_dir() or not os.access(output_parent, os.W_OK):
            diagnostics.append(
                f"output directory is not writable: nearest existing parent {output_parent}"
            )

        indices = _cuda_indices(config.device)
        if indices is not None:
            try:
                torch = importlib.import_module("torch")
                available = bool(torch.cuda.is_available())
                count = int(torch.cuda.device_count()) if available else 0
            except ImportError:
                available = False
                count = 0
            if not available:
                diagnostics.append(
                    f"CUDA device {config.device!r} was requested but CUDA is unavailable"
                )
            elif any(index >= count for index in indices):
                diagnostics.append(
                    f"requested CUDA device {config.device!r}, but only "
                    f"{count} device(s) are visible"
                )
        elif isinstance(config.device, str) and config.device.lower() == "mps":
            try:
                torch = importlib.import_module("torch")
                if not bool(torch.backends.mps.is_available()):
                    diagnostics.append("MPS was requested but is unavailable")
            except (AttributeError, ImportError):
                diagnostics.append("MPS was requested but is unavailable")

        return TrainingPreflight(
            ok=not diagnostics,
            diagnostics=tuple(diagnostics),
            dataset_report=report,
            requested_device=config.device,
            output_directory=config.run_dir,
            model_sha256=model_hash,
        )

    def train(self, config: YoloTrainConfig) -> TrainingResult:
        preflight = self.preflight(config)
        if not preflight.ok or preflight.dataset_report is None:
            raise YoloTrainingError(
                "training preflight failed: " + "; ".join(preflight.diagnostics)
            )
        assert config.base_model_path is not None
        assert config.dataset_yaml is not None
        model_source = config.resume if config.resume is not None else config.base_model_path
        assert model_source is not None
        model = self._create_model(model_source, config.model_family, config.task)
        ultralytics_version, torch_version = self._versions()
        started_at = utc_now_iso()
        start = time.perf_counter()
        kwargs: dict[str, Any] = {
            "data": str(config.dataset_yaml),
            "epochs": config.epochs,
            "imgsz": config.imgsz,
            "batch": config.batch,
            "device": config.device,
            "workers": config.workers,
            "patience": config.patience,
            "seed": config.seed,
            "deterministic": config.deterministic,
            "amp": config.amp,
            "cache": config.cache,
            "project": str(config.project_dir),
            "name": config.run_name,
            "save": True,
            "save_period": -1,
            "exist_ok": False,
            "plots": True,
        }
        if config.resume is not None:
            kwargs["resume"] = True
        try:
            train_result = model.train(**kwargs)
        except Exception as exc:  # concrete framework boundary
            raise YoloTrainingError(f"Ultralytics training failed: {exc}") from exc
        elapsed = time.perf_counter() - start
        finished_at = utc_now_iso()
        run_dir = Path(getattr(train_result, "save_dir", config.run_dir)).resolve()
        best = run_dir / "weights" / "best.pt"
        last = run_dir / "weights" / "last.pt"
        if not best.is_file() or not last.is_file():
            raise YoloTrainingError(
                "training finished without required best.pt and last.pt under "
                f"{run_dir / 'weights'}"
            )
        required_outputs = (run_dir / "results.csv", run_dir / "args.yaml")
        missing_outputs = [str(path) for path in required_outputs if not path.is_file()]
        if missing_outputs:
            raise YoloTrainingError(
                "training finished without required metric/config outputs: "
                + ", ".join(missing_outputs)
            )
        self._prune_intermediate_checkpoints(run_dir / "weights")
        metrics = _result_metrics(train_result)
        self._organize_figures(run_dir)
        manifest = build_model_manifest(
            config=config,
            dataset_report=preflight.dataset_report,
            best_model_path=best,
            started_at=started_at,
            finished_at=finished_at,
            elapsed_s=elapsed,
            metrics=metrics,
            ultralytics_version=ultralytics_version,
            torch_version=torch_version,
        )
        manifest_path = write_model_manifest(run_dir, manifest)
        return TrainingResult(
            run_dir=run_dir,
            best_model_path=best,
            last_model_path=last,
            model_manifest_path=manifest_path,
            elapsed_s=elapsed,
            metrics=metrics,
        )

    @staticmethod
    def _prune_intermediate_checkpoints(weights_dir: Path) -> None:
        """Keep only the two documented resumable/deployable checkpoints."""

        for checkpoint in weights_dir.glob("*.pt"):
            if checkpoint.name not in {"best.pt", "last.pt"}:
                checkpoint.unlink()

    @staticmethod
    def _organize_figures(run_dir: Path) -> None:
        figures = run_dir / "figures"
        for source in run_dir.iterdir():
            if not source.is_file() or source.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            figures.mkdir(exist_ok=True)
            destination = figures / source.name
            if not destination.exists():
                shutil.move(str(source), str(destination))
        figures.mkdir(exist_ok=True)

    def validate(
        self,
        *,
        model_path: str | Path,
        dataset_yaml: str | Path,
        device: str | int,
        imgsz: int = 960,
        task: str = "detect",
        model_family: str = "yolo",
    ) -> ValidationResult:
        model_file = Path(model_path).expanduser().resolve()
        data_file = Path(dataset_yaml).expanduser().resolve()
        family = _normalized_model_family(model_family)
        normalized_task = str(task).strip().lower()
        report = YoloDatasetValidator(task=normalized_task).validate(data_file)
        if not report.ok:
            raise YoloTrainingError(
                "dataset validation failed: "
                + "; ".join(f"{issue.code}: {issue.message}" for issue in report.errors)
            )
        model = self._create_model(model_file, family, normalized_task)
        start = time.perf_counter()
        try:
            result = model.val(data=str(data_file), device=device, imgsz=imgsz, plots=False)
        except Exception as exc:  # concrete framework boundary
            raise YoloTrainingError(f"Ultralytics validation failed: {exc}") from exc
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return self._parse_validation_result(
            result,
            model_path=model_file,
            dataset_yaml=data_file,
            model_family=family,
            task=normalized_task,
            class_names=report.class_names,
            validation_target_area_px=report.target_area_px_by_split["val"],
            validation_image_count=report.split_image_counts.get("val", 0),
            total_elapsed_ms=elapsed_ms,
        )

    @staticmethod
    def _parse_validation_result(
        result: Any,
        *,
        model_path: Path,
        dataset_yaml: Path,
        model_family: str,
        task: str,
        class_names: Sequence[str],
        validation_target_area_px: Mapping[str, float | int | None],
        validation_image_count: int,
        total_elapsed_ms: float,
    ) -> ValidationResult:
        metric_kind = "seg" if task == "segment" else "box"
        fallback_kind = "box" if metric_kind == "seg" else "seg"
        box = getattr(result, metric_kind, None)
        if box is None:
            box = getattr(result, fallback_kind, None)
        raw = _result_metrics(result)
        metric_suffix = "M" if task == "segment" else "B"
        map50 = _float_or_none(getattr(box, "map50", None))
        map50_95 = _float_or_none(getattr(box, "map", None))
        precision = _float_or_none(getattr(box, "mp", None))
        recall = _float_or_none(getattr(box, "mr", None))
        if map50 is None:
            map50 = _float_or_none(raw.get(f"metrics/mAP50({metric_suffix})"))
        if map50_95 is None:
            map50_95 = _float_or_none(raw.get(f"metrics/mAP50-95({metric_suffix})"))
        if precision is None:
            precision = _float_or_none(raw.get(f"metrics/precision({metric_suffix})"))
        if recall is None:
            recall = _float_or_none(raw.get(f"metrics/recall({metric_suffix})"))

        empty_class_metrics = {
            "precision": None,
            "recall": None,
            "mAP50": None,
            "mAP50-95": None,
        }
        per_class: dict[str, dict[str, float | None]] = {
            name: dict(empty_class_metrics) for name in class_names
        }
        maps = getattr(box, "maps", None)
        class_result = getattr(box, "class_result", None)
        if callable(class_result):
            raw_class_indices = getattr(box, "ap_class_index", None)
            if raw_class_indices is None:
                raise YoloTrainingError(
                    "Ultralytics did not expose ap_class_index; refusing to guess "
                    "per-class metric rows"
                )
            try:
                class_indices = list(raw_class_indices)
            except TypeError as exc:
                raise YoloTrainingError(
                    "Ultralytics returned an invalid ap_class_index"
                ) from exc
            for result_index, raw_class_id in enumerate(class_indices):
                try:
                    class_id = int(raw_class_id)
                except (TypeError, ValueError) as exc:
                    raise YoloTrainingError(
                        "Ultralytics returned a non-integer ap_class_index"
                    ) from exc
                if class_id < 0 or class_id >= len(class_names):
                    raise YoloTrainingError(
                        f"Ultralytics ap_class_index {class_id} is outside the "
                        f"dataset class range [0, {len(class_names)})"
                    )
                try:
                    values = tuple(class_result(result_index))
                except (IndexError, TypeError, ValueError):
                    values = ()
                per_class[class_names[class_id]] = {
                    "precision": _float_or_none(values[0]) if len(values) > 0 else None,
                    "recall": _float_or_none(values[1]) if len(values) > 1 else None,
                    "mAP50": _float_or_none(values[2]) if len(values) > 2 else None,
                    "mAP50-95": _float_or_none(values[3]) if len(values) > 3 else None,
                }
        elif maps is not None:
            try:
                values = list(maps)
            except TypeError:
                values = []
            for index, name in enumerate(class_names):
                per_class[name] = {
                    "precision": None,
                    "recall": None,
                    "mAP50": None,
                    "mAP50-95": (
                        _float_or_none(values[index]) if index < len(values) else None
                    ),
                }
        speed = getattr(result, "speed", {})
        latency: dict[str, float | None] = {
            "preprocess": None,
            "inference": None,
            "postprocess": None,
            "total_validation": float(total_elapsed_ms),
            "wall_per_validation_image": (
                float(total_elapsed_ms) / validation_image_count
                if validation_image_count > 0
                else None
            ),
        }
        if isinstance(speed, Mapping):
            for key, value in speed.items():
                parsed = _float_or_none(value)
                if parsed is not None:
                    latency[str(key)] = parsed
        small_values = {
            "precision": _float_or_none(raw.get("metrics/precision(small)")),
            "recall": _float_or_none(raw.get("metrics/recall(small)")),
            "mAP50": _float_or_none(raw.get("metrics/mAP50(small)")),
            "mAP50-95": _float_or_none(raw.get("metrics/mAP50-95(small)")),
        }
        small_available = all(
            value is not None and 0.0 <= value <= 1.0
            for value in small_values.values()
        )
        small_metrics = {
            "definition": "bbox area < 32^2 pixels",
            "validation_ground_truth": _plain(validation_target_area_px),
            "available": small_available,
            **small_values,
            "note": (
                None
                if small_available
                else (
                    "Ultralytics did not expose a complete, finite set of area-binned "
                    "small-object metrics; validation and export are blocked and no "
                    "aggregate was fabricated"
                )
            ),
        }
        primary_metrics = (map50, map50_95, precision, recall)
        passed = small_available and all(
            value is not None and 0.0 <= value <= 1.0 for value in primary_metrics
        )
        return ValidationResult(
            model_path=model_path,
            model_sha256=sha256_file(model_path),
            dataset_yaml=dataset_yaml,
            model_family=model_family,
            task=task,
            passed=passed,
            map50=map50,
            map50_95=map50_95,
            precision=precision,
            recall=recall,
            per_class=per_class,
            small_target_metrics=small_metrics,
            latency_ms=latency,
            raw_metrics=raw,
        )

    def predict(
        self,
        *,
        model_path: str | Path,
        source: str | Path,
        device: str | int,
        confidence: float = 0.25,
        imgsz: int = 960,
        task: str = "detect",
        model_family: str = "yolo",
    ) -> PredictionResult:
        model_file = Path(model_path).expanduser().resolve()
        if not 0.0 <= confidence <= 1.0 or not math.isfinite(confidence):
            raise YoloTrainingError("prediction confidence must be finite and in [0, 1]")
        source_value = str(source)
        if not source_value:
            raise YoloTrainingError("prediction source cannot be empty")
        model = self._create_model(model_file, model_family, task)
        start = time.perf_counter()
        try:
            raw = model.predict(
                source=source_value,
                device=device,
                conf=confidence,
                imgsz=imgsz,
                save=False,
                stream=False,
            )
            results = tuple(raw)
        except Exception as exc:  # concrete framework boundary
            raise YoloTrainingError(f"Ultralytics prediction failed: {exc}") from exc
        return PredictionResult(
            model_path=model_file,
            source=source_value,
            image_count=len(results),
            elapsed_s=time.perf_counter() - start,
            results=results,
        )

    def export(
        self,
        *,
        model_path: str | Path,
        validation_report: str | Path,
        format: str,
        device: str | int,
        imgsz: int = 960,
        model_family: str = "yolo",
        freeze_yoloe_prompts: bool = False,
        half: bool = False,
    ) -> ExportResult:
        model_file = Path(model_path).expanduser().resolve()
        report_path = Path(validation_report).expanduser().resolve()
        family = _normalized_model_family(model_family)
        try:
            gate = load_validation_gate(
                report_path,
                model_path=model_file,
                model_family=family,
            )
        except ModelRegistryError as exc:
            raise YoloTrainingError(str(exc)) from exc
        normalized_format = format.strip().lower()
        if normalized_format == "tensorrt":
            normalized_format = "engine"
        if normalized_format not in {"onnx", "engine"}:
            raise YoloTrainingError("export format must be onnx or tensorrt/engine")
        if family == "yoloe" and not freeze_yoloe_prompts:
            raise YoloTrainingError(
                "YOLOE export statically freezes prompts; pass freeze_yoloe_prompts=True "
                "only when that behavior is intentional"
            )
        self._require_export_runtime(normalized_format, device)
        model = self._create_model(model_file, family, str(gate["task"]))
        try:
            exported = model.export(
                format=normalized_format,
                device=device,
                imgsz=imgsz,
                half=half,
                simplify=False,
            )
        except Exception as exc:  # concrete framework boundary
            raise YoloTrainingError(f"Ultralytics export failed: {exc}") from exc
        exported_path = Path(str(exported)).expanduser().resolve()
        if not exported_path.exists():
            raise YoloTrainingError(
                f"Ultralytics reported export {exported_path}, but the artifact is missing"
            )
        names = getattr(model, "names", ())
        if isinstance(names, Mapping):
            frozen_classes = [
                str(names[key]) for key in sorted(names, key=lambda item: str(item))
            ]
        elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
            frozen_classes = [str(name) for name in names]
        else:
            frozen_classes = []
        export_manifest_path = write_json(
            exported_path.with_suffix(exported_path.suffix + ".manifest.json"),
            {
                "schema_version": 1,
                "exported_at": utc_now_iso(),
                "source_model": {
                    "path": str(model_file),
                    "sha256": sha256_file(model_file),
                },
                "exported_model": {
                    "path": str(exported_path),
                    "sha256": sha256_file(exported_path),
                },
                "validation_report": str(report_path),
                "format": normalized_format,
                "model_family": family,
                "task": str(gate["task"]),
                "dynamic_prompts_supported": False,
                "yoloe_prompts_statically_frozen": family == "yoloe",
                "frozen_classes": frozen_classes,
            },
        )
        return ExportResult(
            model_path=model_file,
            exported_path=exported_path,
            format=normalized_format,
            validation_report=report_path,
            export_manifest_path=export_manifest_path,
            dynamic_prompts_supported=False,
        )

    def _require_export_runtime(self, format: str, device: str | int) -> None:
        """Fail before Ultralytics can try to install exporter dependencies."""

        # Injected models are an explicit test/application boundary and must not
        # depend on the host's optional exporter runtimes.
        if self._model_factory is not None:
            return
        try:
            importlib.import_module("onnx")
        except ImportError as exc:
            raise YoloTrainingError(
                "ONNX export dependency is unavailable. Recreate the pinned "
                "yolo_perception environment (onnx==1.22.0); automatic dependency "
                "installation is disabled."
            ) from exc
        if format != "engine":
            return
        try:
            indices = _cuda_indices(device)
        except (AttributeError, ValueError) as exc:
            raise YoloTrainingError(
                "TensorRT export requires an integer GPU index or cuda:<index>"
            ) from exc
        if indices is None or not indices or any(index < 0 for index in indices):
            raise YoloTrainingError(
                "TensorRT export requires an explicit CUDA device, not CPU/MPS"
            )
        try:
            torch = importlib.import_module("torch")
            cuda = getattr(torch, "cuda")
            visible_count = int(cuda.device_count()) if cuda.is_available() else 0
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            raise YoloTrainingError(
                "TensorRT export requires a working CUDA-enabled PyTorch runtime"
            ) from exc
        unavailable = [index for index in indices if index >= visible_count]
        if unavailable:
            raise YoloTrainingError(
                "TensorRT export requested unavailable CUDA device(s): "
                + ", ".join(str(index) for index in unavailable)
            )
        try:
            importlib.import_module("tensorrt")
        except ImportError as exc:
            raise YoloTrainingError(
                "TensorRT export is optional and its runtime is not installed. "
                "Install NVIDIA TensorRT manually for this host's CUDA/driver, then "
                "retry; automatic platform-package installation is disabled."
            ) from exc
