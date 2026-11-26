"""Integrity-checked RV-031 artifact and source-input loading; no alternate runner."""

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from raven.experiments.config import ExperimentConfig
from raven.experiments.runner import FileIdentity, FrameInputs, load_experiment_frame
from raven.experiments.splits import split_membership
from raven.perception.data.frame_manifest import validate_frame_ids
from raven.perception.detection.predictions import detections_from_document


class EvaluationInvalid(ValueError):
    """Invalid/corrupted/inconsistent evidence; no metric should be interpreted."""


class EvaluationBlocked(RuntimeError):
    """Required authentic metadata or supported evaluation scope is unavailable."""


def strict_json(raw):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise EvaluationInvalid(f"duplicate JSON key: {k}")
            result[k] = v
        return result

    def constant(value):
        raise EvaluationInvalid(f"nonfinite JSON constant: {value}")

    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    json.dumps(result, allow_nan=False)
    return result


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise EvaluationInvalid("expected lowercase SHA256")
    return value


def _read(path):
    if not path.is_file():
        raise EvaluationInvalid(f"required run artifact unavailable: {path}")
    return path.read_bytes()


@dataclass(frozen=True)
class CompletedRun:
    manifest_path: Path
    manifest_sha256: str
    config: ExperimentConfig
    results: dict
    provenance: dict
    artifact_identities: dict


def load_completed_run(path: str | Path) -> CompletedRun:
    path = Path(path).resolve()
    raw = _read(path)
    manifest = strict_json(raw)
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("run_schema_version")) is not int
        or manifest["run_schema_version"] != 1
    ):
        raise EvaluationInvalid("unsupported run manifest schema")
    if manifest.get("status") != "success":
        raise EvaluationInvalid("evaluation requires a completed successful run")
    ids = validate_frame_ids(manifest.get("frame_ids"))
    digest = _sha(manifest.get("config_sha256"))
    records = manifest.get("artifacts")
    required = {"configuration", "frames", "results", "provenance"}
    if not isinstance(records, dict) or not required <= records.keys():
        raise EvaluationInvalid("missing required run artifact records")
    loaded, identities, filenames = {}, {}, set()
    for key, record in records.items():
        if not isinstance(record, dict) or set(record) != {"file", "sha256"}:
            raise EvaluationInvalid("malformed artifact identity")
        name = record["file"]
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in (".", "..")
            or name in filenames
        ):
            raise EvaluationInvalid("unsafe or duplicate artifact filename")
        filenames.add(name)
        artifact = path.parent / name
        if artifact.resolve().parent != path.parent or artifact.is_symlink():
            raise EvaluationInvalid("artifact redirects outside its run directory")
        data = _read(artifact)
        checksum = hashlib.sha256(data).hexdigest()
        if checksum != _sha(record["sha256"]):
            raise EvaluationInvalid(f"artifact checksum mismatch: {name}")
        identities[key] = {"file": name, "sha256": checksum}
        loaded[key] = data.decode("utf-8").splitlines() if key == "frames" else strict_json(data)
    if tuple(loaded["frames"]) != ids:
        raise EvaluationInvalid("frame manifest disagrees with run identity")
    config = ExperimentConfig.from_dict(loaded["configuration"], base_dir=path.parent)
    if config.config_sha256 != digest or config.frame_ids != ids:
        raise EvaluationInvalid("normalized configuration identity disagrees with run manifest")
    results, provenance = loaded["results"], loaded["provenance"]
    for document in (results, provenance):
        if (
            not isinstance(document, dict)
            or type(document.get("run_schema_version")) is not int
            or document["run_schema_version"] != 1
        ):
            raise EvaluationInvalid("unsupported result/provenance schema")
    summary = results.get("summary", {})
    if (
        type(summary.get("run_schema_version")) is not int
        or summary.get("run_schema_version") != 1
        or summary.get("status") != "success"
        or type(summary.get("failed")) is not int
        or summary.get("failed") != 0
        or summary.get("failures") != []
        or type(summary.get("succeeded")) is not int
        or summary["succeeded"] != len(ids)
        or summary.get("selected_frame_ids") != list(ids)
        or summary.get("config_sha256") != digest
    ):
        raise EvaluationInvalid("run summary is not a complete successful execution")
    frames = results.get("frames")
    if (
        not isinstance(frames, list)
        or len(frames) != len(ids)
        or {f.get("frame_id") for f in frames} != set(ids)
    ):
        raise EvaluationInvalid("result frame set is missing, duplicated or unexpected")
    if any(f.get("status") != "success" for f in frames):
        raise EvaluationInvalid("failed frame cannot contribute evaluation metrics")
    if (
        provenance.get("config_sha256") != digest
        or provenance.get("configuration") != config.to_dict()
    ):
        raise EvaluationInvalid("run provenance/configuration disagreement")
    if provenance.get("selection", {}).get("selected_frame_ids") != list(ids):
        raise EvaluationInvalid("provenance selection mismatch")
    return CompletedRun(
        path, hashlib.sha256(raw).hexdigest(), config, results, provenance, identities
    )


def prepare_sources(run: CompletedRun, *, allow_training_smoke: bool):
    config = run.config
    if config.dataset.split == "test":
        raise EvaluationBlocked("unlabeled KITTI test split cannot provide measured AP")
    if config.dataset.split != "val" and not allow_training_smoke:
        raise EvaluationBlocked(
            "meaningful evaluation requires val; training needs explicit --allow-training-smoke"
        )
    members, evidence = split_membership(config.dataset.split)
    if not set(config.frame_ids) <= members:
        raise EvaluationInvalid("evaluated frame is outside declared canonical split")
    if run.provenance.get("split") != evidence:
        raise EvaluationInvalid("split evidence differs from the source run")
    source_records = run.provenance.get("inputs")
    if not isinstance(source_records, list) or len(source_records) != len(config.frame_ids):
        raise EvaluationInvalid("missing per-frame input provenance")
    source_by_id = {v["frame_id"]: v["files"] for v in source_records}
    if set(source_by_id) != set(config.frame_ids):
        raise EvaluationInvalid("input provenance frame mismatch")
    result_by_id = {v["frame_id"]: v for v in run.results["frames"]}
    frames, warnings = [], []
    for frame_id in config.frame_ids:
        record = result_by_id[frame_id]
        files = record.get("inputs")
        if files != source_by_id[frame_id]:
            raise EvaluationInvalid("frame input identities disagree with provenance")
        root = config.dataset.scan_directory.parent
        paths = {
            "scan": config.dataset.scan_directory / f"{frame_id}.bin",
            "labels": root / "label_2" / f"{frame_id}.txt",
            "calibration": root / "calib" / f"{frame_id}.txt",
        }
        if config.model.source == "saved":
            paths["predictions"] = config.model.predictions_dir / f"{frame_id}.json"
        if not isinstance(files, dict) or set(files) != set(paths):
            raise EvaluationInvalid("missing or unexpected frame input identities")
        identities = []
        for key, path in paths.items():
            item = files[key]
            if not isinstance(item, dict) or set(item) != {"path", "sha256", "bytes"}:
                raise EvaluationInvalid("malformed source file identity")
            if (
                Path(item["path"]) != path.resolve()
                or type(item["bytes"]) is not int
                or item["bytes"] < 0
            ):
                raise EvaluationInvalid("source path/size identity mismatch")
            identities.append(
                (key, FileIdentity(Path(item["path"]), _sha(item["sha256"]), item["bytes"]))
            )
        try:
            _, cloud, labels, calibration, saved, source = load_experiment_frame(
                config, frame_id, FrameInputs(frame_id, tuple(identities))
            )
        except FileNotFoundError as exc:
            raise EvaluationBlocked(f"source input unavailable: {exc}") from exc
        detections = detections_from_document(record.get("predictions"), expected_frame_id=frame_id)
        if (
            type(record.get("input_count")) is not int
            or record["input_count"] != cloud.count
            or type(record.get("prediction_count")) is not int
            or record["prediction_count"] != len(detections)
            or record.get("class_counts") != dict(Counter(d.label for d in detections))
        ):
            raise EvaluationInvalid("source result point/prediction counts disagree")
        if type(record.get("processed_count")) is not int or record["processed_count"] < 0:
            raise EvaluationInvalid("invalid processed point count")
        _sha(record.get("processed_xyzi_sha256"))
        if config.model.source == "saved" and (
            record["processed_count"] != cloud.count
            or record["processed_xyzi_sha256"] != files["scan"]["sha256"]
            or record.get("preprocessing", {}).get("stages") != []
        ):
            raise EvaluationInvalid("saved baseline cannot claim new point-cloud preprocessing")
        for annotation in labels.records:
            if annotation.object_type in ("Car", "Pedestrian", "Cyclist") and (
                annotation.occlusion < 0
                or annotation.truncation < 0
                or all(v == -1 for v in annotation.bbox_xyxy)
            ):
                raise EvaluationBlocked(
                    f"required ground-truth visibility/image-box metadata unavailable: {frame_id}"
                )
        expected_gt = {
            "coordinate_frame": labels.coordinate_frame,
            "objects": [asdict(v) for v in labels.objects],
            "ignored_regions": [asdict(v) for v in labels.ignored_regions],
            "calibration": {k: v.tolist() for k, v in calibration.matrices.items()},
        }
        if record.get("ground_truth") != json.loads(json.dumps(expected_gt)):
            raise EvaluationInvalid(
                "persisted ground truth/calibration disagrees with hashed source"
            )
        supplied_source = record.get("prediction_source", {})
        if config.model.source == "saved":
            if detections != saved or supplied_source != source:
                raise EvaluationInvalid("persisted detections/provenance differ from saved source")
            if source["input_binding"] != "sha256_verified":
                raise EvaluationBlocked("saved prediction-to-scan binding is unavailable")
            warnings.append(
                "saved predictions: GPU execution/model provenance is declared, not independently authenticated"
            )
        else:
            if (
                supplied_source.get("mode")
                != ("remote_adapter" if config.model.source == "modal" else "live_adapter")
                or supplied_source.get("input_binding") != "executed_on_recorded_processed_cloud"
            ):
                raise EvaluationInvalid("live-source execution binding is incompatible")
            model = supplied_source.get("model_provenance")
            if not isinstance(model, dict) or not model:
                raise EvaluationBlocked("live model provenance unavailable")
            if config.model.source == "modal":
                from raven.inference.evidence import validate_remote_evidence

                if validate_remote_evidence(model, record, config) != detections:
                    raise EvaluationInvalid("remote response differs from persisted detections")
                warnings.append(
                    "remote worker evidence is consistency-checked, not independently attested GPU execution"
                )
            if model.get("synthetic_test") or model.get("not_real_gpu_inference"):
                raise EvaluationBlocked(
                    "source declares mocked inference, not genuine GPU evidence"
                )
        frames.append(
            (frame_id, labels, calibration, detections, root / "image_2" / f"{frame_id}.png", files)
        )
    purpose = (
        "training_smoke"
        if config.dataset.split == "train"
        else ("validation_full" if set(config.frame_ids) == members else "validation_subset")
    )
    if purpose == "training_smoke":
        warnings.append("training smoke only; not representative validation accuracy")
    if purpose == "validation_subset":
        warnings.append("validation subset only; not the complete canonical validation benchmark")
    return frames, purpose, evidence, list(dict.fromkeys(warnings))
