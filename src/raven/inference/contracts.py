"""Strict v1 remote contract. Binary XYZI never uses pickle/np.load or user paths."""

import hashlib
import json
import math
import re
from datetime import datetime

import numpy as np

from raven.core.point_cloud import PointCloud
from raven.perception.detection.predictions import detections_from_document

MAX_POINTS = 2_000_000
MAX_RESPONSE_BYTES = 16_000_000
REVISION = "233f849829b6ac19afb8af8837a0246890908755"
CHECKPOINT = "4c83fc0fa02575b9b3e9dec676f698e7a70bb5a795e89f91df8a96b916fa19e2"
MODEL_CONFIG = "170a9ffe76cfd8509d1044cfbcf1cbd44c5d320fda81bf0089a8d5efaf1c91c8"
BASE_CONFIG = "a6c44a7b0f15b94e135946cecbd8b5541906a69d37811705e9884a411e7b1797"
PACKAGES = {
    "torch": "2.2.2+cu118",
    "spconv-cu118": "2.3.6",
    "cumm-cu118": "0.4.11",
    "numpy": "1.26.4",
    "pcdet": "0.6.0+233f849",
}


def model_identity():
    return {
        "name": "pointpillars",
        "openpcdet_revision": REVISION,
        "checkpoint_sha256": CHECKPOINT,
        "config_sha256": MODEL_CONFIG,
        "base_config_sha256": BASE_CONFIG,
    }


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def keys(value, required, context):
    if not isinstance(value, dict) or set(value) != set(required):
        raise ValueError(f"{context}: expected fields {sorted(required)}")


def sha(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("expected lowercase SHA256")
    return value


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise ValueError("invalid remote identity")
    return value


def timestamp(value):
    if not isinstance(value, str):
        raise TypeError("timestamp must be timezone-aware ISO8601")
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset().total_seconds() != 0:
        raise ValueError("timestamp must be UTC")
    return result


def duration(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("duration must be finite nonnegative seconds")


def context_metadata(value):
    allowed = {"vehicle_id", "sensor_id", "session_id", "capture_timestamp_utc", "pose"}
    if not isinstance(value, dict) or value.keys() - allowed or len(canonical(value)) > 8192:
        raise ValueError("unsupported/oversized context metadata")
    for k in ("vehicle_id", "sensor_id", "session_id"):
        if k in value:
            identifier(value[k])
    if "capture_timestamp_utc" in value:
        timestamp(value["capture_timestamp_utc"])
    if "pose" in value:
        pose = value["pose"]
        keys(pose, {"coordinate_reference", "matrix"}, "pose")
        identifier(pose["coordinate_reference"])
        a = np.asarray(pose["matrix"])
        if a.shape != (4, 4) or a.dtype.kind not in "iuf" or not np.isfinite(a).all():
            raise ValueError("pose must be a finite numeric 4x4 matrix")
        if not np.array_equal(a[3], [0, 0, 0, 1]):
            raise ValueError("pose must be homogeneous")
    return value


def request_metadata(request):
    return {k: v for k, v in request.items() if k != "points"}


def create_request(
    cloud,
    *,
    request_id,
    experiment_id,
    config_sha256,
    preprocessing_sha256,
    submitted_at_utc,
    context=None,
):
    payload = cloud.points.astype("<f4").tobytes(order="C")
    request = {
        "schema_version": 1,
        "request_id": request_id,
        "experiment_id": experiment_id,
        "experiment_config_sha256": config_sha256,
        "frame_id": cloud.frame_id,
        "coordinate_frame": cloud.coordinate_frame,
        "preprocessing_sha256": preprocessing_sha256,
        "model": model_identity(),
        "input": {
            "points": cloud.count,
            "shape": [cloud.count, 4],
            "dtype": "<f4",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        },
        "submitted_at_utc": submitted_at_utc,
        "context": {} if context is None else context,
        "points": payload,
    }
    request["request_sha256"] = digest(request_metadata(request))
    validate_request(request)
    return request


def validate_request_metadata(request):
    keys(
        request,
        {
            "schema_version",
            "request_id",
            "experiment_id",
            "experiment_config_sha256",
            "frame_id",
            "coordinate_frame",
            "preprocessing_sha256",
            "model",
            "input",
            "submitted_at_utc",
            "context",
            "request_sha256",
        },
        "request",
    )
    if type(request["schema_version"]) is not int or request["schema_version"] != 1:
        raise ValueError("unsupported remote request schema")
    identifier(request["request_id"])
    identifier(request["experiment_id"])
    if not isinstance(request["frame_id"], str) or not re.fullmatch(
        r"[0-9]{6}", request["frame_id"]
    ):
        raise ValueError("invalid request frame_id")
    for k in ("experiment_config_sha256", "preprocessing_sha256", "request_sha256"):
        sha(request[k])
    timestamp(request["submitted_at_utc"])
    context_metadata(request["context"])
    if len(canonical(request)) > 16_384:
        raise ValueError("oversized request metadata")
    if request["coordinate_frame"] != "kitti_velodyne" or request["model"] != model_identity():
        raise ValueError("unsupported remote coordinate/model convention")
    source = request["input"]
    keys(source, {"points", "shape", "dtype", "sha256", "bytes"}, "input")
    n = source["points"]
    if type(n) is not int or not 1 <= n <= MAX_POINTS:
        raise ValueError("remote input requires 1..2,000,000 points")
    if (
        source["shape"] != [n, 4]
        or any(type(v) is not int for v in source["shape"])
        or source["dtype"] != "<f4"
        or type(source["bytes"]) is not int
        or source["bytes"] != n * 16
    ):
        raise ValueError("invalid XYZI shape/dtype/size")
    sha(source["sha256"])
    metadata = {k: v for k, v in request.items() if k != "request_sha256"}
    if digest(metadata) != request["request_sha256"]:
        raise ValueError("request envelope checksum mismatch")


def validate_request(request):
    if not isinstance(request, dict) or "points" not in request:
        raise ValueError("request requires binary points")
    validate_request_metadata(request_metadata(request))
    n = request["input"]["points"]
    source = request["input"]
    payload = request["points"]
    if type(payload) is not bytes or len(payload) != n * 16:
        raise ValueError("payload size/type mismatch")
    if hashlib.sha256(payload).hexdigest() != sha(source["sha256"]):
        raise ValueError("remote point-cloud checksum mismatch")
    points = np.frombuffer(payload, dtype="<f4").reshape(n, 4)
    if not np.isfinite(points).all():
        raise ValueError("remote points must be finite")
    return PointCloud(points, request["frame_id"], "kitti_velodyne")


def response_digest(response):
    return digest({k: v for k, v in response.items() if k != "response_sha256"})


def validate_response(response, request):
    validate_request(request)
    return validate_response_metadata(response, request_metadata(request))


def validate_response_metadata(response, request):
    validate_request_metadata(request)
    if len(canonical(response)) > MAX_RESPONSE_BYTES:
        raise ValueError("oversized response")
    keys(
        response,
        {
            "schema_version",
            "status",
            "request_id",
            "frame_id",
            "request_sha256",
            "input_sha256",
            "model",
            "predictions",
            "worker",
            "timings",
            "response_sha256",
        },
        "response",
    )
    if (
        type(response["schema_version"]) is not int
        or response["schema_version"] != 1
        or response["status"] != "success"
    ):
        raise ValueError("remote execution did not succeed with supported schema")
    for k in ("request_id", "frame_id", "request_sha256", "model"):
        if response[k] != request[k]:
            raise ValueError(f"response {k} mismatch")
    if response["input_sha256"] != request["input"]["sha256"]:
        raise ValueError("response input binding mismatch")
    if sha(response["response_sha256"]) != response_digest(response):
        raise ValueError("response checksum mismatch")
    detections = detections_from_document(
        response["predictions"], expected_frame_id=request["frame_id"]
    )
    worker = response["worker"]
    keys(
        worker,
        {
            "provider",
            "worker_id",
            "image_recipe_sha256",
            "gpu",
            "compute_capability",
            "python",
            "packages",
            "cuda_runtime",
            "model_provenance",
            "raven_source_sha256",
        },
        "worker",
    )
    if (
        worker["provider"] != "modal"
        or worker["gpu"] != "Tesla T4"
        or worker["compute_capability"] != [7, 5]
    ):
        raise ValueError("unsupported worker GPU/provider")
    identifier(worker["worker_id"])
    sha(worker["image_recipe_sha256"])
    sha(worker["raven_source_sha256"])
    from raven.inference.modal_image import recipe_sha256
    from raven.inference.worker import runtime_source_sha256

    if (
        worker["image_recipe_sha256"] != recipe_sha256()
        or worker["raven_source_sha256"] != runtime_source_sha256()
    ):
        raise ValueError("worker recipe/Raven source mismatch")
    if (
        worker["packages"] != PACKAGES
        or worker["python"] != "3.10.16"
        or worker["cuda_runtime"] != "11.8"
    ):
        raise ValueError("worker dependency mismatch")
    model = worker["model_provenance"]
    if (
        not isinstance(model, dict)
        or model.get("openpcdet_commit") != REVISION
        or model.get("checkpoint_sha256") != CHECKPOINT
        or model.get("config", {}).get("sha256") != MODEL_CONFIG
        or model.get("synthetic_test")
        or model.get("not_real_gpu_inference")
    ):
        raise ValueError("unverified worker model provenance")
    times = response["timings"]
    keys(
        times,
        {
            "worker_started_at_utc",
            "worker_finished_at_utc",
            "model_load_seconds",
            "detect_seconds",
            "worker_total_seconds",
            "clock",
            "gpu_synchronized",
        },
        "timings",
    )
    if timestamp(times["worker_finished_at_utc"]) < timestamp(times["worker_started_at_utc"]):
        raise ValueError("worker timestamps reversed")
    for k in ("model_load_seconds", "detect_seconds", "worker_total_seconds"):
        duration(times[k])
    if times["worker_total_seconds"] < times["detect_seconds"]:
        raise ValueError("worker duration is shorter than detector duration")
    if times["clock"] != "time.perf_counter" or times["gpu_synchronized"] is not True:
        raise ValueError("unsupported measurement boundary")
    return detections
