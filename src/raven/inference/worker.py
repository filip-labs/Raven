"""CPU-importable worker logic; genuine existing PointPillars adapter loaded on GPU startup."""

import hashlib
import importlib.metadata
import platform
import time
import uuid
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from raven.inference.contracts import (
    CHECKPOINT,
    PACKAGES,
    model_identity,
    response_digest,
    supported_python,
    validate_request,
    validate_response,
)
from raven.inference.image_validation import validate_cpu_environment
from raven.inference.json_native import normalize_json_containers
from raven.inference.modal_image import recipe_sha256
from raven.perception.detection.predictions import detections_to_document


def runtime_source_sha256():
    root = Path(__file__).resolve().parents[1]
    names = [
        "inference/contracts.py",
        "inference/json_native.py",
        "inference/worker.py",
        "inference/modal_image.py",
        "inference/patch_argo.py",
        "inference/image_validation.py",
        "perception/detection/pointpillars.py",
        "perception/detection/_openpcdet.py",
        "perception/detection/base.py",
        "perception/detection/predictions.py",
        "core/point_cloud.py",
    ]
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


class WorkerInitializationError(RuntimeError):
    """Safe phase/cause code for caught startup failures returned through the ordinary method."""


@contextmanager
def startup_stage(stage):
    try:
        yield
    except Exception as exc:
        # Detailed cause/traceback remains in container logs; no arbitrary remote message is exposed.
        if isinstance(exc, RuntimeError) and "Checkpoint loading failed" in str(exc):
            stage = "safe_checkpoint_loading"
        raise WorkerInitializationError(
            f"Raven worker initialization failed; stage={stage}; cause={type(exc).__name__}"
        ) from exc


class WorkerSession:
    """One model per container; bounded recent-response replay cache, never persistent exactly-once."""

    def __init__(self):
        started = time.perf_counter()
        import os

        with startup_stage("dependency_import"):
            import torch

        from raven.perception.detection.pointpillars import PointPillarsDetector

        with startup_stage("runtime_profile"):
            packages = {name: str(importlib.metadata.version(name)) for name in PACKAGES}
            if (
                not supported_python(platform.python_version())
                or packages != PACKAGES
                or not torch.cuda.is_available()
                or torch.version.cuda != "11.8"
                or torch.cuda.get_device_capability(0) != (7, 5)
                or torch.cuda.get_device_name(0) != "Tesla T4"
            ):
                raise RuntimeError("candidate image/GPU compatibility check failed")
        root = Path("/opt/OpenPCDet")
        with startup_stage("dependency_source_validation"):
            cpu_validation = validate_cpu_environment(root)
        with startup_stage("model_initialization"):
            self.detector = PointPillarsDetector(
                openpcdet_root=root,
                config=root / "tools/cfgs/kitti_models/pointpillar.yaml",
                checkpoint="/weights/pointpillar.pth",
                checkpoint_sha256=CHECKPOINT,
                trust_checkpoint=os.environ.get("RAVEN_TRUST_CHECKPOINT") == "1",
            )
            try:
                self.detector.initialize()
                provenance = self.detector.provenance
                checked = cpu_validation["source"]
                if (
                    provenance["openpcdet_commit"] != checked["openpcdet_commit"]
                    or provenance["openpcdet_modified_files"] != checked["openpcdet_modified_files"]
                ):
                    raise ValueError("adapter/source provenance mismatch")
                provenance.update(checked)
                provenance["image_cpu_validation"] = cpu_validation
            except Exception:
                self.detector.close()
                raise
        with startup_stage("cuda_ready"):
            torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - started
        self.metadata = normalize_json_containers(
            {
                "provider": "modal",
                "worker_id": uuid.uuid4().hex,
                "image_recipe_sha256": recipe_sha256(),
                "gpu": str(torch.cuda.get_device_name(0)),
                "compute_capability": [int(v) for v in torch.cuda.get_device_capability(0)],
                "python": platform.python_version(),
                "packages": packages,
                "cuda_runtime": str(torch.version.cuda),
                "model_provenance": provenance,
                "raven_source_sha256": runtime_source_sha256(),
            }
        )
        self.cache = {}

    def infer(self, request):
        import torch

        started, utc = time.perf_counter(), now()
        cloud = validate_request(request)
        key = request["request_id"]
        if key in self.cache:
            envelope, response = self.cache[key]
            if envelope != request["request_sha256"]:
                raise ValueError("request ID reused for different input")
            validate_response(response, request)
            return deepcopy(response)
        torch.cuda.synchronize()
        detect_started = time.perf_counter()
        predictions = self.detector.detect(cloud)
        torch.cuda.synchronize()
        detect_seconds = time.perf_counter() - detect_started
        response = {
            "schema_version": 1,
            "status": "success",
            "request_id": key,
            "frame_id": cloud.frame_id,
            "request_sha256": request["request_sha256"],
            "input_sha256": request["input"]["sha256"],
            "model": model_identity(),
            "predictions": detections_to_document(predictions, frame_id=cloud.frame_id),
            "worker": self.metadata,  # Copied/checked by normalize_json_containers below.
            "timings": {
                "worker_started_at_utc": utc,
                "worker_finished_at_utc": now(),
                "model_load_seconds": self.load_seconds,
                "detect_seconds": detect_seconds,
                "worker_total_seconds": time.perf_counter() - started,
                "clock": "time.perf_counter",
                "gpu_synchronized": True,
            },
        }
        response = normalize_json_containers(response)
        response["response_sha256"] = response_digest(response)
        validate_response(response, request)
        if len(self.cache) == 16:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = (request["request_sha256"], deepcopy(response))
        return response

    def close(self):
        self.detector.close()
