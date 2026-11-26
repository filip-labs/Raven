"""CPU-importable worker logic; genuine existing PointPillars adapter loaded on GPU startup."""

import hashlib
import importlib.metadata
import platform
import subprocess
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from raven.inference.contracts import (
    BASE_CONFIG,
    CHECKPOINT,
    MODEL_CONFIG,
    PACKAGES,
    model_identity,
    response_digest,
    validate_request,
    validate_response,
)
from raven.inference.modal_image import recipe_sha256
from raven.perception.detection.predictions import detections_to_document


def runtime_source_sha256():
    root = Path(__file__).resolve().parents[1]
    names = [
        "inference/contracts.py",
        "inference/worker.py",
        "inference/modal_image.py",
        "inference/patch_argo.py",
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


class WorkerSession:
    """One model per container; bounded recent-response replay cache, never persistent exactly-once."""

    def __init__(self):
        started = time.perf_counter()
        import os

        import torch

        from raven.perception.detection.pointpillars import PointPillarsDetector

        packages = {name: importlib.metadata.version(name) for name in PACKAGES}
        if (
            platform.python_version() != "3.10.16"
            or packages != PACKAGES
            or not torch.cuda.is_available()
            or torch.version.cuda != "11.8"
            or torch.cuda.get_device_capability(0) != (7, 5)
        ):
            raise RuntimeError("candidate image/GPU compatibility check failed")
        root = Path("/opt/OpenPCDet")
        for path, sha in (
            (root / "tools/cfgs/kitti_models/pointpillar.yaml", MODEL_CONFIG),
            (root / "tools/cfgs/dataset_configs/kitti_dataset.yaml", BASE_CONFIG),
        ):
            if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                raise ValueError("worker model configuration checksum mismatch")
        self.detector = PointPillarsDetector(
            openpcdet_root=root,
            config=root / "tools/cfgs/kitti_models/pointpillar.yaml",
            checkpoint="/weights/pointpillar.pth",
            checkpoint_sha256=CHECKPOINT,
            trust_checkpoint=os.environ.get("RAVEN_TRUST_CHECKPOINT") == "1",
        )
        self.detector.initialize()
        provenance = self.detector.provenance
        if provenance["openpcdet_modified_files"] != ["pcdet/datasets/__init__.py"]:
            raise ValueError("unexpected upstream source modifications")
        patch = subprocess.check_output(["git", "-C", str(root), "diff", "--binary", "HEAD", "--"])
        provenance["openpcdet_diff_sha256"] = hashlib.sha256(patch).hexdigest()
        from raven.inference.patch_argo import PREIMAGE

        backup = root / "pcdet/datasets/__init__.py.bak"
        if hashlib.sha256(backup.read_bytes()).hexdigest() != PREIMAGE:
            raise ValueError("Argoverse workaround backup checksum mismatch")
        provenance["argo_workaround"] = {
            "backup_sha256": PREIMAGE,
            "patched_sha256": hashlib.sha256(
                (root / "pcdet/datasets/__init__.py").read_bytes()
            ).hexdigest(),
        }
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - started
        self.metadata = {
            "provider": "modal",
            "worker_id": uuid.uuid4().hex,
            "image_recipe_sha256": recipe_sha256(),
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "python": platform.python_version(),
            "packages": packages,
            "cuda_runtime": torch.version.cuda,
            "model_provenance": provenance,
            "raven_source_sha256": runtime_source_sha256(),
        }
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
            "worker": deepcopy(self.metadata),
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
        response["response_sha256"] = response_digest(response)
        validate_response(response, request)
        if len(self.cache) == 16:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = (request["request_sha256"], deepcopy(response))
        return response

    def close(self):
        self.detector.close()
