"""Billable integration test: disabled unless explicitly configured AND both opt-ins are set."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from raven.evaluation.evaluate import evaluate_experiment
from raven.experiments.artifacts import write_json_exclusive
from raven.experiments.config import load_experiment_config
from raven.experiments.runner import run_experiment
from raven.inference.modal_transport import check_opt_in
from raven.perception.data.kitti_loader import load_kitti_bin
from raven.perception.detection.predictions import load_detections
from raven.perception.visualization.boxes import prediction_corners


@pytest.mark.skipif(
    not os.environ.get("RAVEN_MODAL_SMOKE_CONFIG"), reason="live Modal GPU explicitly opt-in"
)
def test_authorized_modal_frame_000000():
    check_opt_in()
    config = load_experiment_config(os.environ["RAVEN_MODAL_SMOKE_CONFIG"])
    assert config.model.source == "modal" and config.frame_ids == ("000000",)
    scan = config.dataset.scan_directory / "000000.bin"
    before = hashlib.sha256(scan.read_bytes()).hexdigest()
    assert before == "0e09c85e3f6078ecbdd1e706ee9624519f1bd29417437167a9ed7fbe6f54b4b1"
    assert load_kitti_bin(scan).count == 115384
    result = run_experiment(config)
    assert result.status == "success", result.failures
    records = json.loads(config.artifact_paths["results"].read_text())["frames"]
    frame = records[0]
    document = frame["predictions"]
    document["input"].update(sha256=frame["processed_xyzi_sha256"], points=frame["processed_count"])
    document["preprocessing"] = frame["preprocessing"]
    document["remote_execution"] = frame["prediction_source"]["model_provenance"]
    path = config.run_directory / "000000.predictions.json"
    write_json_exclusive(path, document)
    detections = load_detections(path, expected_frame_id="000000")
    for box in detections:
        assert np.isfinite(prediction_corners(box, "kitti_velodyne")).all()
    evaluation = evaluate_experiment(
        config.run_directory / f"{config.output.artifact_prefix}.manifest.json",
        allow_training_smoke=True,
    )
    # Actual image/GT metadata can block AP; contract incompatibility cannot count as success.
    assert evaluation.status in ("success", "blocked"), evaluation.failures
    write_json_exclusive(config.run_directory / "smoke-evaluation.json", evaluation.to_dict())
    assert hashlib.sha256(scan.read_bytes()).hexdigest() == before
    assert Path(path).is_file()
