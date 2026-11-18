import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from raven.cli import main
from raven.experiments.config import (
    ExperimentConfig,
    ExperimentConfigurationError,
    FrameSelection,
    load_experiment_config,
    save_experiment_config,
)
from raven.perception.preprocessing import pipeline


@pytest.fixture
def document():
    return {
        "schema_version": 1,
        "name": "Raw baseline",
        "experiment_id": "baseline-001",
        "dataset": {"root": "dataset", "split": "val"},
        "frames": {"ids": ["000003", "000001", "000002", "000000", "000004"]},
        "model": {"name": "pointpillars", "source": "saved", "predictions_dir": "predictions"},
        "preprocessing": [],
        "seed": 42,
        "output": {"directory": "outputs", "artifact_prefix": "baseline"},
    }


def config(document, tmp_path):
    return ExperimentConfig.from_dict(document, base_dir=tmp_path)


def inference(document):
    document["model"] = {
        "name": "pointpillars",
        "source": "inference",
        "config": "model.yaml",
        "checkpoint": "weights.pth",
        "checkpoint_sha256": "a" * 64,
        "openpcdet_revision": "233f849829b6ac19afb8af8837a0246890908755",
    }
    return document


def test_roundtrip_identity_order_detachment_and_artifact_names(document, tmp_path):
    original = copy.deepcopy(document)
    c = config(document, tmp_path)
    assert document == original
    assert c.frame_ids == tuple(original["frames"]["ids"])
    assert c.dataset.scan_directory == tmp_path / "dataset/training/velodyne"
    normalized = c.to_dict()
    expected = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    assert c.config_sha256 == expected
    save_experiment_config(tmp_path / "saved.json", c)
    loaded = load_experiment_config(tmp_path / "saved.json")
    assert loaded == c and loaded.config_sha256 == c.config_sha256
    assert (
        loaded.artifact_paths["configuration"]
        == tmp_path / "outputs/baseline-001/baseline.config.json"
    )
    normalized["frames"]["ids"].clear()
    document["frames"]["ids"].clear()
    assert c.frame_ids == tuple(original["frames"]["ids"])
    with pytest.raises(FrozenInstanceError):
        c.seed = 5
    with pytest.raises(FileExistsError):
        save_experiment_config(tmp_path / "saved.json", c)


def test_seeded_selection_golden_order_and_no_global_rng_change(document, tmp_path):
    document["frames"]["sample_size"] = 3
    before = np.random.get_state()
    c = config(document, tmp_path)
    assert c.frame_ids == ("000003", "000000", "000004")  # PCG64(42), indexes 0,3,4.
    assert c.frame_ids == c.frame_ids
    assert replace(c, seed=7).frame_ids != c.frame_ids
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert config(dict(reversed(list(document.items()))), tmp_path).config_sha256 == c.config_sha256
    assert replace(c, seed=7).config_sha256 != c.config_sha256


def test_manifest_hash_snapshot_reload_change_and_sampling(document, tmp_path):
    manifest = tmp_path / "frames.txt"
    raw = b"000003\r\n000001\r\n000000\r\n"
    manifest.write_bytes(raw)
    document["frames"] = {"manifest": "frames.txt", "sample_size": 2}
    c = config(document, tmp_path)
    assert c.frames.sha256 == hashlib.sha256(raw).hexdigest()
    save_experiment_config(tmp_path / "config.json", c)
    assert load_experiment_config(tmp_path / "config.json") == c
    ids, identity = c.frame_ids, c.config_sha256
    manifest.write_text("000000\n000001\n000003\n")
    assert c.frame_ids == ids and c.config_sha256 == identity  # Frozen snapshot.
    with pytest.raises(ExperimentConfigurationError, match="SHA256 mismatch"):
        load_experiment_config(tmp_path / "config.json")


@pytest.mark.parametrize(
    "stages",
    [
        [],
        [{"name": "noise", "position_std": 0, "seed": 7}],
        [{"name": "voxel_downsample", "voxel_size": [0.1, 0.2, 0.3]}],
    ],
)
def test_existing_stage_format_without_execution(document, tmp_path, monkeypatch, stages):
    inference(document)["preprocessing"] = stages
    operation = pytest.fail
    monkeypatch.setattr(pipeline, "run_preprocessing_pipeline", operation)
    c = config(document, tmp_path)
    assert c.to_dict()["preprocessing"] == stages
    assert ExperimentConfig.from_dict(c.to_dict(), base_dir=tmp_path) == c
    if c.preprocessing:
        with pytest.raises(TypeError):
            c.preprocessing[0]["name"] = "other"


def test_repository_configs_and_stage_order_preserved(document, tmp_path):
    root = Path(__file__).resolve().parents[1]
    for path in (root / "configs/preprocessing").glob("*.json"):
        stages = json.loads(path.read_text())
        inference(document)["preprocessing"] = stages
        c = config(document, tmp_path)
        assert c.to_dict()["preprocessing"] == stages
    stages = [
        {"name": "noise", "position_std": 0.01, "seed": 12},
        {"name": "voxel_downsample", "voxel_size": 0.2},
    ]
    document["preprocessing"] = stages + stages
    c = config(document, tmp_path)
    assert [s["name"] for s in c.preprocessing] == [s["name"] for s in stages + stages]
    document["preprocessing"] = list(reversed(stages)) + stages
    assert config(document, tmp_path).config_sha256 != c.config_sha256
    assert c.preprocessing[0]["seed"] == 12  # Experiment seed does not replace stage seeds.


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "name",
        "experiment_id",
        "dataset",
        "frames",
        "model",
        "preprocessing",
        "seed",
        "output",
    ],
)
def test_missing_fields(document, tmp_path, field):
    del document[field]
    with pytest.raises(ExperimentConfigurationError, match="missing fields"):
        config(document, tmp_path)


@pytest.mark.parametrize(
    "section,key,value",
    [
        (None, "schema_version", True),
        (None, "schema_version", 2),
        (None, "name", " "),
        (None, "name", "\ud800"),
        (None, "experiment_id", "../evil"),
        (None, "seed", True),
        (None, "seed", -1),
        (None, "seed", 1.5),
        (None, "seed", 2**64),
        (None, "extra", 1),
        ("dataset", "split", "training"),
        ("dataset", "root", ""),
        ("dataset", "root", 4),
        ("dataset", "root", "../data"),
        ("dataset", "root", "~/data"),
        ("frames", "ids", []),
        ("frames", "ids", ["000000", "000000"]),
        ("frames", "ids", ["１２３４５６"]),
        ("frames", "ids", [0]),
        ("frames", "ids", "000000"),
        ("frames", "sample_size", 0),
        ("frames", "sample_size", 6),
        ("frames", "sample_size", True),
        ("frames", "sample_size", None),
        ("frames", "sha256", "a" * 64),
        ("model", "name", "other"),
        ("model", "source", "other"),
        ("model", "checkpoint", "weights.pth"),
        ("model", "config", None),
        ("output", "artifact_prefix", "../file"),
        ("output", "directory", "/"),
        ("output", "directory", str(Path.home())),
        ("output", "directory", "dataset/sub"),
        ("output", "directory", "predictions/sub"),
        ("output", "directory", "${HOME}/output"),
    ],
)
def test_invalid_settings(document, tmp_path, section, key, value):
    (document if section is None else document[section])[key] = value
    with pytest.raises(ExperimentConfigurationError):
        config(document, tmp_path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("checkpoint_sha256", "bad"),
        ("openpcdet_revision", "abc"),
        ("config", None),
        ("predictions_dir", "predictions"),
    ],
)
def test_inference_incompatible_settings(document, tmp_path, key, value):
    inference(document)["model"][key] = value
    with pytest.raises(ExperimentConfigurationError):
        config(document, tmp_path)


def test_inference_missing_field_and_saved_preprocessing(document, tmp_path):
    document["preprocessing"] = [{"name": "noise", "position_std": 0, "seed": 42}]
    with pytest.raises(ExperimentConfigurationError, match="saved predictions"):
        config(document, tmp_path)
    inference(document)
    del document["model"]["checkpoint"]
    with pytest.raises(ExperimentConfigurationError, match="inference model requires"):
        config(document, tmp_path)


@pytest.mark.parametrize(
    "stage",
    [
        {"name": "voxel_downsample", "voxel_size": 0},
        {"name": "range_filter", "x_range": [1, 0], "y_range": [0, 1], "z_range": [0, 1]},
        {"name": "noise", "position_std": float("nan"), "seed": 42},
        {"name": "noise", "position_std": 0.1},
        {"name": "statistical_outlier_removal", "nb_neighbors": 0, "std_ratio": 1},
    ],
)
def test_invalid_preprocessing_uses_existing_errors(document, tmp_path, stage):
    inference(document)["preprocessing"] = [stage]
    with pytest.raises(ExperimentConfigurationError, match="Stage 1"):
        config(document, tmp_path)


def test_unsafe_path_aliases_and_existing_files(document, tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(dataset, target_is_directory=True)
    document["output"]["directory"] = "alias"
    with pytest.raises(ExperimentConfigurationError, match="disjoint"):
        config(document, tmp_path)
    document["output"]["directory"] = "outputs"
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "baseline-001").symlink_to(dataset, target_is_directory=True)
    with pytest.raises(ExperimentConfigurationError, match="symlink"):
        config(document, tmp_path)
    (outputs / "baseline-001").unlink()
    (outputs / "baseline-001").write_text("protected")
    with pytest.raises(ExperimentConfigurationError, match="existing file"):
        config(document, tmp_path)
    assert (outputs / "baseline-001").read_text() == "protected"


@pytest.mark.parametrize(
    "text",
    ['{"schema_version":1,"schema_version":1}', "NaN", "[]", "{", '{"seed":Infinity}', "\ufeff{}"],
)
def test_malformed_json(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ExperimentConfigurationError):
        load_experiment_config(path)


@pytest.mark.parametrize("contents", [b"", b"000000\n\n", b"000000\n000000\n", b"\xff"])
def test_invalid_manifest(document, tmp_path, contents):
    (tmp_path / "frames.txt").write_bytes(contents)
    document["frames"] = {"manifest": "frames.txt"}
    with pytest.raises(ExperimentConfigurationError):
        config(document, tmp_path)


def test_missing_manifest_and_exclusive_selection(document, tmp_path):
    document["frames"] = {"manifest": "missing.txt"}
    with pytest.raises(ExperimentConfigurationError, match="missing.txt"):
        config(document, tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        FrameSelection(ids=("000000",), manifest=tmp_path / "manifest.txt")
    with pytest.raises(ExperimentConfigurationError):
        load_experiment_config(tmp_path / "missing.json")


def test_cli_inspection_no_artifacts_or_optional_dependencies(
    document, tmp_path, monkeypatch, capsys
):
    inference(document)["preprocessing"] = [
        {"name": "statistical_outlier_removal", "nb_neighbors": 16, "std_ratio": 1}
    ]
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(document))
    for name in ("torch", "spconv", "cumm", "pcdet", "open3d", "scipy"):
        monkeypatch.setitem(sys.modules, name, None)
    main(["inspect-experiment", str(path)])
    report = json.loads(capsys.readouterr().out)
    assert report["configuration"]["schema_version"] == 1
    assert report["frame_ids"] == document["frames"]["ids"]
    assert len(report["config_sha256"]) == 64
    assert not (tmp_path / "outputs").exists()
    # Same configuration in a fresh process, with optional imports actively blocked.
    code = """import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch','spconv','cumm','pcdet','open3d','scipy'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from raven.cli import main
main(['inspect-experiment', sys.argv[1]])
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    fresh = subprocess.run(
        [sys.executable, "-c", code, str(path)], env=env, capture_output=True, text=True, check=True
    )
    assert json.loads(fresh.stdout) == report


def test_cli_help_and_error_exit(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["inspect-experiment", "--help"])
    assert exc.value.code == 0
    assert "schema-v1" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(["inspect-experiment", str(tmp_path / "missing.json")])
    assert exc.value.code == 2
    assert "missing.json" in capsys.readouterr().err


def test_test_split_and_ancestor_file_rejection(document, tmp_path):
    document["dataset"]["split"] = "test"
    assert (
        config(document, tmp_path).dataset.scan_directory == tmp_path / "dataset/testing/velodyne"
    )
    (tmp_path / "file").write_text("protected")
    document["output"]["directory"] = "file/child"
    with pytest.raises(ExperimentConfigurationError, match="ancestors"):
        config(document, tmp_path)


def test_output_cannot_contain_inputs_or_redirect_artifacts(document, tmp_path):
    inference(document)
    document["model"]["checkpoint"] = "outputs/weights.pth"
    with pytest.raises(ExperimentConfigurationError, match="cannot contain input"):
        config(document, tmp_path)
    document["model"]["checkpoint"] = "weights.pth"
    run = tmp_path / "outputs/baseline-001"
    run.mkdir(parents=True)
    (run / "baseline.results.json").symlink_to(tmp_path / "protected.json")
    with pytest.raises(ExperimentConfigurationError, match="redirect"):
        config(document, tmp_path)


def test_manifest_roundtrip_after_changing_working_directory(document, tmp_path, monkeypatch):
    (tmp_path / "frames.txt").write_text("000001\n000002\n")
    document["frames"] = {"manifest": "frames.txt"}
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(document))
    before = load_experiment_config(path)
    monkeypatch.chdir(tmp_path.parent)
    assert load_experiment_config(path) == before
    output = tmp_path / "outputs/baseline-001"
    output.mkdir(parents=True)
    save_experiment_config(output / "baseline.config.json", before)
    assert load_experiment_config(output / "baseline.config.json") == before


def test_public_stage_validator_does_not_execute_and_detaches(monkeypatch):
    stages = [{"name": "range_filter", "x_range": [0, 1], "y_range": [0, 1], "z_range": [0, 1]}]
    monkeypatch.setitem(
        pipeline._OPERATIONS, "range_filter", lambda **kwargs: pytest.fail("execution")
    )
    validated = pipeline.validate_preprocessing_stages(stages)
    stages[0]["x_range"][0] = -10
    assert validated[0]["x_range"] == (0.0, 1.0)
    with pytest.raises(TypeError):
        validated[0]["x_range"] = (0, 2)
