import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "dataset_validation", Path(__file__).resolve().parents[1] / "scripts/validate_kitti_dataset.py"
)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


def test_pinned_split_deterministic_disjoint_and_complete():
    first = validation.validate_splits()
    assert first == validation.validate_splits()
    assert {key: len(value) for key, value in first.items()} == {"train": 3712, "val": 3769}
    assert not set(first["train"]) & set(first["val"])
    assert len(set(first["train"]) | set(first["val"])) == 7481
    assert "000000" in first["train"]


@pytest.mark.parametrize("text", ["", "1\n", "00000a\n", " 000001\n", "000001\n000001\n"])
def test_bad_ids(tmp_path, text):
    path = tmp_path / "ids.txt"
    path.write_text(text)
    with pytest.raises(ValueError):
        validation.read_ids(path)


def test_manifest_detects_tampering(tmp_path):
    for path in validation.SPLITS.iterdir():
        (tmp_path / path.name).write_bytes(path.read_bytes())
    path = tmp_path / "train.txt"
    path.write_text(path.read_text().replace("000000", "999999"))
    with pytest.raises(ValueError, match="checksum"):
        validation.validate_splits(tmp_path)


def test_missing_inventory(tmp_path):
    with pytest.raises(ValueError, match="missing="):
        validation.validate_inventory(tmp_path)


@pytest.mark.parametrize("frame", ["../000000", "00000x", "007481", "00000"])
def test_bad_sample_ids(tmp_path, frame):
    with pytest.raises(ValueError, match="training ID"):
        validation.validate_sample(tmp_path, frame)


def test_missing_sample(tmp_path):
    with pytest.raises(FileNotFoundError):
        validation.validate_sample(tmp_path, "000000")


def test_synthetic_sample_is_reproducible_and_read_only(tmp_path):
    base = tmp_path / "training"
    for kind in ("velodyne", "calib", "label_2"):
        (base / kind).mkdir(parents=True)
    np.array([[1, 2, 3, 0.5]], dtype="<f4").tofile(base / "velodyne/000000.bin")
    (base / "calib/000000.txt").write_text(
        "R0_rect: 1 0 0 0 1 0 0 0 1\nTr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n"
    )
    (base / "label_2/000000.txt").write_text("")
    first = validation.validate_sample(tmp_path, "000000")
    assert first == validation.validate_sample(tmp_path, "000000")
    assert first["points"] == 1 and first["objects"] == 0
    (base / "velodyne/000000.bin").write_bytes(b"123")
    with pytest.raises(ValueError, match="file size"):
        validation.validate_sample(tmp_path, "000000")


def test_cli_without_dataset(capsys):
    assert validation.main([]) == 0
    assert json.loads(capsys.readouterr().out)["split_counts"]["val"] == 3769


def test_cli_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        validation.main(["--root", str(tmp_path)])
    assert error.value.code == 1
    assert "KITTI validation failed" in capsys.readouterr().err


def test_cli_sample_requires_root():
    with pytest.raises(SystemExit) as error:
        validation.main(["--sample", "000000"])
    assert error.value.code == 2
