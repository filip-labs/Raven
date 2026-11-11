import json
import sys

import pytest

from raven.cli import main
from raven.perception.data.kitti_annotations import load_kitti_labels, parse_kitti_annotation

VALID = "Car 0.1 1 -0.2 10 20 100 200 1.5 1.6 4.0 2.0 1.8 30.0 0.5"
IGNORE = "DontCare -1 -1 -10 10 20 100 200 -1 -1 -1 -1000 -1000 -1000 -10"


def test_all_fields_and_camera_conventions():
    record = parse_kitti_annotation(VALID)
    assert record.object_type == "Car" and record.is_known_class
    assert record.truncation == 0.1 and record.occlusion == 1 and record.alpha == -0.2
    assert record.bbox_xyxy == (10, 20, 100, 200)
    assert record.dimensions_hwl == (1.5, 1.6, 4)
    assert record.location_xyz == (2, 1.8, 30) and record.rotation_y == 0.5
    assert record.source_score is None
    box = record.ground_truth
    assert box.size_lwh == (4, 1.6, 1.5)
    assert box.position_xyz == record.location_xyz  # No geometric-center shift.
    assert box.position_reference == "bottom_center"
    assert box.coordinate_frame == "kitti_rectified_camera"
    assert box.rotation_axis == "y" and box.rotation_rad == 0.5
    assert not hasattr(box, "score") and not hasattr(box, "yaw_rad")


@pytest.mark.parametrize("score", ["0.8", "2.5", "-3"])
def test_optional_score_is_preserved_not_reinterpreted(score):
    record = parse_kitti_annotation(VALID + " " + score)
    assert record.source_score == float(score)
    assert not hasattr(record.ground_truth, "score")


def test_dontcare_preserves_sentinels_without_building_box():
    record = parse_kitti_annotation(IGNORE)
    assert record.is_ignored and record.is_known_class
    assert record.ground_truth is None
    assert record.dimensions_hwl == (-1, -1, -1)
    assert record.location_xyz == (-1000, -1000, -1000)
    assert record.rotation_y == -10 and record.alpha == -10
    assert record.bbox_xyxy == (10, 20, 100, 200)


def test_unknown_classes_are_retained_explicitly():
    record = parse_kitti_annotation(VALID.replace("Car", "UnrecognizedVehicle"))
    assert not record.is_known_class and not record.is_ignored
    assert record.ground_truth.label == "UnrecognizedVehicle"


def test_unavailable_result_metadata():
    record = parse_kitti_annotation(VALID.replace("0.1 1 -0.2", "-1 -1 -10") + " 2.5")
    assert (record.truncation, record.occlusion, record.alpha) == (-1, -1, -10)
    assert record.source_score == 2.5


@pytest.mark.parametrize(
    "line", ["", "Car 1 2", VALID + " 0.5 extra", " ".join(VALID.split()[:-1])]
)
def test_malformed_field_counts(line):
    with pytest.raises(ValueError, match="15 or 16"):
        parse_kitti_annotation(line)


@pytest.mark.parametrize(
    "index,value",
    [
        (1, "nan"),
        (2, "1.5"),
        (2, "4"),
        (3, "inf"),
        (4, "oops"),
        (5, "-inf"),
        (8, "0"),
        (9, "-1"),
        (10, "nan"),
        (11, "inf"),
        (12, "nan"),
        (13, "-inf"),
        (14, "nan"),
        (1, "1.1"),
        (3, "4"),
        (14, "4"),
        (6, "5"),
        (7, "5"),
    ],
)
def test_invalid_fields(index, value):
    fields = VALID.split()
    fields[index] = value
    with pytest.raises(ValueError):
        parse_kitti_annotation(" ".join(fields))


@pytest.mark.parametrize("score", ["nan", "inf", "bad"])
def test_invalid_optional_scores(score):
    with pytest.raises(ValueError):
        parse_kitti_annotation(VALID + " " + score)


def test_file_loading_multiple_objects_ignores_and_determinism(tmp_path):
    path = tmp_path / "000123.txt"
    path.write_text(VALID + "\n\n" + IGNORE + "\n" + VALID.replace("Car", "Cyclist") + "\n")
    labels = load_kitti_labels(path)
    assert labels == load_kitti_labels(path)
    assert labels.frame_id == "000123"
    assert labels.coordinate_frame == "kitti_rectified_camera"
    assert [obj.label for obj in labels.objects] == ["Car", "Cyclist"]
    assert len(labels.ignored_regions) == 1 and len(labels.records) == 3
    assert labels.records[1].is_ignored


def test_empty_and_whitespace_file(tmp_path):
    path = tmp_path / "empty.txt"
    for contents in ["", "  \n\t\n"]:
        path.write_text(contents)
        labels = load_kitti_labels(path)
        assert labels.records == labels.objects == labels.ignored_regions == ()


def test_file_errors_include_source_and_physical_line_number(tmp_path):
    path = tmp_path / "invalid.txt"
    path.write_text(VALID + "\n\nCar bad\n")
    with pytest.raises(ValueError, match=r"invalid\.txt:3: Expected"):
        load_kitti_labels(path)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_kitti_labels(tmp_path / "missing.txt")


def test_label_cli_without_open3d(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "open3d", None)
    path = tmp_path / "000000.txt"
    path.write_text(VALID + "\n" + IGNORE + "\n" + VALID.replace("Car", "Unknown"))
    main(["inspect-labels", str(path)])
    assert json.loads(capsys.readouterr().out) == {
        "frame_id": "000000",
        "coordinate_frame": "kitti_rectified_camera",
        "objects": 2,
        "class_counts": {"Car": 1, "Unknown": 1},
        "ignored_regions": 1,
        "unknown_classes": ["Unknown"],
    }


@pytest.mark.parametrize("error", ["missing", "invalid"])
def test_label_cli_errors(tmp_path, capsys, error):
    path = tmp_path / "000000.txt"
    if error == "invalid":
        path.write_text("Car bad")
    with pytest.raises(SystemExit) as exc:
        main(["inspect-labels", str(path)])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "error:" in err and "000000.txt" in err and "Traceback" not in err


@pytest.mark.parametrize(
    "name", ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]
)
def test_all_standard_kitti_classes_are_retained(name):
    record = parse_kitti_annotation(VALID.replace("Car", name))
    assert record.is_known_class and record.ground_truth.label == name


@pytest.mark.parametrize("index", [3, 8, 11])
def test_nonfinite_ignore_metadata_rejected(index):
    fields = IGNORE.split()
    fields[index] = "nan"
    with pytest.raises(ValueError, match="finite"):
        parse_kitti_annotation(" ".join(fields))


@pytest.mark.parametrize(
    "changes",
    [
        {"bbox_xyxy": (1, 2, 3)},
        {"dimensions_hwl": ((1, 2, 3), 2, 3)},
        {"location_xyz": ("1", 2, 3)},
        {"source_score": True},
    ],
)
def test_annotation_model_rejects_malformed_metadata(changes):
    from dataclasses import replace

    with pytest.raises(ValueError):
        replace(parse_kitti_annotation(IGNORE), **changes)
