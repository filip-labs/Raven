"""Synthetic conversion/schema cases; no captured predictions or GPU dependencies."""

import json
import subprocess
import sys
from copy import deepcopy

import numpy as np
import pytest

from raven.perception.detection.base import Detection3D
from raven.perception.detection.pointpillars import convert_predictions as adapter_convert
from raven.perception.detection.predictions import (
    CLASS_NAMES,
    convert_predictions,
    detections_from_document,
    detections_to_document,
    load_detections,
    save_detections,
)


@pytest.fixture
def arrays():
    return [
        np.array(
            [
                [0.1, -2, 3, 4, 2, 1, 7.5],
                [5, 6, -1, 0.5, 0.6, 1.8, -0.2],
                [7, 8, 9, 1.7, 0.7, 1.5, -7.5],
            ],
            dtype=np.float32,
        ),
        np.array([0, 0.9, 1], dtype=np.float32),
        np.array([1, 2, 3], dtype=np.int64),
    ]


@pytest.mark.parametrize("count", [0, 1, 3])
def test_preserve_geometry_classes_confidence_and_input(arrays, count):
    values = [array[:count] for array in arrays]
    original = [array.copy() for array in values]
    for array in values:
        array.setflags(write=False)
    detections = convert_predictions(*values, CLASS_NAMES)
    assert adapter_convert is convert_predictions  # Existing adapter helper path is preserved.
    assert len(detections) == count
    for index, box in enumerate(detections):
        assert isinstance(box, Detection3D)
        assert box.center_xyz == tuple(float(v) for v in values[0][index, :3])
        assert box.size_lwh == tuple(float(v) for v in values[0][index, 3:6])
        assert box.yaw_rad == float(values[0][index, 6])  # No angle wrapping.
        assert box.score == float(values[1][index])
        assert box.label == CLASS_NAMES[index]
        assert box.coordinate_frame == "kitti_velodyne" and box.rotation_axis == "z"
        assert type(box.score) is float and type(box.yaw_rad) is float
        assert all(type(value) is float for value in (*box.center_xyz, *box.size_lwh))
    for array, before in zip(values, original, strict=True):
        np.testing.assert_array_equal(array, before)
        assert not array.flags.writeable


@pytest.mark.parametrize(
    "field,value,message",
    [
        (0, np.zeros(7), r"shape \(N, 7\)"),
        (0, np.zeros((1, 6)), r"shape \(N, 7\)"),
        (0, np.zeros((1, 8)), r"shape \(N, 7\)"),
        (0, np.zeros((1, 1, 7)), r"shape \(N, 7\)"),
        (0, [[1, 2], [1]], "rectangular"),
        (1, np.array(0.5), "shape"),
        (1, np.ones((3, 1)), "shape"),
        (1, np.ones(2), "shape"),
        (2, np.ones((3, 1)), "shape"),
        (2, np.ones(2), "shape"),
        (2, np.array([0, 2, 3]), "one-based"),
        (2, np.array([-1, 2, 3]), "one-based"),
        (2, np.array([4, 2, 3]), "one-based"),
        (2, np.array([1.5, 2, 3]), "one-based"),
        (1, np.array([-0.1, 0.5, 1]), "probability"),
        (1, np.array([0, 0.5, 1.01]), "probability"),
        (1, np.array([True, False, True]), "real numeric"),
        (1, [True, 0.5, 1], "real numeric"),
        (2, [True, 2, 3], "real numeric"),
        (1, np.array(["0", ".5", "1"]), "real numeric"),
        (1, np.array([0, 0.5, 1], dtype=object), "real numeric"),
        (1, np.array([0, 0.5, 1], dtype=complex), "real numeric"),
        (1, [None, 0.5, 1], "real numeric"),
    ],
)
def test_invalid_shapes_types_scores_and_indices(arrays, field, value, message):
    arrays[field] = value
    with pytest.raises(ValueError, match=message):
        convert_predictions(*arrays, CLASS_NAMES)


@pytest.mark.parametrize("column", range(7))
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_all_box_components_must_be_finite(arrays, column, value):
    arrays[0][0, column] = value
    with pytest.raises(ValueError, match="finite"):
        convert_predictions(*arrays, CLASS_NAMES)


@pytest.mark.parametrize("field", [1, 2])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_scores_and_indices_must_be_finite(arrays, field, value):
    arrays[field] = arrays[field].astype(float)
    arrays[field][0] = value
    with pytest.raises(ValueError, match="finite"):
        convert_predictions(*arrays, CLASS_NAMES)


@pytest.mark.parametrize("column", [3, 4, 5])
@pytest.mark.parametrize("value", [0, -1])
def test_dimensions_must_be_positive(arrays, column, value):
    arrays[0][0, column] = value
    with pytest.raises(ValueError, match="positive"):
        convert_predictions(*arrays, CLASS_NAMES)


@pytest.mark.parametrize(
    "names", [[], ["Car", "Car"], [" Car"], ["Car "], [""], [None], "Car", None]
)
def test_invalid_class_names(arrays, names):
    with pytest.raises((ValueError, TypeError), match="Class names"):
        convert_predictions(*arrays, names)


def test_float32_precision_and_json_roundtrip(arrays):
    boxes = convert_predictions(*arrays, CLASS_NAMES)
    assert boxes[0].center_xyz[0] == float(np.float32(0.1))
    assert boxes[0].center_xyz[0] != 0.1  # Promoting a rounded float32 does not recover real 0.1.
    document = detections_to_document(boxes, frame_id="000000")
    assert detections_from_document(json.loads(json.dumps(document, allow_nan=False))) == boxes
    arrays[0][:] = 100  # No references to source arrays remain in Detection3D.
    assert boxes[0].center_xyz[0] == float(np.float32(0.1))


@pytest.fixture
def document(arrays):
    return detections_to_document(convert_predictions(*arrays, CLASS_NAMES), frame_id="000000")


def test_reuse_existing_runner_schema(document):
    assert document["schema_version"] == 1
    assert document["predictions"][0]["label_id"] == 1
    assert document["detections"][0]["label"] == "Car"
    assert document["class_counts"] == {"Car": 1, "Pedestrian": 1, "Cyclist": 1}
    before = deepcopy(document)
    boxes = detections_from_document(document, expected_frame_id="000000")
    assert document == before
    # Synthetic legacy RV-022 envelope: class mapping lived in effective_config.
    legacy = deepcopy(document)
    legacy["effective_config"] = {"CLASS_NAMES": legacy.pop("class_names")}
    legacy["environment"] = {"synthetic_fixture": True}
    assert detections_from_document(legacy) == boxes
    del legacy["detections"]  # Existing standalone raw-only mode is supported too.
    assert detections_from_document(legacy) == boxes


@pytest.mark.parametrize(
    "path,value,message",
    [
        (("schema_version",), 2, "schema_version"),
        (("schema_version",), True, "schema_version"),
        (("schema_version",), 1.0, "schema_version"),
        (("input", "frame_id"), 0, "frame_id"),
        (("input", "frame_id"), "00000", "frame_id"),
        (("input", "frame_id"), "../000000", "frame_id"),
        (("input", "frame_id"), " 000000", "frame_id"),
        (("input", "coordinate_frame"), "lidar", "coordinate conventions"),
        (("coordinate_frame",), "camera", "coordinate conventions"),
        (("rotation_axis",), "y", "coordinate conventions"),
        (("position_reference",), "bottom_center", "coordinate conventions"),
        (("box_format",), ["x", "y", "z", "h", "w", "l", "heading"], "coordinate conventions"),
        (("units", "heading"), "degrees", "coordinate conventions"),
        (("units", "position"), "cm", "coordinate conventions"),
        (("total_predictions",), 4, "counts"),
        (("total_predictions",), True, "counts"),
        (("class_counts", "Car"), True, "counts"),
        (("class_counts", "Car"), 2, "counts"),
        (("predictions",), None, "predictions must be a list"),
        (("detections",), None, "detections must be a list"),
    ],
)
def test_reject_invalid_schema(document, path, value, message):
    parent = document
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises((ValueError, TypeError), match=message):
        detections_from_document(document)


@pytest.mark.parametrize(
    "field,value",
    [
        ("center_xyz", [100, 2, 3]),
        ("size_lwh", [1, 2, 3]),
        ("yaw_rad", 0.123),
        ("score", 0.5),
        ("label", "Cyclist"),
        ("coordinate_frame", "camera"),
        ("rotation_axis", "y"),
    ],
)
def test_duplicate_representations_must_agree(document, field, value):
    document["detections"][0][field] = value
    with pytest.raises(ValueError, match="disagree"):
        detections_from_document(document)


def test_missing_detection_field_does_not_fall_back_to_legacy_default(document):
    del document["detections"][0]["coordinate_frame"]
    with pytest.raises(ValueError, match="all seven"):
        detections_from_document(document)


def test_invalid_detection_record_is_revalidated(document):
    document["detections"][0]["score"] = True
    with pytest.raises(ValueError, match=r"detections\[0\]"):
        detections_from_document(document)


def test_conflicting_class_mapping(document):
    document["effective_config"] = {"CLASS_NAMES": ["Cyclist", "Pedestrian", "Car"]}
    with pytest.raises(ValueError, match="class_names disagree"):
        detections_from_document(document)


def test_raw_class_name_must_match_index(document):
    document["predictions"][0]["class_name"] = "Cyclist"
    with pytest.raises(ValueError, match="class_name disagrees"):
        detections_from_document(document)


@pytest.mark.parametrize("index", [True, 1.0, "1"])
def test_json_label_id_requires_integer(document, index):
    document["predictions"][0]["label_id"] = index
    with pytest.raises(ValueError, match="integer label_id"):
        detections_from_document(document)


def test_missing_class_vocabulary(document):
    del document["class_names"]
    with pytest.raises(ValueError, match="Class names"):
        detections_from_document(document)


def test_frame_mismatch(document):
    with pytest.raises(ValueError, match="expected_frame_id"):
        detections_from_document(document, expected_frame_id="000001")


@pytest.mark.parametrize("frame_id", ["", "00000x", "../000000", 0])
def test_export_frame_id_validation(frame_id):
    with pytest.raises(ValueError, match="frame_id"):
        detections_to_document([], frame_id=frame_id)


@pytest.mark.parametrize("frame,axis", [("lidar", "z"), ("camera", "y"), ("kitti_velodyne", "y")])
def test_export_requires_explicit_supported_conventions(frame, axis):
    box = Detection3D((0, 0, 0), (1, 1, 1), 0, "Car", 0.5, frame, axis)
    with pytest.raises(ValueError, match="rotation_axis=z"):
        detections_to_document([box], frame_id="000000")


@pytest.mark.parametrize("count", [0, 1, 3])
def test_file_roundtrip_is_read_only_and_never_overwrites(tmp_path, arrays, count):
    boxes = convert_predictions(*(a[:count] for a in arrays), CLASS_NAMES)
    path = tmp_path / "synthetic_predictions.json"
    save_detections(path, boxes, frame_id="000000")
    original = path.read_bytes()
    assert load_detections(path, expected_frame_id="000000") == boxes
    assert path.read_bytes() == original
    with pytest.raises(FileExistsError):
        save_detections(path, [], frame_id="000000")
    assert path.read_bytes() == original


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonstandard_json_constants_are_rejected(tmp_path, document, value):
    document["predictions"][0]["score"] = value
    path = tmp_path / "nonfinite.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Nonfinite JSON"):
        load_detections(path)


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="Duplicate JSON field"):
        load_detections(path)


def test_cpu_only_export_and_import_in_fresh_interpreter(tmp_path):
    program = """
import importlib.abc
import sys
blocked = {"torch", "pcdet", "OpenPCDet", "spconv", "cumm", "open3d"}
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise AssertionError("Unexpected GPU/GUI import: " + fullname)
sys.meta_path.insert(0, Block())
from raven.perception.detection.base import Detection3D
from raven.perception.detection.predictions import save_detections, load_detections
box = Detection3D((1, 2, 3), (4, 2, 1), .5, "Car", .9, "kitti_velodyne")
save_detections(sys.argv[1], [box], frame_id="000000")
assert load_detections(sys.argv[1], expected_frame_id="000000") == [box]
assert "raven.perception.detection._openpcdet" not in sys.modules
assert not any(name.split(".")[0] in blocked for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", program, str(tmp_path / "cpu.json")],
        check=True,
        capture_output=True,
        text=True,
    )
