"""KITTI object label parsing without calibration or coordinate conversion."""

import math
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import ClassVar

from raven.perception.detection.base import GroundTruth3D

KITTI_CAMERA_FRAME = "kitti_rectified_camera"
KITTI_OBJECT_CLASSES = frozenset(
    {"Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"}
)


@dataclass(frozen=True)
class KittiAnnotation:
    """Original numeric fields plus an optional validated ground-truth box.

    DontCare records preserve finite sentinel values and have no 3D box. An
    optional source_score is a raw KITTI ranking score, not a probability and
    not a confidence assigned to ground truth. Unknown class names are retained.
    """

    object_type: str
    truncation: float
    occlusion: int
    alpha: float
    bbox_xyxy: tuple[float, float, float, float]
    dimensions_hwl: tuple[float, float, float]
    location_xyz: tuple[float, float, float]
    rotation_y: float
    source_score: float | None = None
    ground_truth: GroundTruth3D | None = field(init=False)
    coordinate_frame: ClassVar[str] = KITTI_CAMERA_FRAME

    def __post_init__(self) -> None:
        if (
            not isinstance(self.object_type, str)
            or len(self.object_type.split()) != 1
            or self.object_type != self.object_type.strip()
        ):
            raise ValueError("object type must be a single nonempty class name")
        for name, count in (("bbox_xyxy", 4), ("dimensions_hwl", 3), ("location_xyz", 3)):
            try:
                values = tuple(getattr(self, name))
                if len(values) != count:
                    raise ValueError(f"{name} must contain {count} numbers")
                object.__setattr__(self, name, values)
            except TypeError as exc:
                raise ValueError(f"{name} must contain {count} numbers") from exc
        numbers = (
            self.truncation,
            self.alpha,
            *self.bbox_xyxy,
            *self.dimensions_hwl,
            *self.location_xyz,
            self.rotation_y,
        )
        if self.source_score is not None:
            numbers += (self.source_score,)
        try:
            if any(
                isinstance(n, bool) or not isinstance(n, Real) or not math.isfinite(n)
                for n in numbers
            ):
                raise ValueError("annotation numeric fields must be finite")
        except (TypeError, OverflowError) as exc:
            raise ValueError("annotation numeric fields must be finite numbers") from exc
        if isinstance(self.occlusion, bool) or not isinstance(self.occlusion, int):
            raise TypeError("occlusion must be an integer")
        left, top, right, bottom = self.bbox_xyxy
        if left > right or top > bottom:
            raise ValueError("2D bounding box must have left <= right and top <= bottom")
        if self.is_ignored:
            object.__setattr__(self, "ground_truth", None)
            return
        if self.truncation != -1 and not 0 <= self.truncation <= 1:
            raise ValueError("truncation must be in [0, 1] or the unavailable sentinel -1")
        if self.occlusion not in (-1, 0, 1, 2, 3):
            raise ValueError("occlusion must be -1, 0, 1, 2, or 3")
        if self.alpha != -10 and not -math.pi <= self.alpha <= math.pi:
            raise ValueError("alpha must be in [-pi, pi] or the unavailable sentinel -10")
        if not -math.pi <= self.rotation_y <= math.pi:
            raise ValueError("rotation_y must be in [-pi, pi]")
        height, width, length = self.dimensions_hwl
        box = GroundTruth3D(
            position_xyz=self.location_xyz,
            size_lwh=(length, width, height),
            rotation_rad=self.rotation_y,
            label=self.object_type,
            coordinate_frame=KITTI_CAMERA_FRAME,
            rotation_axis="y",
            position_reference="bottom_center",
        )
        object.__setattr__(self, "ground_truth", box)

    @property
    def is_ignored(self) -> bool:
        return self.object_type == "DontCare"

    @property
    def is_known_class(self) -> bool:
        return self.object_type in KITTI_OBJECT_CLASSES or self.is_ignored


@dataclass(frozen=True)
class KittiLabels:
    """A label file in source order, including raw ignore-region metadata."""

    frame_id: str
    records: tuple[KittiAnnotation, ...]
    coordinate_frame: ClassVar[str] = KITTI_CAMERA_FRAME

    @property
    def objects(self) -> tuple[GroundTruth3D, ...]:
        return tuple(
            record.ground_truth for record in self.records if record.ground_truth is not None
        )

    @property
    def ignored_regions(self) -> tuple[KittiAnnotation, ...]:
        return tuple(record for record in self.records if record.is_ignored)


def parse_kitti_annotation(line: str) -> KittiAnnotation:
    """Parse exactly 15 fields, or 16 with an optional finite source score.

    Blank lines are skipped by load_kitti_labels, not interpreted as records.
    Truncation/occlusion -1 and alpha -10 are preserved as unavailable metadata.
    """
    fields = line.split()
    if len(fields) not in (15, 16):
        raise ValueError(f"Expected 15 or 16 KITTI fields, got {len(fields)}")
    try:
        occlusion = int(fields[2])
        numeric = [float(token) for token in fields[1:]]
    except ValueError as exc:
        raise ValueError("Invalid KITTI numeric field or noninteger occlusion") from exc
    return KittiAnnotation(
        object_type=fields[0],
        truncation=numeric[0],
        occlusion=occlusion,
        alpha=numeric[2],
        bbox_xyxy=tuple(numeric[3:7]),
        dimensions_hwl=tuple(numeric[7:10]),
        location_xyz=tuple(numeric[10:13]),
        rotation_y=numeric[13],
        source_score=numeric[14] if len(fields) == 16 else None,
    )


def load_kitti_labels(path: str | Path) -> KittiLabels:
    """Load a UTF-8 KITTI label file, reporting filename and line on errors."""
    path = Path(path)
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(parse_kitti_annotation(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return KittiLabels(frame_id=path.stem, records=tuple(records))
