"""CPU-only detector doubles shared by contract and future runner tests."""

from collections.abc import Iterable

from raven.core.point_cloud import PointCloud
from raven.perception.detection.base import Detection3D


class FixedDetector:
    """Return configured immutable boxes in a fresh list on every call.

    This is a test fixture, not an inference backend. Incompatible output frames
    fail explicitly; no transformations or predictions are fabricated.
    """

    def __init__(self, detections: Iterable[Detection3D] = ()) -> None:
        self._detections = tuple(detections)
        if any(not isinstance(box, Detection3D) for box in self._detections):
            raise TypeError("FixedDetector requires validated Detection3D objects")

    def detect(self, cloud: PointCloud) -> list[Detection3D]:
        if any(box.coordinate_frame != cloud.coordinate_frame for box in self._detections):
            raise ValueError("Detection coordinate frame must match the input cloud")
        return list(self._detections)
