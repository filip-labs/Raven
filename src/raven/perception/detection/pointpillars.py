"""Optional OpenPCDet PointPillars adapter; import and construction are CPU-only."""

from pathlib import Path

from raven.core.point_cloud import PointCloud
from raven.perception.detection._openpcdet import OpenPCDetSession
from raven.perception.detection.base import Detection3D
from raven.perception.detection.predictions import convert_predictions

__all__ = ["PointPillarsDetector", "convert_predictions"]


class PointPillarsDetector:
    """Reusable ObjectDetector implementation for the pinned KITTI PointPillar model.

    Validates assets on construction without importing GPU frameworks. initialize()
    or the first detect() loads one model for subsequent calls. Use close() when
    finished; this instance cannot be reused afterwards. See the adapter guide.
    """

    def __init__(
        self,
        *,
        openpcdet_root: str | Path,
        config: str | Path,
        checkpoint: str | Path,
        checkpoint_sha256: str,
        device_index: int = 0,
        trust_checkpoint: bool = False,
    ):
        self._session = OpenPCDetSession(
            openpcdet_root=openpcdet_root,
            config=config,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            device_index=device_index,
            trust_checkpoint=trust_checkpoint,
        )

    def initialize(self) -> None:
        self._session.initialize()

    def detect(self, cloud: PointCloud) -> list[Detection3D]:
        arrays, names = self._session.predict(cloud)
        return convert_predictions(*arrays, names)

    @property
    def provenance(self) -> dict:
        """Defensive copy of runtime/model evidence, available after initialization."""
        return self._session.provenance

    def close(self) -> None:
        self._session.close()
