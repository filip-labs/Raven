"""Shared headless OpenPCDet execution. GPU dependencies are imported only on initialize."""

import hashlib
import importlib.metadata
import os
import pickle
import platform
import re
import subprocess
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import RLock

import numpy as np

# Preserve the existing helper import path for the runner and external callers.
from raven.perception.detection.predictions import serialize_predictions

__all__ = ["OpenPCDetSession", "serialize_predictions"]

PINNED_REVISION = "233f849829b6ac19afb8af8837a0246890908755"
# Upstream config resolution changes process cwd; serialize our sessions.
_SESSION_LOCK = RLock()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plain_config(value):
    if isinstance(value, Mapping):
        return {str(key): plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_config(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"Unsupported configuration value: {type(value).__name__}")


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def command_output(arguments):
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def validate_assets(root, config, checkpoint, checksum):
    root, config, checkpoint = (
        Path(path).expanduser().resolve() for path in (root, config, checkpoint)
    )
    if not (root / "tools").is_dir() or not (root / "pcdet/__init__.py").is_file():
        raise ValueError("Provide an OpenPCDet checkout containing tools/ and pcdet/__init__.py")
    if not config.is_file():
        raise FileNotFoundError(f"PointPillar configuration not found: {config}")
    if not checkpoint.is_file() or not checkpoint.stat().st_size:
        raise ValueError(f"Checkpoint must be an existing nonempty file: {checkpoint}")
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", checksum):
        raise ValueError("Checkpoint SHA256 must contain exactly 64 hexadecimal characters")
    if sha256_file(checkpoint) != checksum.lower():
        raise ValueError("Checkpoint SHA256 mismatch; inference was not started")
    return root, config, checkpoint


def validate_config(cfg):
    """Limit this adapter to the pinned KITTI XYZI PointPillar architecture."""
    try:
        model, data = cfg.MODEL, cfg.DATA_CONFIG
        encoding = data.POINT_FEATURE_ENCODING
        expected = ["x", "y", "z", "intensity"]
        supported = (
            model.NAME == "PointPillar"
            and model.VFE.NAME == "PillarVFE"
            and model.MAP_TO_BEV.NAME == "PointPillarScatter"
            and model.DENSE_HEAD.NAME == "AnchorHeadSingle"
            and model.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.BOX_CODER == "ResidualCoder"
            and model.POST_PROCESSING.OUTPUT_RAW_SCORE is False
            and model.BACKBONE_2D.NAME == "BaseBEVBackbone"
            and all(
                processor.NAME
                in {
                    "mask_points_and_boxes_outside_range",
                    "shuffle_points",
                    "transform_points_to_voxels",
                }
                for processor in data.DATA_PROCESSOR
            )
            and data.DATASET == "KittiDataset"
            and not data.get("DATABASE_WITH_FAKELIDAR", False)
            and encoding.encoding_type == "absolute_coordinates_encoding"
            and list(encoding.src_feature_list) == expected
            and list(encoding.used_feature_list) == expected
            and list(cfg.CLASS_NAMES) == ["Car", "Pedestrian", "Cyclist"]
        )
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError("Incomplete KITTI PointPillar configuration") from exc
    if not supported:
        raise ValueError(
            "Unsupported configuration: requires KITTI XYZI PointPillar, "
            "standard class order and probability scores"
        )


def configure_environment():
    for name in ("CUMM_DISABLE_JIT", "SPCONV_DISABLE_JIT"):
        if os.environ.get(name) is None and any(
            module.split(".")[0] in ("spconv", "cumm") for module in sys.modules
        ):
            raise RuntimeError(f"Set {name}=1 in a fresh process before importing spconv/cumm")
        if os.environ.setdefault(name, "1") != "1":
            raise ValueError(f"{name}=1 must be set before launching this process")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")


class OpenPCDetSession:
    """One lazily loaded model, shared by the adapter and standalone runner.

    Operations are serialized across Raven sessions because upstream config
    loading changes cwd. Other threads must not change cwd during initialization.
    close() releases our references; it does not reset CUDA or clear global caches.
    """

    def __init__(
        self,
        *,
        openpcdet_root,
        config,
        checkpoint,
        checkpoint_sha256,
        device_index=0,
        trust_checkpoint=False,
    ):
        self.root, self.config_path, self.checkpoint = validate_assets(
            openpcdet_root, config, checkpoint, checkpoint_sha256
        )
        if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
            raise ValueError("device_index must be a nonnegative integer")
        if not isinstance(trust_checkpoint, bool):
            raise TypeError("trust_checkpoint must be a boolean")
        self.checksum = checkpoint_sha256.lower()
        self.device_index = device_index
        self.trust_checkpoint = trust_checkpoint
        self._state = None
        self._closed = False
        self._metadata = None

    def initialize(self):
        with _SESSION_LOCK:
            self._initialize()

    def _initialize(self):
        if self._closed:
            raise RuntimeError("PointPillars detector is closed")
        if self._state is not None:
            return
        # Recheck immediately before loading, even if construction happened earlier.
        validate_assets(self.root, self.config_path, self.checkpoint, self.checksum)
        configure_environment()
        sys.path.insert(0, str(self.root))
        try:
            import pcdet
            import torch
            from pcdet.config import cfg, cfg_from_yaml_file
            from pcdet.datasets import DatasetTemplate
            from pcdet.models import build_network, load_data_to_gpu
            from pcdet.utils import common_utils
        except ImportError as exc:
            raise RuntimeError(
                "External GPU dependencies are unavailable. See docs/pointpillars-adapter.md; "
                "for missing av2, review the explicit Argoverse workaround. "
                f"Original error: {exc}"
            ) from exc
        finally:
            sys.path.remove(str(self.root))
        if Path(pcdet.__file__).resolve().parent != self.root / "pcdet":
            raise ValueError("Imported pcdet does not match the requested OpenPCDet checkout")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "An external CUDA GPU is required; CPU/macOS inference is unsupported"
            )
        if self.device_index >= torch.cuda.device_count():
            raise ValueError("device_index is outside the visible CUDA device range")
        revision = subprocess.run(
            ["git", "-C", str(self.root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if revision != PINNED_REVISION:
            raise ValueError(f"Unsupported OpenPCDet revision; expected {PINNED_REVISION}")
        dirty = subprocess.run(
            ["git", "-C", str(self.root), "diff", "--name-only", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        # Copy upstream globals: multiple sessions must not inherit each other's YAML overrides.
        config = deepcopy(cfg)
        with working_directory(self.root / "tools"), torch.cuda.device(self.device_index):
            cfg_from_yaml_file(str(self.config_path), config)
            validate_config(config)
            dataset = DatasetTemplate(
                dataset_cfg=config.DATA_CONFIG,
                class_names=config.CLASS_NAMES,
                training=False,
                root_path=self.root,
                logger=common_utils.create_logger(),
            )
            model = build_network(
                model_cfg=config.MODEL, num_class=len(config.CLASS_NAMES), dataset=dataset
            )
            try:
                checkpoint = torch.load(
                    self.checkpoint, map_location="cpu", weights_only=not self.trust_checkpoint
                )
            except pickle.UnpicklingError as exc:
                raise RuntimeError(
                    "Checkpoint loading failed; review provenance before explicitly enabling "
                    "trust_checkpoint=True (--trust-checkpoint). No unsafe fallback was attempted. "
                    f"Original error: {exc}"
                ) from exc
            if not isinstance(checkpoint, Mapping) or not isinstance(
                checkpoint.get("model_state"), Mapping
            ):
                raise TypeError("Expected an OpenPCDet checkpoint containing a model_state mapping")
            model.load_state_dict(checkpoint["model_state"], strict=True)
            model.cuda().eval()
            metadata = {
                "environment": {
                    "python": platform.python_version(),
                    "pytorch": torch.__version__,
                    "torch_cuda_runtime": torch.version.cuda,
                    "cuda_toolkit_nvcc": command_output(["nvcc", "--version"]),
                    "gpu_driver": command_output(
                        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
                    ),
                    "spconv_cu126": package_version("spconv-cu126"),
                    "cumm_cu126": package_version("cumm-cu126"),
                    "numpy": np.__version__,
                    "openpcdet": pcdet.__version__,
                    "gpu": torch.cuda.get_device_name(self.device_index),
                    "compute_capability": list(torch.cuda.get_device_capability(self.device_index)),
                    "visible_gpu_count": torch.cuda.device_count(),
                    "device_index": self.device_index,
                    "environment_flags": {
                        key: os.environ[key]
                        for key in (
                            "CUMM_DISABLE_JIT",
                            "SPCONV_DISABLE_JIT",
                            "TORCH_CUDA_ARCH_LIST",
                        )
                    },
                },
                "raven_version": package_version("raven-vision"),
                "openpcdet_commit": revision,
                "openpcdet_modified_files": dirty,
                "effective_config": plain_config(config),
                "config": {"path": str(self.config_path), "sha256": sha256_file(self.config_path)},
                "checkpoint_sha256": self.checksum,
                "pickle_loading_allowed": self.trust_checkpoint,
            }
        # Publish only a fully initialized model; exceptions cannot cache partial state.
        self._metadata = metadata
        self._state = (torch, dataset, model, load_data_to_gpu, list(config.CLASS_NAMES))

    @property
    def provenance(self):
        with _SESSION_LOCK:
            if self._metadata is None:
                raise RuntimeError("Initialize or detect before requesting runtime provenance")
            return deepcopy(self._metadata)

    def predict(self, cloud):
        with _SESSION_LOCK:
            if cloud.coordinate_frame != "kitti_velodyne":
                raise ValueError("PointPillars requires coordinate_frame='kitti_velodyne'")
            if not cloud.count:
                raise ValueError("PointPillars requires a nonempty point cloud")
            self._initialize()
            torch, dataset, model, load_data_to_gpu, names = self._state
            with torch.cuda.device(self.device_index), torch.no_grad():
                prepared = dataset.prepare_data(
                    {"points": cloud.points.copy(), "frame_id": cloud.frame_id}
                )
                batch = dataset.collate_batch([prepared])
                load_data_to_gpu(batch)
                results, _ = model.forward(batch)
                if len(results) != 1:
                    raise ValueError("Expected one prediction result for one point cloud")
                arrays = [
                    results[0][key].detach().cpu().numpy().copy()
                    for key in ("pred_boxes", "pred_scores", "pred_labels")
                ]
            return arrays, list(names)

    def close(self):
        with _SESSION_LOCK:
            self._state = None
            self._closed = True
