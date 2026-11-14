"""Ordered composition of existing preprocessing operations, with detached provenance."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from numbers import Real
from platform import python_version
from time import perf_counter
from types import MappingProxyType
from typing import Any

import numpy as np

from raven.core.point_cloud import PointCloud
from raven.perception.preprocessing.noise import _parameters as _noise_parameters
from raven.perception.preprocessing.noise import inject_noise
from raven.perception.preprocessing.range_filter import _validated_range, filter_range
from raven.perception.preprocessing.statistical_outlier_removal import (
    _parameters as _statistical_parameters,
)
from raven.perception.preprocessing.statistical_outlier_removal import statistical_outlier_removal
from raven.perception.preprocessing.voxel_downsample import _voxel_sizes, voxel_downsample

_OPERATIONS = {
    "range_filter": filter_range,
    "voxel_downsample": voxel_downsample,
    "statistical_outlier_removal": statistical_outlier_removal,
    "noise": inject_noise,
}
_KEYS = {
    "range_filter": {"x_range", "y_range", "z_range"},
    "voxel_downsample": {"voxel_size"},
    "statistical_outlier_removal": {"nb_neighbors", "std_ratio"},
    "noise": {"position_std", "seed"},
}


class PipelineConfigurationError(ValueError):
    """Invalid configuration; no processing stage has run."""


class PipelineExecutionError(RuntimeError):
    """A validated stage failed at runtime; no successful partial result exists."""


@dataclass(frozen=True)
class StageExecution:
    position: int
    name: str
    parameters: Mapping[str, Any]
    input_count: int
    output_count: int
    points_removed: int | None
    duration_seconds: float
    frame_id: str
    coordinate_frame: str


@dataclass(frozen=True)
class PipelineResult:
    cloud: PointCloud
    stages: tuple[StageExecution, ...]
    duration_seconds: float
    reproducibility: Mapping[str, str | None]

    def to_metadata_dict(self) -> dict:
        """Return detached JSON-compatible provenance, without point data."""
        return {
            "input_count": self.stages[0].input_count if self.stages else self.cloud.count,
            "output_count": self.cloud.count,
            "frame_id": self.cloud.frame_id,
            "coordinate_frame": self.cloud.coordinate_frame,
            "duration_seconds": self.duration_seconds,
            "reproducibility": dict(self.reproducibility),
            "stages": [
                {
                    "position": s.position,
                    "name": s.name,
                    "parameters": {
                        k: list(v) if isinstance(v, tuple) else v for k, v in s.parameters.items()
                    },
                    "input_count": s.input_count,
                    "output_count": s.output_count,
                    "points_removed": s.points_removed,
                    "duration_seconds": s.duration_seconds,
                    "frame_id": s.frame_id,
                    "coordinate_frame": s.coordinate_frame,
                }
                for s in self.stages
            ],
        }


def _reusable_sequence(value, parameter):
    # Avoid consuming iterators/generators embedded in the caller's configuration.
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise TypeError(f"{parameter} must be a list, tuple or NumPy array")
    return value


def _normalize(name, values):
    if name == "range_filter":
        return {
            key: _validated_range(_reusable_sequence(values[key], key), axis)
            for key, axis in [("x_range", "X"), ("y_range", "Y"), ("z_range", "Z")]
        }
    if name == "voxel_downsample":
        value = values["voxel_size"]
        if not isinstance(value, Real):
            _reusable_sequence(value, "voxel_size")
        sizes = _voxel_sizes(value)
        # Preserve scalar vs per-axis intent in provenance.
        return {
            "voxel_size": float(sizes[0]) if isinstance(value, Real) else tuple(map(float, sizes))
        }
    if name == "statistical_outlier_removal":
        k, ratio = _statistical_parameters(values["nb_neighbors"], values["std_ratio"])
        return {"nb_neighbors": k, "std_ratio": ratio}
    std, seed = _noise_parameters(values["position_std"], values["seed"])
    return {"position_std": std, "seed": seed}


def _validate(stages):
    if isinstance(stages, (str, bytes)) or not isinstance(stages, Sequence):
        raise PipelineConfigurationError("stages must be an ordered sequence of stage mappings")
    validated = []
    for position, stage in enumerate(stages, 1):
        name = None
        try:
            if not isinstance(stage, Mapping):
                raise TypeError(
                    "configuration must be a mapping with a name and explicit parameters"
                )
            values = dict(stage)
            name = values.pop("name", None)
            if not isinstance(name, str) or name not in _OPERATIONS:
                raise ValueError(
                    f"unknown stage name {name!r}; supported: {', '.join(_OPERATIONS)}"
                )
            missing = _KEYS[name] - values.keys()
            extra = values.keys() - _KEYS[name]
            if missing or extra:
                raise ValueError(
                    f"missing parameters: {sorted(missing)}; unsupported keys: {sorted(extra, key=str)}"
                )
            parameters = MappingProxyType(_normalize(name, values))
        except (TypeError, ValueError, OverflowError) as exc:
            raise PipelineConfigurationError(f"Stage {position} ({name!r}): {exc}") from exc
        validated.append((name, parameters))
    return tuple(validated)


def run_preprocessing_pipeline(cloud: PointCloud, stages: Sequence[Mapping]) -> PipelineResult:
    """Prevalidate all explicit stages, run in order, and return cloud plus provenance.

    Canonical names: range_filter, voxel_downsample, statistical_outlier_removal,
    noise. Each mapping has a name and flat operation parameters, without defaults.
    Repetitions are valid. Empty stages produce an independent identity copy.
    Data-dependent/backend failures are wrapped with 1-based position and name.
    """
    start = perf_counter()
    validated = _validate(stages)
    current = PointCloud(cloud.points, cloud.frame_id, cloud.coordinate_frame)
    records = []
    for position, (name, parameters) in enumerate(validated, 1):
        before = current.count
        stage_start = perf_counter()
        try:
            current = _OPERATIONS[name](current, **parameters)
        except Exception as exc:
            raise PipelineExecutionError(f"Stage {position} ({name!r}) failed: {exc}") from exc
        duration = perf_counter() - stage_start
        records.append(
            StageExecution(
                position,
                name,
                parameters,
                before,
                current.count,
                None if name == "noise" else before - current.count,
                duration,
                current.frame_id,
                current.coordinate_frame,
            )
        )
    environment = {"python": python_version(), "numpy": np.__version__}
    if any(name == "noise" for name, _ in validated):
        environment["noise_rng"] = "numpy.random.Generator(PCG64)"
    if any(name == "statistical_outlier_removal" for name, _ in validated):
        try:
            environment["scipy"] = version("scipy")
        except PackageNotFoundError:
            environment["scipy"] = None
    return PipelineResult(
        current, tuple(records), perf_counter() - start, MappingProxyType(environment)
    )
