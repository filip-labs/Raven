"""Schema-v1 experiment declarations and deterministic frame selection, without execution."""

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from raven.perception.data.frame_manifest import read_frame_manifest, validate_frame_ids
from raven.perception.preprocessing.pipeline import validate_preprocessing_stages


class ExperimentConfigurationError(ValueError):
    """Malformed, incomplete or incompatible experiment declaration."""


def _keys(value, required, optional=(), context="configuration"):
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise ValueError(f"{context} must be an object with string keys")
    missing = set(required) - value.keys()
    extra = value.keys() - set(required) - set(optional)
    if missing or extra:
        raise ValueError(
            f"{context}: missing fields {sorted(missing)}; unsupported fields {sorted(extra)}"
        )


def _slug(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise ValueError(f"{name} must be a 1–64 character filename-safe identifier")


def _integer(value, name, minimum=0):
    if type(value) is not int or not minimum <= value < 2**64:
        raise ValueError(f"{name} must be an integer in [{minimum}, 2**64)")


def _digest(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA256")


def _path(value, base):
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError("paths must be nonempty strings without NUL")
    value.encode("utf-8")
    path = Path(value)
    if ".." in path.parts or "~" in path.parts or "$" in value:
        raise ValueError("paths cannot contain traversal, ~ or environment substitutions")
    return (base / path).resolve()


def _absolute(value, name):
    if not isinstance(value, Path) or not value.is_absolute() or value != value.resolve():
        raise ValueError(f"{name} must be a resolved absolute Path")


def _contains(parent, child):
    return child == parent or parent in child.parents


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    split: Literal["train", "val", "test"]

    def __post_init__(self):
        _absolute(self.root, "dataset.root")
        if self.split not in ("train", "val", "test"):
            raise ValueError("dataset.split must be train, val or test")
        if self.root.exists() and not self.root.is_dir():
            raise ValueError("dataset.root must be a directory")

    @property
    def scan_directory(self) -> Path:
        return self.root / ("testing" if self.split == "test" else "training") / "velodyne"


@dataclass(frozen=True)
class FrameSelection:
    ids: tuple[str, ...] | None = None
    manifest: Path | None = None
    sha256: str | None = None
    sample_size: int | None = None
    _candidates: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self):
        if (self.ids is None) == (self.manifest is None):
            raise ValueError("frames requires exactly one of ids or manifest")
        if self.manifest is not None:
            _absolute(self.manifest, "frames.manifest")
            ids, checksum = read_frame_manifest(self.manifest)
            if self.sha256 is not None:
                _digest(self.sha256, "frames.sha256")
                if checksum != self.sha256:
                    raise ValueError("frames.manifest SHA256 mismatch")
            object.__setattr__(self, "sha256", checksum)
        else:
            ids = validate_frame_ids(self.ids)
            object.__setattr__(self, "ids", ids)
            if self.sha256 is not None:
                raise ValueError("frames.sha256 requires a manifest")
        if self.sample_size is not None:
            _integer(self.sample_size, "frames.sample_size", 1)
            if self.sample_size > len(ids):
                raise ValueError("frames.sample_size exceeds available frame count")
        object.__setattr__(self, "_candidates", ids)

    def select(self, seed: int) -> tuple[str, ...]:
        _integer(seed, "seed")
        if self.sample_size is None:
            return self._candidates
        rng = np.random.Generator(np.random.PCG64(seed))
        indexes = sorted(rng.choice(len(self._candidates), self.sample_size, replace=False))
        return tuple(self._candidates[i] for i in indexes)

    def to_dict(self) -> dict:
        result = (
            {"ids": list(self.ids)}
            if self.ids is not None
            else {"manifest": str(self.manifest), "sha256": self.sha256}
        )
        if self.sample_size is not None:
            result["sample_size"] = self.sample_size
        return result


@dataclass(frozen=True)
class ModelConfig:
    name: Literal["pointpillars"]
    source: Literal["saved", "inference"]
    predictions_dir: Path | None = None
    config: Path | None = None
    checkpoint: Path | None = None
    checkpoint_sha256: str | None = None
    openpcdet_revision: str | None = None

    def __post_init__(self):
        if self.name != "pointpillars" or self.source not in ("saved", "inference"):
            raise ValueError("model requires name pointpillars and source saved or inference")
        inference = (self.config, self.checkpoint, self.checkpoint_sha256, self.openpcdet_revision)
        if self.source == "saved":
            if self.predictions_dir is None or any(v is not None for v in inference):
                raise ValueError(
                    "saved model requires only predictions_dir, not inference settings"
                )
            _absolute(self.predictions_dir, "model.predictions_dir")
            if self.predictions_dir.exists() and not self.predictions_dir.is_dir():
                raise ValueError("model.predictions_dir must be a directory")
        else:
            if self.predictions_dir is not None or any(v is None for v in inference):
                raise ValueError(
                    "inference model requires config, checkpoint, checkpoint_sha256 and openpcdet_revision only"
                )
            _absolute(self.config, "model.config")
            _absolute(self.checkpoint, "model.checkpoint")
            _digest(self.checkpoint_sha256, "model.checkpoint_sha256")
            if not isinstance(self.openpcdet_revision, str) or not re.fullmatch(
                r"[0-9a-f]{40}", self.openpcdet_revision
            ):
                raise ValueError("model.openpcdet_revision must be a full lowercase Git SHA")
            for path in (self.config, self.checkpoint):
                if path.exists() and not path.is_file():
                    raise ValueError("model config/checkpoint must name files")

    def to_dict(self) -> dict:
        fields = (
            ("predictions_dir",)
            if self.source == "saved"
            else ("config", "checkpoint", "checkpoint_sha256", "openpcdet_revision")
        )
        return {
            "name": self.name,
            "source": self.source,
            **{k: str(getattr(self, k)) for k in fields},
        }


@dataclass(frozen=True)
class OutputConfig:
    directory: Path
    artifact_prefix: str

    def __post_init__(self):
        _absolute(self.directory, "output.directory")
        _slug(self.artifact_prefix, "output.artifact_prefix")
        if self.directory in (Path(self.directory.anchor), Path.home().resolve()):
            raise ValueError("output.directory cannot be a filesystem root or home directory")
        if any(p.exists() and not p.is_dir() for p in (self.directory, *self.directory.parents)):
            raise ValueError("output.directory and its existing ancestors must be directories")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    experiment_id: str
    dataset: DatasetConfig
    frames: FrameSelection
    model: ModelConfig
    preprocessing: tuple[Mapping, ...]
    seed: int
    output: OutputConfig
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported experiment schema_version; expected 1")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be nonempty text")
        self.name.encode("utf-8")
        _slug(self.experiment_id, "experiment_id")
        _integer(self.seed, "seed")
        for value, cls in (
            (self.dataset, DatasetConfig),
            (self.frames, FrameSelection),
            (self.model, ModelConfig),
            (self.output, OutputConfig),
        ):
            if not isinstance(value, cls):
                raise TypeError(f"expected validated {cls.__name__}")
        stages = validate_preprocessing_stages(self.preprocessing)
        object.__setattr__(self, "preprocessing", stages)
        if self.model.source == "saved" and stages:
            raise ValueError(
                "saved predictions cannot be paired with new preprocessing; use inference"
            )
        protected = [self.dataset.root]
        if self.model.predictions_dir is not None:
            protected.append(self.model.predictions_dir)
        if any(
            _contains(p, self.output.directory) or _contains(self.output.directory, p)
            for p in protected
        ):
            raise ValueError(
                "output.directory must be disjoint from dataset/predictions directories"
            )
        run = self.run_directory
        if run.resolve() != run or (run.exists() and not run.is_dir()):
            raise ValueError("run_directory cannot be a symlink or an existing file")
        if any(p.resolve() != p for p in self.artifact_paths.values()):
            raise ValueError("artifact paths cannot redirect through symlinks")
        inputs = [self.frames.manifest, self.model.config, self.model.checkpoint]
        if any(p is not None and _contains(self.output.directory, p) for p in inputs):
            raise ValueError("output.directory cannot contain input manifests/configs/checkpoints")

    @property
    def frame_ids(self) -> tuple[str, ...]:
        return self.frames.select(self.seed)

    @property
    def selection_provenance(self) -> dict:
        return {
            "algorithm": (
                "input-order"
                if self.frames.sample_size is None
                else "PCG64.choice-without-replacement/input-order/v1"
            ),
            "numpy_version": np.__version__,
            "manifest_sha256": self.frames.sha256,
            "selected_frame_ids": list(self.frame_ids),
        }

    @property
    def run_directory(self) -> Path:
        return self.output.directory / self.experiment_id

    @property
    def artifact_paths(self) -> dict[str, Path]:
        return {
            name: self.run_directory / f"{self.output.artifact_prefix}.{suffix}"
            for name, suffix in (
                ("configuration", "config.json"),
                ("frames", "frames.txt"),
                ("provenance", "provenance.json"),
                ("results", "results.json"),
            )
        }

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "experiment_id": self.experiment_id,
            "dataset": {"root": str(self.dataset.root), "split": self.dataset.split},
            "frames": self.frames.to_dict(),
            "model": self.model.to_dict(),
            "preprocessing": [
                {k: list(v) if isinstance(v, tuple) else v for k, v in stage.items()}
                for stage in self.preprocessing
            ],
            "seed": self.seed,
            "output": {
                "directory": str(self.output.directory),
                "artifact_prefix": self.output.artifact_prefix,
            },
        }

    @property
    def config_sha256(self) -> str:
        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, document: Mapping, *, base_dir: str | Path) -> "ExperimentConfig":
        """Resolve paths relative to an explicit base; no assets other than a manifest are read."""
        try:
            base = Path(base_dir).resolve()
            _keys(
                document,
                (
                    "schema_version",
                    "name",
                    "experiment_id",
                    "dataset",
                    "frames",
                    "model",
                    "preprocessing",
                    "seed",
                    "output",
                ),
            )
            d, f, m, o = (document[k] for k in ("dataset", "frames", "model", "output"))
            _keys(d, ("root", "split"), context="dataset")
            _keys(f, (), ("ids", "manifest", "sha256", "sample_size"), "frames")
            _keys(
                m,
                ("name", "source"),
                (
                    "predictions_dir",
                    "config",
                    "checkpoint",
                    "checkpoint_sha256",
                    "openpcdet_revision",
                ),
                "model",
            )
            _keys(o, ("directory", "artifact_prefix"), context="output")
            if any(k in f and f[k] is None for k in f):
                raise ValueError("frames fields cannot be null")
            for k in m:
                if m[k] is None:
                    raise ValueError(f"model.{k} cannot be null")
            frames = FrameSelection(
                **{k: _path(v, base) if k == "manifest" else v for k, v in f.items()}
            )
            model = ModelConfig(
                **{
                    k: _path(v, base) if k in ("predictions_dir", "config", "checkpoint") else v
                    for k, v in m.items()
                }
            )
            return cls(
                document["name"],
                document["experiment_id"],
                DatasetConfig(_path(d["root"], base), d["split"]),
                frames,
                model,
                document["preprocessing"],
                document["seed"],
                OutputConfig(_path(o["directory"], base), o["artifact_prefix"]),
                document["schema_version"],
            )
        except (OSError, ValueError, TypeError, OverflowError) as exc:
            raise ExperimentConfigurationError(str(exc)) from exc


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Read strict UTF-8 JSON; resolve relative paths against the config's directory."""
    path = Path(path).resolve()
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_object, parse_constant=_constant
        )
        return ExperimentConfig.from_dict(document, base_dir=path.parent)
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        raise ExperimentConfigurationError(f"{path}: {exc}") from exc


def save_experiment_config(path: str | Path, config: ExperimentConfig) -> None:
    """Export normalized JSON exclusively; never overwrite an existing file or create directories."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(config.to_dict(), stream, indent=2, allow_nan=False)
        stream.write("\n")
