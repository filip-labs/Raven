"""Small separate benchmark specification; experiment schema v1 remains unchanged."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from raven.experiments.config import _constant, _object, _slug


@dataclass(frozen=True)
class BenchmarkSpec:
    mode: str = "preprocessing"
    warmup: int = 5
    repetitions: int = 30
    include_loading: bool = False
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported benchmark schema_version")
        if self.mode not in ("preprocessing", "inference"):
            raise ValueError("benchmark mode must be preprocessing or inference")
        if type(self.warmup) is not int or not 0 <= self.warmup <= 20:
            raise ValueError("warmup must be integer in [0,20]")
        if type(self.repetitions) is not int or not 1 <= self.repetitions <= 100:
            raise ValueError("repetitions must be integer in [1,100]")
        if type(self.include_loading) is not bool:
            raise ValueError("include_loading must be boolean")

    def to_dict(self):
        return asdict(self)

    def validate_size(self, frame_ids):
        if len(frame_ids) > 32 or len(frame_ids) * (self.warmup + self.repetitions) > 1000:
            raise ValueError("benchmark limited to 32 frames and 1000 total iterations")


def load_benchmark_spec(path):
    value = json.loads(
        Path(path).read_text(encoding="utf-8"), object_pairs_hook=_object, parse_constant=_constant
    )
    if not isinstance(value, dict) or set(value) != set(BenchmarkSpec().to_dict()):
        raise ValueError(
            "benchmark spec requires exactly schema_version, mode, warmup, repetitions, include_loading"
        )
    return BenchmarkSpec(**value)


def benchmark_directory(config, benchmark_id):
    _slug(benchmark_id, "benchmark_id")
    path = config.output.directory / f"{benchmark_id}.benchmark"
    if path.resolve() != path or path == config.run_directory:
        raise ValueError("benchmark output must be a separate nonsymlink directory")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing existing benchmark directory: {path}")
    return path
