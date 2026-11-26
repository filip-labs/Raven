"""Integer monotonic intervals and explicit population/linear-percentile statistics."""

from dataclasses import asdict, dataclass
from time import perf_counter_ns

import numpy as np

MAX_DURATION_NS = 86_400_000_000_000


def duration_ns(value):
    if type(value) is not int or not 0 <= value <= MAX_DURATION_NS:
        raise ValueError("duration must be integer nanoseconds in [0,24 hours]")
    return value


@dataclass(frozen=True)
class TimingSummary:
    count: int
    mean_ms: float
    median_ms: float
    minimum_ms: float
    maximum_ms: float
    std_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    percentile_method: str = "linear"
    std_ddof: int = 0

    def to_dict(self):
        return asdict(self)


def summarize(samples) -> TimingSummary:
    values = tuple(duration_ns(v) for v in samples)
    if not values:
        raise ValueError("cannot summarize empty timing samples")
    a = np.asarray(values, dtype=np.float64) / 1_000_000
    p50, p90, p95, p99 = np.percentile(a, [50, 90, 95, 99], method="linear")
    return TimingSummary(
        len(a),
        float(a.mean()),
        float(np.median(a)),
        float(a.min()),
        float(a.max()),
        float(a.std(ddof=0)),
        float(p50),
        float(p90),
        float(p95),
        float(p99),
    )


class Timer:
    def __init__(self, clock=None):
        self.clock = perf_counter_ns if clock is None else clock

    def start(self):
        value = self.clock()
        if type(value) is not int or value < 0:
            raise ValueError("clock must return nonnegative integer nanoseconds")
        return value

    def stop(self, start):
        return duration_ns(self.start() - start)

    def call(self, operation, synchronize=None):
        if synchronize is not None:
            synchronize()
        start = self.start()
        value = operation()
        if synchronize is not None:
            synchronize()
        return value, self.stop(start)
