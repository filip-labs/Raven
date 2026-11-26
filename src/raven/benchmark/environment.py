"""Measured host/source metadata without importing inference packages."""

import os
import platform
import subprocess
import time
from pathlib import Path

from raven.experiments.artifacts import file_sha256
from raven.experiments.runner import _environment


def environment():
    info = _environment()  # Uses distribution metadata only, never imports torch.
    cpu, source = platform.processor(), "platform.processor"
    if platform.system() == "Darwin":
        try:
            cpu = subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            source = "sysctl machdep.cpu.brand_string"
        except (OSError, subprocess.SubprocessError):
            pass
    elif platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu, source = line.split(":", 1)[1].strip(), "/proc/cpuinfo model name"
                    break
        except OSError:
            pass
    clock = time.get_clock_info("perf_counter")
    root = Path(__file__).parents[1]
    files = [
        *Path(__file__).parent.glob("*.py"),
        *(root / "perception/preprocessing").glob("*.py"),
        *(root / "perception/detection").glob("*.py"),
        root / "core/point_cloud.py",
        root / "perception/data/kitti_loader.py",
        root / "experiments/config.py",
        root / "experiments/splits.py",
    ]
    return {
        **info,
        "cpu_model": cpu or None,
        "cpu_model_source": source if cpu else "unavailable",
        "logical_cpu_count": os.cpu_count(),
        "hardware_identity": "OS-reported, not independently authenticated",
        "thread_environment": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "clock": {
            "name": "time.perf_counter_ns",
            "implementation": clock.implementation,
            "monotonic": clock.monotonic,
            "resolution_seconds": clock.resolution,
        },
        "benchmark_source_sha256": {
            str(p.relative_to(Path(__file__).parents[1])): file_sha256(p) for p in files
        },
    }
