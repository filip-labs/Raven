"""Strict KITTI frame identifiers shared by dataset inspection and experiments."""

import hashlib
import re
from pathlib import Path


def validate_frame_ids(values) -> tuple[str, ...]:
    """Preserve explicit order; reject empty, duplicate and non-six-digit IDs."""
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("expected nonempty six-digit frame identifiers")
    if any(not isinstance(v, str) or not re.fullmatch(r"[0-9]{6}", v) for v in values):
        raise ValueError("expected nonempty six-digit frame identifiers")
    if len(values) != len(set(values)):
        raise ValueError("duplicate frame identifiers")
    return tuple(values)


def read_frame_manifest(path: str | Path) -> tuple[tuple[str, ...], str]:
    """Read once; hash the exact UTF-8 bytes parsed, including line endings."""
    path = Path(path)
    raw = path.read_bytes()
    try:
        return validate_frame_ids(raw.decode("utf-8").splitlines()), hashlib.sha256(raw).hexdigest()
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
