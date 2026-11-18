"""Strict KITTI frame identifiers shared by dataset inspection and experiments."""

import hashlib
import json
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


def read_verified_split_manifests(directory: str | Path) -> tuple[dict, dict, str]:
    """Verify existing KITTI train/val policy once for dataset tools and experiments."""
    directory = Path(directory)
    raw = (directory / "provenance.json").read_bytes()
    metadata = json.loads(raw)
    groups = {}
    for name in ("train", "val"):
        ids, checksum = read_frame_manifest(directory / f"{name}.txt")
        expected = metadata["files"][name]
        if checksum != expected["sha256"] or len(ids) != expected["count"]:
            raise ValueError(f"invalid {name} split evidence: checksum/count mismatch")
        groups[name] = ids
    train, val = set(groups["train"]), set(groups["val"])
    if train & val or train | val != {f"{i:06d}" for i in range(7481)}:
        raise ValueError("invalid KITTI split evidence: overlap or incomplete membership")
    return groups, metadata, hashlib.sha256(raw).hexdigest()
