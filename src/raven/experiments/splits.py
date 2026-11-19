"""Offline verification of the existing pinned KITTI train/validation manifests."""

from pathlib import Path

from raven.perception.data.frame_manifest import read_verified_split_manifests


def split_membership(split: str) -> tuple[frozenset[str], dict]:
    """Return verified IDs/evidence; absent evidence is never inferred from a directory."""
    if split == "test":
        raise FileNotFoundError(
            "authoritative test split manifest unavailable; testing is unlabeled"
        )
    packaged = Path(__file__).with_name("kitti_splits")
    directory = (
        packaged
        if packaged.is_dir()
        else Path(__file__).resolve().parents[3] / "configs/kitti_splits"
    )
    groups, provenance, metadata_hash = read_verified_split_manifests(directory)
    record = provenance["files"][split]
    return frozenset(groups[split]), {
        "status": "verified",
        "split": split,
        "manifest_sha256": record["sha256"],
        "source_url": record["url"],
        "upstream_commit": provenance["upstream_commit"],
        "metadata_sha256": metadata_hash,
    }
