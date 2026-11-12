"""Read-only KITTI split/inventory checks; sample content uses existing Raven parsers."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from raven.perception.data.kitti_annotations import load_kitti_labels
from raven.perception.data.kitti_calibration import load_kitti_calibration
from raven.perception.data.kitti_loader import load_kitti_bin

SPLITS = Path(__file__).resolve().parents[1] / "configs/kitti_splits"


def read_ids(path):
    ids = Path(path).read_text().splitlines()
    if not ids or any(not re.fullmatch(r"[0-9]{6}", value) for value in ids):
        raise ValueError(f"{path}: expected nonempty six-digit frame identifiers")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate frame identifiers")
    return ids


def validate_splits(directory=SPLITS):
    directory = Path(directory)
    manifest = json.loads((directory / "provenance.json").read_text())
    groups = {}
    for name in ("train", "val"):
        path = directory / f"{name}.txt"
        groups[name] = read_ids(path)
        recorded = manifest["files"][name]
        if (
            len(groups[name]) != recorded["count"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != recorded["sha256"]
        ):
            raise ValueError(f"{path}: provenance checksum/count mismatch")
    if set(groups["train"]) & set(groups["val"]):
        raise ValueError("train/validation overlap")
    if set(groups["train"]) | set(groups["val"]) != {f"{i:06d}" for i in range(7481)}:
        raise ValueError("split union must be labeled training IDs 000000–007480")
    return groups


def validate_inventory(root):
    counts = {}
    for subset, total in (("training", 7481), ("testing", 7518)):
        expected = {f"{i:06d}" for i in range(total)}
        for kind, suffix in (("velodyne", ".bin"), ("calib", ".txt"), ("label_2", ".txt")):
            if subset == "testing" and kind == "label_2":
                continue
            directory = Path(root) / subset / kind
            actual = {p.stem for p in directory.glob(f"*{suffix}") if p.is_file()}
            if actual != expected:
                missing, extra = sorted(expected - actual), sorted(actual - expected)
                raise ValueError(f"{directory}: missing={missing[:3]}, unexpected={extra[:3]}")
            counts[f"{subset}/{kind}"] = len(actual)
    return counts


def validate_sample(root, frame):
    if not re.fullmatch(r"[0-9]{6}", frame) or int(frame) >= 7481:
        raise ValueError("sample must be a labeled training ID 000000–007480")
    base = Path(root) / "training"
    paths = [
        base / kind / f"{frame}{suffix}"
        for kind, suffix in (("velodyne", ".bin"), ("calib", ".txt"), ("label_2", ".txt"))
    ]
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    cloud = load_kitti_bin(paths[0])
    load_kitti_calibration(paths[1])
    labels = load_kitti_labels(paths[2])
    after = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    if before != after:
        raise ValueError(f"{frame}: source bytes changed during inspection")
    return {
        "frame": frame,
        "points": len(cloud.points),
        "objects": len(labels.objects),
        "sha256": dict(zip(("velodyne", "calib", "label_2"), before)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="check complete inventory and selected samples")
    parser.add_argument("--sample", action="append", help="training ID; repeatable; default 000000")
    args = parser.parse_args(argv)
    if args.sample and args.root is None:
        parser.error("--sample requires --root")
    try:
        groups = validate_splits()
        report = {"split_counts": {name: len(ids) for name, ids in groups.items()}}
        if args.root is not None:
            report["inventory"] = validate_inventory(args.root)
            report["samples"] = [
                validate_sample(args.root, frame) for frame in (args.sample or ["000000"])
            ]
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"KITTI validation failed: {exc}\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
