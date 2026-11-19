"""Exclusive, atomic minimal run persistence; a final manifest is the commit marker."""

import hashlib
import json
import os
import tempfile
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"required regular input/artifact file unavailable: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_exclusive(path: Path, text: str) -> None:
    """Publish a fully flushed file without replacing an existing destination."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".raven-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic and fails on any existing file/symlink.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json_exclusive(path: Path, document: dict) -> None:
    write_text_exclusive(path, json.dumps(document, indent=2, allow_nan=False) + "\n")


def reserve_run(config) -> Path:
    base = config.output.directory
    if base.resolve() != base or config.run_directory.resolve() != config.run_directory:
        raise ValueError("output path changed through a symlink")
    base.mkdir(parents=True, exist_ok=True)
    config.run_directory.mkdir(mode=0o700)  # No exist_ok: also excludes concurrent runs.
    if base.resolve() != base or config.run_directory.resolve() != config.run_directory:
        raise ValueError("output path changed while reserving run")
    return config.run_directory


def persist_run(config, summary, frames: list[dict], provenance: dict) -> Path:
    paths = {
        **config.artifact_paths,
        "started": config.run_directory / f"{config.output.artifact_prefix}.started.json",
    }
    write_json_exclusive(paths["configuration"], config.to_dict())
    write_text_exclusive(paths["frames"], "".join(f"{v}\n" for v in config.frame_ids))
    write_json_exclusive(
        paths["results"],
        {
            "run_schema_version": 1,
            "summary": summary.to_dict(),
            "frames": frames,
        },
    )
    write_json_exclusive(paths["provenance"], provenance)
    manifest_path = config.run_directory / f"{config.output.artifact_prefix}.manifest.json"
    write_json_exclusive(
        manifest_path,
        {
            "run_schema_version": 1,
            "status": summary.status,
            "config_sha256": config.config_sha256,
            "frame_ids": list(config.frame_ids),
            "artifacts": {k: {"file": p.name, "sha256": file_sha256(p)} for k, p in paths.items()},
        },
    )
    return manifest_path
