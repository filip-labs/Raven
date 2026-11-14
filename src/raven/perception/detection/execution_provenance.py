"""Capture source/run identity without importing an inference backend."""

import hashlib
import subprocess
import uuid
from pathlib import Path


def _git(root, *arguments):
    return subprocess.run(
        ["git", "-C", str(root), *arguments], check=True, capture_output=True
    ).stdout


def execution_provenance(openpcdet_root):
    """Call once per real inference invocation; no invented fallback commit IDs."""
    raven_root = Path(__file__).resolve().parents[4]
    try:
        commit = _git(raven_root, "rev-parse", "HEAD").decode().strip()
        dirty = _git(raven_root, "status", "--porcelain").decode().splitlines()
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    try:
        patch = _git(openpcdet_root, "diff", "--binary", "HEAD", "--")
        patch_digest = hashlib.sha256(patch).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        patch_digest = None
    return {
        "raven_git_commit": commit,
        "raven_modified_files": dirty,
        "inference_run_id": uuid.uuid4().hex,
        "openpcdet_diff_sha256": patch_digest,
    }
