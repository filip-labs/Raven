"""Explicit container-only optional Argoverse import workaround; never patches Raven."""

import hashlib
from pathlib import Path

PREIMAGE = "2253ce236c5fb203db788131df2d2d348a9beb473db064a5ecd566b59f876da8"


def patched_source(raw):
    if hashlib.sha256(raw).hexdigest() != PREIMAGE:
        raise ValueError("Argoverse workaround preimage mismatch")
    text = raw.decode().replace("from .argo2.argo2_dataset import Argo2Dataset\n", "")
    text = text.replace("    'Argo2Dataset': Argo2Dataset\n", "")
    if "Argo2Dataset" in text:
        raise ValueError("Argoverse workaround inventory mismatch")
    return text.encode()


def validate_patch(root):
    path = Path(root) / "pcdet/datasets/__init__.py"
    backup = path.with_suffix(".py.bak").read_bytes()
    expected = patched_source(backup)
    if path.read_bytes() != expected:
        raise ValueError("Argoverse workaround content differs from the exact reviewed patch")
    return {
        "backup_sha256": hashlib.sha256(backup).hexdigest(),
        "patched_sha256": hashlib.sha256(expected).hexdigest(),
    }


def patch(root):
    path = Path(root) / "pcdet/datasets/__init__.py"
    raw = path.read_bytes()
    patched = patched_source(raw)
    with path.with_suffix(".py.bak").open("xb") as backup:
        backup.write(raw)
    path.write_bytes(patched)


if __name__ == "__main__":
    patch("/opt/OpenPCDet")
