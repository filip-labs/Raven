"""Explicit container-only optional Argoverse import workaround; never patches Raven."""

import hashlib
from pathlib import Path

PREIMAGE = "2253ce236c5fb203db788131df2d2d348a9beb473db064a5ecd566b59f876da8"


def patch(root):
    path = Path(root) / "pcdet/datasets/__init__.py"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PREIMAGE:
        raise ValueError("Argoverse workaround preimage mismatch")
    text = raw.decode().replace("from .argo2.argo2_dataset import Argo2Dataset\n", "")
    text = text.replace("    'Argo2Dataset': Argo2Dataset\n", "")
    if "Argo2Dataset" in text:
        raise ValueError("Argoverse workaround inventory mismatch")
    with path.with_suffix(".py.bak").open("xb") as backup:
        backup.write(raw)
    path.write_text(text)


if __name__ == "__main__":
    patch("/opt/OpenPCDet")
