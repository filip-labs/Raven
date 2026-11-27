"""Reviewed candidate image recipe. Constructing the graph does not build/deploy it."""

import hashlib
import json
from pathlib import Path

from raven.inference.contracts import REVISION, digest


def image_spec():
    return json.loads(Path(__file__).with_name("image-spec.json").read_text())


def recipe_sha256():
    return digest(
        {
            "spec": image_spec(),
            "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )


def build_image():
    import modal

    spec = image_spec()
    build = [f"{name}=={version}" for name, version in spec["build_packages"].items()]
    packages = spec["packages"]
    cuda = [f"{name}=={packages[name]}" for name in ("torch", "torchvision")]
    other = [
        f"{name}=={version}"
        for name, version in packages.items()
        if name not in {"torch", "torchvision", "pcdet"}
    ]
    # No assets/weights; constructing this graph does not build it or change the workspace builder.
    return (
        modal.Image.from_registry(spec["base"], add_python=spec["python"])
        .apt_install("git", "build-essential", "ninja-build", "clang")
        .env(
            {
                "CUMM_DISABLE_JIT": "1",
                "SPCONV_DISABLE_JIT": "1",
                "TORCH_CUDA_ARCH_LIST": "7.5",
                "CUDA_HOME": "/usr/local/cuda",
                "RAVEN_MODAL_GPU": "T4",
                "PYTHONPATH": "/root",
            }
        )
        .pip_install(*build, f"numpy=={packages['numpy']}")
        .pip_install(*cuda, index_url="https://download.pytorch.org/whl/cu118")
        .pip_install(*other)
        .run_commands(
            "git clone https://github.com/open-mmlab/OpenPCDet.git /opt/OpenPCDet",
            f"git -C /opt/OpenPCDet checkout --detach {REVISION}",
            "cd /opt/OpenPCDet && python -m pip install --no-build-isolation --no-deps -e .",
        )
        .add_local_python_source("raven", copy=True, ignore=[])
        # Explicit upstream patch; CPU import/native/config checks fail BEFORE GPU startup.
        .run_commands(
            "PYTHONPATH=/root python -m raven.inference.patch_argo",
            "python -m pip check",
            "PYTHONPATH=/root python -m raven.inference.image_validation --root /opt/OpenPCDet --output /opt/raven-image-validation.json",
        )
    )
