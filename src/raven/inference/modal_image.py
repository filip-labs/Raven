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

    # No weights/data in image layers. Only the explicitly named Raven source package is included.
    return (
        modal.Image.from_registry(image_spec()["base"], add_python="3.10")
        .apt_install("git", "build-essential", "ninja-build")
        .env(
            {
                "CUMM_DISABLE_JIT": "1",
                "SPCONV_DISABLE_JIT": "1",
                "TORCH_CUDA_ARCH_LIST": "7.5",
                "CUDA_HOME": "/usr/local/cuda",
                "RAVEN_MODAL_GPU": "T4",
            }
        )
        .pip_install("pip==24.3.1", "setuptools==69.5.1", "wheel==0.43.0", "numpy==1.26.4")
        .pip_install("torch==2.2.2+cu118", index_url="https://download.pytorch.org/whl/cu118")
        .pip_install(
            "spconv-cu118==2.3.6",
            "cumm-cu118==0.4.11",
            "numba==0.59.1",
            "scipy==1.12.0",
            "easydict==1.13",
            "PyYAML==6.0.2",
            "scikit-image==0.22.0",
            "tensorboardX==2.6.2.2",
            "tqdm==4.67.1",
            "SharedArray==3.2.4",
            "pandas==2.2.3",
            "opencv-python-headless==4.10.0.84",
            "pyquaternion==0.9.9",
            "pybind11==2.13.6",
            "ninja==1.11.1.3",
        )
        .run_commands(
            "git clone https://github.com/open-mmlab/OpenPCDet.git /opt/OpenPCDet",
            f"git -C /opt/OpenPCDet checkout --detach {REVISION}",
            "cd /opt/OpenPCDet && python -m pip install --no-build-isolation --no-deps -e .",
        )
        .add_local_python_source("raven", copy=True, ignore=[])
        # Explicit, hash-guarded optional Argoverse-only patch; backup/diff are preserved.
        .run_commands("python -m raven.inference.patch_argo")
    )
