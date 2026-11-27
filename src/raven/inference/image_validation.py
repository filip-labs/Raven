"""Fail-fast CPU-compatible candidate-image checks. No checkpoint or GPU operation."""

import hashlib
import importlib
import importlib.metadata
import json
import platform
import subprocess
from copy import deepcopy
from pathlib import Path

from raven.inference.contracts import (
    BASE_CONFIG,
    MODEL_CONFIG,
    PACKAGES,
    REVISION,
    supported_python,
)
from raven.inference.patch_argo import validate_patch
from raven.perception.detection._openpcdet import validate_config, working_directory

NATIVE_MODULES = (
    "pcdet.ops.iou3d_nms.iou3d_nms_cuda",
    "pcdet.ops.roiaware_pool3d.roiaware_pool3d_cuda",
    "pcdet.ops.roipoint_pool3d.roipoint_pool3d_cuda",
    "pcdet.ops.pointnet2.pointnet2_stack.pointnet2_stack_cuda",
    "pcdet.ops.pointnet2.pointnet2_batch.pointnet2_batch_cuda",
    "pcdet.ops.bev_pool.bev_pool_ext",
    "pcdet.ops.ingroup_inds.ingroup_inds_cuda",
)
IMPORTS = (
    "torch",
    "torchvision",
    "spconv.pytorch",
    "cumm.tensorview",
    "SharedArray",
    "numba",
    "scipy",
    "skimage",
    "cv2",
    "pyquaternion",
    "easydict",
    "yaml",
    "PIL",
    "tensorboardX",
    "tqdm",
    "pcdet.datasets",
    "pcdet.models",
)


def validate_packages():
    python = platform.python_version()
    if not supported_python(python):
        raise RuntimeError(f"candidate image requires Python 3.10.x, observed {python}")
    versions = {}
    for name, expected in PACKAGES.items():
        try:
            observed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"image dependency missing: {name}=={expected}") from exc
        if observed != expected:
            raise RuntimeError(
                f"image dependency mismatch: {name}, expected {expected}, observed {observed}"
            )
        versions[name] = observed
    return versions


def validate_source(root):
    root = Path(root).resolve()

    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True)
        return result.stdout

    try:
        top = Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve()
        revision = git("rev-parse", "HEAD").decode().strip()
        changed = git("diff", "--name-only", "HEAD", "--").decode().splitlines()
        diff = git("diff", "--binary", "HEAD", "--")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"OpenPCDet Git provenance unavailable at {root}") from exc
    if top != root or revision != REVISION:
        raise ValueError("OpenPCDet source root/revision mismatch")
    if changed != ["pcdet/datasets/__init__.py"]:
        raise ValueError(f"unexpected OpenPCDet tracked modifications: {changed}")
    argo = validate_patch(root)
    configs = {}
    for name, expected in (
        ("tools/cfgs/kitti_models/pointpillar.yaml", MODEL_CONFIG),
        ("tools/cfgs/dataset_configs/kitti_dataset.yaml", BASE_CONFIG),
    ):
        observed = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"worker model configuration checksum mismatch: {name}")
        configs[name] = observed
    return {
        "openpcdet_commit": revision,
        "openpcdet_modified_files": changed,
        "openpcdet_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "argo_workaround": argo,
        "model_configurations": configs,
    }


def validate_cpu_environment(root):
    """Imports/linkage/config/CPU voxelization only; constructing model anchors needs GPU."""
    import sys

    from raven.perception.detection._openpcdet import configure_environment

    root = Path(root).resolve()
    configure_environment()  # Before any spconv/cumm import, including build-time checks.
    packages = validate_packages()
    source = validate_source(root)
    sys.path.insert(0, str(root))
    try:
        with working_directory(root):
            pcdet = importlib.import_module("pcdet")
            if Path(pcdet.__file__).resolve().parent != root / "pcdet":
                raise ValueError("imported pcdet is outside verified checkout")
            for name in (*IMPORTS, *NATIVE_MODULES):
                try:
                    importlib.import_module(name)
                except (ImportError, OSError, RuntimeError) as exc:
                    raise RuntimeError(
                        f"image import/native linkage failed: {name}: {type(exc).__name__}"
                    ) from exc
            torch = importlib.import_module("torch")
            vision = importlib.import_module("torchvision")
            if (
                torch.__version__ != PACKAGES["torch"]
                or vision.__version__ != PACKAGES["torchvision"]
                or torch.version.cuda != "11.8"
            ):
                raise ValueError("Torch/torchvision/CUDA runtime pair mismatch")
            if not vision.extension._has_ops():
                raise RuntimeError("torchvision compiled operators unavailable")
            # CPU dispatcher check: no GPU allocation or downloaded model.
            vision.ops.nms(torch.empty((0, 4)), torch.empty((0,)), 0.5)
            config_module = importlib.import_module("pcdet.config")
            cfg = deepcopy(config_module.cfg)
            with working_directory(root / "tools"):
                config_module.cfg_from_yaml_file(
                    str(root / "tools/cfgs/kitti_models/pointpillar.yaml"), cfg
                )
            validate_config(cfg)
            dataset_type = importlib.import_module("pcdet.datasets").DatasetTemplate
            dataset = dataset_type(
                dataset_cfg=cfg.DATA_CONFIG,
                class_names=cfg.CLASS_NAMES,
                training=False,
                root_path=root,
                logger=None,
            )
            import numpy as np

            prepared = dataset.prepare_data(
                {
                    "points": np.array([[20, 0, 0, 0.5], [20.01, 0, 0, 0.6]], dtype=np.float32),
                    "frame_id": "000000",
                }
            )
            if prepared["voxels"].ndim != 3 or len(prepared["voxels"]) == 0:
                raise RuntimeError("synthetic CPU voxelization produced invalid voxels")
    finally:
        sys.path.remove(str(root))
    return {
        "scope": "CPU imports/linkage/configuration/synthetic voxelization; no model construction, weights or GPU kernels",
        "python": platform.python_version(),
        "packages": packages,
        "source": source,
        "imports": list(IMPORTS),
        "native_modules": list(NATIVE_MODULES),
        "gpu_execution": "not_performed",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/opt/OpenPCDet"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate_cpu_environment(args.root)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        "CPU image dependency/native/configuration checks passed; GPU model execution not performed."
    )
