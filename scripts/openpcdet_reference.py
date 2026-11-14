"""Independent headless execution of the pinned upstream DemoDataset/forward path.

Shares only setup/serialization helpers with Raven. Never calls its adapter or
OpenPCDetSession. The upstream DemoDataset class is executed unchanged, while
GUI imports/rendering are omitted. Checkpoint loading is explicitly strict/safe.
"""

import ast
import glob
import os
import pickle
import platform
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import numpy as np
from kaggle_pointpillars import main

from raven.perception.detection._openpcdet import (
    PINNED_REVISION,
    command_output,
    configure_environment,
    package_version,
    plain_config,
    sha256_file,
    validate_assets,
    validate_config,
    working_directory,
)
from raven.perception.detection.reference_validation import REFERENCE_ENTRYPOINT

# SHA256 of tools/demo.py at the pinned OpenPCDet commit (verified upstream).
DEMO_SOURCE_SHA256 = "f75942984a03ae605862bfd29d0e664a9e9676624cf9cb17256ea4219815dcfe"


def upstream_demo_dataset(path, base_class):
    """Load only the original class; do not import tools/demo.py's GUI dependencies."""
    module = ast.parse(path.read_text())
    classes = [
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "DemoDataset"
    ]
    if len(classes) != 1:
        raise ValueError("Pinned upstream demo.py must define exactly one DemoDataset")
    namespace = {"DatasetTemplate": base_class, "np": np, "glob": glob, "Path": Path}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102 -- trusted pinned upstream class; never capture data
    return namespace["DemoDataset"]


def run_reference(args, _cloud):
    if args.adapter:
        raise ValueError("Reference execution cannot use --adapter")
    root, config_path, checkpoint_path = validate_assets(
        args.openpcdet_root, args.config, args.checkpoint, args.checkpoint_sha256
    )
    configure_environment()
    sys.path.insert(0, str(root))
    try:
        import pcdet
        import torch
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.datasets import DatasetTemplate
        from pcdet.models import build_network, load_data_to_gpu
        from pcdet.utils import common_utils
    except ImportError as exc:
        raise RuntimeError(f"External OpenPCDet GPU dependencies unavailable: {exc}") from exc
    finally:
        sys.path.remove(str(root))
    if Path(pcdet.__file__).resolve().parent != root / "pcdet":
        raise ValueError("Imported OpenPCDet does not match requested checkout")
    if not torch.cuda.is_available():
        raise RuntimeError("Independent reference requires an external CUDA GPU")
    if args.device_index < 0 or args.device_index >= torch.cuda.device_count():
        raise ValueError("device-index is outside visible CUDA devices")
    revision = command_output(["git", "-C", str(root), "rev-parse", "HEAD"])
    if revision != PINNED_REVISION:
        raise ValueError(f"Independent reference requires OpenPCDet {PINNED_REVISION}")
    demo_path = root / "tools/demo.py"
    if sha256_file(demo_path) != DEMO_SOURCE_SHA256:
        raise ValueError("Reference tools/demo.py differs from the pinned upstream source")
    demo_class = upstream_demo_dataset(demo_path, DatasetTemplate)
    config = deepcopy(cfg)
    with working_directory(root / "tools"), torch.cuda.device(args.device_index):
        cfg_from_yaml_file(str(config_path), config)
        validate_config(config)
        dataset = demo_class(
            dataset_cfg=config.DATA_CONFIG,
            class_names=config.CLASS_NAMES,
            training=False,
            root_path=args.input,
            ext=".bin",
            logger=common_utils.create_logger(),
        )
        model = build_network(
            model_cfg=config.MODEL, num_class=len(config.CLASS_NAMES), dataset=dataset
        )
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=not args.trust_checkpoint
            )
        except pickle.UnpicklingError as exc:
            raise RuntimeError(
                "Review checkpoint provenance; unrestricted pickle needs explicit "
                "--trust-checkpoint. No fallback was attempted."
            ) from exc
        if not isinstance(checkpoint, Mapping) or not isinstance(
            checkpoint.get("model_state"), Mapping
        ):
            raise TypeError("Expected checkpoint model_state mapping")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.cuda().eval()
        with torch.no_grad():
            data = dataset[
                0
            ]  # Real upstream __getitem__: reads file and calls prepare_data itself.
            batch = dataset.collate_batch([data])
            load_data_to_gpu(batch)
            results, _ = model.forward(batch)
            if len(results) != 1:
                raise ValueError("Reference expected exactly one prediction batch")
            arrays = [
                results[0][key].detach().cpu().numpy().copy()
                for key in ("pred_boxes", "pred_scores", "pred_labels")
            ]
        dirty = command_output(["git", "-C", str(root), "diff", "--name-only", "HEAD"])
        return (
            arrays,
            list(config.CLASS_NAMES),
            {
                "inference_entrypoint": REFERENCE_ENTRYPOINT,
                "runner_sha256": sha256_file(Path(__file__).resolve()),
                "reference_dataset_sha256": sha256_file(demo_path),
                "openpcdet_commit": revision,
                "openpcdet_modified_files": dirty.splitlines() if dirty is not None else None,
                "effective_config": plain_config(config),
                "environment": {
                    "python": platform.python_version(),
                    "pytorch": torch.__version__,
                    "torch_cuda_runtime": torch.version.cuda,
                    "openpcdet": pcdet.__version__,
                    "numpy": np.__version__,
                    "spconv_cu126": package_version("spconv-cu126"),
                    "cumm_cu126": package_version("cumm-cu126"),
                    "gpu": torch.cuda.get_device_name(args.device_index),
                    "compute_capability": list(torch.cuda.get_device_capability(args.device_index)),
                    "visible_gpu_count": torch.cuda.device_count(),
                    "device_index": args.device_index,
                    "cuda_toolkit_nvcc": command_output(["nvcc", "--version"]),
                    "gpu_driver": command_output(
                        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
                    ),
                    "environment_flags": {
                        key: os.environ[key]
                        for key in (
                            "CUMM_DISABLE_JIT",
                            "SPCONV_DISABLE_JIT",
                            "TORCH_CUDA_ARCH_LIST",
                        )
                    },
                },
            },
        )


if __name__ == "__main__":
    raise SystemExit(main(inference=run_reference, description=__doc__))
