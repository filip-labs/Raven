"""Synthetic OpenPCDet boundary for CPU-only adapter/runner tests."""

import sys
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from raven.core.point_cloud import PointCloud
from raven.perception.detection import _openpcdet as backend


class Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@pytest.fixture
def gpu_backend(tmp_path, monkeypatch):
    root = tmp_path / "OpenPCDet"
    (root / "tools").mkdir(parents=True)
    (root / "pcdet").mkdir()
    (root / "pcdet/__init__.py").write_text("# synthetic CPU fixture\n")
    config_path = root / "tools/pointpillar.yaml"
    config_path.write_text("# synthetic config; parser is mocked\n")
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"synthetic checkpoint; torch.load is mocked")
    encoding = Config(
        encoding_type="absolute_coordinates_encoding",
        src_feature_list=["x", "y", "z", "intensity"],
        used_feature_list=["x", "y", "z", "intensity"],
    )
    cfg = Config(
        MODEL=Config(
            NAME="PointPillar",
            VFE=Config(NAME="PillarVFE"),
            BACKBONE_2D=Config(NAME="BaseBEVBackbone"),
            MAP_TO_BEV=Config(NAME="PointPillarScatter"),
            DENSE_HEAD=Config(
                NAME="AnchorHeadSingle", TARGET_ASSIGNER_CONFIG=Config(BOX_CODER="ResidualCoder")
            ),
            POST_PROCESSING=Config(OUTPUT_RAW_SCORE=False),
        ),
        DATA_CONFIG=Config(
            DATASET="KittiDataset",
            POINT_FEATURE_ENCODING=encoding,
            DATA_PROCESSOR=[Config(NAME="transform_points_to_voxels")],
        ),
        CLASS_NAMES=["Car", "Pedestrian", "Cyclist"],
        ROOT_DIR=root,
    )
    model = Mock()
    model.cuda.return_value = model
    model.eval.return_value = model
    raw = [
        np.array([[1.0, 2, 3, 4, 2, 1, 0.5], [5, 6, 7, 0.5, 0.6, 1.8, -0.2]]),
        np.array([0.0, 1.0]),
        np.array([1, 2]),
    ]

    def set_predictions(arrays):
        tensors = [
            SimpleNamespace(
                detach=lambda value=value: SimpleNamespace(
                    cpu=lambda: SimpleNamespace(numpy=lambda: value)
                )
            )
            for value in arrays
        ]
        model.forward.return_value = (
            [dict(zip(("pred_boxes", "pred_scores", "pred_labels"), tensors, strict=True))],
            {},
        )

    set_predictions(raw)
    dataset = Mock()
    dataset.prepare_data.side_effect = lambda data: data
    dataset.collate_batch.side_effect = lambda samples: {"sample": samples[0]}
    build = Mock(return_value=model)
    loader = Mock(return_value={"model_state": {"synthetic_weight": 1}})
    selected = []

    @contextmanager
    def device(index):
        selected.append(index)
        try:
            yield
        finally:
            selected.pop()

    cuda = SimpleNamespace(
        is_available=Mock(return_value=True),
        get_device_name=lambda index: "Fake T4",
        get_device_capability=lambda index: (7, 5),
        device_count=lambda: 2,
        device=Mock(side_effect=device),
    )
    torch = SimpleNamespace(
        cuda=cuda,
        load=loader,
        no_grad=nullcontext,
        __version__="synthetic-test",
        version=SimpleNamespace(cuda="test"),
    )
    configure = Mock()
    transfer = Mock()
    dataset_class = Mock(return_value=dataset)
    modules = {
        "torch": torch,
        "pcdet": SimpleNamespace(__file__=str(root / "pcdet/__init__.py"), __version__="test"),
        "pcdet.config": SimpleNamespace(cfg=cfg, cfg_from_yaml_file=configure),
        "pcdet.datasets": SimpleNamespace(DatasetTemplate=dataset_class),
        "pcdet.models": SimpleNamespace(build_network=build, load_data_to_gpu=transfer),
        "pcdet.utils": SimpleNamespace(common_utils=SimpleNamespace(create_logger=Mock())),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    for name in ("CUMM_DISABLE_JIT", "SPCONV_DISABLE_JIT", "TORCH_CUDA_ARCH_LIST"):
        monkeypatch.delenv(name, raising=False)

    def git_result(*args, **kwargs):
        value = (
            backend.PINNED_REVISION
            if args[0][-1] == "HEAD" and "rev-parse" in args[0]
            else "pcdet/datasets/__init__.py\n"
        )
        return SimpleNamespace(stdout=value if kwargs.get("text") else value.encode())

    monkeypatch.setattr(backend.subprocess, "run", Mock(side_effect=git_result))
    monkeypatch.setattr(backend, "command_output", lambda _: None)
    monkeypatch.setattr(backend, "package_version", lambda _: None)
    return SimpleNamespace(
        options={
            "openpcdet_root": root,
            "config": config_path,
            "checkpoint": checkpoint,
            "checkpoint_sha256": backend.sha256_file(checkpoint),
        },
        root=root,
        cfg=cfg,
        model=model,
        dataset=dataset,
        build=build,
        loader=loader,
        cuda=cuda,
        torch=torch,
        configure=configure,
        transfer=transfer,
        selected=selected,
        dataset_class=dataset_class,
        raw=raw,
        set_predictions=set_predictions,
        cloud=PointCloud(np.array([[1, 2, 3, 0.5], [4, 5, 6, 0.7]]), "000000", "kitti_velodyne"),
    )
