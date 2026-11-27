"""Local audit regressions: source Git is real temporary Git; runtime/SDK calls are mocked."""

import hashlib
import importlib.metadata
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from test_remote_inference import cloud as cloud  # noqa: PLC0414 -- Imported fixture.
from test_remote_inference import make_request

from raven.inference import contracts, modal_image, patch_argo
from raven.inference import image_validation as validation
from raven.inference.modal_transport import ModalTransport, RemoteExecutionError
from raven.inference.worker import WorkerInitializationError, startup_stage


@pytest.mark.parametrize(
    "python,valid",
    [
        ("3.10.16", True),
        ("3.10.99", True),
        ("3.10", False),
        ("3.11.1", False),
        ("3.10.fake", False),
        (None, False),
    ],
)
def test_python_profile_is_exact_major_minor_with_numeric_patch(python, valid):
    assert contracts.supported_python(python) is valid


def test_recipe_installs_reviewed_torchvision_pair_and_build_gate_before_any_gpu(monkeypatch):
    calls = []

    class Image:
        def __getattr__(self, name):
            def record(*args, **kwargs):
                calls.append((name, args, kwargs))
                return self

            return record

    image = Image()
    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(Image=SimpleNamespace(from_registry=image.from_registry)),
    )
    assert modal_image.build_image() is image
    assert calls[0][2]["add_python"] == "3.10"
    assert "clang" in next(args for name, args, kw in calls if name == "apt_install")
    pip = [(args, kw) for name, args, kw in calls if name == "pip_install"]
    torch_install = next((args, kw) for args, kw in pip if any("torch==" in item for item in args))
    assert set(torch_install[0]) == {"torch==2.2.2+cu118", "torchvision==0.17.2+cu118"}
    assert torch_install[1] == {"index_url": "https://download.pytorch.org/whl/cu118"}
    installed = {item for args, kw in pip for item in args}
    spec = modal_image.image_spec()
    assert spec["image_builder"] == "2025.06"
    assert spec["packages"] == contracts.PACKAGES
    assert {
        f"{name}=={version}" for name, version in contracts.PACKAGES.items() if name != "pcdet"
    } <= installed
    assert not any(
        item.startswith(("kornia", "av2", "torch-scatter", "pandas")) for item in installed
    )
    commands = [item for name, args, kw in calls if name == "run_commands" for item in args]
    assert commands.index("PYTHONPATH=/root python -m raven.inference.patch_argo") < next(
        i for i, v in enumerate(commands) if "image_validation --root" in v
    )
    assert "python -m pip check" in commands
    assert all("torchvision" not in command or "--upgrade" not in command for command in commands)


@pytest.mark.parametrize(
    "changed", ["torch", "torchvision", "spconv-cu118", "cumm-cu118", "llvmlite", "SharedArray"]
)
def test_runtime_inventory_rejects_missing_or_mismatched_dependencies(monkeypatch, changed):
    monkeypatch.setattr(validation.platform, "python_version", lambda: "3.10.20")
    monkeypatch.setattr(
        validation.importlib.metadata,
        "version",
        lambda name: "wrong" if name == changed else contracts.PACKAGES[name],
    )
    with pytest.raises(RuntimeError, match=changed.replace("-", ".")):
        validation.validate_packages()

    def missing(name):
        if name == changed:
            raise importlib.metadata.PackageNotFoundError(name)
        return contracts.PACKAGES[name]

    monkeypatch.setattr(validation.importlib.metadata, "version", missing)
    with pytest.raises(RuntimeError, match="missing"):
        validation.validate_packages()


@pytest.fixture
def upstream(tmp_path, monkeypatch):
    """Minimal synthetic source with real Git. Constants are scoped only to this fixture."""
    root = tmp_path / "OpenPCDet"
    (root / "pcdet/datasets").mkdir(parents=True)
    raw = b"from .argo2.argo2_dataset import Argo2Dataset\n__all__ = {\n    'Argo2Dataset': Argo2Dataset\n}\n"
    (root / "pcdet/datasets/__init__.py").write_bytes(raw)
    monkeypatch.setattr(patch_argo, "PREIMAGE", hashlib.sha256(raw).hexdigest())
    for relative, constant in [
        ("tools/cfgs/kitti_models/pointpillar.yaml", "MODEL_CONFIG"),
        ("tools/cfgs/dataset_configs/kitti_dataset.yaml", "BASE_CONFIG"),
    ]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic " + relative)
        monkeypatch.setattr(validation, constant, hashlib.sha256(path.read_bytes()).hexdigest())

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init")
    git("add", ".")
    git(
        "-c",
        "user.name=Synthetic Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-m",
        "synthetic source",
    )
    monkeypatch.setattr(validation, "REVISION", git("rev-parse", "HEAD"))
    patch_argo.patch(root)
    return root


def test_real_git_source_validation_is_independent_of_process_cwd(upstream, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    checked = validation.validate_source(upstream)
    assert checked["openpcdet_commit"] == validation.REVISION
    assert checked["openpcdet_modified_files"] == ["pcdet/datasets/__init__.py"]
    assert checked["argo_workaround"] == patch_argo.validate_patch(upstream)
    assert len(checked["openpcdet_diff_sha256"]) == 64


@pytest.mark.parametrize(
    "corruption", ["missing_git", "revision", "extra_file", "patch", "backup", "config"]
)
def test_source_audit_rejects_actual_git_patch_and_config_corruption(
    upstream, monkeypatch, corruption
):
    if corruption == "missing_git":
        (upstream / ".git").rename(upstream / ".git-preserved")
    elif corruption == "revision":
        monkeypatch.setattr(validation, "REVISION", "a" * 40)
    elif corruption == "extra_file":
        p = upstream / "tools/cfgs/kitti_models/pointpillar.yaml"
        p.write_text("tampered")
    elif corruption == "patch":
        p = upstream / "pcdet/datasets/__init__.py"
        p.write_bytes(p.read_bytes() + b"# extra edit\n")
    elif corruption == "backup":
        (upstream / "pcdet/datasets/__init__.py.bak").write_text("wrong backup")
    else:
        monkeypatch.setattr(validation, "MODEL_CONFIG", "a" * 64)
    with pytest.raises((RuntimeError, ValueError)):
        validation.validate_source(upstream)


@pytest.fixture
def cpu_graph(upstream, monkeypatch):
    monkeypatch.setattr(validation, "validate_packages", lambda: dict(contracts.PACKAGES))
    torch = SimpleNamespace(
        __version__=contracts.PACKAGES["torch"],
        version=SimpleNamespace(cuda="11.8"),
        empty=lambda shape: np.empty(shape),
    )
    vision = SimpleNamespace(
        __version__=contracts.PACKAGES["torchvision"],
        extension=SimpleNamespace(_has_ops=lambda: True),
        ops=SimpleNamespace(nms=Mock()),
    )
    cfg = SimpleNamespace(DATA_CONFIG="synthetic", CLASS_NAMES=["Car", "Pedestrian", "Cyclist"])
    configure = Mock()
    check = Mock()
    monkeypatch.setattr(validation, "validate_config", check)

    class Dataset:
        def __init__(self, **kwargs):
            assert kwargs["training"] is False

        def prepare_data(self, row):
            assert row["points"].shape == (2, 4)
            return {"voxels": np.zeros((1, 2, 4), dtype=np.float32)}

    modules = {
        "pcdet": SimpleNamespace(__file__=str(upstream / "pcdet/__init__.py")),
        "torch": torch,
        "torchvision": vision,
        "pcdet.config": SimpleNamespace(cfg=cfg, cfg_from_yaml_file=configure),
        "pcdet.datasets": SimpleNamespace(DatasetTemplate=Dataset),
    }
    calls = []

    def load(name):
        calls.append(name)
        return modules.get(name, SimpleNamespace())

    monkeypatch.setattr(validation.importlib, "import_module", load)
    return upstream, modules, calls, configure, check


def test_cpu_build_validation_imports_all_native_modules_without_constructing_a_model(cpu_graph):
    root, modules, calls, configure, check = cpu_graph
    cwd = Path.cwd()
    path = list(sys.path)
    result = validation.validate_cpu_environment(root)
    assert Path.cwd() == cwd and sys.path == path
    assert set(validation.NATIVE_MODULES) <= set(calls)
    assert set(validation.IMPORTS) <= set(calls)
    assert result["gpu_execution"] == "not_performed"
    configure.assert_called_once()
    check.assert_called_once()
    modules["torchvision"].ops.nms.assert_called_once()


@pytest.mark.parametrize("missing", ["torchvision", "pcdet.models", validation.NATIVE_MODULES[0]])
def test_build_gate_fails_on_missing_import_or_native_linkage(cpu_graph, monkeypatch, missing):
    root, *_ = cpu_graph
    original = validation.importlib.import_module

    def load(name):
        if name == missing:
            raise ModuleNotFoundError("synthetic missing import")
        return original(name)

    monkeypatch.setattr(validation.importlib, "import_module", load)
    with pytest.raises(RuntimeError, match="import/native linkage"):
        validation.validate_cpu_environment(root)


@pytest.mark.parametrize("broken", ["torch_version", "vision_version", "cuda", "vision_ops"])
def test_build_gate_rejects_torch_vision_runtime_abi_mismatch(cpu_graph, broken):
    root, modules, *_ = cpu_graph
    if broken == "torch_version":
        modules["torch"].__version__ = "2.3.0"
    elif broken == "vision_version":
        modules["torchvision"].__version__ = "0.18.0"
    elif broken == "cuda":
        modules["torch"].version.cuda = "12.1"
    else:
        modules["torchvision"].extension._has_ops = lambda: False
    with pytest.raises((ValueError, RuntimeError), match="mismatch|operators"):
        validation.validate_cpu_environment(root)


def test_worker_startup_stage_keeps_dependency_cause_and_safe_loading_policy():
    with (
        pytest.raises(
            WorkerInitializationError, match="stage=dependency_import; cause=ModuleNotFoundError"
        ) as error,
        startup_stage("dependency_import"),
    ):
        raise ModuleNotFoundError("torchvision")
    assert isinstance(error.value.__cause__, ModuleNotFoundError)
    with (
        pytest.raises(WorkerInitializationError, match="stage=safe_checkpoint_loading"),
        startup_stage("model_initialization"),
    ):
        raise RuntimeError("Checkpoint loading failed; no unsafe fallback")


def test_failed_enter_does_not_crash_loop_or_reinitialize_and_method_fails(monkeypatch):
    from raven.inference import worker

    class App:
        def __init__(self, *a):
            pass

        def cls(self, **kw):
            assert kw["retries"] == 0
            return lambda cls: cls

    fake = SimpleNamespace(
        App=App,
        is_local=lambda: False,
        Volume=SimpleNamespace(from_name=lambda *a, **kw: "private-existing-volume"),
        enter=lambda: lambda f: f,
        method=lambda: lambda f: f,
        exit=lambda: lambda f: f,
    )
    monkeypatch.setitem(sys.modules, "modal", fake)
    monkeypatch.setattr(modal_image, "build_image", lambda: "unbuilt-image")
    initialize = Mock(
        side_effect=WorkerInitializationError(
            "Raven worker initialization failed; stage=dependency_source_validation; cause=RuntimeError"
        )
    )
    monkeypatch.setattr(worker, "WorkerSession", initialize)
    scope = runpy.run_path(str(Path(__file__).resolve().parents[1] / "providers/modal/app.py"))
    container = scope["PointPillarsWorker"]()
    container.initialize()
    container.initialize()
    initialize.assert_called_once()
    for _ in range(2):
        with pytest.raises(WorkerInitializationError, match="dependency_source_validation"):
            container.infer({})
    container.shutdown()
    assert not hasattr(container, "session")


@pytest.mark.parametrize(
    "error,category,cancel",
    [
        (TimeoutError("private token"), "client_result_timeout", True),
        (
            WorkerInitializationError(
                "Raven worker initialization failed; stage=dependency_source_validation; cause=RuntimeError"
            ),
            "worker_startup_failure",
            False,
        ),
    ],
)
def test_transport_distinguishes_startup_from_result_timeout_without_resubmitting(
    monkeypatch, cloud, error, category, cancel
):
    monkeypatch.setenv("RAVEN_ALLOW_MODAL_GPU", "1")
    monkeypatch.setenv("RAVEN_MODAL_ASSET_RIGHTS_CONFIRMED", "1")
    call = Mock(object_id="fc-audit")
    call.get.side_effect = error
    spawn = Mock(return_value=call)
    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(
            Cls=SimpleNamespace(
                from_name=lambda *a, **kw: (
                    lambda: SimpleNamespace(infer=SimpleNamespace(spawn=spawn))
                )
            )
        ),
    )
    from raven.experiments.config import ModalConfig

    with pytest.raises(RemoteExecutionError, match=category) as failure:
        ModalTransport(ModalConfig("raven-pointpillars", "main")).submit(make_request(cloud))
    assert "private token" not in str(failure.value)
    assert "cancellation_confirmed=False" in str(failure.value)
    spawn.assert_called_once()
    if cancel:
        call.cancel.assert_called_once_with(terminate_containers=True)
    else:
        call.cancel.assert_not_called()


@pytest.mark.parametrize("failure", [False, True])
def test_adapter_imports_upstream_in_its_git_root_and_restores_cwd(
    gpu_backend, monkeypatch, failure
):
    import builtins

    from raven.perception.detection.pointpillars import PointPillarsDetector

    original = builtins.__import__
    imported = []
    previous = Path.cwd()

    def load(name, *args, **kwargs):
        if name == "pcdet":
            imported.append(Path.cwd())
            assert Path.cwd() == Path(gpu_backend.options["openpcdet_root"])
            if failure:
                raise ModuleNotFoundError("synthetic dependency failure")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", load)
    if failure:
        with pytest.raises(RuntimeError, match="External GPU dependencies"):
            PointPillarsDetector(**gpu_backend.options).initialize()
    else:
        PointPillarsDetector(**gpu_backend.options).initialize()
    assert imported and Path.cwd() == previous


@pytest.mark.parametrize("failure", ["dependency", "safe_loading"])
def test_worker_validation_precedes_weights_and_never_enables_unsafe_fallback(monkeypatch, failure):
    from raven.inference import worker
    from raven.perception.detection import pointpillars

    monkeypatch.setattr(worker.platform, "python_version", lambda: "3.10.20")
    monkeypatch.setattr(worker.importlib.metadata, "version", lambda name: contracts.PACKAGES[name])
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            version=SimpleNamespace(cuda="11.8"),
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_capability=lambda i: (7, 5),
                get_device_name=lambda i: "Tesla T4",
            ),
        ),
    )
    initialize = Mock()
    factory = Mock(return_value=initialize)
    monkeypatch.setattr(pointpillars, "PointPillarsDetector", factory)
    monkeypatch.delenv("RAVEN_TRUST_CHECKPOINT", raising=False)
    validate = Mock()
    if failure == "dependency":
        validate.side_effect = ModuleNotFoundError("torchvision")
    else:
        initialize.initialize.side_effect = RuntimeError(
            "Checkpoint loading failed; safe loader refused legacy payload"
        )
    monkeypatch.setattr(worker, "validate_cpu_environment", validate)
    with pytest.raises(
        WorkerInitializationError, match="dependency_source_validation|safe_checkpoint_loading"
    ):
        worker.WorkerSession()
    if failure == "dependency":
        factory.assert_not_called()
    else:
        assert factory.call_args.kwargs["trust_checkpoint"] is False
        initialize.initialize.assert_called_once()
        initialize.close.assert_called_once()


def test_adapter_records_actual_cu118_packages_without_dropping_legacy_fields(
    gpu_backend, monkeypatch
):
    from raven.perception.detection import _openpcdet
    from raven.perception.detection.pointpillars import PointPillarsDetector

    versions = {"spconv-cu118": "2.3.6", "cumm-cu118": "0.4.11"}
    monkeypatch.setattr(_openpcdet, "package_version", versions.get)
    detector = PointPillarsDetector(**gpu_backend.options)
    detector.initialize()
    environment = detector.provenance["environment"]
    assert environment["spconv_cu118"] == "2.3.6"
    assert environment["cumm_cu118"] == "0.4.11"
    assert environment["spconv_cu126"] is None and environment["cumm_cu126"] is None
