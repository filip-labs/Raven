"""Export correctness tests use synthetic source fixtures, not new performance claims."""

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_benchmark import Clock
from test_kitti_evaluation import completed as completed  # noqa: PLC0414 -- Imported pytest fixture.

from raven.benchmark.experiment import benchmark_experiment, digest
from raven.benchmark.spec import BenchmarkSpec
from raven.cli import main
from raven.evaluation.evaluate import evaluate_experiment, save_evaluation_result
from raven.exporting import export as publisher
from raven.exporting.csv_tables import (
    BENCHMARK_COLUMNS,
    EVALUATION_COLUMNS,
    RAW_COLUMNS,
    SUMMARY_COLUMNS,
    protect_text,
    restore_text,
)
from raven.exporting.evidence import Snapshot, load_evidence
from raven.exporting.export import ExportWriteError, export_results


@pytest.fixture
def sources(completed, tmp_path):
    run, config = completed
    evaluation = tmp_path / "evaluation.json"
    save_evaluation_result(evaluation, evaluate_experiment(run))
    result = benchmark_experiment(
        config,
        specification=BenchmarkSpec(warmup=2, repetitions=3),
        benchmark_id="synthetic-benchmark",
        _clock=Clock(),
    )
    assert result.status == "success"
    return {
        "run": run,
        "evaluation": evaluation,
        "benchmark": Path(result.output_directory) / "manifest.json",
        "config": config,
    }


def json_source(path):
    return json.loads(Path(path).read_text())


def rows(directory, name):
    with (directory / name).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def tamper_benchmark(path, mutate):
    manifest = json_source(path)
    result_path = path.parent / manifest["artifacts"]["benchmark.json"]["file"]
    value = json_source(result_path)
    mutate(value)
    result_path.write_text(json.dumps(value))
    manifest["artifacts"]["benchmark.json"]["sha256"] = hashlib.sha256(
        result_path.read_bytes()
    ).hexdigest()
    path.write_text(json.dumps(manifest))


def test_all_sources_export_versioned_json_csv_without_execution_or_mutation(
    sources, tmp_path, monkeypatch
):
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError("export reran a computation")

    monkeypatch.setattr("raven.experiments.runner.run_experiment", forbidden)
    monkeypatch.setattr("raven.evaluation.evaluate.evaluate_experiment", forbidden)
    monkeypatch.setattr("raven.benchmark.experiment.benchmark_experiment", forbidden)
    monkeypatch.setattr(
        "raven.perception.preprocessing.pipeline.run_preprocessing_pipeline", forbidden
    )
    output = tmp_path / "export"
    summary = export_results(
        [sources["run"], sources["evaluation"], sources["benchmark"]],
        output,
        format="all",
        raw_timings=True,
    )
    assert summary["status"] == "success" and summary["source_count"] == 3
    doc = json_source(output / "results.json")
    assert doc["export_schema_version"] == 1
    content = doc["content"]
    assert (
        doc["content_sha256"]
        == hashlib.sha256(
            json.dumps(
                content, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            ).encode()
        ).hexdigest()
    )
    assert [r["source_type"] for r in content["sources"]] == [
        "experiment",
        "evaluation",
        "benchmark",
    ]
    assert [r["evidence_kind"] for r in content["sources"]] == [
        "saved_predictions",
        "saved_predictions",
        "synthetic_test",
    ]
    assert content["sources"][1]["payload"]["purpose"] == "validation_subset"
    assert content["sources"][2]["configuration"] == sources["config"].to_dict()
    assert content["sources"][0]["unavailable"]["benchmark"] == "not provided"
    manifest = json_source(output / "manifest.json")
    assert manifest["status"] == "success"
    for entry in manifest["artifacts"].values():
        raw = (output / entry["file"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"] and len(raw) == entry["bytes"]
    for path, raw in before.items():
        assert path.read_bytes() == raw


def test_stable_csv_headers_precise_values_nulls_warmups_and_synthetic_label(sources, tmp_path):
    output = tmp_path / "export"
    export_results(
        [sources["evaluation"], sources["benchmark"]], output, format="csv", raw_timings=True
    )
    assert (output / "metadata.json").is_file() and not (output / "results.json").exists()
    header, summary = rows(output, "experiment_summary.csv")
    assert header == list(SUMMARY_COLUMNS) and len(summary) == 2
    header, metrics = rows(output, "evaluation_metrics.csv")
    assert header == list(EVALUATION_COLUMNS)
    assert len(metrics) == 18
    source = json_source(sources["evaluation"])
    for row, metric in zip(metrics, source["metrics"], strict=True):
        assert row["metric"] == metric["metric"]
        assert float(row["iou_threshold"]) == metric["iou_threshold"]
        if metric["ap_r40_percent"] is None:
            assert (
                row["ap_r40_percent"] == ""
                and row["metric_availability"] == "no_eligible_ground_truth"
            )
        else:
            assert float(row["ap_r40_percent"]) == metric["ap_r40_percent"]
    header, bench = rows(output, "benchmark_summaries.csv")
    assert header == list(BENCHMARK_COLUMNS)
    assert len(bench) == 2 and all(
        r["timing_unit"] == "ms" and r["evidence_kind"] == "synthetic_test" for r in bench
    )
    assert all(r["warmup"] == "2" and r["repetitions"] == r["count"] == "3" for r in bench)
    header, raw = rows(output, "raw_timing_samples.csv")
    assert header == list(RAW_COLUMNS)
    assert len(raw) == 10 and sum(r["phase"] == "warmup" for r in raw) == 4
    for row in bench:
        samples = [
            int(r["duration"])
            for r in raw
            if r["phase"] == "measured" and r["stage"] == row["stage"]
        ]
        assert len(samples) == 3 and sum(samples) / 3 / 1_000_000 == float(row["mean_ms"])
        assert (
            row["benchmark_status"] == "success"
            and row["percentile_method"] == "linear"
            and row["std_ddof"] == "0"
        )


@pytest.mark.parametrize(
    "text",
    [
        "＝1+1",
        "＋1",
        "－1",
        "＠SUM(A1)",
        "=1+1",
        "+SUM(A1)",
        "-1+2",
        "@SUM(A1)",
        "\t=1",
        "\r=1",
        "\n=1",
        "  =1",
        "'already quoted",
        'comma,quote"\nUnicode čćž',
        "normal",
        " leading space",
    ],
)
def test_csv_formula_protection_is_reversible_without_changing_json(text):
    protected = protect_text(text)
    assert restore_text(protected) == text
    if text.lstrip().startswith(("=", "+", "-", "@", "＝", "＋", "－", "＠")) or text.startswith(
        ("'", "\t", "\r", "\n")
    ):
        assert protected.startswith("'")


def test_csv_escaping_and_numeric_precision_in_actual_source_projection(sources, tmp_path):
    from dataclasses import replace

    from raven.exporting.csv_tables import tables

    record = load_evidence(sources["evaluation"], Snapshot())
    name = '=SUM(A1), "čć"\nnext row'
    changed = replace(record, experiment_name=name)
    data = tables([changed])["experiment_summary.csv"]
    parsed = next(csv.DictReader(data.splitlines(keepends=True)))
    assert (
        parsed["experiment_name"].startswith("'=")
        and restore_text(parsed["experiment_name"]) == name
    )
    assert changed.to_dict()["experiment_name"] == name
    # Preserve the source binary64 value; do not round the stored result for presentation.
    precise = 0.12345678901234566
    record.payload["metrics"][0]["ap_r40_percent"] = precise
    parsed = next(csv.DictReader(tables([record])["evaluation_metrics.csv"].splitlines()))
    assert float(parsed["ap_r40_percent"]) == precise


def test_deterministic_content_and_csv_exclude_generation_metadata(sources, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    export_results([sources["benchmark"]], a, format="all", raw_timings=True)
    export_results([sources["benchmark"]], b, format="all", raw_timings=True)
    first, second = json_source(a / "results.json"), json_source(b / "results.json")
    assert (
        first["content"] == second["content"]
        and first["content_sha256"] == second["content_sha256"]
    )
    assert first["generation"]["generated_at_utc"] != second["generation"]["generated_at_utc"]
    for name in (
        "experiment_summary.csv",
        "evaluation_metrics.csv",
        "benchmark_summaries.csv",
        "raw_timing_samples.csv",
    ):
        assert (a / name).read_bytes() == (b / name).read_bytes()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda v: v.update(benchmark_schema_version=2),
        lambda v: v["specification"].update(warmup=100),
        lambda v: v.update(specification_sha256="0" * 64),
        lambda v: v["frames"][0]["observations"][0].update(phase="measured"),
        lambda v: v["frames"][0]["series"][0]["summary"].update(mean_ms=999.0),
        lambda v: v["frames"][0]["series"][0]["raw_ns"].append(10),
        lambda v: v["frames"][0]["series"][0]["summary"].update(mean_ms=None),
        lambda v: v["frames"][0]["observations"][0]["durations_ns"].update(preprocessing=-1),
        lambda v: v["frames"][0].update(frame_id="000002"),
        lambda v: v["experiment"]["inputs"][0].update(sha256="bad"),
        lambda v: v["frames"][0]["observations"][0].update(processed_xyzi_sha256="bad"),
        lambda v: v.update(evidence_kind="local_cpu"),
        lambda v: v.update(frames=[]),
        lambda v: v["methodology"].update(summary_unit="s"),
        lambda v: v["frames"][0]["series"].append(v["frames"][0]["series"][0]),
    ],
)
def test_corrupt_benchmark_rejected_even_after_manifest_hashes_updated(sources, tmp_path, mutation):
    tamper_benchmark(sources["benchmark"], mutation)
    output = tmp_path / "export"
    with pytest.raises((ValueError, TypeError)):
        export_results([sources["benchmark"]], output)
    assert not output.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(evaluation_schema_version=2),
        lambda d: d.update(run_manifest_sha256="0" * 64),
        lambda d: d.update(config_sha256="0" * 64),
        lambda d: d.update(purpose="training_smoke"),
        lambda d: d["metrics"][0].update(ap_r40_percent=101.0),
        lambda d: d["metrics"][0].update(iou_threshold=0.1),
        lambda d: d["metrics"][0].update(eligible_ground_truth=True),
        lambda d: d["metrics"][0].update(ap_r40_percent=None),
        lambda d: d["metrics"].append(d["metrics"][0]),
        lambda d: next(m for m in d["metrics"] if m["eligible_ground_truth"] == 0).update(
            ap_r40_percent=0
        ),
        lambda d: d.update(diagnostics=[]),
        lambda d: d.update(evaluated_frame_count=True),
    ],
)
def test_corrupt_evaluation_never_exports_measured_metrics(sources, tmp_path, mutation):
    value = json_source(sources["evaluation"])
    mutation(value)
    sources["evaluation"].write_text(json.dumps(value))
    with pytest.raises(ValueError):
        export_results([sources["evaluation"]], tmp_path / "export")
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize("problem", ["checksum", "missing", "incomplete", "unsafe", "symlink"])
def test_run_manifest_corruption_and_missing_evidence_are_rejected(sources, tmp_path, problem):
    path = sources["run"]
    doc = json_source(path)
    artifact = path.parent / doc["artifacts"]["results"]["file"]
    if problem == "checksum":
        artifact.write_text("{}")
    if problem == "missing":
        artifact.unlink()
    if problem == "incomplete":
        doc["status"] = "started"
        path.write_text(json.dumps(doc))
    if problem == "unsafe":
        doc["artifacts"]["results"]["file"] = "../escape.json"
        path.write_text(json.dumps(doc))
    if problem == "symlink":
        raw = artifact.read_bytes()
        artifact.unlink()
        target = tmp_path / "elsewhere.json"
        target.write_bytes(raw)
        artifact.symlink_to(target)
    with pytest.raises(ValueError):
        export_results([path], tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_blocked_invalid_evaluation_remains_explicit_without_fake_metrics(sources, tmp_path):
    from dataclasses import replace

    result = evaluate_experiment(sources["run"])
    for status in ("blocked", "invalid"):
        outcome = replace(
            result,
            status=status,
            metrics=(),
            diagnostics=(),
            failures=("required metadata unavailable",),
        )
        path = tmp_path / f"{status}.json"
        save_evaluation_result(path, outcome)
        dest = tmp_path / f"{status}-export"
        export_results([path], dest, format="all")
        record = json_source(dest / "results.json")["content"]["sources"][0]
        assert record["status"] == status and record["payload"]["metrics"] == []
        assert record["payload"]["evaluated_frame_count"] == 0
        assert rows(dest, "evaluation_metrics.csv")[1] == []


def test_failed_completed_benchmark_exports_diagnostics_without_steady_state_summary(
    sources, tmp_path
):
    def fail(value):
        value["status"] = "failed"
        value["failures"] = [{"phase": "measurement", "message": "synthetic test failure"}]
        frame = value["frames"][0]
        frame["status"] = "failed"
        frame["series"] = []
        frame["observations"] = frame["observations"][:1] + [
            {
                "iteration": 1,
                "phase": "warmup",
                "status": "failed",
                "durations_ns": {},
                "operations": [],
                "failure": "failure",
            }
        ]

    tamper_benchmark(sources["benchmark"], fail)
    manifest = json_source(sources["benchmark"])
    manifest["status"] = "failed"
    sources["benchmark"].write_text(json.dumps(manifest))
    export_results([sources["benchmark"]], tmp_path / "export", format="all", raw_timings=True)
    assert rows(tmp_path / "export", "benchmark_summaries.csv")[1] == []
    assert all(
        r["benchmark_status"] == "failed"
        for r in rows(tmp_path / "export", "raw_timing_samples.csv")[1]
    )


def test_comparison_exposes_differences_and_never_assumes_comparability(sources, tmp_path):
    export_results([sources["evaluation"], sources["benchmark"]], tmp_path / "export")
    comparison = json_source(tmp_path / "export/results.json")["content"]["comparison"]
    pair = comparison["pairs"][0]
    assert pair["comparability"] == "not_assumed"
    assert (
        "source_type" in pair["different_dimensions"]
        and "evidence_kind" in pair["different_dimensions"]
    )
    assert "timing_scope" in pair["missing_dimensions"]
    assert "improvement" not in pair


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        "[]",
        "NaN",
        '{"evaluation_schema_version":1,"evaluation_schema_version":1}',
        '{"run_schema_version":1,"benchmark_schema_version":1}',
    ],
)
def test_malformed_and_ambiguous_json_rejected(tmp_path, raw):
    path = tmp_path / "input"
    path.write_text(raw)
    with pytest.raises(ValueError):
        export_results([path], tmp_path / "export")


def test_no_filename_guessing_and_duplicate_sources_rejected(sources, tmp_path):
    renamed = tmp_path / "not-an-evaluation.txt"
    renamed.write_bytes(sources["evaluation"].read_bytes())
    export_results([renamed], tmp_path / "export")
    with pytest.raises(ValueError, match="duplicate"):
        export_results([sources["run"], sources["run"]], tmp_path / "duplicate")


def test_collision_traversal_source_directory_and_symlink_outputs(sources, tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep").write_text("unchanged")
    for dest in (existing, tmp_path / "child/../traversal", sources["run"].parent / "export"):
        with pytest.raises(ValueError):
            export_results([sources["run"]], dest)
    assert (existing / "keep").read_text() == "unchanged"
    link = tmp_path / "link"
    link.symlink_to(existing, target_is_directory=True)
    with pytest.raises(ValueError):
        export_results([sources["run"]], link)


def test_partial_publication_rolls_back_and_never_leaves_success_manifest(
    sources, tmp_path, monkeypatch
):
    original = publisher.write_text_exclusive

    def fail(path, text):
        original(path, text)
        if path.name == "manifest.json":
            raise OSError("injected after publication failure")

    monkeypatch.setattr(publisher, "write_text_exclusive", fail)
    with pytest.raises(ExportWriteError):
        export_results([sources["run"]], tmp_path / "export", format="all")
    assert not (tmp_path / "export").exists()


def test_source_changed_during_export_aborts_and_rolls_back(sources, tmp_path, monkeypatch):
    original = publisher.write_text_exclusive

    def change(path, text):
        original(path, text)
        if path.name == "results.json":
            sources["evaluation"].write_text("{}")

    monkeypatch.setattr(publisher, "write_text_exclusive", change)
    with pytest.raises(ExportWriteError):
        export_results([sources["evaluation"]], tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_cli_success_and_invalid_source_exit(sources, tmp_path, capsys):
    main(
        [
            "export-results",
            str(sources["benchmark"]),
            "--format",
            "csv",
            "--raw-timings",
            "--output",
            str(tmp_path / "export"),
        ]
    )
    assert json.loads(capsys.readouterr().out)["status"] == "success"
    with pytest.raises(SystemExit) as error:
        main(
            ["export-results", str(tmp_path / "missing"), "--output", str(tmp_path / "missing-out")]
        )
    assert error.value.code == 2


def test_export_subprocess_requires_no_gpu_modal_gui_or_scipy(sources, tmp_path):
    script = """
import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,name,*args):
        if name.split('.')[0] in {'torch','spconv','cumm','pcdet','modal','open3d','scipy','shapely'}:
            raise AssertionError('forbidden import '+name)
sys.meta_path.insert(0,Block())
from raven.exporting.export import export_results
assert export_results([sys.argv[1]],sys.argv[2],format='all')['status']=='success'
"""
    env = dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    subprocess.run(
        [sys.executable, "-c", script, str(sources["benchmark"]), str(tmp_path / "export")],
        env=env,
        check=True,
        capture_output=True,
    )


def test_training_smoke_cannot_be_presented_as_validation(sources, tmp_path):
    import shutil

    from raven.experiments.config import ExperimentConfig
    from raven.experiments.runner import run_experiment

    # Only isolated pytest fixture copies are writable; source artifacts remain read-only.
    fixture_config = sources["config"]
    isolated_dataset = tmp_path / "isolated-training-dataset"
    isolated_predictions = tmp_path / "isolated-training-predictions"
    shutil.copytree(fixture_config.dataset.root, isolated_dataset)
    shutil.copytree(fixture_config.model.predictions_dir, isolated_predictions)
    for folder, ext in [
        ("velodyne", "bin"),
        ("calib", "txt"),
        ("label_2", "txt"),
        ("image_2", "png"),
    ]:
        parent = isolated_dataset / "training" / folder
        with (parent / f"000000.{ext}").open("xb") as stream:
            stream.write((parent / f"000001.{ext}").read_bytes())
    capture = json_source(isolated_predictions / "000001.json")
    capture["input"]["frame_id"] = "000000"
    with (isolated_predictions / "000000.json").open("x") as stream:
        json.dump(capture, stream)
    document = fixture_config.to_dict()
    document["dataset"] = {"root": str(isolated_dataset), "split": "train"}
    document["model"]["predictions_dir"] = str(isolated_predictions)
    document["frames"]["ids"] = ["000000"]
    document["experiment_id"] = "training-smoke"
    document["output"]["directory"] = str(tmp_path / "isolated-training-runs")
    training = ExperimentConfig.from_dict(document, base_dir=tmp_path)
    assert run_experiment(training).status == "success"
    value = evaluate_experiment(
        training.run_directory / "result.manifest.json", allow_training_smoke=True
    )
    assert value.status == "success" and value.purpose == "training_smoke"
    path = tmp_path / "training.json"
    save_evaluation_result(path, value)
    export_results([path], tmp_path / "export", format="all")
    data = rows(tmp_path / "export", "evaluation_metrics.csv")[1]
    assert all(
        r["evaluation_purpose"] == "training_smoke" and r["dataset_split"] == "train" for r in data
    )


def test_valid_unavailable_outcome_without_source_run_has_null_identity(tmp_path):
    outcome = evaluate_experiment(tmp_path / "missing.manifest.json")
    assert outcome.status == "invalid"
    path = tmp_path / "invalid.json"
    save_evaluation_result(path, outcome)
    export_results([path], tmp_path / "export", format="all")
    value = json_source(tmp_path / "export/results.json")["content"]["sources"][0]
    assert (
        value["configuration"] is None and value["model"] is None and value["config_sha256"] is None
    )
    assert value["evidence_kind"] == "unavailable" and value["status"] == "invalid"
    assert rows(tmp_path / "export", "benchmark_summaries.csv")[1] == []


def test_failed_initialization_benchmark_remains_unavailable_without_zero_latency(
    sources, tmp_path
):
    def fail(v):
        v.update(
            status="failed",
            frames=[],
            initialization=None,
            model_provenance=None,
            evidence_kind="local_cuda",
            failures=[{"phase": "measurement", "message": "initialization failed"}],
        )
        v["specification"]["mode"] = "inference"
        v["specification_sha256"] = digest(v["specification"])
        v["environment"]["cuda_used"] = False
        v["environment"]["effective_clock"] = "time.perf_counter_ns"

    tamper_benchmark(sources["benchmark"], fail)
    manifest = json_source(sources["benchmark"])
    manifest["status"] = "failed"
    sources["benchmark"].write_text(json.dumps(manifest))
    export_results([sources["benchmark"]], tmp_path / "export", format="all")
    assert rows(tmp_path / "export", "benchmark_summaries.csv")[1] == []
    record = json_source(tmp_path / "export/results.json")["content"]["sources"][0]
    assert record["status"] == "failed" and record["payload"]["initialization"] is None


def test_changed_hardware_is_explicit_comparison_dimension(sources, tmp_path):
    import shutil

    first = sources["benchmark"]
    copied = tmp_path / "second"
    shutil.copytree(first.parent, copied)
    second = copied / "manifest.json"

    def change(value):
        value["environment"]["cpu_model"] = "Different CPU"
        value["environment"]["machine"] = "x86_64"

    tamper_benchmark(second, change)
    export_results([first, second], tmp_path / "export")
    pair = json_source(tmp_path / "export/results.json")["content"]["comparison"]["pairs"][0]
    assert "hardware" in pair["different_dimensions"] and pair["comparability"] == "not_assumed"


def test_cli_publication_failure_exit_is_one(sources, tmp_path, monkeypatch, capsys):
    def denied(*args):
        raise OSError("write denied")

    monkeypatch.setattr(publisher, "write_text_exclusive", denied)
    with pytest.raises(SystemExit) as error:
        main(["export-results", str(sources["run"]), "--output", str(tmp_path / "export")])
    assert error.value.code == 1 and "write denied" in capsys.readouterr().err
    assert not (tmp_path / "export").exists()


def test_standalone_blocked_benchmark_keeps_no_latency_rows(sources, tmp_path):
    result = benchmark_experiment(
        sources["config"],
        specification=BenchmarkSpec(mode="inference", warmup=0, repetitions=1),
        benchmark_id="blocked",
    )
    assert result.status == "blocked" and not result.frames
    path = tmp_path / "blocked-benchmark.json"
    path.write_text(json.dumps(result.to_dict()))
    export_results([path], tmp_path / "export", format="all", raw_timings=True)
    record = json_source(tmp_path / "export/results.json")["content"]["sources"][0]
    assert record["status"] == "blocked" and record["payload"]["initialization"] is None
    assert rows(tmp_path / "export", "benchmark_summaries.csv")[1] == []
    assert rows(tmp_path / "export", "raw_timing_samples.csv")[1] == []


def test_ignored_detection_diagnostics_need_not_account_for_every_eligible_gt(sources, tmp_path):
    # Pinned KITTI matching may consume eligible GT with an ignored DT, counting neither TP nor FN.
    value = json_source(sources["evaluation"])
    row = next(
        r
        for r in value["diagnostics"]
        if r["metric"] == "3d" and r["class_name"] == "Car" and r["difficulty"] == "easy"
    )
    row.update(tp=0, fn=0)
    sources["evaluation"].write_text(json.dumps(value))
    export_results([sources["evaluation"]], tmp_path / "export")
    actual = json_source(tmp_path / "export/results.json")["content"]["sources"][0]["payload"]
    assert actual["diagnostics"] == value["diagnostics"]


def test_float_raw_nanoseconds_are_rejected_even_when_numerically_equal(sources, tmp_path):
    def change(value):
        series = value["frames"][0]["series"][0]
        series["raw_ns"][0] = float(series["raw_ns"][0])

    tamper_benchmark(sources["benchmark"], change)
    with pytest.raises(ValueError, match="integer nanoseconds"):
        export_results([sources["benchmark"]], tmp_path / "export")
