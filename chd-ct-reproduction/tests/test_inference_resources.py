"""Resource budgets, phase boundaries and real CPU workers on synthetic data."""

from pathlib import Path

import numpy as np
import pytest
import test_chd_main

from chd_ct import cpu_runtime
from chd_ct.imagechd import predict, preprocess
from chd_ct.recovery import file_hash, read_json, write_json
from chd_ct.server.inference_phases import gpu_ready, operations

ROOT = Path(__file__).resolve().parents[1]


def test_budget_accounts_for_load_memory_and_all_concurrent_tasks():
    hardware = dict(cpus=256, idle_fraction=0.97, memory_available=200 * 2**30)
    selected = cpu_runtime.choose(hardware, cases=100, case_bytes=30 * 2**30, parallel=True)
    assert selected["processes"] == 3 and selected["threads"] == 1
    assert cpu_runtime.choose(dict(hardware, cpus=2), concurrency=3)["threads"] == 1
    assert cpu_runtime.choose(dict(hardware, idle_fraction=0.01), cases=100, parallel=True)["processes"] == 2
    assert cpu_runtime.choose(dict(hardware, memory_available=0), cases=100, parallel=True)["processes"] == 1


def test_training_environment_is_not_changed(tmp_path, monkeypatch):
    monkeypatch.setattr(cpu_runtime, "inspect", lambda: pytest.fail("training entered new CPU tuning"))
    env = {"OMP_NUM_THREADS": "2", "sentinel": "unchanged"}
    assert cpu_runtime.configure_env("train", tmp_path, env, 3) is env
    assert not list(tmp_path.iterdir())


def test_postprocess_estimate_uses_largest_restoration_grid():
    rows = [dict(cache_shape=[16] * 3, canonical_shape=[512] * 3, native_shape=[512] * 3)]
    estimated = cpu_runtime.volume_bytes(rows, 48, native=True)
    assert estimated >= 512**3 * 48
    selected = cpu_runtime.choose(
        dict(cpus=32, idle_fraction=1, memory_available=4 * 2**30),
        cases=20,
        case_bytes=estimated,
        parallel=True,
    )
    assert selected["processes"] == 1


@pytest.mark.parametrize("auto", ["0", "1"])
def test_preprocess_limit_does_not_inspect_unused_corrupt_case(tmp_path, monkeypatch, auto):
    raw = test_chd_main.raw_chd.__wrapped__(tmp_path)
    (raw / "ct_3_image.nii.gz").write_bytes(b"not a nifti")
    monkeypatch.setenv("CHD_CPU_AUTO", auto)
    report = preprocess.prepare(raw, tmp_path / "cache", size=16, limit=3, no_resample=True)
    assert len(report["cases"]) == 3


def test_test_workers_receive_outer_output_lock(tmp_path, monkeypatch):
    import os

    from chd_ct.imagechd import test

    seen = []
    monkeypatch.setattr(test, "read_prepared", lambda _: (None, {}))

    def capture(*args, **kwargs):
        fds = kwargs["inherited_lock_fds"]
        assert len(fds) == 1
        os.fstat(fds[0])
        seen.append(fds)
        return {"split": "test", "cases_evaluated": 1}

    monkeypatch.setattr(test, "predict", capture)
    monkeypatch.setattr(test, "evaluate", capture)
    test.run_test("cache", "models", tmp_path / "test")
    assert len(seen) == 2 and seen[0] == seen[1]


@pytest.mark.parametrize("mode", ["full", "simple", "none"])
@pytest.mark.parametrize("prior_status", ["pending_cpu", "failed"])
def test_interactive_cpu_tail_resume_does_not_query_gpu(tmp_path, monkeypatch, mode, prior_status):
    from chd_ct.server import inference_phases, resume
    from chd_ct.server.launcher import make_parser

    step = dict(
        name="predict", output=str(tmp_path / "predict"), arguments=dict(task="predict", device="cuda")
    )
    write_json(
        tmp_path / "plan.json",
        dict(schema_version=1, directory=str(tmp_path), runtime=str(tmp_path / "runtime"), steps=[step]),
    )
    write_json(
        tmp_path / "job_state.json", dict(status="interrupted", steps=[dict(step, status=prior_status)])
    )
    if prior_status == "failed":
        (Path(step["output"]) / "predict").mkdir(parents=True)
        write_json(Path(step["output"]) / "predict/gpu-phase-ready.json", {})
    monkeypatch.setattr(inference_phases, "gpu_ready", lambda _: pytest.fail("hidden full cache read"))
    monkeypatch.setattr(resume, "show_gpus", lambda: pytest.fail("CPU-only resume queried GPU"))
    monkeypatch.setattr("builtins.input", lambda _: "")  # Accept the default simple check.
    args = make_parser().parse_args(["--resume-job", str(tmp_path), "--resume-check", mode])
    plan = resume.build_resume_plan(args, {}, True)
    assert plan["steps"][0]["arguments"]["resume"]


def test_legacy_cap_is_normalized_before_holder_creation():
    from chd_ct.server.inference_phases import normalize_caps

    plan = dict(
        steps=[
            dict(reuse={}, arguments=dict(task="train", device="cuda", gpu_count=3)),
            dict(arguments=dict(task="predict", device="cuda")),
        ]
    )
    normalize_caps(plan)
    assert plan["steps"][1]["arguments"]["gpu_count"] == 3


def test_training_log_identifies_waiting_gpu(tmp_path, capsys):
    from chd_ct.server.log_progress import show_progress

    step = tmp_path / "step"
    train = step / "train"
    train.mkdir(parents=True)
    write_json(
        tmp_path / "job_state.json", dict(steps=[dict(name="train", output=str(step), status="running")])
    )
    write_json(step / "status.json", dict(task="train", phase="train", phase_started_at=1))
    write_json(train / "queue.json", dict(status="waiting_gpu", sampled_at=2))
    show_progress(tmp_path)
    assert "等待 GPU" in capsys.readouterr().out


def test_parallel_preprocess_matches_serial_and_preserves_complete_cache(tmp_path, monkeypatch):
    raw = test_chd_main.raw_chd.__wrapped__(tmp_path)
    expected = preprocess.prepare(raw, tmp_path / "serial", size=16)
    monkeypatch.setenv("CHD_CPU_AUTO", "1")
    monkeypatch.setenv("CHD_CPU_REPORT_DIR", str(tmp_path / "status"))
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    monkeypatch.setattr(
        cpu_runtime, "inspect", lambda: dict(cpus=8, idle_fraction=1, memory_available=8 * 2**30)
    )
    actual = preprocess.prepare(raw, tmp_path / "parallel", size=16)
    assert actual == expected
    for row in actual["cases"]:
        with (
            np.load(tmp_path / "serial" / row["cache"]) as a,
            np.load(tmp_path / "parallel" / row["cache"]) as b,
        ):
            assert all(np.array_equal(a[k], b[k]) for k in a.files)
    saved = file_hash(tmp_path / "parallel/dataset.json")
    preprocess.prepare(raw, tmp_path / "parallel", size=16, resume=True)
    assert file_hash(tmp_path / "parallel/dataset.json") == saved
    progress = read_json(tmp_path / "status/progress.json")
    assert progress["completed"] == progress["total"] == 4


def test_gpu_phase_then_cpu_resume_without_inference(tmp_path, monkeypatch):
    raw = test_chd_main.raw_chd.__wrapped__(tmp_path)
    cache = tmp_path / "cache"
    preprocess.prepare(raw, cache, size=16)
    models = tmp_path / "models"
    models.mkdir()
    write_json(models / "models.json", {"fixture": True})
    collection = dict(mode="smoke", prepared_sha256="fixture", config={"threads": 1})
    monkeypatch.setattr(predict, "read_collection", lambda *a: (models, collection))
    monkeypatch.setattr(predict, "predict_volume", lambda image, *a: (np.zeros(image.shape, np.uint8), {}))
    output = tmp_path / "step/predict"
    result = predict.predict(cache, models, output, phase="gpu", allow_smoke=True)
    assert result["status"] == "pending_cpu" and not list(output.glob("*_seg.nii.gz"))
    step = dict(
        output=str(output.parent),
        arguments=dict(task="predict", device="cuda", prepared=str(cache), models=str(models), resume=True),
    )
    assert gpu_ready(step)
    assert operations(dict(steps=[step])) == [(0, "cpu")]
    monkeypatch.setattr(predict, "predict_volume", lambda *a: pytest.fail("CPU resume used GPU inference"))
    result = predict.predict(cache, models, output, phase="cpu", resume=True, allow_smoke=True)
    assert result["status"] == "passed" and len(result["cases"]) == 4
    (output / ".gpu-results/ct_0.npy").write_bytes(b"damaged")
    assert not gpu_ready(step)


def test_phase_order_and_gpu_cap_for_test_and_predict(tmp_path):
    from chd_ct.quickstart.commands import build_command
    from chd_ct.server.launcher import make_parser
    from chd_ct.server.settings import task_configuration

    steps = []
    for task in ("preprocess", "train", "test", "predict", "evaluate", "demo"):
        steps.append(
            dict(
                output=str(tmp_path / task),
                arguments=dict(task=task, device="cpu" if task in {"preprocess", "evaluate"} else "cuda"),
            )
        )
    assert operations(dict(steps=steps)) == [
        (0, "all"),
        (1, "all"),
        (2, "gpu"),
        (3, "gpu"),
        (5, "all"),
        (2, "cpu"),
        (3, "cpu"),
        (4, "all"),
    ]
    for task in ("test", "predict"):
        args = make_parser().parse_args(
            [
                "--task",
                task,
                "--gpu-count",
                "3",
                "--device",
                "cuda",
                "--prepared",
                "cache",
                "--models",
                "models",
            ]
        )
        assert task_configuration(args)["gpu_count"] == 3
        args.gpu_ids, args.gpu_limit, args.inference_phase = ["GPU-a", "GPU-b"], 3, "gpu"
        command = build_command(args, ROOT, "python", tmp_path, "cuda")
        assert command[command.index("--gpu-limit") + 1] == "3"
        assert command[command.index("--phase") + 1] == "gpu"


def test_log_progress_eta_is_readonly_and_waits_are_unknown(tmp_path, capsys):
    from chd_ct.server.log_progress import show_progress

    step = tmp_path / "step"
    step.mkdir()
    write_json(
        tmp_path / "job_state.json", dict(steps=[dict(name="predict", output=str(step), status="running")])
    )
    write_json(
        step / "progress.json",
        dict(
            phase="gpu_inference",
            completed=2,
            total=4,
            elapsed_seconds=10,
            eta_seconds=10,
            status="running",
            sampled_at=1,
        ),
    )
    before = file_hash(step / "progress.json")
    show_progress(tmp_path)
    assert "50.0%" in capsys.readouterr().out
    assert file_hash(step / "progress.json") == before
    record = read_json(step / "progress.json")
    write_json(step / "progress.json", dict(record, status="waiting_gpu"))
    show_progress(tmp_path)
    assert "暂无法估算" in capsys.readouterr().out
