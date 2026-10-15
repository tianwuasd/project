"""Fault injection at durable boundaries; equivalence to an uninterrupted run."""

from pathlib import Path

import numpy as np
import pytest
import test_chd_main
import torch

from chd_ct.imagechd import preprocess
from chd_ct.recovery import file_hash, output_lock, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def raw_chd(tmp_path):
    return test_chd_main.raw_chd.__wrapped__(tmp_path)


def test_legacy_partial_cache_keeps_valid_and_repairs_truncated(raw_chd, tmp_path):
    output = tmp_path / "cache"
    expected = preprocess.prepare(raw_chd, output, size=16)
    (output / "dataset.json").unlink()
    (output / "preprocess-contract.json").unlink()
    for path in (output / ".resume").glob("*.json"):
        path.unlink()
    initial = dict(expected, status="running", cases=[], excluded=[])
    write_json(output / "preprocess-report.json", initial)
    good = output / "ct_0.npz"
    stamp, digest = good.stat().st_mtime_ns, file_hash(good)
    (output / "ct_2.npz").write_bytes(b"PK truncated")
    (output / "ct_3.npz").unlink()
    recovered = preprocess.prepare(raw_chd, output, size=16, resume=True)
    assert recovered == expected
    assert good.stat().st_mtime_ns == stamp and file_hash(good) == digest
    assert len(list((output / ".resume").glob("*.json"))) == 4


def test_resume_skips_processing_verified_cases_and_preserves_manifest(raw_chd, tmp_path, monkeypatch):
    output = tmp_path / "cache"
    preprocess.prepare(raw_chd, output, size=16)
    manifest_hash = file_hash(output / "dataset.json")
    monkeypatch.setattr(preprocess, "prepare_case", lambda *a: pytest.fail("valid cache recomputed"))
    preprocess.prepare(raw_chd, output, size=16, resume=True)
    assert file_hash(output / "dataset.json") == manifest_hash
    with pytest.raises(FileExistsError):
        preprocess.prepare(raw_chd, output, size=16)
    with pytest.raises(ValueError, match="改变"):
        preprocess.prepare(raw_chd, output, size=32, resume=True)
    with (raw_chd / "ct_0_label.nii.gz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="改变"):
        preprocess.prepare(raw_chd, output, size=16, resume=True)
    assert file_hash(output / "dataset.json") == manifest_hash


def test_case_interrupt_records_progress_and_does_not_publish_manifest(raw_chd, tmp_path, monkeypatch):
    output = tmp_path / "cache"
    original = preprocess.prepare_case

    def interrupt(case, *args):
        if case == "ct_2":
            raise KeyboardInterrupt("simulated stop")
        return original(case, *args)

    monkeypatch.setattr(preprocess, "prepare_case", interrupt)
    with pytest.raises(KeyboardInterrupt):
        preprocess.prepare(raw_chd, output, size=16)
    assert not (output / "dataset.json").exists()
    report = read_json(output / "preprocess-report.json")
    assert report["status"] == "interrupted" and len(report["cases"]) == 2
    monkeypatch.setattr(preprocess, "prepare_case", original)
    assert len(preprocess.prepare(raw_chd, output, size=16, resume=True)["cases"]) == 4


def test_output_lock_blocks_concurrent_writer_and_releases(tmp_path):
    with output_lock(tmp_path):
        with pytest.raises(ValueError, match="正在写入"):
            with output_lock(tmp_path):
                pytest.fail("two writers")
    with output_lock(tmp_path):
        pass


def test_training_resume_matches_uninterrupted_optimizer_rng_and_best(raw_chd, tmp_path, monkeypatch):
    from chd_ct.imagechd import train as training
    from chd_ct.imagechd.config import load_config

    cache = tmp_path / "cache"
    data = preprocess.prepare(raw_chd, cache, size=16)
    config = load_config(ROOT / "configs/chd-smoke.yaml")
    config.update(device="cpu", augment=True)
    config["stages"]["crop64"]["epochs"] = 3
    config["stages"]["crop64"]["base"] = 2
    rows = training.training_rows(data)
    digest = file_hash(cache / "dataset.json")
    full, interrupted = tmp_path / "full", tmp_path / "interrupted"
    full.mkdir()
    interrupted.mkdir()
    training.train_stage(cache, rows, "crop64", config, full, "train", digest)
    original = training.write_json

    def stop_after_epoch(path, value):
        original(path, value)
        if str(path).endswith("history.json") and len(value) == 1:
            raise KeyboardInterrupt("simulated interruption after durable epoch")

    monkeypatch.setattr(training, "write_json", stop_after_epoch)
    with pytest.raises(KeyboardInterrupt):
        training.train_stage(cache, rows, "crop64", config, interrupted, "train", digest)
    monkeypatch.setattr(training, "write_json", original)
    training.train_stage(cache, rows, "crop64", config, interrupted, "train", digest, resume=True)
    expected = torch.load(full / "crop64.last.pt", weights_only=True)
    actual = torch.load(interrupted / "crop64.last.pt", weights_only=True)
    assert expected["history"] == actual["history"]
    assert torch.equal(expected["torch_rng"], actual["torch_rng"])
    assert all(torch.equal(v, actual["state_dict"][k]) for k, v in expected["state_dict"].items())
    assert np.array_equal(expected["numpy_rng"][1], actual["numpy_rng"][1])
    for key, value in expected["optimizer"]["state"].items():
        assert all(torch.equal(v, actual["optimizer"]["state"][key][k]) for k, v in value.items())
    best_hash = file_hash(interrupted / "crop64.pt")
    training.train_stage(cache, rows, "crop64", config, interrupted, "train", digest, resume=True)
    assert file_hash(interrupted / "crop64.pt") == best_hash
    config["seed"] += 1
    with pytest.raises(ValueError, match="来源不同"):
        training.train_stage(cache, rows, "crop64", config, interrupted, "train", digest, resume=True)


def test_prediction_resume_only_recomputes_missing_or_damaged_cases(raw_chd, tmp_path, monkeypatch):
    from chd_ct.imagechd import predict as inference

    cache = tmp_path / "cache"
    preprocess.prepare(raw_chd, cache, size=16)
    models = tmp_path / "models"
    models.mkdir()
    write_json(models / "models.json", {"fixture": True})
    collection = {"mode": "smoke", "prepared_sha256": "fixture", "config": {"threads": 1}}
    monkeypatch.setattr(inference, "read_collection", lambda *a: (models, collection))
    calls = []

    def predict(image, *args):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt()
        return np.zeros(image.shape, np.uint8), {}

    monkeypatch.setattr(inference, "predict_volume", predict)
    output = tmp_path / "predictions"
    with pytest.raises(KeyboardInterrupt):
        inference.predict(cache, models, output, allow_smoke=True)
    assert len(read_json(output / "prediction-report.json")["cases"]) == 2
    kept = file_hash(output / "ct_0_seg.nii.gz")
    (output / "ct_1_grid.npz").write_bytes(b"broken")
    result = inference.predict(cache, models, output, allow_smoke=True, resume=True)
    assert len(calls) == 6 and len(result["cases"]) == 4
    assert file_hash(output / "ct_0_seg.nii.gz") == kept
    original_hash = inference.file_hash

    def stop_during_validation(path):
        if str(path).endswith("ct_0_seg.nii.gz"):
            raise KeyboardInterrupt("second interruption while validating committed output")
        return original_hash(path)

    monkeypatch.setattr(inference, "file_hash", stop_during_validation)
    with pytest.raises(KeyboardInterrupt):
        inference.predict(cache, models, output, allow_smoke=True, resume=True)
    assert len(read_json(output / "prediction-report.json")["cases"]) == 4
    monkeypatch.setattr(inference, "file_hash", original_hash)
    inference.predict(cache, models, output, allow_smoke=True, resume=True)
    assert len(calls) == 6
    with pytest.raises(ValueError, match="改变"):
        inference.predict(cache, models, output, case_id="ct_0", allow_smoke=True, resume=True)


def test_resume_plan_uses_frozen_settings_and_preserves_completed_steps(tmp_path, monkeypatch):
    from chd_ct.server import jobs, launcher
    from chd_ct.server.resume import build_resume_plan

    runtime, folder = tmp_path / "runtime", tmp_path / "job"
    runtime.mkdir()
    folder.mkdir()
    plan = dict(
        schema_version=1,
        runtime=str(runtime),
        directory=str(folder),
        log=str(folder / "launcher.log"),
        steps=[
            dict(name="environment", output="env", arguments={"task": "environment", "device": "cpu"}),
            dict(
                name="preprocess",
                output="prep",
                arguments={"task": "preprocess", "prepared": "old-cache", "dataset": "old-data"},
            ),
            dict(name="diagnosis-demo", output="demo", arguments={"task": "diagnosis-demo"}),
        ],
    )
    state = jobs.initial_state(plan)
    state["status"] = "running"
    state["steps"][0]["status"] = "passed"
    write_json(folder / "plan.json", plan)
    write_json(folder / "job_state.json", state)
    write_json(folder / "receipt.json", dict(pid=999, identity={"boot": "old", "start": "1"}))
    write_json(runtime / "last-job.json", dict(directory=str(folder)))
    monkeypatch.setattr(jobs, "identity", lambda pid: None)
    args = launcher.make_parser().parse_args(["--resume-job", "--runtime", str(runtime)])
    result = build_resume_plan(args, {"runtime": str(runtime)}, False)
    assert result["steps"][0]["reuse"]["status"] == "passed"
    assert result["steps"][1]["arguments"]["resume"]
    assert result["steps"][1]["arguments"]["prepared"] == "old-cache"
    assert result["steps"][2]["restart"]


def test_gpu_status_precedes_choice_and_cpu_never_queries(tmp_path, monkeypatch, capsys):
    from test_server_jobs import arguments

    from chd_ct.server import launcher
    from chd_ct.server.settings import resolve_settings

    args, profile = arguments(tmp_path, "--task", "train", "--device", "cuda")
    rows = [
        dict(index="0", uuid="GPU-0", name="3090", free=23000, util=70, busy=True),
        dict(index="7", uuid="GPU-7", name="3090", free=24570, util=0, busy=False),
    ]
    monkeypatch.setattr(launcher, "query_gpus", lambda: rows)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    seen = []

    def answer(prompt):
        if "GPU" in prompt:
            if not seen:
                text = capsys.readouterr().out
                assert "当前显卡状态" in text and "范围外" in text and "范围内" in text
            seen.append(prompt)
        return ""

    monkeypatch.setattr("builtins.input", answer)
    resolve_settings(args, profile, True)
    assert len(seen) == 2


def test_six_stage_resume_keeps_encoder_hash_and_rejects_config_without_mutation(raw_chd, tmp_path):
    import yaml

    from chd_ct.imagechd.checkpoints import read_collection
    from chd_ct.imagechd.train import train

    cache, models = tmp_path / "cache", tmp_path / "models"
    preprocess.prepare(raw_chd, cache, size=16)
    config = ROOT / "configs/chd-smoke.yaml"
    train(cache, models, config, mode="smoke")
    before = file_hash(models / "blood2d.pt")
    train(cache, models, config, mode="smoke", resume=True)
    assert read_collection(models, allow_smoke=True)[1]["status"] == "complete"
    assert file_hash(models / "blood2d.pt") == before
    saved = {p: p.read_bytes() for p in (models / "models.json", models / "report.json")}
    changed = yaml.safe_load(config.read_text(encoding="utf-8"))
    changed["seed"] += 1
    bad = tmp_path / "config.yaml"
    bad.write_text(yaml.safe_dump(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="改变"):
        train(cache, models, bad, mode="smoke", resume=True)
    assert all(p.read_bytes() == value for p, value in saved.items())


def test_second_interrupt_keeps_reused_step_proofs():
    from chd_ct.server.jobs import initial_state

    prior = dict(
        name="diagnosis-labels", output="labels", status="passed", snapshot={"inputs": {}, "outputs": {}}
    )
    plan = dict(
        directory="job", log="log", steps=[dict(name="diagnosis-labels", output="labels", reuse=prior)]
    )
    state = initial_state(plan)
    assert state["steps"][0]["status"] == "reused"
    assert state["steps"][0]["snapshot"] == prior["snapshot"]


def test_snapshot_rejects_new_case_and_changed_diagnosis_labels(tmp_path):
    from chd_ct.server.resume import snapshot, verify_snapshot

    labels = tmp_path / "labels"
    labels.mkdir()
    write_json(labels / "labels.json", {"value": 1})
    step = dict(
        output=str(tmp_path / "step"), arguments=dict(task="diagnosis-train", diagnosis_labels=str(labels))
    )
    record = snapshot(step)
    write_json(labels / "labels.json", {"value": 2})
    with pytest.raises(ValueError, match="改变"):
        verify_snapshot(record)

    write_json(labels / "labels.json", {"value": 1})
    write_json(labels / "new.json", {})
    with pytest.raises(ValueError, match="改变"):
        verify_snapshot(record)


@pytest.mark.parametrize("stale_receipt", [False, True])
def test_stop_uses_current_state_when_receipt_missing_or_stale(tmp_path, monkeypatch, stale_receipt):
    import signal

    from chd_ct.server import jobs

    job = tmp_path / "job"
    job.mkdir()
    write_json(tmp_path / "last-job.json", {"directory": str(job)})
    identity = {"boot": "boot", "start": "1"}
    write_json(job / "job_state.json", {"pid": 5432, "identity": identity})
    if stale_receipt:
        write_json(job / "receipt.json", {"pid": 123, "identity": identity})
    monkeypatch.setattr(jobs, "identity", lambda pid: identity if pid == 5432 else None)
    signals = []
    monkeypatch.setattr(jobs.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    assert jobs.stop_job(tmp_path) == 0
    assert signals == [(5432, signal.SIGTERM)]
