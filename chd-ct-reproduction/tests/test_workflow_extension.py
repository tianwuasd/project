"""Completed steps remain byte-for-byte intact while a named workflow grows."""

import json
from pathlib import Path

import pytest

from chd_ct.server import extension, jobs, launcher, panel, results
from chd_ct.server.resume import build_resume_plan, snapshot


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def finished(tmp_path, monkeypatch):
    job, runtime, prepared = tmp_path / "results/job_workflow", tmp_path / "runtime", tmp_path / "cache"
    config = tmp_path / "model.yaml"
    config.write_text("seed: 42\n")
    save(prepared / "dataset.json", {})
    output = job / "steps/01_train"
    save(output / "train/models.json", {"mode": "train", "stages": {}})
    (output / "train/crop64.pt").write_bytes(b"trained checkpoint")
    (output / "train/crop64.last.pt").write_bytes(b"last checkpoint")
    profile = dict(runtime=str(runtime), results=str(job.parent), prepared=str(prepared), gpu_count=3)
    step = dict(
        name="train",
        output=str(output),
        arguments=dict(
            task="train",
            mode="train",
            prepared=str(prepared),
            config=str(config),
            device="cpu",
        ),
    )
    plan = dict(
        schema_version=1,
        name="已完成训练",
        runtime=str(runtime),
        directory=str(job),
        log=str(job / "launcher.log"),
        profile=profile,
        steps=[step],
    )
    save(job / "plan.json", plan)
    state = jobs.initial_state(plan)
    state.update(status="passed")
    state["steps"][0].update(status="passed", snapshot=snapshot(step))
    save(job / "job_state.json", state)
    save(runtime / "last-job.json", {"directory": str(job)})
    save(
        runtime / ".server-settings.json",
        {
            "prepared": str(tmp_path / "another_cache"),
            "last_models": "wrong-models",
            "last_predictions": "wrong-predictions",
        },
    )
    monkeypatch.setattr(jobs, "identity", lambda _: None)
    return job, runtime, plan, profile


def arguments(job, *extra):
    return launcher.make_parser().parse_args(
        ["--resume-job", str(job), "--resume-check", "full", "--device", "cpu", *extra]
    )


def test_append_uses_original_data_and_links_new_outputs(finished):
    job, runtime, old, profile = finished
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in job.rglob("*") if p.is_file()}
    plan = build_resume_plan(arguments(job, "--append-tasks", "test,predict,evaluate"), profile, False)
    assert plan["name"] == old["name"] and plan["directory"] == str(job)
    train, test, predict, evaluate = plan["steps"]
    assert train["reuse"]["status"] == "passed"
    assert (
        test["arguments"]["models"] == predict["arguments"]["models"] == str(Path(train["output"]) / "train")
    )
    assert all(s["arguments"]["prepared"] == old["steps"][0]["arguments"]["prepared"] for s in plan["steps"])
    assert evaluate["arguments"]["predictions"] == str(Path(predict["output"]) / "predict")
    assert Path(test["output"]).name == "02_test"
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in job.rglob("*") if p.is_file()}


def test_simple_append_skips_outputs_that_will_be_produced(finished):
    job, _, _, profile = finished
    plan = build_resume_plan(
        arguments(job, "--resume-check", "simple", "--append-tasks", "test,predict,evaluate"), profile, False
    )
    check = plan["resume_verification"]
    assert check["mode"] == "simple" and check["files"] == 4
    assert not any("evaluate/predictions" in dep for dep in check["checked_dependencies"])
    assert plan["steps"][0]["reuse"]


def test_append_does_not_replace_frozen_config_digest(finished):
    from chd_ct.server.resume import file_hash

    job, _, old, profile = finished
    config = Path(old["steps"][0]["arguments"]["config"])
    old["frozen_files"] = {str(config): file_hash(config)}
    save(job / "plan.json", old)
    config.write_text("seed: 999\n")
    # A plan-building refresh must not silently trust the changed file.
    plan = build_resume_plan(
        arguments(job, "--resume-check", "none", "--append-tasks", "test"), profile, False
    )
    assert plan["frozen_files"][str(config)] == old["frozen_files"][str(config)]
    with pytest.raises(ValueError, match="改变"):
        build_resume_plan(arguments(job, "--append-tasks", "test"), profile, False)


def test_execute_only_new_steps_and_results_include_them(finished, monkeypatch, capsys):
    from chd_ct.server import imagechd

    job, _, _, profile = finished
    plan = build_resume_plan(arguments(job, "--append-tasks", "test,predict,evaluate"), profile, False)
    root = Path(plan["steps"][0]["output"])
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
    executed = []

    def run(args, profile, output_dir, **kwargs):
        executed.append(args.task)
        if args.task in {"test", "predict"}:
            target = output_dir / ("test/predictions" if args.task == "test" else "predict")
            save(target / "predictions.json", {"cases": []})
        else:
            save(output_dir / "evaluate/evaluation.json", {"mean_dice": 0.8})
        save(output_dir / "status.json", {})
        return 0

    monkeypatch.setattr(imagechd, "run", run)
    save(job / "plan.json", plan)
    assert jobs.execute(plan) == 0
    assert executed == ["test", "predict", "evaluate"]
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
    results.show_results(job)
    text = capsys.readouterr().out
    assert "已完成训练" in text and "evaluate" in text and "predict" in text
    # A later append can validate the same completed artifacts again.
    next_plan = build_resume_plan(arguments(job, "--append-tasks", "diagnosis-demo"), profile, False)
    assert len(next_plan["steps"]) == 5
    assert all(s.get("reuse") for s in next_plan["steps"][:4])


@pytest.mark.parametrize("selection", ["train", "preprocess", "train,test"])
def test_existing_steps_and_bound_cache_cannot_be_replaced(finished, selection):
    job, _, _, profile = finished
    with pytest.raises(ValueError, match="不能"):
        build_resume_plan(arguments(job, "--append-tasks", selection), profile, False)


@pytest.mark.parametrize("damage", ["checkpoint", "snapshot"])
def test_completed_step_invalidity_never_turns_into_rerun(finished, damage):
    job, _, plan, profile = finished
    if damage == "checkpoint":
        (Path(plan["steps"][0]["output"]) / "train/crop64.pt").write_bytes(b"changed")
    else:
        state = json.loads((job / "job_state.json").read_text())
        state["steps"][0].pop("snapshot")
        save(job / "job_state.json", state)
    with pytest.raises(ValueError, match="已完成"):
        build_resume_plan(arguments(job, "--append-tasks", "test"), profile, False)


def test_failed_original_steps_resume_before_added_steps(finished):
    job, _, _, profile = finished
    state = json.loads((job / "job_state.json").read_text())
    state["status"] = state["steps"][0]["status"] = "interrupted"
    state["steps"][0].pop("snapshot")
    save(job / "job_state.json", state)
    plan = build_resume_plan(arguments(job, "--append-tasks", "test"), profile, False)
    assert plan["steps"][0]["arguments"]["resume"]
    assert not plan["steps"][0].get("reuse")
    assert plan["steps"][1]["name"] == "test"


def test_append_accepts_new_diagnosis_input_without_changing_old_steps(finished):
    job, _, original, profile = finished
    source = job.parent.parent / "new-labels.csv"
    source.write_text("id,label\n1,1\n")
    plan = build_resume_plan(
        arguments(job, "--append-tasks", "diagnosis-labels", "--diagnosis-source", str(source)),
        profile,
        False,
    )
    assert plan["steps"][1]["arguments"]["diagnosis_source"] == str(source)
    assert plan["steps"][0]["arguments"] == original["steps"][0]["arguments"]


@pytest.mark.parametrize("field", ["prepared", "models", "config"])
def test_append_rejects_rebinding_existing_inputs(finished, field):
    job, _, _, profile = finished
    with pytest.raises(ValueError, match="已绑定"):
        build_resume_plan(
            arguments(job, "--append-tasks", "test", "--" + field, str(job / "other-input")), profile, False
        )


def test_external_models_are_frozen_in_interactive_append(finished, monkeypatch):
    job, _, original, profile = finished
    old = original["steps"][0]
    test = dict(
        name="test",
        output=str(job / "steps/01_test"),
        arguments={
            "task": "test",
            "prepared": old["arguments"]["prepared"],
            "models": str(Path(old["output"]) / "train"),
            "device": "cpu",
        },
    )
    plan = dict(original, steps=[test])
    save(job / "plan.json", plan)
    state = jobs.initial_state(plan)
    state["status"] = "passed"
    state["steps"][0].update(status="passed", snapshot=snapshot(test))
    save(job / "job_state.json", state)
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("Frozen model must not be prompted"))
    extended = build_resume_plan(arguments(job, "--append-tasks", "predict"), profile, True)
    assert extended["steps"][1]["arguments"]["models"] == test["arguments"]["models"]


def test_stale_append_is_rejected_before_submit_writes(finished):
    job, _, _, profile = finished
    plan = build_resume_plan(arguments(job, "--append-tasks", "test"), profile, False)
    save(job / "receipt.json", {"pid": 123})
    with pytest.raises(ValueError, match="已改变"):
        jobs.submit(plan, None)
    assert not (job / "attempts").exists()
    assert len(json.loads((job / "plan.json").read_text())["steps"]) == 1


def test_source_stamp_detects_plan_changes(finished):
    job, _, _, profile = finished
    plan = build_resume_plan(arguments(job, "--append-tasks", "test"), profile, False)
    save(job / "plan.json", {})
    with pytest.raises(ValueError, match="已改变"):
        extension.verify_source(plan)


def test_submit_archives_previous_attempt_and_keeps_directory(finished, monkeypatch):
    from types import SimpleNamespace

    job, runtime, _, profile = finished
    originals = {name: (job / name).read_bytes() for name in ("plan.json", "job_state.json")}
    plan = build_resume_plan(arguments(job, "--append-tasks", "test"), profile, False)
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: SimpleNamespace(pid=123))
    with (runtime / "test.lock").open("w") as lock:
        assert jobs.submit(plan, lock) == 0
    (attempt,) = (job / "attempts").iterdir()
    assert all((attempt / name).read_bytes() == data for name, data in originals.items())
    state = json.loads((job / "job_state.json").read_text())
    assert state["steps"][0]["status"] == "reused"
    assert state["steps"][1]["status"] == "pending"
    assert json.loads((runtime / "last-job.json").read_text())["directory"] == str(job)


def test_completed_task_can_be_chosen_from_panel(finished, monkeypatch):
    job, runtime, _, profile = finished
    monkeypatch.setattr("builtins.input", lambda _: "1")
    assert panel.pick_history(runtime, profile["results"], resume=True) == job


def test_completed_cli_requires_explicit_append_and_menu_lists_only_missing(finished, monkeypatch, capsys):
    job, _, _, profile = finished
    with pytest.raises(ValueError, match="append-tasks"):
        build_resume_plan(arguments(job), profile, False)
    monkeypatch.setattr("builtins.input", lambda _: "test")
    plan = build_resume_plan(arguments(job), profile, True)
    assert plan["appended_steps"] == ["test"]
    assert "正式训练 (train)" not in capsys.readouterr().out
