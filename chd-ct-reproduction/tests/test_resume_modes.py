"""Selectable resume checks without rerunning completed steps."""

import json
from pathlib import Path

import pytest

from chd_ct.server import jobs, launcher, verification
from chd_ct.server.resume import build_resume_plan, snapshot


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    root = tmp_path / "job"
    root.mkdir()
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "image.bin").write_bytes(b"original CT bytes")
    steps = []
    for index, task in enumerate(("preprocess", "diagnosis-labels", "diagnosis-features", "diagnosis-train")):
        output = root / "steps" / f"{index:02d}_{task}"
        arguments = dict(task=task, device="cpu", dataset=str(raw))
        if task == "preprocess":
            arguments["prepared"] = str(tmp_path / "prepared")
            target = Path(arguments["prepared"]) / "dataset.json"
        else:
            target = (
                output
                / task
                / {
                    "diagnosis-labels": "labels.json",
                    "diagnosis-features": "features.json",
                    "diagnosis-train": "classifier.json",
                }[task]
            )
        if task != "diagnosis-train":
            target.parent.mkdir(parents=True)
            target.write_text('{"status": "passed"}', encoding="utf-8")
        steps.append(dict(name=task, output=str(output), arguments=arguments))
    steps[-1]["arguments"].update(
        features=str(Path(steps[2]["output"]) / "diagnosis-features"),
        diagnosis_labels=str(Path(steps[1]["output"]) / "diagnosis-labels"),
    )
    plan = dict(
        schema_version=1,
        name="resume",
        directory=str(root),
        runtime=str(tmp_path / "runtime"),
        log=str(root / "launcher.log"),
        steps=steps,
    )
    state = jobs.initial_state(plan)
    state["status"] = "failed"
    for step, prior in zip(steps[:-1], state["steps"][:-1]):
        prior.update(status="passed", snapshot=snapshot(step))
    state["steps"][-1]["status"] = "failed"
    (root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (root / "job_state.json").write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(jobs, "identity", lambda _: None)
    return root, raw, plan


def resume(workflow, mode=None, interactive=False):
    root, _, plan = workflow
    argv = ["--resume-job", str(root)] + (["--resume-check", mode] if mode else [])
    args = launcher.make_parser().parse_args(argv)
    return build_resume_plan(args, dict(runtime=plan["runtime"]), interactive)


def test_default_simple_checks_direct_dependencies_not_raw_ct(workflow, monkeypatch):
    root, raw, _ = workflow
    (raw / "image.bin").write_bytes(b"changed old raw data")
    read = []
    original = verification.hash_one

    def counted(path, *args):
        read.append(path.name)
        return original(path, *args)

    monkeypatch.setattr(verification, "hash_one", counted)
    result = resume(workflow)
    assert result["resume_verification"]["mode"] == "simple"
    assert sorted(read) == ["features.json", "labels.json"]
    assert all(s.get("reuse") for s in result["steps"][:-1])
    assert result["steps"][-1]["restart"]


def test_full_checks_old_raw_content(workflow):
    _, raw, _ = workflow
    (raw / "image.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="改变"):
        resume(workflow, "full")


def test_simple_rejects_changed_or_missing_direct_input(workflow):
    _, _, plan = workflow
    target = Path(plan["steps"][-1]["arguments"]["features"]) / "features.json"
    target.write_text("changed")
    with pytest.raises(ValueError, match="改变"):
        resume(workflow, "simple")
    target.unlink()
    with pytest.raises(ValueError, match="不存在|改变"):
        resume(workflow, "simple")


def test_none_skips_artifact_checks_but_records_scope(workflow, monkeypatch):
    _, _, plan = workflow
    (Path(plan["steps"][-1]["arguments"]["features"]) / "features.json").unlink()
    monkeypatch.setattr(verification, "verify_snapshots", lambda *a, **k: pytest.fail("must not verify"))
    result = resume(workflow, "none")
    assert result["resume_verification"]["mode"] == "none"
    assert result["resume_verification"]["status"] == "skipped"
    assert all(s.get("reuse") for s in result["steps"][:-1])


@pytest.mark.parametrize("mode", ["full", "simple", "none"])
def test_all_modes_reject_live_children(workflow, monkeypatch, mode):
    monkeypatch.setattr("chd_ct.server.resume.live_commands", lambda _: [dict(pid=123)])
    with pytest.raises(ValueError, match="存活"):
        resume(workflow, mode)


@pytest.mark.parametrize("answer,mode", [("1", "full"), ("2", "simple"), ("3", "none"), ("", "simple")])
def test_panel_choice(workflow, monkeypatch, answer, mode):
    monkeypatch.setattr("builtins.input", lambda _: answer)
    assert resume(workflow, interactive=True)["resume_verification"]["mode"] == mode


@pytest.mark.parametrize("answer,error", [("b", "PanelBack"), ("q", "UserExit")])
def test_check_prompt_navigation(workflow, monkeypatch, answer, error):
    from chd_ct.server import prompts

    monkeypatch.setattr("builtins.input", lambda _: answer)
    with pytest.raises(getattr(prompts, error)):
        resume(workflow, interactive=True)


def test_resume_check_cannot_be_used_for_new_run():
    assert launcher.main(["--resume-check", "none", "--non-interactive"]) == 1


def test_legacy_simple_discloses_missing_hashes_and_full_refuses(workflow):
    root, _, _ = workflow
    state_path = root / "job_state.json"
    state = json.loads(state_path.read_text())
    for step in state["steps"]:
        step.pop("snapshot", None)
    state_path.write_text(json.dumps(state))
    result = resume(workflow, "simple")["resume_verification"]
    assert len(result["unrecorded_roots"]) == 2
    assert result["metadata_only_files"] == 2 and result["bytes"] == 0
    with pytest.raises(ValueError, match="缺少校验记录"):
        resume(workflow, "full")


def test_simple_accepts_artifact_file_with_custom_extension(workflow):
    root, _, plan = workflow
    artifact = root / "features.data"
    artifact.write_text("{}")
    plan["steps"][-1]["arguments"]["features"] = str(artifact)
    (root / "plan.json").write_text(json.dumps(plan))
    result = resume(workflow, "simple")["resume_verification"]
    assert str(artifact.resolve()) in result["unrecorded_roots"]
    assert result["files"] == 2
