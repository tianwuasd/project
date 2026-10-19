"""Workflow GPU retention contracts without allocating any real GPU."""

import os
import subprocess
from pathlib import Path

import pytest

from chd_ct.recovery import read_json, write_json
from chd_ct.server import gpu_hold, jobs, launcher, resources, resume


@pytest.fixture
def registry(tmp_path, monkeypatch):
    identities = {os.getpid(): dict(boot="test", start="owner")}
    monkeypatch.setenv(gpu_hold.ENV, "")
    monkeypatch.setattr(jobs, "identity", identities.get)
    plan = dict(directory=str(tmp_path), steps=[dict(arguments=dict(device="cuda", gpu_count=2))])
    folder = gpu_hold.begin(plan)
    gpu_hold.configure(["GPU-a", "GPU-b", "GPU-c"])

    def holder(pid, gpu):
        identities[pid] = dict(boot="test", start=str(pid))
        write_json(folder / f"{pid}.holder.json", dict(pid=pid, identity=identities[pid], gpu=gpu))

    return folder, identities, holder


def test_retained_cards_are_preferred_and_cap_prevents_extra_acquisition(registry, monkeypatch):
    _, _, holder = registry
    rows = [dict(uuid=gpu) for gpu in ("GPU-c", "GPU-b", "GPU-a")]
    holder(123, "GPU-a")
    assert [r["uuid"] for r in gpu_hold.candidates(rows)] == ["GPU-a", "GPU-c", "GPU-b"]
    holder(124, "GPU-b")
    assert {r["uuid"] for r in gpu_hold.candidates(rows)} == {"GPU-a", "GPU-b"}
    monkeypatch.setattr(gpu_hold.subprocess, "Popen", lambda *a, **kw: pytest.fail("extra holder spawned"))
    assert gpu_hold.acquire("GPU-a") is True
    assert gpu_hold.acquire("GPU-c") is False
    with pytest.raises(ValueError, match="获准"):
        gpu_hold.acquire("GPU-outside")


def test_changed_scope_and_dead_or_reused_holder_are_rejected(registry):
    _, identities, holder = registry
    holder(123, "GPU-a")
    with pytest.raises(ValueError, match="排除了"):
        gpu_hold.configure(["GPU-b"])
    identities[123] = dict(boot="test", start="another-process")
    with pytest.raises(ValueError, match="意外退出"):
        gpu_hold.owned_processes()


def test_gpu_query_ignores_only_our_identity_verified_holder(registry, monkeypatch):
    _, _, holder = registry
    holder(123, "GPU-a")

    def query(command, **kwargs):
        output = (
            "0, GPU-a, GPU, 20000, 0\n1, GPU-b, GPU, 20000, 0\n"
            if "--query-gpu=" in command[1]
            else "GPU-a, 123\nGPU-b, 123\n"
        )
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(launcher.subprocess, "run", query)
    rows = launcher.query_gpus()
    assert rows[0]["busy"] is False and rows[1]["busy"] is True

    def other_user(command, **kwargs):
        result = query(command, **kwargs)
        if "--query-compute-apps=" in command[1]:
            result.stdout += "GPU-a, 999\n"
        return result

    monkeypatch.setattr(launcher.subprocess, "run", other_user)
    assert launcher.query_gpus()[0]["busy"] is True


def test_claim_race_waits_without_counting_failed_claim_as_ready(tmp_path, monkeypatch):
    monkeypatch.delenv(gpu_hold.ENV, raising=False)
    attempts, waits = [], []
    rows = [dict(uuid="GPU-a", busy=False, free=20000, util=0)]

    def claim(gpu):
        attempts.append(gpu)
        return len(attempts) > 1

    monkeypatch.setattr(resources.time, "sleep", waits.append)
    result = resources.wait_for_gpus(["GPU-a"], 2, lambda: rows, lambda *a: None, claim=claim)
    assert result == rows and waits == [5]


@pytest.mark.parametrize("outcome", [0, 7, "interrupt"])
def test_workflow_releases_before_cpu_step(tmp_path, monkeypatch, outcome):
    from chd_ct.server import imagechd

    monkeypatch.setenv(gpu_hold.ENV, "")
    monkeypatch.setattr(jobs, "identity", lambda _: dict(boot="test", start="owner"))
    plan = dict(
        directory=str(tmp_path),
        runtime=str(tmp_path / "runtime"),
        log=str(tmp_path / "launcher.log"),
        profile={},
        steps=[
            dict(
                name="train",
                output=str(tmp_path / "train"),
                arguments=dict(task="train", device="cuda", gpu_count=2),
            ),
            dict(
                name="evaluate",
                output=str(tmp_path / "evaluate"),
                arguments=dict(task="evaluate", device="cpu"),
            ),
        ],
    )
    seen = []

    def run(args, profile, **kwargs):
        assert (gpu_hold.directory() is not None) == (args.task == "train")
        seen.append(args.task)
        if args.task == "evaluate":
            if outcome == "interrupt":
                raise KeyboardInterrupt
            return outcome
        return 0

    monkeypatch.setattr(imagechd, "run", run)
    monkeypatch.setattr(
        gpu_hold, "release", lambda folder: seen.append(("release", Path(folder).parent.name))
    )
    assert jobs.execute(plan) == (130 if outcome == "interrupt" else outcome)
    assert seen == ["train", ("release", "gpu-holds"), "evaluate"]
    assert os.environ[gpu_hold.ENV] == ""


def test_cpu_only_workflow_never_starts_holders(tmp_path, monkeypatch):
    monkeypatch.delenv(gpu_hold.ENV, raising=False)
    assert gpu_hold.begin(dict(directory=str(tmp_path), steps=[dict(arguments=dict(device="cpu"))])) is None
    assert not (tmp_path / "gpu-holds").exists()


def test_resume_limit_excludes_reused_steps(tmp_path, monkeypatch):
    monkeypatch.setenv(gpu_hold.ENV, "")
    monkeypatch.setattr(jobs, "identity", lambda _: dict(boot="test", start="owner"))
    reused = dict(reuse={"status": "passed"}, arguments=dict(device="cuda", gpu_count=5))
    active = dict(arguments=dict(device="cuda", gpu_count=1))
    folder = gpu_hold.begin(dict(directory=str(tmp_path), steps=[reused, active]))
    assert read_json(folder / "owner.json")["limit"] == 1
    assert gpu_hold.begin(dict(directory=str(tmp_path), steps=[reused])) is None


def test_unpublished_holder_receipts_do_not_break_queries_or_stop(registry):
    folder, _, holder = registry
    holder(123, "GPU-a")
    (folder / ".atomic.123.holder.json").write_text('{"pid":', encoding="utf-8")
    assert [r["pid"] for r in gpu_hold.records(folder)] == [123]
    plan = dict(directory=str(folder.parent.parent), steps=[])
    assert [r["pid"] for r in resume.live_commands(plan)] == [123]
    (folder / "bad.holder.json").write_text('{"pid":', encoding="utf-8")
    with pytest.raises(ValueError):
        gpu_hold.records(folder)


def test_cancelled_receipt_disappears_during_scan(registry, monkeypatch):
    folder, _, holder = registry
    holder(123, "GPU-a")
    original = gpu_hold.read_json

    def removed(path):
        if path.name.endswith(".holder.json"):
            path.unlink()
        return original(path)

    monkeypatch.setattr(gpu_hold, "read_json", removed)
    assert gpu_hold.records(folder) == []
