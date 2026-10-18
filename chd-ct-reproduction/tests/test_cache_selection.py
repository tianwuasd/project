"""Cache choice, fallback and downstream path propagation without a GPU."""

from pathlib import Path

import pytest
from test_server_jobs import arguments

from chd_ct.recovery import read_json, write_json
from chd_ct.server import jobs
from chd_ct.server.workflow import build_plan


def test_new_cache_choice_generates_unique_path_without_creating_it(tmp_path):
    args, profile = arguments(tmp_path, "--tasks", "preprocess,train,test", "--preprocess-cache", "new")
    Path(profile["dataset"]).mkdir()
    old = Path(profile["prepared"])
    old.mkdir()
    (old / "keep.txt").write_text("unchanged")
    plans = [build_plan(args, profile, False) for _ in range(2)]
    paths = [p["steps"][0]["arguments"]["prepared"] for p in plans]
    assert paths[0] != paths[1] and str(old) not in paths
    assert all(not Path(p).exists() for p in paths)
    assert all(s["arguments"]["prepared"] == paths[0] for s in plans[0]["steps"])
    assert (old / "keep.txt").read_text() == "unchanged"


@pytest.mark.parametrize("reply,changed", [("", False), ("2", True)])
def test_interactive_cache_choice(tmp_path, monkeypatch, capsys, reply, changed):
    args, profile = arguments(tmp_path, "--task", "preprocess")
    Path(profile["dataset"]).mkdir()
    prompts = []

    def answer(prompt):
        prompts.append(prompt)
        return reply if "缓存策略" in prompt else ""

    monkeypatch.setattr("builtins.input", answer)
    plan = build_plan(args, profile, True)
    assert any("缓存策略" in p for p in prompts)
    assert (plan["steps"][0]["arguments"]["prepared"] != profile["prepared"]) == changed
    assert "新建" in capsys.readouterr().out


def workflow(tmp_path):
    old = tmp_path / "cache"
    old.mkdir()
    (old / "keep.txt").write_text("unchanged")
    directory = tmp_path / "workflow"
    directory.mkdir()
    plan = dict(
        directory=str(directory),
        runtime=str(tmp_path / "runtime"),
        log=str(directory / "launcher.log"),
        profile={},
        steps=[
            dict(
                name=name,
                output=str(directory / name),
                arguments=dict(task=name, device="cpu", prepared=str(old), resume=True),
            )
            for name in ("preprocess", "train")
        ],
    )
    write_json(directory / "plan.json", plan)
    return plan, old


def test_reuse_failure_rebinds_plan_once_and_preserves_old_cache(tmp_path, monkeypatch, capsys):
    from chd_ct.server import imagechd

    plan, old = workflow(tmp_path)
    seen = []

    def run(args, profile, output_dir, **kwargs):
        seen.append((args.task, args.prepared))
        if len(seen) == 1:
            output_dir.mkdir()
            write_json(
                output_dir / "cache-reuse-failure.json",
                dict(kind="cache_reuse_failed", prepared=str(old), reason="options.size 改变"),
            )
            return 78
        frozen = read_json(Path(plan["directory"]) / "plan.json")
        assert all(s["arguments"]["prepared"] == args.prepared for s in frozen["steps"])
        return 0

    monkeypatch.setattr(imagechd, "run", run)
    assert jobs.execute(plan) == 0
    assert [s[0] for s in seen] == ["preprocess", "preprocess", "train"]
    assert seen[1][1] == seen[2][1] != str(old)
    assert (old / "keep.txt").read_text() == "unchanged"
    state = read_json(Path(plan["directory"]) / "job_state.json")
    assert state["steps"][0]["cache_selection"]["reason"] == "options.size 改变"
    assert "自动新建" in capsys.readouterr().out


@pytest.mark.parametrize("code,report,started", [(1, True, False), (78, False, False), (78, True, True)])
def test_unrelated_failure_or_existing_downstream_never_rebinds(tmp_path, monkeypatch, code, report, started):
    from chd_ct.server import imagechd

    plan, old = workflow(tmp_path)
    if started:
        Path(plan["steps"][1]["output"]).mkdir()
    calls = []

    def run(args, profile, output_dir, **kwargs):
        calls.append(args.prepared)
        output_dir.mkdir()
        if report:
            write_json(
                output_dir / "cache-reuse-failure.json",
                dict(kind="cache_reuse_failed", prepared=str(old), reason="cache mismatch"),
            )
        return code

    monkeypatch.setattr(imagechd, "run", run)
    assert jobs.execute(plan) != 0
    assert calls == [str(old)]
    assert all(s["arguments"]["prepared"] == str(old) for s in plan["steps"])


def test_cpu_prefix_does_not_wait_for_gpu_and_training_keeps_its_limit(tmp_path):
    args, profile = arguments(
        tmp_path, "--tasks", "environment,preprocess,train", "--device", "cuda", "--gpu-count", "3"
    )
    Path(profile["dataset"]).mkdir()
    plan = build_plan(args, profile, False)
    assert [s["arguments"]["device"] for s in plan["steps"]] == ["cpu", "cpu", "cuda"]
    assert all("gpu_count" not in s["arguments"] for s in plan["steps"][:2])
    assert plan["steps"][2]["arguments"]["gpu_count"] == 3


@pytest.mark.parametrize("with_training", [False, True])
def test_cpu_environment_and_preprocess_dispatch_never_query_or_claim_gpu(
    tmp_path, monkeypatch, with_training
):
    import signal
    import sys
    from types import SimpleNamespace

    from chd_ct.server import gpu_hold, imagechd

    tasks = "environment,preprocess" + (",train" if with_training else "")
    args, profile = arguments(tmp_path, "--tasks", tasks, "--device", "cuda")
    Path(profile["dataset"]).mkdir()
    plan = build_plan(args, profile, False)
    Path(plan["directory"]).mkdir(parents=True)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(signal, "SIGHUP", 1, raising=False)
    monkeypatch.setattr(signal, "signal", lambda *a: None)
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *a: None))
    monkeypatch.setattr(imagechd, "query_gpus", lambda: pytest.fail("CPU step queried GPU"))
    monkeypatch.setattr(imagechd, "wait_for_gpus", lambda *a, **k: pytest.fail("CPU step waited for GPU"))
    monkeypatch.setattr(gpu_hold, "acquire", lambda *a, **k: pytest.fail("CPU step acquired GPU"))
    seen = []

    def command(cmd, env, log):
        assert env["CUDA_VISIBLE_DEVICES"] == ""
        seen.append(list(map(str, cmd)))
        return 0

    monkeypatch.setattr(imagechd, "run_logged", command)
    actual = imagechd.run

    def dispatch(args, *a, **kw):
        if args.task == "train":
            assert any("chd_ct.imagechd.preprocess" in cmd for cmd in seen)
            return 130  # GPU stage reached only after CPU preprocessing completed.
        return actual(args, *a, **kw)

    monkeypatch.setattr(imagechd, "run", dispatch)
    assert jobs.execute(plan) == (130 if with_training else 0)
    assert any("chd_ct.imagechd.preprocess" in cmd for cmd in seen)
    assert not list(Path(plan["directory"]).glob("gpu-holds/*/*.holder.json"))


def test_fallback_retries_only_once_even_if_new_worker_fails(tmp_path, monkeypatch):
    from chd_ct.server import imagechd

    plan, old = workflow(tmp_path)
    calls = []

    def run(args, profile, output_dir, **kwargs):
        calls.append(args.prepared)
        output_dir.mkdir()
        write_json(
            output_dir / "cache-reuse-failure.json",
            dict(kind="cache_reuse_failed", prepared=args.prepared, reason="bad cache"),
        )
        return 78

    monkeypatch.setattr(imagechd, "run", run)
    assert jobs.execute(plan) == 78
    assert len(calls) == 2 and calls[0] == str(old) != calls[1]
    assert "cache_selection" in jobs.initial_state(plan)["steps"][0]


def test_resume_after_fallback_keeps_resolved_path(tmp_path, monkeypatch):
    from chd_ct.server import imagechd, launcher
    from chd_ct.server.resume import build_resume_plan

    plan, old = workflow(tmp_path)
    plan["schema_version"] = 1
    calls = []

    def run(args, profile, output_dir, **kwargs):
        calls.append(args.prepared)
        if len(calls) == 1:
            output_dir.mkdir()
            write_json(
                output_dir / "cache-reuse-failure.json",
                dict(kind="cache_reuse_failed", prepared=str(old), reason="options.size 改变"),
            )
            return 78
        raise KeyboardInterrupt

    monkeypatch.setattr(imagechd, "run", run)
    assert jobs.execute(plan) == 130
    monkeypatch.setattr(jobs, "identity", lambda _: None)
    args = launcher.make_parser().parse_args(["--resume-job", plan["directory"]])
    resumed = build_resume_plan(args, {}, False)
    assert all(s["arguments"]["prepared"] == calls[-1] != str(old) for s in resumed["steps"])
    assert resumed["steps"][0]["arguments"]["resume"] is True
