import pytest

from chd_ct.server.launcher import select_gpu


def devices():
    return [
        dict(index="0", uuid="GPU-a", name="3090", free=23000, util=0, busy=False),
        dict(index="1", uuid="GPU-b", name="3090", free=22000, util=0, busy=False),
    ]


def test_gpu_respects_allocation_and_explicit_choice():
    rows = devices()
    assert select_gpu(rows, visible="1")["uuid"] == "GPU-b"
    assert select_gpu(rows, visible="GPU-a")["uuid"] == "GPU-a"
    with pytest.raises(ValueError):
        select_gpu(rows, requested="0", visible="1")
    with pytest.raises(ValueError):
        select_gpu(rows, visible="")
    with pytest.raises(ValueError):
        select_gpu(rows, visible="-1")


def test_busy_and_insufficient_gpu_are_rejected():
    rows = devices()
    rows[0]["busy"] = True
    rows[1]["free"] = 1000
    with pytest.raises(ValueError):
        select_gpu(rows)
    rows[1].update(free=23000, util=70)
    with pytest.raises(ValueError):
        select_gpu(rows)


def test_environment_cannot_redirect_private_pip_install(tmp_path, monkeypatch):
    import os

    from chd_ct.server.bootstrap import environment

    for key in ("PIP_TARGET", "PIP_PREFIX", "PIP_USER", "PYTHONHOME", "PYTHONPATH"):
        monkeypatch.setenv(key, "external")
    monkeypatch.setenv("HTTPS_PROXY", "http://server-proxy:8080")
    env = environment(tmp_path / "runtime", tmp_path / "code")
    assert not ({"PIP_TARGET", "PIP_PREFIX", "PIP_USER", "PYTHONHOME"} & env.keys())
    assert env["PIP_CONFIG_FILE"] == os.devnull
    assert env["PYTHONPATH"] == str(tmp_path / "code/src")
    assert env["HTTPS_PROXY"] == "http://server-proxy:8080"


def test_process_group_cleanup_reaches_grandchildren(monkeypatch):
    import signal
    from types import SimpleNamespace

    from chd_ct.server import bootstrap

    signals, waits = [], []
    monkeypatch.setattr(bootstrap.os, "name", "posix")
    monkeypatch.setattr(bootstrap.os, "killpg", lambda pid, sig: signals.append((pid, sig)), raising=False)
    # SIGKILL is not exposed by Python on Windows, but this tests the Linux branch.
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    process = SimpleNamespace(pid=1234, wait=lambda **kw: waits.append(kw))
    bootstrap.stop_process_tree(process)
    assert signals == [(1234, signal.SIGTERM), (1234, 9)]
    assert waits == [{"timeout": 15}, {}]


def test_parallel_settings_writes_are_atomic(tmp_path, monkeypatch):
    import json
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from chd_ct.server import launcher

    replace = launcher.os.replace
    barrier = threading.Barrier(2)
    rename_lock = threading.Lock()

    def concurrent_replace(source, destination):
        # Both writes finish first; serialize only the atomic rename so this legal
        # POSIX interleaving can be exercised on Windows (concurrent replace denies access).
        barrier.wait(timeout=10)
        with rename_lock:
            replace(source, destination)

    monkeypatch.setattr(launcher.os, "replace", concurrent_replace)
    path = tmp_path / ".server-settings.json"
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(launcher.write_json, path, {"run": i}) for i in range(2)]
        for future in futures:
            future.result(timeout=15)
    assert json.loads(path.read_text(encoding="utf-8")) in [{"run": 0}, {"run": 1}]
    assert not list(tmp_path.glob("*.tmp"))


def test_existing_conda_is_reused_without_downloading(tmp_path, monkeypatch):
    from chd_ct.server import bootstrap

    existing = tmp_path / "old/miniforge/bin/conda"
    existing.parent.mkdir(parents=True)
    existing.write_text("sentinel")
    root, base = tmp_path / "code", tmp_path / "runtime"
    root.mkdir()
    (root / "pyproject.toml").write_text("test")
    calls = []

    def checked(command, env, log):
        calls.append([str(x) for x in command])
        if "create" in command:
            python = base / "envs/chd_py311/bin/python"
            python.parent.mkdir(parents=True)
            python.touch()

    monkeypatch.setattr(bootstrap, "checked", checked)
    monkeypatch.setattr(
        bootstrap.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("不应下载新的 Miniforge")
    )
    python = bootstrap.ensure_python(base, root, {}, tmp_path / "log", str(existing))
    assert python.is_file() and calls[0][0] == str(existing)
    assert existing.read_text() == "sentinel"
    assert not (base / "tools/miniforge3").exists()


@pytest.mark.parametrize(
    "task,expected",
    [
        ("preprocess", "preprocess.py"),
        ("train", "train.py"),
        ("predict", "predict.py"),
        ("evaluate", "evaluate.py"),
    ],
)
def test_primary_commands_are_independent(task, expected, tmp_path):
    import argparse

    from chd_ct.quickstart.commands import add_task_arguments, build_command

    parser = argparse.ArgumentParser()
    add_task_arguments(parser)
    args = parser.parse_args(
        [
            "--task",
            task,
            "--dataset",
            str(tmp_path),
            "--prepared",
            str(tmp_path),
            "--models",
            str(tmp_path),
            "--predictions",
            str(tmp_path),
        ]
    )
    command = build_command(args, tmp_path, "python", tmp_path / "out", "cpu")
    assert command[:3] == ["python", "-m", "chd_ct.imagechd." + expected.removesuffix(".py")]
    assert "paper" not in command


def test_server_does_not_train_after_preflight_failure(tmp_path, monkeypatch):
    import argparse
    import signal
    import sys
    from types import SimpleNamespace

    from chd_ct.quickstart.commands import add_task_arguments
    from chd_ct.server import imagechd

    parser = argparse.ArgumentParser()
    add_task_arguments(parser)
    args = parser.parse_args(["--task", "train", "--mode", "train", "--device", "cpu", "--non-interactive"])
    args.runtime = str(tmp_path / "runtime")
    args.results = str(tmp_path / "results")
    args.prepared = str(tmp_path / "prepared")
    args.python = sys.executable
    args.gpu = "auto"
    (tmp_path / "prepared").mkdir()
    (tmp_path / "prepared/dataset.json").write_text("{}")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(signal, "SIGHUP", 1, raising=False)
    monkeypatch.setattr(signal, "signal", lambda *a: None)
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *a: None))
    calls = []

    def run(command, env, log):
        calls.append(list(map(str, command)))
        return 1 if "preflight" in command else 0

    monkeypatch.setattr(imagechd, "run_logged", run)
    profile = {
        "runtime": args.runtime,
        "results": args.results,
        "prepared": args.prepared,
        "source_updated": "2026-09-26",
        "threads": 2,
    }
    assert imagechd.run(args, profile) == 1
    assert len(calls) == 2
    assert "preflight" in calls[-1]
    assert not any("train" == c[c.index("--mode") + 1] for c in calls if "--mode" in c)


def test_existing_output_report_is_never_overwritten(tmp_path):
    import json

    from chd_ct.quickstart.wizard import main

    folder = tmp_path / "existing"
    folder.mkdir()
    original = {"status": "passed", "result": "keep this earlier result"}
    (folder / "report.json").write_text(json.dumps(original))
    assert (
        main(["--task", "environment", "--output", str(folder), "--device", "cpu", "--non-interactive"]) == 1
    )
    assert json.loads((folder / "report.json").read_text(encoding="utf-8")) == original


def test_installed_layout_runs_primary_tasks(tmp_path):
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    site = tmp_path / "installation/Lib/site-packages"
    shutil.copytree(root / "src/chd_ct", site / "chd_ct", ignore=shutil.ignore_patterns("__pycache__"))
    outside = tmp_path / "outside"
    outside.mkdir()
    env = dict(os.environ, PYTHONPATH=str(site), PYTHONIOENCODING="utf-8")
    command = [
        sys.executable,
        "-m",
        "chd_ct",
        "--task",
        "demo",
        "--device",
        "cpu",
        "--non-interactive",
        "--output",
        str(outside / "result"),
    ]
    result = subprocess.run(
        command, cwd=outside, env=env, capture_output=True, text=True, encoding="utf-8", timeout=90
    )
    assert result.returncode == 0, result.stdout + result.stderr
    import json

    assert json.loads((outside / "result/demo/report.json").read_text())["status"] == "passed"
