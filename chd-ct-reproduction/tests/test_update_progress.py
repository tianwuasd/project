"""Observable, bounded downloads using real local child processes, no network."""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import test_server_update

from chd_ct.server import update_progress, updater

repository = test_server_update.repository
local_git = test_server_update.local_git


def child(monkeypatch, script):
    original = subprocess.Popen
    spawned = []

    def spawn(command, **kwargs):
        process = original([sys.executable, "-u", "-c", script], **kwargs)
        spawned.append(process)
        return process

    monkeypatch.setattr(updater.subprocess, "Popen", spawn)
    return spawned


def test_carriage_return_progress_arrives_before_exit_and_drains_both_pipes(tmp_path, monkeypatch):
    child(
        monkeypatch,
        "import sys,time\nsys.stdout.write('x'*200000);sys.stdout.flush()\nsys.stderr.write('Receiving objects: 10% (1/10), 1.0 MiB | 1.0 MiB/s\\r');sys.stderr.flush()\ntime.sleep(.4)\nsys.stderr.write('Receiving objects: 100% (10/10), done.\\n')",
    )
    printed = []
    monkeypatch.setattr(
        "builtins.print", lambda *args, **kwargs: printed.append((time.monotonic(), " ".join(map(str, args))))
    )
    result = updater.git(tmp_path, "fetch", timeout=3, progress_log=tmp_path / "fetch.log", stage="对象下载")
    finished = time.monotonic()
    events = [t for t, line in printed if "10%" in line]
    assert events and events[0] < finished - 0.2
    assert result.returncode == 0 and len(result.stdout) <= update_progress.TAIL_CHARS
    assert "100%" in (tmp_path / "fetch.log").read_text(encoding="utf-8")


def test_total_deadline_is_not_extended_by_progress_or_heartbeat(tmp_path, monkeypatch, capsys):
    processes = child(
        monkeypatch,
        "import sys,time\nwhile True:\n sys.stderr.write('Receiving objects: 1% (1/100)\\r');sys.stderr.flush();time.sleep(.03)",
    )
    monkeypatch.setattr(update_progress, "HEARTBEAT_SECONDS", 0.05)
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as error:
        updater.git(tmp_path, "fetch", timeout=0.3, progress_log=tmp_path / "fetch.log", stage="对象下载")
    assert time.monotonic() - start < 3
    assert processes[0].poll() is not None
    assert "总超时" in str(error.value) and "fetch.log" in str(error.value)
    assert "等待" in capsys.readouterr().out


def test_failure_has_stage_reason_log_and_no_credentials(tmp_path, monkeypatch):
    child(
        monkeypatch,
        "import sys\nsys.stderr.write('Permission denied (publickey).\\nAuthorization: Bearer synthetic-secret\\nhttps://alice:synthetic-secret@example.invalid/repo?token=synthetic-secret\\n');sys.exit(128)",
    )
    log = tmp_path / "fetch.log"
    with pytest.raises(ValueError, match="对象下载.*认证") as error:
        updater.git(tmp_path, "fetch", timeout=2, progress_log=log, stage="对象下载")
    assert "synthetic-secret" not in str(error.value) + log.read_text(encoding="utf-8")
    assert "fetch.log" in str(error.value)


def test_log_size_is_bounded(tmp_path, monkeypatch):
    child(monkeypatch, "import sys\nfor i in range(300): sys.stderr.write('remote: '+ 'x'*4096+'\\r')")
    monkeypatch.setattr(update_progress, "MAX_LOG_BYTES", 8000)
    monkeypatch.setattr("builtins.print", lambda *a, **kw: None)
    log = tmp_path / "fetch.log"
    updater.git(tmp_path, "fetch", timeout=5, progress_log=log, stage="对象下载")
    assert log.stat().st_size <= 8000
    assert "日志达到" in log.read_text(encoding="utf-8")


def test_failure_tail_survives_log_size_cap(tmp_path, monkeypatch):
    child(
        monkeypatch,
        "import sys\nfor i in range(50): sys.stderr.write('remote: '+ 'x'*4096+'\\r')\nsys.stderr.write('fatal: Permission denied (publickey).\\n');sys.exit(128)",
    )
    monkeypatch.setattr(update_progress, "MAX_LOG_BYTES", 8000)
    log = tmp_path / "fetch.log"
    with pytest.raises(ValueError, match="认证") as error:
        updater.git(tmp_path, "fetch", timeout=3, progress_log=log, stage="对象下载")
    assert "Permission denied" in error.value.output_tail
    assert "Permission denied" not in log.read_text(encoding="utf-8")


def test_cancel_reaps_process_and_reader_threads(tmp_path, monkeypatch):
    processes = child(monkeypatch, "import time;time.sleep(60)")
    real_queue = update_progress.queue.Queue

    class CancelledQueue(real_queue):
        def get(self, **kwargs):
            raise KeyboardInterrupt

    monkeypatch.setattr(update_progress.queue, "Queue", CancelledQueue)
    with pytest.raises(KeyboardInterrupt):
        updater.git(tmp_path, "fetch", timeout=3, progress_log=tmp_path / "fetch.log", stage="对象下载")
    assert processes[0].poll() is not None
    assert not [t for t in threading.enumerate() if t.name.startswith("chd-git-")]


def test_archive_binary_stdout_never_enters_progress_log(repository, tmp_path, monkeypatch):
    repo, remote, _, after, _ = repository
    monkeypatch.setattr(updater, "MIRRORS", (str(remote),))
    runtime = tmp_path / "runtime"
    assert updater.update(repo, runtime)
    report = json.loads(next(runtime.glob("updates/*/update.json")).read_text(encoding="utf-8"))
    assert report["after"] == after and report["limits"]["archive"] == 180
    assert "\x00" not in Path(report["archive_log"]).read_text(encoding="utf-8")
    assert report["archive_elapsed_seconds"] >= 0 and report["apply_elapsed_seconds"] >= 0


def test_update_records_stage_limits_timings_and_keeps_checkout_on_timeout(repository, tmp_path, monkeypatch):
    repo, remote, before, _, calls = repository
    monkeypatch.setattr(updater, "MIRRORS", (str(remote),))
    original = updater.git
    observed = []

    def timeout(root, *args, **kwargs):
        if args[0] == "fetch":
            observed.append(kwargs)
            assert "--progress" in args
            raise subprocess.TimeoutExpired("git fetch", kwargs["timeout"])
        return original(root, *args, **kwargs)

    monkeypatch.setattr(updater, "git", timeout)
    runtime = tmp_path / "runtime"
    with pytest.raises(ValueError, match="下载.*超时"):
        updater.update(repo, runtime)
    assert observed[0]["timeout"] == 900
    report = json.loads(next(runtime.glob("updates/*/update.json")).read_text(encoding="utf-8"))
    assert report["attempts"][0]["elapsed_seconds"] >= 0
    assert report["attempts"][0]["error_kind"] == "timeout"
    assert report["attempts"][0]["log"].endswith("fetch-1.log")
    assert local_git(Path(repo["root"]), "rev-parse", "HEAD") == before
    assert not any(args[0] in {"archive", "merge"} for args in calls)
