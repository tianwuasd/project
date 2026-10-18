"""Real Git fast-forward/backup behavior with local, failure-injected mirrors."""

import json
import subprocess
import tarfile
from contextlib import contextmanager
from pathlib import Path

import pytest

from chd_ct.server import inspection, jobs, updater


def local_git(root, *args):
    return subprocess.run(
        ["git", "-c", "safe.directory=" + str(root), "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    remote.mkdir()
    local_git(remote, "init", "-b", "main")
    local_git(remote, "config", "user.email", "test@example.invalid")
    local_git(remote, "config", "user.name", "Test")
    (remote / "chd-ct-reproduction").mkdir()
    source = remote / "chd-ct-reproduction/code.py"
    source.write_bytes(b"old\n")
    local_git(remote, "add", ".")
    local_git(remote, "commit", "-m", "initial")
    old = local_git(remote, "rev-parse", "HEAD")
    checkout = tmp_path / "checkout"
    local_git(tmp_path, "clone", str(remote), str(checkout))
    local_git(checkout, "remote", "set-url", "origin", updater.UPSTREAM)
    source.write_text("new\n")
    local_git(remote, "commit", "-am", "next")
    new = local_git(remote, "rev-parse", "HEAD")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "safe.directory")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(checkout))
    actual_git = updater.git
    calls = []

    def offline_git(root, *args, **kwargs):
        calls.append(args)
        if args[0] == "ls-remote":
            raise subprocess.TimeoutExpired("GitHub", 12)
        return actual_git(root, *args, **kwargs)

    monkeypatch.setattr(updater, "git", offline_git)
    monkeypatch.setattr(updater, "MIRRORS", (str(tmp_path / "missing"), str(remote)))
    repo = updater.inspect_repository(checkout / "chd-ct-reproduction")
    return repo, remote, old, new, calls


def test_failed_mirror_falls_back_and_backs_up_before_fast_forward(repository, tmp_path):
    repo, remote, old, new, calls = repository
    runtime = tmp_path / "runtime"
    assert updater.update(repo, runtime)
    root = Path(repo["root"])
    assert local_git(root, "rev-parse", "HEAD") == new
    assert local_git(root, "remote", "get-url", "origin") == updater.UPSTREAM
    assert not local_git(root, "for-each-ref", "refs/chd-update")
    report_path = next(runtime.glob("updates/*/update.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert [x["status"] for x in report["attempts"]] == ["failed", "verified"]
    assert report["after"] == new and report["before"] == old
    with tarfile.open(report["backup"]) as archive:
        assert archive.extractfile("chd-ct-reproduction/code.py").read() == b"old\n"
    assert (root / "chd-ct-reproduction/code.py").read_text() == "new\n"


def test_all_mirrors_fail_preserves_checkout(repository, tmp_path, monkeypatch):
    repo, _, old, _, _ = repository
    monkeypatch.setattr(updater, "MIRRORS", (str(tmp_path / "missing"),))
    with pytest.raises(ValueError, match="所有更新线路"):
        updater.update(repo, tmp_path / "runtime")
    assert local_git(Path(repo["root"]), "rev-parse", "HEAD") == old


def test_equal_stale_mirror_does_not_prevent_trying_newer_one(repository, tmp_path, monkeypatch):
    repo, remote, old, new, calls = repository
    stale = tmp_path / "stale"
    local_git(tmp_path, "clone", str(remote), str(stale))
    local_git(stale, "reset", "--hard", old)
    monkeypatch.setattr(updater, "MIRRORS", (str(stale), str(remote)))
    assert updater.update(repo, tmp_path / "runtime")
    assert local_git(Path(repo["root"]), "rev-parse", "HEAD") == new
    assert len([c for c in calls if c[0] == "fetch"]) == 2


def test_known_upstream_rejects_stale_mirror(repository, tmp_path, monkeypatch):
    repo, remote, old, new, _ = repository
    stale = tmp_path / "stale"
    local_git(tmp_path, "clone", str(remote), str(stale))
    local_git(stale, "reset", "--hard", old)
    previous = updater.git

    def known(root, *args, **kwargs):
        if args[0] == "ls-remote":
            return subprocess.CompletedProcess(args, 0, new + "\trefs/heads/main\n", "")
        return previous(root, *args, **kwargs)

    monkeypatch.setattr(updater, "git", known)
    monkeypatch.setattr(updater, "MIRRORS", (str(stale), str(remote)))
    assert updater.update(repo, tmp_path / "runtime")
    report = json.loads(
        next((tmp_path / "runtime").glob("updates/*/update.json")).read_text(encoding="utf-8")
    )
    assert report["attempts"][0]["status"] == "failed"
    assert report["upstream_head"] == new


@pytest.mark.parametrize("change", ["dirty", "untracked", "branch", "origin"])
def test_unsafe_checkout_rejected_before_download(repository, tmp_path, change):
    repo, _, _, _, calls = repository
    root = Path(repo["root"])
    if change == "dirty":
        (root / "chd-ct-reproduction/code.py").write_text("user work")
    elif change == "untracked":
        (root / "notes.txt").write_text("private notes")
    elif change == "branch":
        local_git(root, "switch", "-c", "experiment")
    else:
        local_git(root, "remote", "set-url", "origin", "https://example.invalid/unrelated.git")
    with pytest.raises(ValueError):
        updater.update(repo, tmp_path / "runtime")
    assert not any(c[0] == "fetch" for c in calls)


def test_diverged_local_history_is_not_reset(repository, tmp_path):
    repo, _, old, _, _ = repository
    root = Path(repo["root"])
    local_git(root, "config", "user.email", "test@example.invalid")
    local_git(root, "config", "user.name", "Test")
    (root / "chd-ct-reproduction/code.py").write_text("local commit\n")
    local_git(root, "commit", "-am", "local")
    repo = updater.inspect_repository(root / "chd-ct-reproduction")
    with pytest.raises(ValueError, match="所有更新线路"):
        updater.update(repo, tmp_path / "runtime")
    assert local_git(root, "rev-parse", "HEAD") == repo["head"] != old


def test_ignored_local_data_cannot_be_overwritten_by_new_tracked_path(repository, tmp_path):
    repo, remote, old, _, _ = repository
    root = Path(repo["root"])
    # Both sides share an ignore rule; the newer upstream then tracks that path.
    local_git(remote, "reset", "--hard", old)
    (remote / ".gitignore").write_text("data/\n")
    local_git(remote, "add", ".gitignore")
    local_git(remote, "commit", "-m", "ignore data")
    local_git(root, "fetch", str(remote), "main")
    local_git(root, "merge", "--ff-only", "FETCH_HEAD")
    for checkout, content in [(root, "USER DATA"), (remote, "UPSTREAM FILE")]:
        target = checkout / "chd-ct-reproduction/data/example.txt"
        target.parent.mkdir(parents=True)
        target.write_text(content)
    local_git(remote, "add", "-f", "chd-ct-reproduction/data/example.txt")
    local_git(remote, "commit", "-m", "track example")
    repo = updater.inspect_repository(root / "chd-ct-reproduction")
    with pytest.raises(ValueError, match="Git 操作失败"):
        updater.update(repo, tmp_path / "runtime")
    assert (root / "chd-ct-reproduction/data/example.txt").read_text() == "USER DATA"
    assert local_git(root, "rev-parse", "HEAD") == repo["head"]


def test_idle_lock_waits_and_holds_until_update_finishes(tmp_path, monkeypatch):
    state = {"tries": 0, "held": False}

    @contextmanager
    def acquire(runtime):
        state["tries"] += 1
        if state["tries"] == 1:
            raise ValueError("busy")
        state["held"] = True
        try:
            yield
        finally:
            state["held"] = False

    monkeypatch.setattr(jobs, "acquire_lock", acquire)
    monkeypatch.setattr(inspection, "latest", lambda _: None)
    monkeypatch.setattr(inspection, "job_active", lambda _: False)
    monkeypatch.setattr(updater.time, "sleep", lambda _: None)
    with updater.idle_lock(tmp_path, None):
        assert state["held"]
    assert not state["held"] and state["tries"] == 2


def test_idle_lock_timeout_never_enters_update(tmp_path, monkeypatch):
    @contextmanager
    def busy(runtime):
        raise ValueError("busy")
        yield

    monkeypatch.setattr(jobs, "acquire_lock", busy)
    with pytest.raises(ValueError, match="尚未完全退出"):
        with updater.idle_lock(tmp_path, None, timeout=0):
            pytest.fail("updated while running")


def test_changed_target_releases_lock_and_rejects(tmp_path, monkeypatch):
    from contextlib import nullcontext

    monkeypatch.setattr(jobs, "acquire_lock", lambda _: nullcontext())
    monkeypatch.setattr(inspection, "latest", lambda _: tmp_path / "new-job")
    with pytest.raises(ValueError, match="记录已改变"):
        with updater.idle_lock(tmp_path, tmp_path / "old-job"):
            pytest.fail("updated changed target")
