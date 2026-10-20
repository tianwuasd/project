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


def test_failed_mirror_falls_back_and_backs_up_before_fast_forward(repository, tmp_path, monkeypatch):
    repo, remote, old, new, calls = repository
    monkeypatch.setattr(
        updater, "MIRRORS", (str(tmp_path / "missing"), str(remote), str(tmp_path / "unneeded"))
    )
    runtime = tmp_path / "runtime"
    assert updater.update(repo, runtime)
    root = Path(repo["root"])
    assert local_git(root, "rev-parse", "HEAD") == new
    assert local_git(root, "remote", "get-url", "origin") == updater.UPSTREAM
    assert not local_git(root, "for-each-ref", "refs/chd-update")
    report_path = next(runtime.glob("updates/*/update.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert [x["status"] for x in report["attempts"]] == ["failed", "verified"]
    assert [c[4] for c in calls if c[0] == "fetch"] == [str(tmp_path / "missing"), str(remote)]
    assert report["after"] == new and report["before"] == old
    with tarfile.open(report["backup"]) as archive:
        assert archive.extractfile("chd-ct-reproduction/code.py").read() == b"old\n"
    assert (root / "chd-ct-reproduction/code.py").read_text() == "new\n"


@pytest.mark.parametrize("suffix", [".git", "", ".git/", "/"])
def test_legacy_mirror_origin_can_update_without_rewriting_origin(repository, tmp_path, suffix):
    repo, _, _, new, _ = repository
    root = Path(repo["root"])
    origin = "https://gh-proxy.com/https://github.com/tianwuasd/project" + suffix
    local_git(root, "remote", "set-url", "origin", origin)
    assert updater.inspect_repository(root / "chd-ct-reproduction") == repo
    assert updater.update(repo, tmp_path / "runtime")
    assert local_git(root, "rev-parse", "HEAD") == new
    assert local_git(root, "remote", "get-url", "origin") == origin


@pytest.mark.parametrize("origin", [*updater.MIRRORS, "git@github.com:tianwuasd/project.git"])
def test_existing_origin_forms_remain_supported(repository, origin):
    repo, _, _, _, _ = repository
    root = Path(repo["root"])
    local_git(root, "remote", "set-url", "origin", origin)
    assert updater.inspect_repository(root / "chd-ct-reproduction") == repo


@pytest.mark.parametrize(
    "origin",
    [
        "https://gh-proxy.com/https://github.com/other/project.git",
        "https://gh-proxy.com/https://github.com/tianwuasd/other.git",
        "https://gh-proxy.com.evil.invalid/https://github.com/tianwuasd/project.git",
        "https://gh-proxy.com@evil.invalid/https://github.com/tianwuasd/project.git",
        "https://gh-proxy.com/https://github.com/tianwuasd/project.git/extra",
        "https://gh-proxy.com/https://github.com/tianwuasd/project.git?redirect=other",
        "https://gh-proxy.com/https://github.com/tianwuasd/project.git#other",
    ],
)
def test_similar_or_unrelated_mirror_origin_is_rejected(repository, origin):
    repo, _, old, _, calls = repository
    root = Path(repo["root"])
    local_git(root, "remote", "set-url", "origin", origin)
    with pytest.raises(ValueError, match="tianwuasd/project"):
        updater.inspect_repository(root / "chd-ct-reproduction")
    assert local_git(root, "rev-parse", "HEAD") == old
    assert not any(c[0] in {"fetch", "ls-remote"} for c in calls)


def test_all_mirrors_fail_preserves_checkout(repository, tmp_path, monkeypatch):
    repo, _, old, _, _ = repository
    monkeypatch.setattr(updater, "MIRRORS", (str(tmp_path / "missing"),))
    with pytest.raises(ValueError, match="所有更新线路"):
        updater.update(repo, tmp_path / "runtime")
    assert local_git(Path(repo["root"]), "rev-parse", "HEAD") == old


@pytest.mark.parametrize("known_upstream", [False, True])
def test_first_current_mirror_stops_without_checking_later_routes(
    repository, tmp_path, monkeypatch, capsys, known_upstream
):
    repo, remote, old, new, calls = repository
    current = tmp_path / "current"
    local_git(tmp_path, "clone", str(remote), str(current))
    local_git(current, "reset", "--hard", old)
    if known_upstream:
        previous = updater.git

        def known(root, *args, **kwargs):
            if args[0] == "ls-remote":
                return subprocess.CompletedProcess(args, 0, old + "\trefs/heads/main\n", "")
            return previous(root, *args, **kwargs)

        monkeypatch.setattr(updater, "git", known)
    monkeypatch.setattr(updater, "MIRRORS", (str(current), str(remote)))
    runtime = tmp_path / "runtime"
    assert not updater.update(repo, runtime)
    root = Path(repo["root"])
    assert local_git(root, "rev-parse", "HEAD") == old != new
    assert [c[4] for c in calls if c[0] == "fetch"] == [str(current)]
    assert not any(c[0] == "merge" for c in calls)
    assert not local_git(root, "for-each-ref", "refs/chd-update")
    report = json.loads(next(runtime.glob("updates/*/update.json")).read_text(encoding="utf-8"))
    assert report["status"] == ("current" if known_upstream else "mirror_current")
    assert report["selected"] == old
    assert [x["status"] for x in report["attempts"]] == ["verified"]
    assert "backup" not in report
    if not known_upstream:
        assert "当前线路与本地版本一致；GitHub 最新状态未能确认。" in capsys.readouterr().out


@pytest.mark.parametrize("later_route", ["newer", "diverged", "unreachable"])
def test_first_valid_mirror_stops_without_comparing_later_routes(
    repository, tmp_path, monkeypatch, later_route
):
    repo, remote, old, new, calls = repository
    first = tmp_path / "first"
    local_git(tmp_path, "clone", str(remote), str(first))
    later = remote
    if later_route == "unreachable":
        later = tmp_path / "missing"
    else:
        if later_route == "diverged":
            local_git(remote, "reset", "--hard", old)
        (remote / "chd-ct-reproduction/code.py").write_text("another update\n")
        local_git(remote, "commit", "-am", "later route")
    monkeypatch.setattr(updater, "MIRRORS", (str(first), str(later)))
    runtime = tmp_path / "runtime"
    assert updater.update(repo, runtime)
    root = Path(repo["root"])
    assert local_git(root, "rev-parse", "HEAD") == new
    assert [c[4] for c in calls if c[0] == "fetch"] == [str(first)]
    assert not local_git(root, "for-each-ref", "refs/chd-update")
    report = json.loads(next(runtime.glob("updates/*/update.json")).read_text(encoding="utf-8"))
    assert report["selected"] == new
    assert [x["status"] for x in report["attempts"]] == ["verified"]


def test_known_upstream_rejects_stale_mirror(repository, tmp_path, monkeypatch):
    repo, remote, old, new, calls = repository
    stale = tmp_path / "stale"
    local_git(tmp_path, "clone", str(remote), str(stale))
    local_git(stale, "reset", "--hard", old)
    previous = updater.git

    def known(root, *args, **kwargs):
        if args[0] == "ls-remote":
            return subprocess.CompletedProcess(args, 0, new + "\trefs/heads/main\n", "")
        return previous(root, *args, **kwargs)

    monkeypatch.setattr(updater, "git", known)
    monkeypatch.setattr(updater, "MIRRORS", (str(stale), str(remote), str(tmp_path / "unneeded")))
    assert updater.update(repo, tmp_path / "runtime")
    report = json.loads(
        next((tmp_path / "runtime").glob("updates/*/update.json")).read_text(encoding="utf-8")
    )
    assert [x["status"] for x in report["attempts"]] == ["failed", "verified"]
    assert report["upstream_head"] == new
    assert [c[4] for c in calls if c[0] == "fetch"] == [str(stale), str(remote)]


@pytest.mark.parametrize("invalid_history", ["behind", "diverged"])
def test_invalid_history_continues_until_a_fast_forward_candidate(
    repository, tmp_path, monkeypatch, invalid_history
):
    repo, remote, old, new, calls = repository
    root = Path(repo["root"])
    local_git(root, "fetch", str(remote), "main")
    local_git(root, "merge", "--ff-only", "FETCH_HEAD")
    repo = updater.inspect_repository(root / "chd-ct-reproduction")
    invalid = tmp_path / "invalid"
    local_git(tmp_path, "clone", str(remote), str(invalid))
    local_git(invalid, "reset", "--hard", old)
    if invalid_history == "diverged":
        local_git(invalid, "config", "user.email", "test@example.invalid")
        local_git(invalid, "config", "user.name", "Test")
        (invalid / "chd-ct-reproduction/code.py").write_text("divergent update\n")
        local_git(invalid, "commit", "-am", "divergent")
    (remote / "chd-ct-reproduction/code.py").write_text("newest\n")
    local_git(remote, "commit", "-am", "newest")
    newest = local_git(remote, "rev-parse", "HEAD")
    monkeypatch.setattr(updater, "MIRRORS", (str(invalid), str(remote), str(tmp_path / "unneeded")))
    runtime = tmp_path / "runtime"
    assert updater.update(repo, runtime)
    assert local_git(root, "rev-parse", "HEAD") == newest != new
    assert [c[4] for c in calls if c[0] == "fetch"] == [str(invalid), str(remote)]
    report = json.loads(next(runtime.glob("updates/*/update.json")).read_text(encoding="utf-8"))
    assert [x["status"] for x in report["attempts"]] == ["failed", "verified"]


@pytest.mark.parametrize("failure", ["timeout", "invalid_ref"])
def test_failed_candidate_validation_continues_to_next_route(
    repository, tmp_path, monkeypatch, failure
):
    repo, remote, _, new, _ = repository
    previous = updater.git
    first = str(tmp_path / "broken")
    attempted = []

    def broken(root, *args, **kwargs):
        if args[0] == "fetch":
            attempted.append(args[4])
            if args[4] == first:
                if failure == "timeout":
                    raise subprocess.TimeoutExpired("git fetch", kwargs["timeout"])
                # A successful transfer alone is insufficient: its commit ref must resolve.
                return subprocess.CompletedProcess(args, 0, "", "")
        return previous(root, *args, **kwargs)

    monkeypatch.setattr(updater, "git", broken)
    monkeypatch.setattr(updater, "MIRRORS", (first, str(remote), str(tmp_path / "unneeded")))
    runtime = tmp_path / "runtime"
    assert updater.update(repo, runtime)
    assert local_git(Path(repo["root"]), "rev-parse", "HEAD") == new
    assert attempted == [first, str(remote)]
    report = json.loads(next(runtime.glob("updates/*/update.json")).read_text(encoding="utf-8"))
    assert [x["status"] for x in report["attempts"]] == ["failed", "verified"]


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
