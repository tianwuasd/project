"""Code publication previews, rejects divergence, and pushes only reviewed CT commits."""

import subprocess

import pytest

from chd_ct_v2.server import code_sync, network, ssh_upload, updater


def run(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def source(tmp_path, monkeypatch):
    remote, root = tmp_path / "remote.git", tmp_path / "code"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    root.mkdir()
    run(root, "init", "-b", "codex/ct-panel-v2")
    run(root, "config", "user.name", "Test")
    run(root, "config", "user.email", "test@example.invalid")
    project = root / "chd-ct-panel-v2"
    project.mkdir()
    (project / "code.py").write_text("old\n")
    run(root, "add", ".")
    run(root, "commit", "-m", "initial")
    run(root, "push", str(remote), "codex/ct-panel-v2")
    run(root, "remote", "add", "origin", updater.UPSTREAM)
    key = tmp_path / "synthetic-key"
    key.write_text("NOT A REAL KEY")
    runtime = tmp_path / "runtime"
    network.save(runtime, git_transport="github-ssh", ssh_key=str(key))
    original = ssh_upload.git
    calls = []

    def redirect(root, *args, **kwargs):
        calls.append(args)
        if args[:2] == ("ls-remote", "--get-url"):
            return subprocess.CompletedProcess(args, 0, network.SSH_UPSTREAM + "\n", "")
        return original(root, *(str(remote) if x == network.SSH_UPSTREAM else x for x in args), **kwargs)

    monkeypatch.setattr(ssh_upload, "git", redirect)
    return root, project, remote, runtime, calls


def change(root, path="chd-ct-panel-v2/code.py"):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("new\n")
    run(root, "add", "--", path)
    run(root, "commit", "-m", "change")


def test_preview_then_push_verified_commit(source):
    root, project, remote, runtime, calls = source
    change(root)
    before = run(remote, "rev-parse", "codex/ct-panel-v2")
    with network.use(network.load(runtime)):
        preview = code_sync.prepare(project)
        assert preview["files"] == ["chd-ct-panel-v2/code.py"]
        assert run(remote, "rev-parse", "codex/ct-panel-v2") == before
        code_sync.publish(preview, runtime)
        assert run(remote, "rev-parse", "codex/ct-panel-v2") == run(root, "rev-parse", "HEAD")
    assert len(list((runtime / "code-uploads").glob("*.json"))) == 1
    assert not any("--force" in x for x in calls)


@pytest.mark.parametrize(
    "path",
    [
        "AAOCA/code.py",
        "chd-ct-panel-v2/.env",
        "chd-ct-panel-v2/back/private.md",
        "chd-ct-panel-v2/model.pt",
    ],
)
def test_reject_unrelated_or_private_commit_even_if_later_deleted(source, path):
    root, project, _, runtime, calls = source
    change(root, path)
    run(root, "rm", "--", path)
    run(root, "commit", "-m", "remove")
    with network.use(network.load(runtime)), pytest.raises(ValueError, match="代码范围"):
        code_sync.prepare(project)
    assert not any(x[:1] == ("push",) for x in calls)


def test_dirty_and_changed_after_preview_cannot_push(source):
    root, project, _, runtime, calls = source
    change(root)
    with network.use(network.load(runtime)):
        preview = code_sync.prepare(project)
        (project / "code.py").write_text("unsaved review\n")
        with pytest.raises(ValueError, match="修改|改变"):
            code_sync.publish(preview, runtime)
    assert not any(x[:1] == ("push",) for x in calls)


def test_divergence_requires_update_without_force(source, monkeypatch):
    root, project, _, runtime, calls = source
    change(root)
    monkeypatch.setattr(updater, "ancestor", lambda *args: False)
    with network.use(network.load(runtime)), pytest.raises(ValueError, match="更新|分叉"):
        code_sync.prepare(project)
    assert not any(x[:1] == ("push",) for x in calls)
