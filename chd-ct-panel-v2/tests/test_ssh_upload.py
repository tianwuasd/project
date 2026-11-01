"""Actual local Git/LFS storage with remote calls redirected to a local bare repo."""

import json
import subprocess

import pytest
import test_server_upload

from chd_ct_v2.server import network, ssh_upload, upload

workflow = test_server_upload.workflow


@pytest.fixture
def local_remote(tmp_path, monkeypatch):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    key = tmp_path / "synthetic-key"
    key.write_text("NOT A REAL KEY")
    original = ssh_upload.git
    seen = []

    def redirected(root, *args, **kwargs):
        seen.append(args)
        if args[:2] == ("ls-remote", "--get-url"):
            return subprocess.CompletedProcess(args, 0, network.SSH_UPSTREAM + "\n", "")
        args = tuple(str(remote) if a == network.SSH_UPSTREAM else a for a in args)
        return original(root, *args, **kwargs)

    monkeypatch.setattr(ssh_upload, "git", redirected)
    return remote, key, seen


def configured_package(workflow, local_remote):
    job, runtime = workflow
    _, key, _ = local_remote
    network.save(runtime, git_transport="github-ssh", ssh_key=str(key))
    return upload.prepare_upload(job, runtime)


def test_git_lfs_roundtrip_and_same_commit_retry(workflow, local_remote, tmp_path):
    package = configured_package(workflow, local_remote)
    url = upload.upload_package(package)
    receipt = json.loads((package / "receipt.json").read_text())
    assert "/tree/chd-v2-artifacts/" in url
    remote, _, seen = local_remote
    head = subprocess.check_output(
        ["git", "--git-dir", str(remote), "rev-parse", receipt["ref"]], text=True
    ).strip()
    assert head == receipt["commit"]
    part = upload._json(package / "manifest.json")["assets"][0]
    pointer = subprocess.check_output(["git", "--git-dir", str(remote), "show", f"{head}:{part['name']}"])
    assert b"version https://git-lfs.github.com/spec/v1" in pointer
    assert part["sha256"].encode() in pointer
    # A clone followed by LFS fetch/checkout must restore the exact archive bytes.
    checkout = tmp_path / "download"
    subprocess.run(
        ["git", "clone", "--branch", receipt["branch"], str(remote), str(checkout)],
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "-C", str(checkout), "lfs", "pull"], check=True, capture_output=True)
    assert (checkout / part["name"]).read_bytes() == (package / part["name"]).read_bytes()
    pushes = sum(a[:1] == ("push",) for a in seen)
    assert upload.upload_package(package) == url
    assert sum(a[:1] == ("push",) for a in seen) == pushes
    assert not any("--force" in a or "--force-with-lease" in a for a in seen)
    assert (
        subprocess.run(
            ["git", "--git-dir", str(remote), "show-ref", "--verify", "refs/heads/main"], capture_output=True
        ).returncode
        != 0
    )


def test_uncertain_push_success_is_verified_on_retry(workflow, local_remote, monkeypatch):
    package = configured_package(workflow, local_remote)
    original = ssh_upload.git
    failed = False

    def uncertain(root, *args, **kwargs):
        nonlocal failed
        result = original(root, *args, **kwargs)
        if args[:1] == ("push",) and not failed:
            failed = True
            raise ValueError("synthetic lost response")
        return result

    monkeypatch.setattr(ssh_upload, "git", uncertain)
    with pytest.raises(ValueError, match="lost response"):
        upload.upload_package(package)
    assert not (package / "receipt.json").exists()
    assert upload.upload_package(package).startswith("https://github.com/")


def test_existing_foreign_ref_is_never_overwritten(workflow, local_remote, monkeypatch):
    package = configured_package(workflow, local_remote)
    monkeypatch.setattr(ssh_upload, "remote_head", lambda *args: "a" * 40)
    with pytest.raises(ValueError, match="不一致|冲突"):
        upload.upload_package(package)
    assert not any(a[:1] == ("push",) for a in local_remote[2])


def test_ssh_configuration_required_without_gh_fallback(workflow):
    package = upload.prepare_upload(*workflow)
    with pytest.raises(ValueError, match="SSH|github-ssh"):
        upload.upload_package(package)


def test_modified_package_is_rejected_before_git(workflow, local_remote):
    package = configured_package(workflow, local_remote)
    name = upload._json(package / "manifest.json")["assets"][0]["name"]
    (package / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="校验失败"):
        upload.upload_package(package)
    assert not local_remote[2]


def test_lfs_failure_keeps_branch_hidden_and_retries_same_commit(workflow, local_remote, monkeypatch):
    package = configured_package(workflow, local_remote)
    original = ssh_upload.git
    failed = False

    def fail_once(root, *args, **kwargs):
        nonlocal failed
        if args[:2] == ("lfs", "push") and not failed:
            failed = True
            raise ValueError("synthetic quota failure")
        return original(root, *args, **kwargs)

    monkeypatch.setattr(ssh_upload, "git", fail_once)
    with pytest.raises(ValueError, match="quota"):
        upload.upload_package(package)
    assert not (package / "receipt.json").exists()
    assert not any(a[:1] == ("push",) for a in local_remote[2])
    head = subprocess.check_output(
        ["git", "-C", str(package / "git-publication"), "rev-parse", "HEAD"], text=True
    ).strip()
    upload.upload_package(package)
    assert upload._json(package / "receipt.json")["commit"] == head


def test_retry_rejects_local_endpoint_override(workflow, local_remote, monkeypatch):
    package = configured_package(workflow, local_remote)
    monkeypatch.setattr(ssh_upload, "remote_head", lambda *args: "b" * 40)
    with pytest.raises(ValueError):
        upload.upload_package(package)
    subprocess.run(
        ["git", "-C", str(package / "git-publication"), "config", "lfs.url", "https://invalid.example"],
        check=True,
    )
    with pytest.raises(ValueError, match="额外网络"):
        upload.upload_package(package)


def test_git_failure_and_timeout_hide_secret_stderr(tmp_path, monkeypatch):
    key = tmp_path / "key"
    key.write_text("SYNTHETIC KEY")
    settings = network.save(tmp_path, git_transport="github-ssh", ssh_key=str(key))
    captured = []

    class Child:
        returncode, pid = 1, 123
        timed_out = False

        def __init__(self, *args, **kwargs):
            captured.append(self)
            self.reaped = False

        def communicate(self, timeout=None):
            if timeout and self.timed_out:
                raise subprocess.TimeoutExpired("git", timeout, stderr="synthetic-secret")
            self.reaped = True
            return "", "synthetic-secret"

        def poll(self):
            return 1

    monkeypatch.setattr(ssh_upload.subprocess, "Popen", Child)
    if hasattr(ssh_upload.os, "killpg"):
        monkeypatch.setattr(ssh_upload.os, "killpg", lambda *args: None)
    with network.use(settings):
        for timeout in (False, True):
            Child.timed_out = timeout
            with pytest.raises(ValueError) as error:
                ssh_upload.git(tmp_path, "status")
            assert "synthetic-secret" not in str(error.value)
            assert captured[-1].reaped
