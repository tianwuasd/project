"""Explicit network routes stay inside Git/LFS children and retain authentication boundaries."""

import importlib
import importlib.util
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest
import test_server_update as update_tests

from chd_ct_v2.server import launcher, updater, upload

REAL_GIT = updater.git
local_git = update_tests.local_git
repository = update_tests.repository


def module():
    assert importlib.util.find_spec("chd_ct_v2.server.network"), "runtime network configuration is missing"
    return importlib.import_module("chd_ct_v2.server.network")


def test_settings_roundtrip_is_local_and_read_does_not_create_runtime(tmp_path):
    network = module()
    runtime = tmp_path / "runtime"
    assert network.load(runtime)["git_transport"] == "github-ssh"
    assert not runtime.exists()
    original = dict(os.environ)
    settings = network.save(runtime, proxy="http://127.0.0.1:17891", git_transport="github-https")
    assert network.load(runtime) == settings
    assert dict(os.environ) == original
    assert set(p.name for p in runtime.iterdir()) == {"network.json"}


@pytest.mark.parametrize("url", [
    "http://user:secret@127.0.0.1:17891", "http://127.0.0.1:17891/?token=secret",
    "https://example.com/subscription", "socks5://127.0.0.1:17891", "http://127.0.0.1:0",
    "http://127.0.0.1:17891\nOTHER=secret",
])
def test_invalid_proxy_cannot_persist_credentials(tmp_path, url):
    network = module()
    with pytest.raises(ValueError):
        network.save(tmp_path, proxy=url, git_transport="github-https")
    assert not (tmp_path / "network.json").exists()


def test_child_proxy_overrides_bypass_without_changing_parent(tmp_path, monkeypatch):
    network = module()
    settings = network.save(tmp_path, proxy="http://127.0.0.1:17891", git_transport="github-https")
    monkeypatch.setenv("HTTPS_PROXY", "http://old.invalid:1")
    monkeypatch.setenv("NO_PROXY", "*")
    parent = dict(os.environ)
    with network.use(settings):
        child = network.child_env()
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            assert child[name] == settings["proxy"]
        assert child["NO_PROXY"] == child["no_proxy"] == ""
        assert dict(os.environ) == parent
    assert network.child_env()["HTTPS_PROXY"] == "http://old.invalid:1"


def test_ssh_route_uses_explicit_key_connect_proxy_and_strict_host_check(tmp_path, monkeypatch):
    network = module()
    key = tmp_path / "key with spaces"
    key.write_text("TEST PLACEHOLDER NOT A KEY")
    monkeypatch.setattr(network.shutil, "which", lambda name: "/usr/bin/" + name)
    settings = network.save(tmp_path, proxy="http://127.0.0.1:17891",
                            git_transport="github-ssh", ssh_key=str(key))
    with network.use(settings):
        command = shlex.split(network.child_env(ssh=True)["GIT_SSH_COMMAND"])
        assert str(key) in command
        assert "StrictHostKeyChecking=yes" in command
        assert "BatchMode=yes" in command
        assert "443" in command
        assert any("ProxyCommand=" in arg and "17891" in arg and "connect" in arg for arg in command)
    assert network.routes(settings) == (network.SSH_UPSTREAM,)


def test_private_https_never_uses_mirrors(tmp_path):
    network = module()
    settings = network.save(tmp_path, proxy="http://127.0.0.1:17891", git_transport="github-https")
    assert network.routes(settings) == (updater.UPSTREAM,)


def test_upload_ssh_gets_scoped_proxy_key_and_no_api_token(tmp_path, monkeypatch):
    from chd_ct_v2.server import ssh_upload

    network = module()
    key = tmp_path / "test-key"
    key.write_text("SYNTHETIC KEY")
    settings = network.save(tmp_path, proxy="http://127.0.0.1:17891", git_transport="github-ssh",
                            ssh_key=str(key))
    captured = {}
    monkeypatch.setenv("GH_TOKEN", "synthetic-test-token")
    monkeypatch.setattr(ssh_upload.shutil, "which", lambda name: "mock-" + name)

    class Child:
        returncode = 0

        def __init__(self, command, **kwargs):
            captured.update(command=command, **kwargs)

        def communicate(self, **kwargs):
            return "ok", ""

    monkeypatch.setattr(ssh_upload.subprocess, "Popen", Child)
    with network.use(settings):
        assert ssh_upload.git(tmp_path, "status").stdout == "ok"
    assert captured["env"]["HTTPS_PROXY"] == settings["proxy"]
    assert "GH_TOKEN" not in captured["env"]
    assert str(key) in captured["env"]["GIT_SSH_COMMAND"]
    assert captured["env"]["GIT_CONFIG_GLOBAL"] == os.devnull


def test_network_cli_can_configure_and_show_without_ml_or_starting_jobs(tmp_path):
    runtime = tmp_path / "runtime"
    arguments = ["--runtime", str(runtime), "--configure-network", "--github-proxy",
                 "http://127.0.0.1:17891", "--github-transport", "github-https"]
    assert launcher.main(arguments) == 0
    original = (runtime / "network.json").read_bytes()
    assert launcher.main(["--runtime", str(runtime), "--show-network"]) == 0
    assert (runtime / "network.json").read_bytes() == original


def test_network_configuration_cannot_be_mixed_with_task_launch(tmp_path):
    assert launcher.main(["--configure-network", "--tasks", "train", "--runtime", str(tmp_path)]) == 1


@pytest.mark.parametrize("transport", ["github-https", "github-ssh"])
def test_authenticated_update_only_fetches_official_route_and_preserves_sparse_checkout(
    repository, tmp_path, monkeypatch, transport
):
    network = module()
    repo, remote, old, new, _ = repository
    root = Path(repo["root"])
    local_git(root, "sparse-checkout", "set", "--cone", "chd-ct-panel-v2")
    runtime = tmp_path / "runtime"
    config = network.save(runtime, proxy="http://127.0.0.1:17891", git_transport=transport,
                          ssh_key="/synthetic/key")
    official = network.routes(config)[0]
    calls = []

    def offline(root, *args, **kwargs):
        calls.append(args)
        if args[:2] == ("ls-remote", "--get-url"):
            return subprocess.CompletedProcess(args, 0, official + "\n", "")
        if args[0] == "ls-remote":
            return subprocess.CompletedProcess(args, 0, new + "\trefs/heads/codex/ct-panel-v2\n", "")
        if args[0] == "fetch":
            assert args[4] == official
            args = (*args[:4], str(remote), *args[5:])
        return REAL_GIT(root, *args, **kwargs)

    monkeypatch.setattr(updater, "git", offline)
    assert updater.update(repo, runtime)
    assert local_git(root, "rev-parse", "HEAD") == new
    assert local_git(root, "sparse-checkout", "list") == "chd-ct-panel-v2"
    assert [c[4] for c in calls if c[0] == "fetch"] == [official]
    report = json.loads(next(runtime.glob("updates/*/update.json")).read_text())
    assert report["before"] == old and report["after"] == new
    assert report["network"] == {"proxy": config["proxy"], "git_transport": transport}


def test_official_url_rewrite_rejected_before_network_or_merge(repository, tmp_path, monkeypatch):
    network = module()
    repo, _, old, _, _ = repository
    runtime = tmp_path / "runtime"
    network.save(runtime, git_transport="github-https")
    calls = []

    def offline(root, *args, **kwargs):
        calls.append(args)
        if args[:2] == ("ls-remote", "--get-url"):
            return subprocess.CompletedProcess(args, 0, "https://mirror.invalid/private.git\n", "")
        assert args[0] not in {"fetch", "merge"}
        return REAL_GIT(root, *args, **kwargs)

    monkeypatch.setattr(updater, "git", offline)
    with pytest.raises(ValueError, match="URL 重写"):
        updater.update(repo, runtime)
    assert local_git(Path(repo["root"]), "rev-parse", "HEAD") == old
    assert not any(c[0] == "ls-remote" and c[1] != "--get-url" for c in calls)


def test_upload_scoped_settings_come_from_package_runtime(tmp_path, monkeypatch):
    network = module()
    runtime = tmp_path / "original-runtime"
    network.save(runtime, proxy="http://127.0.0.1:17891", git_transport="github-https")
    package = tmp_path / "package"
    monkeypatch.setattr(upload, "_validate_package", lambda p: (p, {"runtime": str(runtime)}, []))
    observed = []
    monkeypatch.setattr(upload, "_publish", lambda *args: observed.append(network.current()) or "test-url")
    assert upload.upload_package(package) == "test-url"
    assert observed[0]["proxy"] == "http://127.0.0.1:17891"
    assert network.current()["proxy"] == ""


def test_missing_proxy_helper_fails_without_direct_ssh_fallback(tmp_path, monkeypatch):
    network = module()
    key = tmp_path / "key"
    key.write_text("synthetic")
    settings = network.save(tmp_path, proxy="http://127.0.0.1:17891", git_transport="github-ssh",
                            ssh_key=str(key))
    monkeypatch.setattr(network.shutil, "which", lambda _: None)
    with network.use(settings), pytest.raises(ValueError, match="OpenBSD nc"):
        network.child_env(ssh=True)


@pytest.mark.parametrize("transport,url", [("github-https", updater.UPSTREAM),
                                          ("public-mirrors", updater.MIRRORS[0])])
def test_explicit_proxy_overrides_existing_url_specific_git_proxy(repository, tmp_path, monkeypatch, transport, url):
    network = module()
    repo, _, _, _, _ = repository
    root = Path(repo["root"])
    monkeypatch.setattr(updater, "MIRRORS", (url, updater.UPSTREAM))
    local_git(root, "config", "http." + url + ".proxy", "http://old.invalid:1")
    settings = network.save(tmp_path / "runtime", proxy="http://127.0.0.1:17891",
                            git_transport=transport)
    with network.use(settings):
        resolved = REAL_GIT(root, "config", "--get-urlmatch", "http.proxy", url).stdout.strip()
    assert resolved == settings["proxy"]
    assert local_git(root, "config", "--get-urlmatch", "http.proxy", url) == "http://old.invalid:1"


@pytest.mark.parametrize("cancel", ["b", "q", KeyboardInterrupt])
@pytest.mark.parametrize("step", [0, 1, 2])
def test_network_form_cancel_preserves_config(tmp_path, monkeypatch, step, cancel):
    network = module()
    network.save(tmp_path, git_transport="github-ssh", ssh_key="/original/key")
    original = (tmp_path / "network.json").read_bytes()
    answers = iter(["", "github-ssh", ""][:step] + [cancel])

    def answer(_):
        value = next(answers)
        if value is KeyboardInterrupt:
            raise KeyboardInterrupt
        return value

    monkeypatch.setattr("builtins.input", answer)
    from chd_ct_v2.server.prompts import PanelBack, UserExit
    with pytest.raises((PanelBack, UserExit, KeyboardInterrupt)):
        network.configure_interactive(tmp_path)
    assert (tmp_path / "network.json").read_bytes() == original


def test_corrupt_configuration_has_recovery_guidance_without_overwrite(tmp_path):
    network = module()
    path = tmp_path / "network.json"
    path.write_bytes(b"{broken")
    with pytest.raises(ValueError, match="备份并改名"):
        network.save(tmp_path, git_transport="github-https")
    assert path.read_bytes() == b"{broken"


@pytest.mark.parametrize("transport,expected", [("github-https", "synthetic-helper"),
                                               ("public-mirrors", "")])
def test_only_official_mode_retains_git_credential_helper(repository, tmp_path, transport, expected):
    network = module()
    repo, _, _, _, _ = repository
    root = Path(repo["root"])
    local_git(root, "config", "credential.helper", "synthetic-helper")
    with network.use(network.save(tmp_path / "runtime", git_transport=transport)):
        value = REAL_GIT(root, "config", "--get", "credential.helper").stdout.strip()
    assert value == expected
