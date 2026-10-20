"""Publication requires an explicit choice after a concrete package is prepared."""

import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from chd_ct.server import launcher, results, sharing


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def publication(tmp_path, monkeypatch):
    job, runtime = tmp_path / "job", tmp_path / "runtime"
    package = runtime / "uploads/package"
    save(job / "plan.json", {"runtime": str(runtime), "directory": str(job), "steps": []})
    save(job / "job_state.json", {"status": "passed", "steps": []})
    save(package / "manifest.json", {"job_directory": str(job), "runtime": str(runtime)})
    calls = []
    locked = []

    @contextmanager
    def lock(_):
        locked.append(True)
        yield
        locked.clear()

    def prepare(*args):
        assert locked
        calls.append("prepare")
        return package

    def preview(*args):
        assert locked
        calls.append("preview")

    def upload(*args):
        assert locked
        calls.append("publish")
        return "https://github.com/tianwuasd/project/releases/tag/result-test"

    monkeypatch.setitem(
        sys.modules,
        "chd_ct.server.upload",
        SimpleNamespace(
            prepare_upload=prepare,
            show_preview=preview,
            upload_package=upload,
        ),
    )
    # Imported submodules can already be attributes after another upload test ran.
    import chd_ct.server

    monkeypatch.setattr(chd_ct.server, "upload", sys.modules["chd_ct.server.upload"], raising=False)
    monkeypatch.setattr(sharing.jobs, "acquire_lock", lock)
    monkeypatch.setattr(sharing.inspection, "job_active", lambda _: False)
    return job, runtime, package, calls


def test_noninteractive_prepares_without_publishing(publication, capsys):
    job, runtime, package, calls = publication
    args = launcher.make_parser().parse_args(["--upload-job", str(job)])
    assert sharing.run_upload(args, {"runtime": str(runtime)}, False) == 0
    assert calls == ["prepare", "preview"]
    assert not (job / "uploads").exists()
    assert "--upload-package" in capsys.readouterr().out


def test_confirmed_retry_publishes_package_without_repacking(publication, capsys):
    job, runtime, package, calls = publication
    args = launcher.make_parser().parse_args(["--upload-package", str(package), "--yes-upload"])
    assert sharing.run_upload(args, {"runtime": str(runtime)}, False) == 0
    assert calls == ["preview", "publish"]
    assert (job / "uploads/package.json").is_file()
    results.show_results(job)
    assert "https://github.com/tianwuasd/project/releases/tag/result-test" in capsys.readouterr().out


def test_declined_preview_and_active_target_never_publish(publication, monkeypatch):
    job, runtime, _, calls = publication
    args = launcher.make_parser().parse_args(["--upload-job", str(job)])
    monkeypatch.setattr("builtins.input", lambda _: "n")
    sharing.run_upload(args, {"runtime": str(runtime)}, True)
    assert calls == ["prepare", "preview"]
    monkeypatch.setattr(sharing.inspection, "job_active", lambda _: True)
    with pytest.raises(ValueError, match="仍在运行"):
        sharing.run_upload(args, {"runtime": str(runtime)}, True)
    assert calls == ["prepare", "preview"]


@pytest.mark.parametrize(
    "argv",
    [
        ["--yes-upload"],
        ["--append-tasks", "test"],
        ["--upload-job", "latest", "--query", "results"],
        ["--upload-job", "latest", "--resume-job"],
        ["--upload-package", "pack", "--stop"],
    ],
)
def test_upload_cli_rejects_mixed_actions(argv):
    assert launcher.main(argv) == 1
