"""Release upload uses synthetic artifacts and a fake gh; never a live network."""

import hashlib
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from chd_ct.server import upload


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def blob(path, content=b"synthetic weights"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    job, runtime = tmp_path / "finished_workflow", tmp_path / "runtime"
    train = job / "steps/02_train"
    steps = [{"name": "train", "output": str(train), "arguments": {}}]
    write(job / "plan.json", {"runtime": str(runtime), "directory": str(job), "steps": steps})
    write(job / "job_state.json", {
        "name": "测试任务", "status": "passed",
        "steps": [{"name": "train", "output": str(train), "status": "passed"}],
    })
    write(train / "train/models.json", {"status": "complete"})
    write(train / "train/report.json", {"status": "passed"})
    blob(train / "train/fold_0/blood2d.pt", b"best" * 100)
    blob(train / "train/fold_0/blood2d.last.pt", b"last" * 100)
    blob(train / "train/fold_0/other.pth", b"legacy weights")
    write(train / "train/fold_0/blood2d.history.json", [{"epoch": 1}])
    blob(train / "train/fold_0/stage_logs/blood2d.log", b"epoch=1 loss=0.1")
    blob(job / "launcher.log", b"finished")
    monkeypatch.setattr("chd_ct.server.inspection.job_active", lambda path: False)
    return job, runtime


def manifest(package):
    return json.loads((package / "manifest.json").read_text(encoding="utf-8"))


def test_cpu_settings_and_environment_probe_logs_are_included(workflow):
    job, runtime = workflow
    step = job / "steps/02_train"
    write(step / "cpu-allocation.json", {"threads": 4})
    write(step / ".cpu-runtime/evaluate.json", {"processes": 2})
    write(step / ".cpu-runtime/unrelated-private.json", {"private": True})
    write(step / "environment/0/report.json", {"packages": {"torch": "test"}})
    blob(step / "environment/0/run.log", b"probe complete")
    blob(step / "env/model.pt", b"not an artifact")
    package = upload.prepare_upload(job, runtime)
    names = {row["archive_path"] for row in manifest(package)["files"]}
    assert any(name.endswith("cpu-allocation.json") for name in names)
    assert any(name.endswith(".cpu-runtime/evaluate.json") for name in names)
    assert any(name.endswith("environment/0/report.json") for name in names)
    assert any(name.endswith("environment/0/run.log") for name in names)
    assert not any("unrelated-private" in name or "/env/" in name for name in names)


def test_demo_models_and_stage_reports_are_included(workflow):
    job, runtime = workflow
    blob(job / "steps/03_demo/demo/models/crop64.pt", b"demo best")
    blob(job / "steps/03_demo/demo/models/crop64.last.pt", b"demo last")
    write(job / "steps/02_train/train/fold_0/blood2d.environment.json", {"device": "cpu"})
    write(job / "steps/02_train/train/fold_0/blood2d.result.json", {"file": "blood2d.pt"})
    write(job / "steps/04_test/test/test-report.json", {"passed": True})
    names = {row["archive_path"] for row in manifest(upload.prepare_upload(job, runtime))["files"]}
    for suffix in (
        "demo/models/crop64.pt", "demo/models/crop64.last.pt",
        "blood2d.environment.json", "blood2d.result.json", "test/test-report.json",
    ):
        assert any(name.endswith(suffix) for name in names)


class FakeGh:
    def __init__(self):
        self.release = None
        self.assets = {}
        self.calls = []
        self.uploads = []
        self.fail_upload_once = None
        self.corrupt = False
        self.with_digest = True
        self.create_then_fail = False
        self.patch_then_fail = False

    def run(self, args, **kwargs):
        self.calls.append((args, kwargs))
        assert args[0] == "mock-gh"
        assert kwargs["timeout"] > 0
        assert kwargs["env"]["GH_HOST"] == "github.com"
        assert kwargs["env"]["GH_REPO"] == "tianwuasd/project"
        assert kwargs.get("shell", False) is False
        assert "gh-proxy" not in " ".join(args)
        assert "GH_DEBUG" not in kwargs["env"]
        command = args[1:]
        if command[:2] == ["auth", "status"]:
            assert command == ["auth", "status", "--hostname", "github.com"]
            result = ""
        elif command[0] == "api":
            assert command[1:3] == ["--hostname", "github.com"]
            endpoint = command[3]
            assert endpoint.startswith("repos/tianwuasd/project/")
            if "--method" in command:
                assert command[command.index("--method") + 1] == "PATCH"
                assert command[command.index("-f") + 1] == "make_latest=false"
                assert self.assets and all(a["state"] == "uploaded" for a in self.assets.values())
                self.release["draft"] = False
                if self.patch_then_fail:
                    self.patch_then_fail = False
                    return SimpleNamespace(returncode=1, stdout="", stderr="network lost")
                result = self.release
            elif "/assets?" in endpoint:
                result = list(self.assets.values())
            else:
                result = [self.release] if self.release else []
        elif command[:2] == ["release", "create"]:
            assert self.release is None
            assert command[command.index("--repo") + 1] == "github.com/tianwuasd/project"
            assert "--draft" in command
            assert "--latest=false" in command
            self.release = {
                "id": 123, "tag_name": command[2], "draft": True,
                "body": command[command.index("--notes") + 1],
            }
            if self.create_then_fail:
                self.create_then_fail = False
                return SimpleNamespace(returncode=1, stdout="", stderr="network lost")
            result = ""
        elif command[:2] == ["release", "upload"]:
            assert self.release["draft"]
            assert command[command.index("--repo") + 1] == "github.com/tianwuasd/project"
            path = Path(command[3])
            self.uploads.append(path.name)
            data = path.read_bytes()
            if self.fail_upload_once == path.name:
                self.fail_upload_once = None
                self.assets[path.name] = {"name": path.name, "size": 0, "state": "starter"}
                return SimpleNamespace(returncode=1, stdout="", stderr="do not expose secret")
            self.assets[path.name] = {
                "name": path.name, "size": len(data) + int(self.corrupt), "state": "uploaded",
                "digest": "sha256:" + hashlib.sha256(data).hexdigest() if self.with_digest else None,
            }
            result = ""
        else:
            raise AssertionError(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps(result) if not isinstance(result, str) else result,
                               stderr="")


def fake_gh(monkeypatch):
    fake = FakeGh()
    monkeypatch.setattr(upload.shutil, "which", lambda executable: "mock-gh")
    monkeypatch.setattr(upload.subprocess, "run", fake.run)
    monkeypatch.setenv("GH_HOST", "untrusted.example")
    monkeypatch.setenv("GH_REPO", "someone/else")
    monkeypatch.setenv("GH_DEBUG", "api")
    return fake


def test_collect_only_authorized_layout_and_referenced_models(workflow, tmp_path):
    job, runtime = workflow
    train = job / "steps/02_train/train"
    for relative in (
        "case.nii.gz", "data/case.pt", "cache/metrics.json", ".env", ".git/config",
        ".venv/config.yaml", "private/metrics.json", "secrets.json", "api_token.pt",
        "unrelated/secret.pth", "labels.json", "features.json", "diagnosis.json", "notes.txt",
    ):
        blob(train / relative, b"private or raw")
    external = tmp_path / "other_run/train"
    write(external / "models.json", {"status": "complete"})
    blob(external / "fold_1/best.pt", b"external-best")
    blob(external / "fold_1/last.pth", b"external-last")
    blob(external / "raw/ct.nii.gz", b"raw")
    config = tmp_path / "settings/chd.yaml"
    blob(config, b"epochs: 50\n")
    classifier = tmp_path / "diagnosis_model/classifier.json"
    write(classifier, {"trees": {}})
    plan = upload._json(job / "plan.json")
    plan["steps"][0]["arguments"].update(
        models=str(external), config=str(config), classifier=str(classifier),
    )
    plan["steps"].append({
        "name": "test", "output": str(job / "steps/03_test"),
        "arguments": {"models": str(external)}, "reuse": {"output": str(job / "steps/02_train")},
    })
    write(job / "plan.json", plan)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in job.rglob("*") if p.is_file()}
    package = upload.prepare_upload(job, runtime)
    rows = manifest(package)["files"]
    sources = [Path(row["source"]) for row in rows]
    assert len(sources) == len(set(sources))
    assert external / "fold_1/best.pt" in sources
    assert external / "fold_1/last.pth" in sources
    assert train / "fold_0/blood2d.pt" in sources
    assert train / "fold_0/blood2d.last.pt" in sources
    assert config in sources
    assert classifier in sources
    assert all(p.name not in {"labels.json", "features.json", "diagnosis.json", ".env", "api_token.pt"}
               and "private" not in p.parts and "raw" not in p.parts and "unrelated" not in p.parts
               for p in sources)
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before} == before
    assert package.parent == runtime / "uploads"
    assert not package.is_relative_to(job)


def test_parts_reassemble_tar_and_record_actual_content_hashes(workflow, monkeypatch, capsys):
    monkeypatch.setattr(upload, "CHUNK_BYTES", 1536)
    package = upload.prepare_upload(*workflow)
    record = manifest(package)
    assert len(record["assets"]) > 1
    data = b"".join((package / a["name"]).read_bytes() for a in record["assets"])
    assert all(0 < a["size"] <= 1536 for a in record["assets"])
    for asset in record["assets"]:
        assert hashlib.sha256((package / asset["name"]).read_bytes()).hexdigest() == asset["sha256"]
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        assert set(archive.getnames()) == {r["archive_path"] for r in record["files"]}
        for row in record["files"]:
            content = archive.extractfile(row["archive_path"]).read()
            assert len(content) == row["size"]
            assert hashlib.sha256(content).hexdigest() == row["sha256"]
    assert not (package / "artifacts.tar").exists()
    upload.show_preview(package)
    output = capsys.readouterr().out
    assert "blood2d.last.pt" in output and "github.com/tianwuasd/project" in output


@pytest.mark.parametrize("status", ["running", "queued", "failed", "interrupted", "pending_cpu"])
def test_reject_unfinished_workflow(workflow, status):
    job, runtime = workflow
    value = upload._json(job / "job_state.json")
    value["status"] = status
    write(job / "job_state.json", value)
    with pytest.raises(ValueError, match="已完成"):
        upload.prepare_upload(job, runtime)
    assert not (runtime / "uploads").exists()


def test_reject_live_process_even_if_saved_state_passed(workflow, monkeypatch):
    monkeypatch.setattr("chd_ct.server.inspection.job_active", lambda path: True)
    with pytest.raises(ValueError, match="存活"):
        upload.prepare_upload(*workflow)


def test_reject_wrong_runtime_and_nested_staging(workflow):
    job, runtime = workflow
    with pytest.raises(ValueError, match="原任务"):
        upload.prepare_upload(job, runtime / "other")
    plan = upload._json(job / "plan.json")
    plan["runtime"] = str(job / "runtime")
    write(job / "plan.json", plan)
    with pytest.raises(ValueError, match="之外"):
        upload.prepare_upload(job, job / "runtime")


def test_reject_symlink_and_path_escape(workflow, tmp_path):
    job, runtime = workflow
    root = job / "steps/02_train/train"
    target = tmp_path / "outside.pt"
    blob(target)
    link = root / "linked.pt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable on this host")
    with pytest.raises(ValueError, match="链接"):
        upload.prepare_upload(job, runtime)
    link.unlink()
    plan = upload._json(job / "plan.json")
    plan["steps"][0]["arguments"]["models"] = str(root / ".." / ".." / "outside")
    write(job / "plan.json", plan)
    with pytest.raises(ValueError, match=r"\.\."):
        upload.prepare_upload(job, runtime)


def test_external_models_still_running_are_rejected(workflow, tmp_path):
    job, runtime = workflow
    external = tmp_path / "active_model"
    write(external / "models.json", {"status": "running"})
    plan = upload._json(job / "plan.json")
    plan["steps"][0]["arguments"]["models"] = str(external)
    write(job / "plan.json", plan)
    with pytest.raises(ValueError, match="运行中"):
        upload.prepare_upload(job, runtime)


def test_mutation_during_stream_never_creates_uploadable_manifest(workflow, monkeypatch):
    job, runtime = workflow
    original = upload._HashReader.read
    modified = False

    def mutate(self, size=-1):
        nonlocal modified
        data = original(self, size)
        if not modified:
            modified = True
            with Path(self.stream.name).open("ab") as stream:
                stream.write(b"changed")
        return data

    monkeypatch.setattr(upload._HashReader, "read", mutate)
    with pytest.raises(ValueError, match="发生变化"):
        upload.prepare_upload(job, runtime)
    packages = list((runtime / "uploads").iterdir())
    assert len(packages) == 1
    assert not (packages[0] / "manifest.json").exists()
    assert (packages[0] / "preparation-failed.txt").is_file()


def test_changed_staged_part_rejected_before_any_network(workflow, monkeypatch):
    package = upload.prepare_upload(*workflow)
    part = package / manifest(package)["assets"][0]["name"]
    data = part.read_bytes()
    part.write_bytes(b"x" + data[1:])
    fake = fake_gh(monkeypatch)
    with pytest.raises(ValueError, match="校验失败"):
        upload.upload_package(package)
    assert not fake.calls


@pytest.mark.parametrize("digest", [True, False])
def test_upload_verify_publish_receipt_and_idempotent_retry(workflow, monkeypatch, digest):
    package = upload.prepare_upload(*workflow)
    fake = fake_gh(monkeypatch)
    fake.with_digest = digest
    url = upload.upload_package(package)
    assert url.startswith("https://github.com/tianwuasd/project/releases/tag/chd-artifacts-")
    assert not fake.release["draft"]
    assert upload._json(package / "receipt.json")["url"] == url
    assert not (workflow[0] / "uploads").exists()
    uploaded = list(fake.uploads)
    assert upload.upload_package(package) == url
    assert fake.uploads == uploaded
    assert sum(args[1:3] == ["release", "create"] for args, _ in fake.calls) == 1


def test_partial_failure_keeps_draft_and_resumes_only_missing_assets(workflow, monkeypatch):
    monkeypatch.setattr(upload, "CHUNK_BYTES", 6144)
    package = upload.prepare_upload(*workflow)
    fake = fake_gh(monkeypatch)
    fake.fail_upload_once = manifest(package)["assets"][1]["name"]
    with pytest.raises(ValueError, match="操作失败"):
        upload.upload_package(package)
    assert fake.release["draft"] and not (package / "receipt.json").exists()
    first = manifest(package)["assets"][0]["name"]
    assert fake.uploads.count(first) == 1
    upload.upload_package(package)
    assert fake.uploads.count(first) == 1
    assert not fake.release["draft"]


@pytest.mark.parametrize("stage", ["create_then_fail", "patch_then_fail"])
def test_uncertain_remote_success_reuses_same_release(workflow, monkeypatch, stage):
    package = upload.prepare_upload(*workflow)
    fake = fake_gh(monkeypatch)
    setattr(fake, stage, True)
    with pytest.raises(ValueError):
        upload.upload_package(package)
    upload.upload_package(package)
    assert sum(args[1:3] == ["release", "create"] for args, _ in fake.calls) == 1
    assert not fake.release["draft"]


def test_corrupt_remote_asset_never_publishes(workflow, monkeypatch):
    package = upload.prepare_upload(*workflow)
    fake = fake_gh(monkeypatch)
    fake.corrupt = True
    with pytest.raises(ValueError, match="校验未通过"):
        upload.upload_package(package)
    assert fake.release["draft"] and not (package / "receipt.json").exists()
    fake.corrupt = False
    upload.upload_package(package)
    assert not fake.release["draft"]


def test_unowned_release_never_modified(workflow, monkeypatch):
    package = upload.prepare_upload(*workflow)
    fake = fake_gh(monkeypatch)
    fake.release = {"id": 456, "tag_name": manifest(package)["release_tag"], "draft": True, "body": "unrelated"}
    with pytest.raises(ValueError, match="不属于"):
        upload.upload_package(package)
    assert not fake.uploads and fake.release["draft"]


def test_missing_auth_and_timeout_errors_do_not_leak_stderr(workflow, monkeypatch):
    package = upload.prepare_upload(*workflow)
    monkeypatch.setattr(upload.shutil, "which", lambda executable: None)
    with pytest.raises(ValueError, match="安装"):
        upload.upload_package(package)
    monkeypatch.setattr(upload.shutil, "which", lambda executable: "mock-gh")
    monkeypatch.setattr(upload.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=1, stdout="", stderr="Authorization: Bearer private-token"))
    with pytest.raises(ValueError, match="gh auth login") as error:
        upload.upload_package(package)
    assert "private-token" not in str(error.value)

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"], stderr="private-token")

    monkeypatch.setattr(upload.subprocess, "run", timeout)
    with pytest.raises(ValueError, match="超时") as error:
        upload.upload_package(package)
    assert "private-token" not in str(error.value)


def test_part_asset_count_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(upload, "MAX_ASSETS", 3)
    writer = upload._SplitWriter(tmp_path, 4)
    try:
        with pytest.raises(ValueError, match="1000"):
            writer.write(b"012345678")
    finally:
        writer.close()
    assert len(writer.assets) == 2
    assert sum(a["size"] for a in writer.assets) == 8


def test_module_import_needs_no_site_packages():
    source = Path(upload.__file__).parents[2]
    env = dict(os.environ, PYTHONPATH=str(source))
    result = subprocess.run(
        [os.sys.executable, "-S", "-c", "import chd_ct.server.upload"],
        env=env, capture_output=True, text=True, check=False, timeout=15,
    )
    assert result.returncode == 0, result.stderr
