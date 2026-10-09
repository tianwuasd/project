from copy import deepcopy

import pytest
import yaml

from chd_ct.server.launcher import resolve_manifest, select_gpu
from chd_ct.server.training import run


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


def test_raw_imagechd_cannot_be_misread_as_internal_labels(tmp_path):
    (tmp_path / "ct_1001_label.nii.gz").touch()
    with pytest.raises(ValueError, match="ImageCHD"):
        resolve_manifest(tmp_path)


def test_split_archive_is_not_a_dataset(tmp_path):
    (tmp_path / "ImageCHD_dataset.z01").touch()
    with pytest.raises(ValueError, match="分卷"):
        resolve_manifest(tmp_path)


def test_full_size_preflight_one_epoch_and_separate_output(tmp_path, monkeypatch):
    from chd_ct.server import training

    config = deepcopy(training.load_config("configs/server3090.yaml"))
    calls = []
    monkeypatch.setattr(
        training,
        "train_stage",
        lambda cfg, manifest, stage, output, **kw: calls.append((deepcopy(cfg), stage, kw)),
    )
    output = tmp_path / "preflight"
    run("configs/server3090.yaml", "manifest.csv", output, True)
    assert len(calls) == 8
    for cfg, stage, kw in calls:
        assert cfg["stages"][stage]["epochs"] == 1
        assert cfg["stages"][stage]["size"] == config["stages"][stage]["size"]
        assert kw == {"device": "cuda", "max_steps": 1}
    saved = yaml.safe_load((output / "effective-config.yaml").read_text())
    assert all(s["epochs"] == 1 for s in saved["stages"].values())
    with pytest.raises(FileExistsError):
        run("configs/server3090.yaml", "manifest.csv", output, True)


@pytest.mark.parametrize(
    "step_codes,expected,calls_count",
    [
        ([2], 2, 1),
        ([1], 1, 1),
        ([0, 1], 1, 2),
        ([0, 0, 0], 0, 3),
    ],
)
def test_train_only_continues_after_success(tmp_path, monkeypatch, step_codes, expected, calls_count):
    import json
    import sys
    from types import SimpleNamespace

    from chd_ct.server import launcher

    root = tmp_path / "code"
    root.mkdir()
    (root / "configs").mkdir()
    (root / "configs/server3090.yaml").write_text("placeholder", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("case_id,patient_id,split,image,label\na,a,train,a.nii,a.nii\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "ROOT", root)
    monkeypatch.setattr(launcher.platform, "system", lambda: "Linux")
    monkeypatch.setattr(launcher.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(launcher, "query_gpus", devices)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(launcher.shutil, "disk_usage", lambda p: SimpleNamespace(free=100 * 1024**3))
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *a: None))
    commands, codes = [], iter(step_codes)

    def execute(command, env, log):
        assert env["CUDA_VISIBLE_DEVICES"] == "GPU-b"
        commands.append([str(x) for x in command])
        return next(codes)

    monkeypatch.setattr(launcher, "run_logged", execute)
    code = launcher.main(
        [
            "--non-interactive",
            "--runtime",
            str(tmp_path / "runtime"),
            "--results",
            str(tmp_path / "results"),
            "--dataset",
            str(manifest),
            "--python",
            sys.executable,
            "--mode",
            "train",
        ]
    )
    assert code == expected
    assert len(commands) == calls_count
    state = json.loads(next((tmp_path / "results").glob("*/status.json")).read_text(encoding="utf-8"))
    assert state["status"] == ("passed" if expected == 0 else "partial" if expected == 2 else "failed")
    if calls_count == 3:
        assert "--preflight" in commands[1]
        assert "--preflight" not in commands[2]
        assert (
            commands[1][commands[1].index("--output") + 1] != commands[2][commands[2].index("--output") + 1]
        )


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
