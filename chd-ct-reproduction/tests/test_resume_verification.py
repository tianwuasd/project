"""Full hash validation, bounded readers and observable/cancellable progress."""

import hashlib
import os
import threading
import time
from pathlib import Path

import pytest

from chd_ct import cpu_runtime
from chd_ct.server import verification as v
from chd_ct.server.resume import snapshot, tree_hashes


def fixture_record(tmp_path, count=3):
    cache = tmp_path / "cache"
    cache.mkdir()
    for n in range(count):
        (cache / f"case{n}.bin").write_bytes(bytes([n]) * 512)
    step = dict(output=str(tmp_path / "step"), arguments=dict(task="test", prepared=str(cache)))
    return cache, snapshot(step)


def threads(monkeypatch, number):
    monkeypatch.setattr(v, "storage_kind", lambda _: "ssd")
    monkeypatch.setattr(
        v, "choose_threads", lambda *args: dict(workers=number, cpus=8, budget=6, storage_cap=number)
    )


def test_repeated_steps_hash_each_file_once_without_writing(tmp_path, monkeypatch, capsys):
    cache, record = fixture_record(tmp_path)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in cache.iterdir()}
    calls = []
    original = v.hash_one

    def counted(path, *args):
        calls.append(path)
        return original(path, *args)

    monkeypatch.setattr(v, "hash_one", counted)
    result = v.verify_snapshots([(name, record) for name in ("preprocess", "smoke", "train", "test")])
    assert result["files"] == len(calls) == len(set(calls)) == 3
    assert result["bytes"] == 1536 and result["references"] == 12
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in cache.iterdir()}
    output = capsys.readouterr().out
    assert "合并 9 次重复" in output and "读取剩余" in output and "校验通过" in output


@pytest.mark.parametrize(
    "cpus,idle,memory,kind,count,total,want",
    [
        (256, 0.97, 128 * 1024**3, "network", 100, 100 * v.MIB, 4),
        (256, 0.97, 128 * 1024**3, "ssd", 100, 100 * v.MIB, 8),
        (256, 0.97, 128 * 1024**3, "hdd", 100, 100 * v.MIB, 1),
        (256, 0.97, 128 * 1024**3, "unknown", 100, 100 * v.MIB, 2),
        (2, 0.9, 1024**3, "ssd", 100, 100 * v.MIB, 1),
        (16, 0.1, 1024**3, "ssd", 100, 100 * v.MIB, 1),
        (16, 0.9, 80 * v.MIB, "ssd", 100, 100 * v.MIB, 1),
        (16, None, 1024**3, "ssd", 3, 100 * v.MIB, 3),
        (256, 1, 128 * 1024**3, "network", 100, v.MIB, 1),
    ],
)
def test_auto_threads_respect_load_storage_memory_and_size(cpus, idle, memory, kind, count, total, want):
    hardware = dict(cpus=cpus, idle_fraction=idle, memory_available=memory)
    assert v.choose_threads(hardware, {kind}, count, total)["workers"] == want


def test_mount_selection_uses_longest_real_mount(tmp_path):
    path = tmp_path / "data nas/cache"
    mount = str(tmp_path / "data nas").replace("\\", "/").replace(" ", r"\040")
    lines = [
        f"1 0 8:0 / {tmp_path.as_posix()} rw - ext4 /dev/sda rw",
        f"2 1 0:50 / {mount} rw - nfs4 server:/data rw",
    ]
    assert v.storage_kind(path, lines) == "network"
    assert v.storage_kind(path, []) == "unknown"


def test_tree_membership_matches_legacy_including_single_hidden_file(tmp_path):
    for name in (
        "ok.bin",
        ".hidden",
        "file.log",
        "file.tmp",
        "a.interrupted-1/bin",
        ".cache/blob",
        "sub/data",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"content")
    assert v.tree_files(tmp_path) == set(tree_hashes(tmp_path))
    assert v.tree_files(tmp_path / ".hidden") == set(tree_hashes(tmp_path / ".hidden"))


def test_real_hash_workers_overlap_and_are_joined(tmp_path, monkeypatch):
    _, record = fixture_record(tmp_path, 6)
    threads(monkeypatch, 3)
    barrier = threading.Barrier(3)
    active, maximum = 0, 0
    guard = threading.Lock()
    original = v.hash_one

    def concurrent(*args):
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        try:
            barrier.wait(timeout=5)
            return original(*args)
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(v, "hash_one", concurrent)
    v.verify_snapshots([("train", record)], False)
    assert maximum == 3 and active == 0
    assert not any(t.name.startswith("chd-verify-") for t in threading.enumerate())


def test_same_size_and_mtime_change_still_gets_full_hash(tmp_path):
    cache, record = fixture_record(tmp_path)
    path = cache / "case0.bin"
    stamp = path.stat()
    path.write_bytes(b"x" * stamp.st_size)
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with pytest.raises(ValueError, match="改变"):
        v.verify_snapshots([("train", record)], False)


@pytest.mark.parametrize("mutation", ["new", "remove", "rewrite"])
def test_changes_after_hashing_are_rejected(tmp_path, monkeypatch, mutation):
    cache, record = fixture_record(tmp_path)
    original = v.read_hashes

    def changed(*args):
        original(*args)
        if mutation == "new":
            (cache / "new.bin").write_bytes(b"new")
        elif mutation == "remove":
            (cache / "case0.bin").unlink()
        else:
            (cache / "case0.bin").write_bytes(b"new")

    monkeypatch.setattr(v, "read_hashes", changed)
    with pytest.raises(ValueError, match="改变"):
        v.verify_snapshots([("train", record)], False)


def test_legacy_record_and_conflicting_step_records(tmp_path):
    _, record = fixture_record(tmp_path)
    legacy = {key: record[key] for key in ("inputs", "outputs")}
    v.verify_snapshots([("old", legacy)], False)
    other = dict(legacy, inputs={key: "0" * 64 for key in legacy["inputs"]})
    with pytest.raises(ValueError, match="记录冲突"):
        v.verify_snapshots([("old", legacy), ("new", other)], False)


def test_interrupt_joins_readers_and_progress_without_success(tmp_path, monkeypatch, capsys):
    _, record = fixture_record(tmp_path)
    threads(monkeypatch, 2)
    entered = threading.Event()

    def slow(path, stamp, progress, cancelled):
        entered.set()
        assert cancelled.wait(5), "cancellation was never delivered"
        raise InterruptedError()

    def interrupted(*args, **kwargs):
        assert entered.wait(5)
        raise KeyboardInterrupt()

    monkeypatch.setattr(v, "hash_one", slow)
    monkeypatch.setattr(v, "wait", interrupted)
    with pytest.raises(KeyboardInterrupt):
        v.verify_snapshots([("train", record)])
    output = capsys.readouterr().out
    assert "校验中断" in output and "校验通过" not in output
    assert not any(t.name.startswith("chd-verify-") for t in threading.enumerate())


def test_heartbeat_progress_during_one_large_file_and_scan(capsys):
    with v.ReadProgress(interval=0.02) as progress:
        time.sleep(0.06)
        progress.reading(16 * v.MIB, 1, dict(workers=1, cpus=2, budget=1, storage_cap=2), {"unknown"}, 1)
        progress.advance(4 * v.MIB)
        time.sleep(0.06)
    output = capsys.readouterr().out
    assert output.count("扫描文件清单") >= 2
    assert "25.0%" in output and "4.0/16.0 MiB" in output


def test_hash_detects_modification_during_read(tmp_path):
    path = tmp_path / "blob"
    path.write_bytes(b"0" * (v.BLOCK + 10))

    class MutatingProgress(v.ReadProgress):
        def advance(self, size):
            super().advance(size)
            path.write_bytes(b"changed")

    with pytest.raises(ValueError, match="改变"):
        v.hash_one(path, v.fingerprint(path), MutatingProgress(False), threading.Event())


def test_new_session_always_rereads_content(tmp_path, monkeypatch):
    _, record = fixture_record(tmp_path, 1)
    counts = []
    original = v.hash_one

    def counted(path, *args):
        counts.append(path)
        return original(path, *args)

    monkeypatch.setattr(v, "hash_one", counted)
    monkeypatch.setattr(cpu_runtime, "inspect", lambda: dict(cpus=1, memory_available=1024**3))
    for _ in range(2):
        v.verify_snapshots([("train", record)], False)
    assert len(counts) == 2
    assert hashlib.sha256(counts[0].read_bytes()).hexdigest() in record["inputs"].values()


def test_mixed_storage_budget_does_not_collapse_to_one():
    hardware = dict(cpus=256, idle_fraction=0.97, memory_available=128 * 1024**3)
    assert v.choose_threads(hardware, {"hdd", "network"}, 1111, 135 * 1024**3)["workers"] == 5


def test_read_scheduler_obeys_per_device_limits_without_head_of_line_blocking(tmp_path, monkeypatch):
    names = [str(tmp_path / f"hdd{i}") for i in range(2)] + [str(tmp_path / f"nas{i}") for i in range(4)]
    stamps = {name: (1 if Path(name).name.startswith("hdd") else 2, 0, 64 * v.MIB, 0, 0) for name in names}
    expected = dict.fromkeys(names, "digest")
    active, peak = {1: 0, 2: 0}, {1: 0, 2: 0}
    lock, ready = threading.Lock(), threading.Event()

    def read(path, stamp, progress, cancelled):
        device = stamp[0]
        with lock:
            active[device] += 1
            peak[device] = max(peak[device], active[device])
            if active == {1: 1, 2: 4}:
                ready.set()
        try:
            assert ready.wait(3), "NAS readers were blocked behind HDD readers"
            return "digest"
        finally:
            with lock:
                active[device] -= 1

    monkeypatch.setattr(v, "hash_one", read)
    v.read_hashes(names, stamps, 5, v.ReadProgress(False), expected, {1: 1, 2: 4})
    assert peak == {1: 1, 2: 4} and active == {1: 0, 2: 0}


def test_simple_large_file_metadata_only_is_disclosed_and_full_detects_corruption(tmp_path, monkeypatch):
    directory = tmp_path / "models"
    directory.mkdir()
    large = directory / "best.pt"
    large.write_bytes(b"0" * (v.SIMPLE_HASH_LIMIT + 1))
    manifest = directory / "models.json"
    manifest.write_text('{"status":"passed"}')
    record = dict(inputs=tree_hashes(directory), outputs={}, input_roots=[str(directory)])
    with large.open("r+b") as stream:
        stream.write(b"x")
    calls = []
    original = v.hash_one

    def counted(path, *args):
        calls.append(path.name)
        return original(path, *args)

    monkeypatch.setattr(v, "hash_one", counted)
    result = v.verify_snapshots([("models", record)], False, mode="simple")
    assert calls == ["models.json"]
    assert result["metadata_only_files"] == 1 and result["hashed_files"] == 1
    with pytest.raises(ValueError, match="改变"):
        v.verify_snapshots([("models", record)], False, mode="full")
