"""Exercise real preprocessing and the typed failure protocol with synthetic CT."""

from pathlib import Path

import pytest
import test_chd_main

from chd_ct.imagechd import preprocess
from chd_ct.imagechd.cache_reuse import CacheReuseError
from chd_ct.recovery import read_json, write_json
from chd_ct.server import jobs


@pytest.fixture
def raw(tmp_path):
    return test_chd_main.raw_chd.__wrapped__(tmp_path)


def contents(folder):
    return {
        str(p.relative_to(folder)): p.read_bytes()
        for p in folder.rglob("*")
        if p.is_file() and p.name != ".writer.lock"
    }


@pytest.mark.parametrize(
    "damage",
    [
        "options",
        "source",
        "metadata",
        "cache",
        "missing",
        "unknown",
        "normalization",
        "case_id",
        "partial_geometry",
    ],
)
def test_reuse_failure_preserves_old_cache_and_reports_specific_cause(raw, tmp_path, damage):
    cache = tmp_path / "prepared"
    preprocess.prepare(raw, cache, size=16)
    size = 16
    if damage == "options":
        size = 32
    elif damage == "source":
        with (raw / "ct_3_label.nii.gz").open("ab") as stream:
            stream.write(b"modified")
    elif damage == "metadata":
        (cache / "dataset.json").write_text("bad json")
    elif damage == "cache":
        (cache / "ct_3.npz").write_bytes(b"bad zip")
    elif damage == "missing":
        (cache / "ct_3.npz").unlink()
    elif damage == "normalization":
        data = read_json(cache / "dataset.json")
        data.pop("normalization")
        write_json(cache / "dataset.json", data)
    elif damage in {"case_id", "partial_geometry"}:
        record = cache / ".resume" / "ct_3.json"
        data = read_json(record)
        data["row"].pop("case_id" if damage == "case_id" else "native_affine")
        write_json(record, data)
        if damage == "partial_geometry":
            (cache / "dataset.json").unlink()
    else:
        cache = tmp_path / "unrelated"
        cache.mkdir()
        (cache / "keep.txt").write_text("keep")
    before = contents(cache)
    report = tmp_path / "failure.json"
    assert (
        preprocess.main(
            [
                "--input",
                str(raw),
                "--output",
                str(cache),
                "--size",
                str(size),
                "--resume",
                "--reuse-failure-report",
                str(report),
            ]
        )
        == 78
    )
    error = read_json(report)
    assert error["kind"] == "cache_reuse_failed" and error["reason"]
    if damage == "options":
        assert "options.size" in error["reason"]
    assert before == contents(cache)


def test_raw_data_error_does_not_request_fallback(tmp_path):
    raw = tmp_path / "empty"
    raw.mkdir()
    report = tmp_path / "failure.json"
    write_json(report, {"kind": "cache_reuse_failed", "reason": "stale"})
    with pytest.raises(ValueError) as error:
        preprocess.main(
            ["--input", str(raw), "--output", str(tmp_path / "out"), "--reuse-failure-report", str(report)]
        )
    assert not isinstance(error.value, CacheReuseError)
    assert not report.exists()


@pytest.mark.parametrize("error", [PermissionError("permission denied"), OSError("I/O error")])
def test_legacy_cache_io_errors_do_not_request_fallback(raw, tmp_path, monkeypatch, error):
    cache = tmp_path / "prepared"
    preprocess.prepare(raw, cache, size=16)
    for sidecar in (cache / ".resume").glob("*.json"):
        sidecar.unlink()
    before = contents(cache)
    report = tmp_path / "failure.json"

    def denied(*a, **kw):
        raise error

    monkeypatch.setattr(preprocess.np, "load", denied)
    with pytest.raises(type(error)):
        preprocess.main(
            [
                "--input",
                str(raw),
                "--output",
                str(cache),
                "--size",
                "16",
                "--resume",
                "--reuse-failure-report",
                str(report),
            ]
        )
    assert not report.exists() and contents(cache) == before


def test_real_preprocess_fallback_then_training_reads_new_cache(raw, tmp_path, monkeypatch):
    from test_cache_selection import workflow

    from chd_ct.server import imagechd

    plan, old = workflow(tmp_path)
    (old / "keep.txt").unlink()
    preprocess.prepare(raw, old, size=16, resume=True)
    before = contents(old)
    seen = []

    def run(args, profile, output_dir, **kwargs):
        seen.append(args.task)
        output_dir.mkdir()
        if args.task == "preprocess":
            argv = [
                "--input",
                str(raw),
                "--output",
                args.prepared,
                "--size",
                "32",
                "--reuse-failure-report",
                str(output_dir / "cache-reuse-failure.json"),
            ]
            if args.resume:
                argv.append("--resume")
            return preprocess.main(argv)
        assert Path(args.prepared) != old
        assert read_json(Path(args.prepared) / "dataset.json")["size"] == 32
        return 0

    monkeypatch.setattr(imagechd, "run", run)
    assert jobs.execute(plan) == 0
    assert seen == ["preprocess", "preprocess", "train"]
    assert contents(old) == before
