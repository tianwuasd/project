import json
import os
import sys
import time

import pytest

from academic_agent.backends import MAX_OUTPUT_BYTES, CodexBackend, DemoBackend, ExchangeBackend
from academic_agent.demo import seed_demo
from academic_agent.domain import extract_domain
from academic_agent.filesystem import lexical_io_path
from academic_agent.models import ModelJob
from academic_agent.workflow import Runner
from academic_agent.writing import create_outline
from academic_agent.writing_models import WritingSelection


@pytest.fixture
def job():
    return ModelJob(id="test-job", kind="read", instruction="Return structured test output",
                    payload={"text": "original harmless synthetic input"},
                    output_schema={"type": "object", "properties": {"claims": {"type": "array"}},
                                   "required": ["claims"], "additionalProperties": False},
                    timeout_seconds=2)


def fake_backend(tmp_path, body):
    script = tmp_path / "fake_codex.py"
    script.write_text(
        "import json, os, sys, time\nfrom pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "output = Path(args[args.index('--output-last-message') + 1])\n"
        "schema = Path(args[args.index('--output-schema') + 1])\n"
        "prompt = sys.stdin.read()\n" + body, encoding="utf-8")

    class TestCodexBackend(CodexBackend):
        def _executable(self):
            return [sys.executable, str(script)]

    return TestCodexBackend(scratch_dir=tmp_path / "scratch")


def test_fake_cli_real_subprocess_success_and_isolation(tmp_path, job):
    body = (
        "assert '--ignore-user-config' in args and '--ignore-rules' in args\n"
        "assert '--ephemeral' in args and '--skip-git-repo-check' in args\n"
        "assert args[args.index('--sandbox') + 1] == 'read-only'\n"
        "assert Path.cwd() == output.parent == schema.parent\n"
        "assert 'shell_tool' in args and 'plugins' in args and 'multi_agent' in args\n"
        "assert 'web_search=\"disabled\"' in args\n"
        "assert json.loads(prompt)['payload']['text'] == 'original harmless synthetic input'\n"
        "assert json.loads(schema.read_text())['required'] == ['claims']\n"
        "output.write_text(json.dumps({'claims': []}))\n"
    )
    backend = fake_backend(tmp_path, body)
    result = backend.run(job)
    assert result.status == "succeeded", result.error
    assert result.data == {"claims": []}
    assert result.usage["cost_status"] == "unknown"
    assert list((tmp_path / "scratch").iterdir()) == []


@pytest.mark.skipif(os.name != "nt", reason="Verifies Windows Job Object descendant lifetime")
@pytest.mark.parametrize("parent_waits", [True, False])
def test_windows_owns_child_even_after_parent_exit(tmp_path, job, parent_waits):
    import ctypes
    from ctypes import wintypes

    pid_path = tmp_path / "child.pid"
    body = (
        "import subprocess\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        f"Path({str(pid_path)!r}).write_text(str(child.pid))\n"
        "output.write_text('{\"claims\": []}')\n"
        + ("time.sleep(20)\n" if parent_waits else "")
    )
    job.timeout_seconds = 1
    result = fake_backend(tmp_path, body).run(job)
    assert result.status == ("timed_out" if parent_waits else "succeeded")
    child_pid = int(pid_path.read_text())
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, child_pid)
    if handle:
        code = wintypes.DWORD()
        assert kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel.CloseHandle(handle)
        assert code.value != 259  # STILL_ACTIVE


def test_fake_cli_timeout_is_bounded(tmp_path, job):
    backend = fake_backend(tmp_path, "time.sleep(20)\n")
    job.timeout_seconds = 1
    start = time.monotonic()
    result = backend.run(job)
    assert result.status == "timed_out"
    assert time.monotonic() - start < 8


def test_fake_cli_nonzero_redacts_diagnostics(tmp_path, job):
    backend = fake_backend(tmp_path, "sys.stderr.write('SECRET-PROMPT sk-test-credential login failed')\nsys.exit(7)\n")
    result = backend.run(job)
    assert result.status == "failed"
    assert "authentication" in result.error
    assert "SECRET" not in result.error and "sk-test" not in result.error


@pytest.mark.parametrize("body", ["output.write_text('{broken')\n", "output.write_text('[]')\n", "pass\n"])
def test_fake_cli_bad_or_missing_json(tmp_path, job, body):
    result = fake_backend(tmp_path, body).run(job)
    assert result.status == "failed"
    assert "structured JSON" in result.error


@pytest.mark.parametrize("destination", ["stdout", "stderr", "output"])
def test_fake_cli_bounded_output(tmp_path, job, destination):
    expression = f"'x' * {MAX_OUTPUT_BYTES + 100}"
    body = (f"output.write_text({expression})\n" if destination == "output"
            else f"sys.{destination}.write({expression})\nsys.{destination}.flush()\n")
    result = fake_backend(tmp_path, body).run(job)
    assert result.status == "failed"
    assert "output" in result.error or "structured JSON" in result.error


def test_codex_refuses_oversized_input_before_launch(job):
    class NoLaunch(CodexBackend):
        def _executable(self):
            pytest.fail("Oversized job must not look up or start Codex")

    job.max_input_chars = 100
    job.payload = {"text": "x" * 200}
    assert NoLaunch().run(job).status == "failed"


def test_proxy_is_explicit_child_only_and_paid_api_keys_are_not_inherited(tmp_path, job, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://untrusted.invalid:9999")
    monkeypatch.setenv("OPENAI_API_KEY", "private-test-key")
    backend = fake_backend(tmp_path, "assert os.environ['HTTP_PROXY'] == 'http://localhost:8123'\n"
                           "assert 'OPENAI_API_KEY' not in os.environ\n"
                           "output.write_text('{\"claims\": []}')\n")
    backend.proxy_url = "http://localhost:8123"
    assert backend.run(job).status == "succeeded"
    assert os.environ["HTTP_PROXY"] == "http://untrusted.invalid:9999"
    assert os.environ["OPENAI_API_KEY"] == "private-test-key"
    backend.proxy_url = None
    result = backend.run(job)
    assert result.status == "failed" and "proxy_url" in result.error


def test_exchange_requires_matching_fingerprint_and_reuses_response(tmp_path, job):
    backend = ExchangeBackend(tmp_path)
    pending = backend.run(job)
    assert pending.status == "needs_input"
    request_path = next(tmp_path.glob("*.request.json"))
    request = json.loads(request_path.read_text(encoding="utf-8"))
    response_path = request_path.with_name(request_path.name.replace(".request.", ".response."))
    response_path.write_text(json.dumps({"fingerprint": "stale", "data": {"claims": []}}), encoding="utf-8")
    assert backend.run(job).status == "failed"
    response_path.write_text(json.dumps({"fingerprint": request["fingerprint"], "data": {"claims": []}}),
                             encoding="utf-8")
    result = backend.run(job)
    assert result.status == "succeeded" and result.data == {"claims": []}
    assert result.usage["model_calls"] == 0
    job.payload["text"] = "different input"
    assert backend.run(job).status == "needs_input"
    assert len(list(tmp_path.glob("*.request.json"))) == 2


def test_exchange_consumes_fingerprinted_response_beyond_windows_path_limit(tmp_path, job):
    logical = tmp_path
    while len(str(logical / ("f" * 64 + ".response.json"))) <= 300:
        logical /= "长路径 交换 #100% " + "x" * 24
    backend = ExchangeBackend(logical)

    pending = backend.run(job)

    assert pending.status == "needs_input"
    physical = lexical_io_path(logical)
    request_path = next(physical.glob("*.request.json"))
    request = json.loads(request_path.read_text(encoding="utf-8"))
    response_path = request_path.with_name(request_path.name.replace(".request.", ".response."))
    response_path.write_text(json.dumps({
        "fingerprint": request["fingerprint"], "data": {"claims": []},
    }), encoding="utf-8")

    result = backend.run(job)

    assert result.status == "succeeded"
    assert result.data == {"claims": []}
    assert backend.directory == logical
    if os.name == "nt":
        assert len(str(logical / response_path.name)) > 260


def test_exchange_rejects_oversized_response(tmp_path, job):
    backend = ExchangeBackend(tmp_path)
    backend.run(job)
    request = next(tmp_path.glob("*.request.json"))
    request.with_name(request.name.replace(".request.", ".response.")).write_bytes(b" " * (MAX_OUTPUT_BYTES + 1))
    assert backend.run(job).status == "failed"


def test_demo_rejects_real_paper_and_incidental_marker(job):
    backend = DemoBackend()
    for text in ("A real article with no original fixture marker", "Citation mentions ORIGINAL_SYNTHETIC_FIXTURE."):
        job.payload = {"full_available_context": [{"text": text}]}
        result = backend.run(job)
        assert result.status == "failed"
        assert "marker" in result.error


@pytest.mark.parametrize("task", [
    "domain-extract-v1",
    "domain-review-v1",
    "writing-outline-v1",
    "writing-draft-v1",
    "writing-segment-v1",
    "writing-review-v1",
])
def test_demo_routes_only_the_six_exact_writing_tasks_and_refuses_external_inputs(job, task):
    job.kind = "synthesize"
    job.payload = {
        "task": task,
        "domain": {"name": "lookalike scheduling pack"},
        "context": {"pack": {"source_path": "external-domain.json"}},
        "research": {"domain_pack_identity": {"source_path": "external-domain.json"}},
    }

    result = DemoBackend().run(job)

    assert result.status == "failed"
    assert "builtin-pack" in result.error
    assert "Exchange" in result.error


@pytest.mark.parametrize("task", ["writing-import-v1", "writing-review-revision-v1"])
def test_demo_does_not_treat_progress_keys_as_model_tasks(job, task):
    job.kind = "synthesize"
    job.payload = {"task": task}
    result = DemoBackend().run(job)
    assert result.status == "failed"
    assert "writing input" not in result.error


def _captured_demo_writing_jobs(root):
    seed_demo(root)
    Runner(root, offline=True).run()
    work_ids = [
        work.id
        for work in sorted(Runner(root, offline=True).store.works(), key=lambda item: item.title)
    ]
    selection = WritingSelection(
        document_id="doc_" + "6" * 32,
        question="How do the two scheduling reports differ by workload?",
        work_ids=work_ids,
        domain_path="builtin:scheduling",
        trend_directory=None,
        candidate_directory=None,
        candidate_id=None,
        assessment_directory=None,
    )

    class CaptureDemo:
        name = "demo"

        def __init__(self):
            self.jobs = []

        def run(self, model_job):
            self.jobs.append(model_job.model_copy(deep=True))
            return DemoBackend().run(model_job)

    domain_backend = CaptureDemo()
    extracted = extract_domain(Runner(root, offline=True, backend=domain_backend), selection)
    assert extracted["status"] == "complete"
    writing_backend = CaptureDemo()
    outlined = create_outline(
        Runner(root, offline=True, backend=writing_backend),
        selection,
        root / extracted["output"],
    )
    assert outlined["status"] == "complete"
    return domain_backend.jobs, writing_backend.jobs


@pytest.fixture(scope="module")
def captured_demo_writing_jobs(tmp_path_factory):
    return _captured_demo_writing_jobs(tmp_path_factory.mktemp("captured-demo-writing"))


def test_demo_rejects_mutated_reviewed_claim_chain(captured_demo_writing_jobs):
    domain_jobs, _ = captured_demo_writing_jobs
    job = next(
        item.model_copy(deep=True)
        for item in domain_jobs
        if item.payload["task"] == "domain-extract-v1"
    )
    job.payload["research"]["claims"][0]["text"] = "A plausible but fabricated reviewed claim."

    result = DemoBackend().run(job)

    assert result.status == "failed"


def test_demo_fails_closed_for_unknown_segmentation_and_review_text(captured_demo_writing_jobs):
    _, writing_jobs = captured_demo_writing_jobs
    review = next(
        item.model_copy(deep=True)
        for item in writing_jobs
        if item.payload["task"] == "writing-review-v1"
    )

    known_review = review.model_copy(deep=True)
    known_review.payload["outline"]["blocks"][0]["units"][0]["text"] = "All schedulers always improve"
    known_result = DemoBackend().run(known_review)
    assert known_result.status == "succeeded"
    assert known_result.data["items"][0]["decision"] == "contradicted"

    neutral_review = review.model_copy(deep=True)
    neutral_review.payload["outline"]["blocks"][0]["units"][0]["text"] = "Scheduling evidence scope"
    neutral_result = DemoBackend().run(neutral_review)
    assert neutral_result.status == "succeeded"
    assert neutral_result.data["items"][0]["decision"] == "supported"

    known_segment = review.model_copy(deep=True)
    known_segment.payload["task"] = "writing-segment-v1"
    known_text = "Study A used 400 simulated participants.\r\n因此两种策略总会改进。🙂 e\u0301"
    known_segment.payload["raw_blocks"] = [{
        "id": "paragraph_" + "3" * 32,
        "role": "paragraph",
        "level": None,
        "text": known_text,
    }]
    known_segment.payload["target_ids"] = [known_segment.payload["raw_blocks"][0]["id"]]
    segmented = DemoBackend().run(known_segment)
    assert segmented.status == "succeeded"
    assert [known_text[span["start"]:span["end"]] for span in segmented.data["blocks"][0]["spans"]] == [
        "Study A used 400 simulated participants.\r\n",
        "因此两种策略总会改进。🙂 e\u0301",
    ]
    assert all("text" not in span for span in segmented.data["blocks"][0]["spans"])

    first_unit = review.payload["outline"]["blocks"][0]["units"][0]
    first_unit["text"] = "The Moon is made of scheduling cheese."
    assert DemoBackend().run(review).status == "failed"

    segment = review.model_copy(deep=True)
    segment.payload["task"] = "writing-segment-v1"
    segment.payload["raw_blocks"] = [{
        "id": "paragraph_" + "4" * 32,
        "role": "paragraph",
        "level": None,
        "text": "The Moon is made of scheduling cheese.",
    }]
    segment.payload["target_ids"] = [segment.payload["raw_blocks"][0]["id"]]
    assert DemoBackend().run(segment).status == "failed"


def test_demo_rejects_duplicate_domain_effective_review(captured_demo_writing_jobs):
    domain_jobs, _ = captured_demo_writing_jobs
    job = next(
        item.model_copy(deep=True)
        for item in domain_jobs
        if item.payload["task"] == "domain-extract-v1"
    )
    job.payload["research"]["effective_reviews"].append(
        dict(job.payload["research"]["effective_reviews"][0])
    )

    result = DemoBackend().run(job)

    assert result.status == "failed"
    assert "exact builtin-pack and original-fixture protocol" in result.error


def test_demo_rejects_domain_work_document_identity_detachment(captured_demo_writing_jobs):
    domain_jobs, _ = captured_demo_writing_jobs
    job = next(
        item.model_copy(deep=True)
        for item in domain_jobs
        if item.payload["task"] == "domain-extract-v1"
    )
    replacement = "work_" + "9" * 32
    job.payload["research"]["work"]["id"] = replacement
    job.payload["research"]["work"]["citekey"] = "ref_" + replacement.removeprefix("work_")

    result = DemoBackend().run(job)

    assert result.status == "failed"
    assert "exact builtin-pack and original-fixture protocol" in result.error


@pytest.mark.parametrize("map_name", ["works", "documents", "evidence", "cards", "decisions"])
def test_demo_rejects_writing_source_map_key_mismatch(
    captured_demo_writing_jobs, map_name
):
    _, writing_jobs = captured_demo_writing_jobs
    job = next(
        item.model_copy(deep=True)
        for item in writing_jobs
        if item.payload["task"] == "writing-outline-v1"
    )
    source_map = job.payload["context"]["research"][map_name]
    old_key = next(iter(source_map))
    if map_name == "works":
        source_map[old_key]["id"] = "work_" + "8" * 32
        source_map[old_key]["citekey"] = "ref_" + "8" * 32
    elif map_name in {"documents", "evidence"}:
        source_map["mismatched-key"] = source_map.pop(old_key)
    elif map_name == "cards":
        source_map["mismatched-key"] = source_map.pop(old_key)
    else:
        source_map["mismatched-key"] = dict(source_map[old_key])

    result = DemoBackend().run(job)

    assert result.status == "failed"
    assert "exact builtin-pack and original-fixture protocol" in result.error


@pytest.mark.parametrize("map_name", ["evidence", "decisions"])
def test_demo_rejects_unconsumed_writing_source_entry(
    captured_demo_writing_jobs, map_name
):
    _, writing_jobs = captured_demo_writing_jobs
    job = next(
        item.model_copy(deep=True)
        for item in writing_jobs
        if item.payload["task"] == "writing-outline-v1"
    )
    source_map = job.payload["context"]["research"][map_name]
    extra = dict(next(iter(source_map.values())))
    if map_name == "evidence":
        extra["id"] = "evidence_" + "7" * 32
        extra["work_id"] = "work_" + "7" * 32
        source_map[extra["id"]] = extra
    else:
        extra["claim_id"] = "claim_" + "7" * 32
        source_map[extra["claim_id"]] = extra

    result = DemoBackend().run(job)

    assert result.status == "failed"
    assert "exact builtin-pack and original-fixture protocol" in result.error
