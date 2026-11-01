"""Replaceable, bounded model transports. No backend claims scholarly accuracy."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from pydantic import ValidationError

from .documents import verify_span
from .filesystem import lexical_io_path
from .models import Document, Evidence, ModelJob, ModelResult, digest

MAX_OUTPUT_BYTES = 2 * 1024 * 1024
SYNTHETIC_MARKER = "ORIGINAL_SYNTHETIC_FIXTURE"


class Backend(Protocol):
    name: str

    def run(self, job: ModelJob) -> ModelResult: ...


def job_prompt(job: ModelJob) -> str:
    """The exact bounded input representation shared with metering callers."""
    return json.dumps({"instruction": job.instruction, "payload": job.payload},
                      ensure_ascii=False, sort_keys=True)


def _failure(name: str, message: str, status: str = "failed") -> ModelResult:
    return ModelResult(status=status, backend=name, error=message)


def _bounded_json(path: Path) -> dict:
    with path.open("rb") as source:
        raw = source.read(MAX_OUTPUT_BYTES + 1)
    if len(raw) > MAX_OUTPUT_BYTES:
        raise ValueError("Structured output exceeds size limit")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Structured output must be a JSON object")
    return data


def _fingerprint(job: ModelJob) -> str:
    return digest(json.dumps(job.model_dump(), ensure_ascii=False, sort_keys=True))


class ExchangeBackend:
    """Explicit local handoff; responses must echo the complete request fingerprint."""

    name = "exchange"

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def run(self, job: ModelJob) -> ModelResult:
        if len(job_prompt(job)) > job.max_input_chars:
            return _failure(self.name, "Complete input exceeds configured character limit")
        fingerprint = _fingerprint(job)
        request = self.directory / f"{fingerprint}.request.json"
        response = self.directory / f"{fingerprint}.response.json"
        physical_directory = lexical_io_path(self.directory)
        physical_request = physical_directory / request.name
        physical_response = physical_directory / response.name
        try:
            physical_directory.mkdir(parents=True, exist_ok=True)
            if not physical_request.exists():
                with physical_request.open("x", encoding="utf-8") as target:
                    json.dump({"fingerprint": fingerprint, "job": job.model_dump(),
                               "response_format": {"fingerprint": fingerprint, "data": {}}},
                              target, ensure_ascii=False, indent=2)
            if not physical_response.exists():
                return _failure(self.name, f"Manual response required: {response.name}", "needs_input")
            result = _bounded_json(physical_response)
            if result.get("fingerprint") != fingerprint:
                return _failure(self.name, "Manual response fingerprint does not match this job")
            if not isinstance(result.get("data"), dict):
                return _failure(self.name, "Manual response data must be a JSON object")
            return ModelResult(status="succeeded", backend=self.name, data=result["data"],
                               usage={"mode": "manual_exchange", "model_calls": 0})
        except (OSError, ValueError, UnicodeError):
            return _failure(self.name, "Manual exchange could not read or write valid bounded JSON")


class DemoBackend:
    """Deterministic original-fixture mechanics, never a substitute for AI review."""

    name = "demo"

    _labels = {
        "problem": ("problem", "问题"), "method": ("method", "方法"),
        "data": ("data", "数据"), "result": ("result", "结果"),
        "limitation": ("limitation", "局限"), "contribution": ("contribution", "贡献"),
    }

    def run(self, job: ModelJob) -> ModelResult:
        if len(job_prompt(job)) > job.max_input_chars:
            return _failure(self.name, "Complete input exceeds configured character limit")
        if job.kind == "synthesize" and job.payload.get("task") in {
            "domain-extract-v1", "domain-review-v1", "writing-outline-v1",
            "writing-draft-v1", "writing-segment-v1", "writing-review-v1",
        }:
            from .demo_writing import run

            return run(job)
        if job.kind == "synthesize":
            from .demo_innovation import run_demo_innovation

            return run_demo_innovation(job)
        pages = job.payload.get("full_available_context", [])
        text = "\n".join(p.get("text", "") for p in pages)
        if not re.search(r"(?m)^\s*" + SYNTHETIC_MARKER + r"\s*$", text):
            return _failure(self.name, "Demo requires an explicit ORIGINAL_SYNTHETIC_FIXTURE marker line")
        evidence = job.payload.get("evidence", [])
        try:
            document = Document.model_validate(job.payload.get("document", {}))
            verified = [Evidence.model_validate(item) for item in evidence]
        except ValidationError:
            return _failure(self.name, "Demo requires a valid document and exact evidence spans")
        if (not verified or pages != [p.model_dump() for p in document.pages]
                or any(not verify_span(item, document) for item in verified)):
            return _failure(self.name, "Demo evidence failed exact document span validation")
        extracted = []
        for category, labels in self._labels.items():
            for match in re.finditer(r"(?mi)^\s*(?:" + "|".join(labels) + r")[：:]\s*(\S[^\n]*)", text):
                sentence = match.group(1).strip()
                refs = [e["id"] for e in evidence if sentence in e.get("text", "")]
                if refs:
                    extracted.append({"text": sentence, "category": category, "evidence_ids": refs})
        if job.kind in {"read", "ask"}:
            if job.kind == "ask":
                # Only explicitly labelled category questions have a deterministic
                # answer. General questions require a real or manual backend.
                question = job.payload.get("question", "").lower()
                categories = {key for key, labels in self._labels.items()
                              if any(label in question for label in labels)}
                extracted = [c for c in extracted if c["category"] in categories]
            data = {"claims": extracted,
                    "missing": [c for c in self._labels if not any(x["category"] == c for x in extracted)]}
        elif job.kind == "review":
            verdicts = []
            for claim in job.payload.get("reading_card", {}).get("claims", []):
                refs = [e["id"] for e in evidence
                        if e["id"] in claim["evidence_ids"] and claim["text"] in e.get("text", "")]
                exact = any(c["text"] == claim["text"] and c["category"] == claim["category"]
                            for c in extracted)
                verdicts.append({"claim_id": claim["id"],
                                 "verdict": "supported" if exact and refs else "insufficient",
                                 "rationale": "Demo: exact labelled fixture sentence; no independent AI evaluation"
                                 if exact and refs else "Demo cannot establish semantic support for this paraphrase",
                                 "evidence_ids": refs if exact else []})
            data = {"verdicts": verdicts}
        else:
            return _failure(self.name, "Demo supports only fixture reading, category questions and exact review")
        return ModelResult(status="succeeded", backend=self.name, data=data,
                           usage={"mode": "synthetic_demo", "model_calls": 0})


def _native_codex() -> Path | None:
    """Prefer a native executable; never invoke a shell, .cmd or .ps1 wrapper."""
    if os.name != "nt":
        found = shutil.which("codex")
        return Path(found).resolve() if found else None
    found = shutil.which("codex.exe")
    if found:
        return Path(found).resolve()
    bases = []
    for wrapper in (shutil.which("codex.cmd"), shutil.which("codex.ps1")):
        if wrapper:
            bases.append(Path(wrapper).parent)
    if os.environ.get("APPDATA"):
        bases.append(Path(os.environ["APPDATA"]) / "npm")
    for base in bases:
        package = base / "node_modules" / "@openai" / "codex" / "node_modules" / "@openai"
        for candidate in sorted(package.glob("codex-win32-*/vendor/*/codex/codex.exe")):
            if candidate.is_file():
                return candidate.resolve()
    return None


def _terminate_tree(process: subprocess.Popen) -> None:
    if os.name == "nt":
        # taskkill is a native program; PID originates from Popen, never user text.
        executable = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "taskkill.exe"
        try:
            subprocess.run([str(executable), "/PID", str(process.pid), "/T", "/F"],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=5, check=False,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def _windows_job(process: subprocess.Popen):
    """Own descendants even if their parent exits before timeout cancellation.

    A kill-on-close Job Object also stops children retaining pipe handles after a
    successful main process exit. Handle declarations use pointer-sized fields.
    """
    import ctypes
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                    ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t),
                    ("max_working_set", ctypes.c_size_t), ("active_process_limit", wintypes.DWORD),
                    ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                    ("scheduling", wintypes.DWORD)]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [("basic", BasicLimits), ("io", ctypes.c_uint64 * 6),
                    ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                    ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateJobObjectW(None, None)
    if not handle:
        return None
    limits = ExtendedLimits()
    limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    configured = kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits))
    if not configured or not kernel.AssignProcessToJobObject(handle, int(process._handle)):
        kernel.CloseHandle(handle)
        return None
    return lambda: kernel.CloseHandle(handle)


class CodexBackend:
    """Use the existing Codex login in an isolated, tool-disabled child process."""

    name = "codex"

    def __init__(self, model: str | None = None, scratch_dir: Path | None = None,
                 proxy_url: str | None = "http://127.0.0.1:7897") -> None:
        self.model = model
        self.scratch_dir = Path(scratch_dir) if scratch_dir is not None else None
        self.proxy_url = proxy_url

    def _executable(self) -> list[str] | None:
        # Private override seam for subprocess tests; no public arbitrary command option.
        executable = _native_codex()
        return [str(executable)] if executable is not None else None

    def _command(self, executable: list[str], cwd: Path, schema: Path, output: Path) -> list[str]:
        command = [*executable, "exec", "--ignore-user-config", "--ignore-rules", "--ephemeral",
                   "--skip-git-repo-check", "--sandbox", "read-only", "-C", str(cwd),
                   "--output-schema", str(schema), "--output-last-message", str(output),
                   "-c", 'web_search="disabled"', "-c", "project_doc_max_bytes=0"]
        for feature in ("shell_tool", "unified_exec", "apps", "browser_use", "browser_use_external",
                        "browser_use_full_cdp_access", "computer_use", "plugins", "hooks", "multi_agent",
                        "view_image", "image_generation", "skill_search", "workspace_dependencies", "code_mode_host"):
            command.extend(["--disable", feature])
        command.extend(["--enable", "skip_host_skill_discovery"])
        if self.model:
            command.extend(["--model", self.model])
        return [*command, "-"]

    def run(self, job: ModelJob) -> ModelResult:
        prompt = job_prompt(job)
        if len(prompt) > job.max_input_chars:
            return _failure(self.name, "Complete input exceeds configured character limit")
        try:
            route = urlsplit(self.proxy_url or "")
            if route.scheme not in {"http", "https", "socks5", "socks5h"} or not route.hostname:
                raise ValueError("Missing route")
            route.port  # Validate a supplied port without exposing the route in errors.
        except ValueError:
            return _failure(self.name, "Foreign Codex traffic requires an explicit valid proxy_url; no direct fallback")
        executable = self._executable()
        if not executable:
            return _failure(self.name, "Native Codex executable unavailable; install Codex or use manual exchange")
        try:
            if self.scratch_dir:
                self.scratch_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="academic-codex-", dir=self.scratch_dir) as temp:
                return self._execute(job, prompt, executable, Path(temp))
        except (OSError, ValueError):
            return _failure(self.name, "Codex could not start or access isolated scratch files; check local permissions")

    def _execute(self, job: ModelJob, prompt: str, executable: list[str], cwd: Path) -> ModelResult:
        schema = cwd / "output.schema.json"
        output = cwd / "output.json"
        schema.write_text(json.dumps(job.output_schema), encoding="utf-8")
        # Feed a regular file, avoiding a blocked stdin writer outside the timeout.
        prompt_file = cwd / "input.json"
        prompt_file.write_text(prompt, encoding="utf-8")
        environment = os.environ.copy()
        # Foreign OpenAI traffic has an explicit route; loopback stays direct.
        remove_keys = {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "CODEX_THREAD_ID",
                       "CODEX_INTERNAL_ORIGINATOR_OVERRIDE", "CODEX_CONFIG_OVERRIDES", "OPENAI_API_KEY",
                       "CODEX_API_KEY", "OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID"}
        environment = {key: value for key, value in environment.items() if key.upper() not in remove_keys}
        environment.update({"HTTP_PROXY": self.proxy_url, "HTTPS_PROXY": self.proxy_url,
                            "ALL_PROXY": self.proxy_url, "NO_PROXY": "localhost,127.0.0.1,::1"})
        options = ({"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
                   if os.name == "nt" else {"start_new_session": True})
        with prompt_file.open("rb") as input_stream:
            process = subprocess.Popen(self._command(executable, cwd, schema, output), cwd=cwd,
                                       env=environment, stdin=input_stream, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, **options)
            close_job = _windows_job(process) if os.name == "nt" else None
            if os.name == "nt" and close_job is None:
                _terminate_tree(process)
                process.stdout.close()
                process.stderr.close()
                return _failure(self.name, "Codex process-tree isolation unavailable; check Windows permissions")
            over_limit = threading.Event()
            diagnostics: list[bytearray] = [bytearray(), bytearray()]

            def drain(stream, buffer: bytearray) -> None:
                try:
                    while chunk := stream.read(8192):
                        room = MAX_OUTPUT_BYTES - len(buffer)
                        buffer.extend(chunk[:max(room, 0)])
                        if len(chunk) > room:
                            over_limit.set()
                finally:
                    stream.close()

            readers = [threading.Thread(target=drain, args=(stream, buffer), daemon=True)
                       for stream, buffer in zip((process.stdout, process.stderr), diagnostics)]
            for reader in readers:
                reader.start()
            deadline = time.monotonic() + job.timeout_seconds
            failure = None
            try:
                while process.poll() is None:
                    if over_limit.is_set() or (output.exists() and output.stat().st_size > MAX_OUTPUT_BYTES):
                        failure = _failure(self.name, "Codex output exceeded the bounded output limit")
                        break
                    if time.monotonic() >= deadline:
                        failure = _failure(self.name, "Codex exceeded the configured timeout", "timed_out")
                        break
                    time.sleep(0.025)
            finally:
                if close_job is not None:
                    close_job()
                    process.wait(timeout=5)
                elif os.name != "nt" or process.poll() is None or failure is not None:
                    _terminate_tree(process)
                for reader in readers:
                    reader.join(timeout=1)
            if failure:
                return failure
            if over_limit.is_set():
                return _failure(self.name, "Codex output exceeded the bounded output limit")
            if process.returncode != 0:
                diagnostic = b" ".join(diagnostics).lower()
                if any(word in diagnostic for word in (b"login", b"unauthorized", b"401", b"profile token")):
                    message = "Codex authentication/profile unavailable in this process; verify login in a normal user terminal"
                else:
                    message = "Codex exited unsuccessfully; check local login, sandbox permissions and CLI compatibility"
                return _failure(self.name, message)
            try:
                data = _bounded_json(output)
            except (OSError, ValueError, UnicodeError):
                return _failure(self.name, "Codex returned missing, oversized or malformed structured JSON")
            return ModelResult(status="succeeded", backend=self.name, data=data,
                               usage={"mode": "codex_login", "model_calls": 1, "cost_status": "unknown"})
