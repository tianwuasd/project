"""A durable job owner: survives the HTTP server and the SSH connection."""
import json
import os
from pathlib import Path
import subprocess
import sys

from .operations import apply_license, check_device, install, probe, python_path, runtime_env
from .service_state import file_lock, process_identity, read_json, redact, timestamp, write_json


def run(workspace, job_id):
    workspace = Path(workspace).resolve()
    job = workspace / "jobs" / job_id
    request = read_json(job / "request.json")
    state = read_json(job / "state.json")
    write_json(job / "owner.json", process_identity())
    action, config = request["action"], request["config"]
    with file_lock(workspace / "operation.lock"):
        state.update(status="running")
        write_json(job / "state.json", state)
        try:
            if action == "install":
                install(workspace, config)
            elif action == "probe":
                probe(workspace)
                print("环境检查完成。", flush=True)
            elif action == "license":
                apply_license(workspace)
            else:
                info = probe(workspace)
                if action == "run":
                    check_device(config, info)
                command = [str(python_path(workspace)), "-u", str(job / "code" / "run.py"),
                           action, "--manifest", str(job / "manifest.csv"),
                           "--output-root", config["output_root"], "--device", config["device"],
                           "--threads", str(config["threads"])]
                if action == "run":
                    command += ["--stop-file", str(job / "stop")]
                # Child inherits this detached session; no terminal descriptors exist.
                fd = os.open(job / "operation.log", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "wb") as log:
                    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                             stderr=subprocess.STDOUT, env=runtime_env(workspace), close_fds=True)
                    write_json(job / "child.json", process_identity(child.pid))
                    code = child.wait()
                if action == "plan":
                    text = (job / "operation.log").read_text(encoding="utf-8")
                    try:
                        write_json(job / "plan.json", json.loads(text))
                    except ValueError:
                        pass
                state["exit_code"] = code
                if code == 130:
                    state["status"] = "stopped"
                elif code != 0:
                    raise ValueError(f"{action} 返回 {code}，请查看详细日志")
            if state["status"] == "running":
                state["status"] = "completed"
        except Exception as error:
            state.update(status="failed", error=redact(error))
            print(redact(error), flush=True)
        finally:
            state["ended_at"] = timestamp()
            write_json(job / "state.json", state)
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1], sys.argv[2]))
