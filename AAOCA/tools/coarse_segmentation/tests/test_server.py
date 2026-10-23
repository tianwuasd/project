"""Synthetic server tests: no models, internet, licenses or patient data."""
import csv
import http.client
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aaoca_seg.managed import ManagedController
from aaoca_seg.operations import choose_build, stage_license, apply_license, check_device, runtime_env
from aaoca_seg.panel import make_server
from aaoca_seg.server import import_profile
from aaoca_seg.service_state import (CODE_ROOT, alive, detached, file_lock,
                                    process_identity, read_json, write_json, write_private)


def wait_for(predicate, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.05)
    raise AssertionError("Timed out waiting for test process")


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.controller = ManagedController(self.base / "runtime")
        self.workspace = self.controller.workspace

    def tearDown(self):
        self.temp.cleanup()

    def configure(self, **changes):
        config = {k: v for k, v in self.controller.config.items() if k != "gpu_uuid"}
        config.update(changes)
        return self.controller.configure(config)

    def test_no_third_party_dependencies_needed_for_console(self):
        result = subprocess.run([sys.executable, "-S", str(CODE_ROOT / "server.py"), "--help"], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.controller.status()["cases"], [])

    def test_private_state_atomic_and_process_identity_rejects_reuse(self):
        path = self.workspace / "private" / "test.json"
        write_json(path, {"ok": True})
        self.assertEqual(read_json(path), {"ok": True})
        if os.name != "nt":
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        identity = process_identity()
        self.assertTrue(alive(identity))
        self.assertFalse(alive(dict(identity, started="different")))

    def test_os_lock_releases_without_deleting_file(self):
        path = self.workspace / "lock"
        with file_lock(path):
            with self.assertRaises(ValueError):
                with file_lock(path):
                    pass
        self.assertTrue(path.exists())
        with file_lock(path):
            pass

    def test_driver_selection_and_visibility_preserved(self):
        for driver, expected in [("580.173.02", "cu126"), ("535.54.03", "cu118")]:
            self.assertEqual(choose_build("auto", {"gpus": [{"driver": driver}]}), expected)
        self.assertEqual(choose_build("auto", {"gpus": []}), "cpu")
        with self.assertRaises(ValueError):
            choose_build("auto", {"gpus": [{"driver": "470.0"}]})
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-private-synthetic"}):
            self.assertEqual(runtime_env(self.workspace)["CUDA_VISIBLE_DEVICES"], "GPU-private-synthetic")

    def test_gpu_requires_allocation_and_stable_uuid(self):
        info = {"cuda_available": True, "devices": [{"index": 0, "uuid": "GPU-test"}]}
        with patch("aaoca_seg.managed.read_json", wraps=read_json):
            write_json(self.workspace / "environment.json", info)
            with self.assertRaises(ValueError):
                self.configure(device="gpu:0", gpu_confirmed=False)
            self.configure(device="gpu:0", gpu_confirmed=True)
            self.assertEqual(self.controller.config["gpu_uuid"], "GPU-test")
            info["devices"][0]["uuid"] = "GPU-changed"
            with self.assertRaises(ValueError):
                check_device(self.controller.config, info)
        check_device({"device": "cpu"}, {})

    def test_upload_manifest_and_reject_relative_unconfirmed_and_unsafe_paths(self):
        image = self.base / "source" / "demo.nii.gz"
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(["case_id", "image", "spacing_confirmed", "hu_confirmed"])
        writer.writerow(["demo", str(image), "true", "true"])
        response = self.controller.import_manifest({"csv": stream.getvalue()})
        self.assertEqual(response["count"], 1)
        self.assertEqual(self.controller.status()["cases"][0]["case_id"], "demo")
        with self.assertRaises(ValueError):
            self.configure(output_root=str(image.parent))
        with self.assertRaises(ValueError):
            self.controller.import_manifest({"csv": stream.getvalue().replace(str(image), "relative.nii.gz")})
        with self.assertRaises(ValueError):
            self.controller.import_manifest({"csv": stream.getvalue().replace("true,true", "false,true")})

    def test_import_profile_keeps_license_out_of_settings_and_api(self):
        secret = "aca_" + "S" * 14
        profile = self.base / "profile.json"
        write_json(profile, {"license": secret, "output_root": "results"})
        import_profile(self.controller, profile)
        self.assertTrue(self.controller.status()["license_pending"])
        self.assertNotIn(secret, self.controller.config_path.read_text(encoding="utf-8"))
        self.assertNotIn(secret, json.dumps(self.controller.status()))
        self.assertNotIn(secret, self.controller.logs()["text"])
        self.assertEqual(self.controller.config["output_root"], str(self.base / "results"))

    def test_license_uses_stdin_and_hides_upstream_errors(self):
        secret = "aca_" + "T" * 14
        stage_license(self.workspace, secret)
        result = Mock(returncode=1, stdout=secret, stderr=secret)
        with patch("aaoca_seg.operations.subprocess.run", return_value=result) as call:
            with self.assertRaises(ValueError) as error:
                apply_license(self.workspace)
        self.assertEqual(call.call_args.kwargs["input"], secret)
        self.assertNotIn(secret, str(call.call_args.args[0]))
        self.assertNotIn(secret, str(error.exception))
        self.assertTrue((self.workspace / "private" / "license.input").exists())

    def test_restored_controller_can_see_and_stop_own_job(self):
        job_id = "a" * 32
        job = self.workspace / "jobs" / job_id
        write_json(job / "state.json", {"id": job_id, "action": "run", "status": "running"})
        write_json(self.workspace / "current.json", {"id": job_id})
        code = "import pathlib,time,sys; p=pathlib.Path(sys.argv[1]);\nwhile not p.exists(): time.sleep(.05)"
        child = detached([sys.executable, "-c", code, str(job / "stop")], job / "worker.log")
        write_json(job / "owner.json", process_identity(child.pid))
        try:
            restored = ManagedController(self.workspace)
            self.assertTrue(restored.status()["active"])
            with self.assertRaises(ValueError):
                self.configure(threads=4)
            with self.assertRaises(ValueError):
                restored.submit("install")
            restored.stop()
            child.wait(timeout=10)
            self.assertFalse(restored.active())
            self.assertEqual(restored.job()[1]["status"], "interrupted")
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=5)

    def test_orphan_child_blocks_environment_mutation(self):
        job_id = "b" * 32
        job = self.workspace / "jobs" / job_id
        write_json(job / "state.json", {"id": job_id, "action": "plan", "status": "running"})
        write_json(job / "owner.json", None)
        write_json(job / "child.json", process_identity())
        write_json(self.workspace / "current.json", {"id": job_id})
        self.assertTrue(self.controller.active())
        with self.assertRaises(ValueError):
            self.configure(threads=4)

    def test_stale_inference_lock_is_not_removed(self):
        lock = Path(self.controller.config["output_root"]) / ".run.lock"
        write_json(lock, {"pid": 999999})
        with self.assertRaises(ValueError):
            self.controller.submit("install")
        self.assertTrue(lock.exists())

    def test_worker_runs_frozen_synthetic_preflight_and_persists_failure(self):
        import nibabel as nib
        import numpy as np
        from aaoca_seg.worker import run
        image = self.base / "source" / "demo.nii.gz"
        image.parent.mkdir()
        volume = nib.Nifti1Image(np.arange(125, dtype=np.float32).reshape(5, 5, 5), np.eye(4))
        volume.header.set_xyzt_units("mm")
        nib.save(volume, image)
        manifest = self.base / "manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["case_id", "image", "spacing_confirmed", "hu_confirmed"])
            writer.writerow(["demo", str(image), "true", "true"])
        self.configure(manifest=str(manifest))
        mock_child = Mock(pid=os.getpid())
        mock_child.poll.return_value = None
        with patch("aaoca_seg.managed.python_path", return_value=Path(sys.executable)), patch("aaoca_seg.managed.detached", return_value=mock_child):
            self.controller.submit("plan", {"case_ids": ["demo"]})
        job, _ = self.controller.job()
        self.assertNotEqual((job / "code" / "run.py").resolve(), CODE_ROOT / "run.py")
        self.assertTrue((job / "manifest.csv").is_file())
        with patch("aaoca_seg.worker.probe", return_value={}), patch("aaoca_seg.worker.python_path", return_value=Path(sys.executable)):
            self.assertEqual(run(self.workspace, job.name), 0)
        self.assertEqual(read_json(job / "plan.json")["ready"], 1)
        request = read_json(job / "request.json")
        request["action"] = "probe"
        write_json(job / "request.json", request)
        with patch("aaoca_seg.worker.probe", side_effect=ValueError("synthetic failure")):
            self.assertEqual(run(self.workspace, job.name), 1)
        self.assertEqual(read_json(job / "state.json")["status"], "failed")

    def test_detached_child_survives_parent_exit(self):
        result = self.base / "finished"
        parent = "from aaoca_seg.service_state import detached; import sys; detached([sys.executable,'-c',\"import sys,time,pathlib,json; from aaoca_seg.service_state import process_identity; time.sleep(.3); pathlib.Path(sys.argv[1]).write_text(json.dumps(process_identity()))\",sys.argv[1]],sys.argv[2])"
        launch = subprocess.run([sys.executable, "-c", parent, str(result), str(self.base / "detached.log")], cwd=CODE_ROOT, capture_output=True, timeout=10)
        self.assertEqual(launch.returncode, 0, launch.stderr)
        wait_for(result.exists)
        identity = read_json(result)
        self.assertIn("pid", identity)
        wait_for(lambda: not alive(identity))

    def test_http_setup_auth_and_unknown_commands(self):
        server, token = make_server(self.controller, 0, "test-token")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def request(path, payload, auth=True):
            conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            conn.request("POST", path, json.dumps(payload), {"Authorization": "Bearer " + token} if auth else {})
            response = conn.getresponse()
            result = response.status, response.read()
            conn.close()
            return result
        try:
            self.assertEqual(request("/api/setup/install", {}, False)[0], 401)
            self.assertEqual(request("/api/setup/command", {"command": "anything"})[0], 400)
            self.assertEqual(request("/api/setup/license", {"license": "bad"})[0], 400)
            self.assertEqual(request("/api/setup/configure", {"command": "bad"})[0], 400)
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=5)

    def test_background_launcher_restart_uses_existing_private_token(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        base = [sys.executable, "-S", str(CODE_ROOT / "server.py"), "--workspace", str(self.workspace), "--port", str(port)]
        try:
            first = subprocess.run(base, capture_output=True, timeout=20)
            self.assertEqual(first.returncode, 0, first.stderr)
            owner = read_json(self.workspace / "service.json")["owner"]
            token = (self.workspace / "private" / "panel.token").read_text()
            again = subprocess.run(base, capture_output=True, timeout=20)
            self.assertEqual(again.returncode, 0, again.stderr)
            self.assertEqual(owner, read_json(self.workspace / "service.json")["owner"])
            self.assertEqual(token, (self.workspace / "private" / "panel.token").read_text())
        finally:
            subprocess.run(base + ["--stop-panel"], capture_output=True, timeout=15)
            wait_for(lambda: not alive(read_json(self.workspace / "service.json", {}).get("owner")))


if __name__ == "__main__":
    unittest.main()
