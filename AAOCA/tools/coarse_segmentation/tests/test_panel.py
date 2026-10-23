import csv
import http.client
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch, Mock

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aaoca_seg.panel import Controller, make_server
from aaoca_seg.schema import FIELDS


class PanelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        source = self.root / "source"
        source.mkdir()
        image = nib.Nifti1Image(np.arange(125, dtype=np.float32).reshape(5, 5, 5), np.eye(4))
        image.header.set_xyzt_units("mm")
        nib.save(image, str(source / "demo.nii.gz"))
        self.manifest = self.root / "input.csv"
        with self.manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerow({"case_id": "demo", "image": str(source / "demo.nii.gz"),
                             "spacing_confirmed": "true", "hu_confirmed": "true"})
        self.controller = Controller(self.manifest, self.root / "results", "cpu")
        self.server, self.token = make_server(self.controller, 0, "synthetic-test-token")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.temp.cleanup()

    def request(self, path, payload=None, token=True):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        headers = {"Authorization": "Bearer " + self.token} if token else {}
        body = json.dumps(payload) if payload is not None else None
        connection.request("POST" if body is not None else "GET", path, body=body, headers=headers)
        response = connection.getresponse()
        data = response.read()
        connection.close()
        return response.status, data

    def test_auth_and_static_assets(self):
        self.assertEqual(self.request("/api/status", token=False)[0], 401)
        self.assertEqual(self.request("/api/stop", {}, token=False)[0], 401)
        for path in ("/", "/app.js", "/style.css"):
            code, data = self.request(path, token=False)
            self.assertEqual(code, 200)
            self.assertTrue(data)
        self.assertEqual(self.server.server_address[0], "127.0.0.1")

    def test_plan_selection_and_path_injection(self):
        code, body = self.request("/api/plan", {"case_ids": ["demo"]})
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["ready"], 1)
        for payload in ({"case_ids": []}, {"case_ids": ["../escape"]},
                        {"case_ids": ["demo"], "command": "bad"}):
            self.assertEqual(self.request("/api/start", payload)[0], 400)
        self.assertFalse(self.controller.root.exists())

    def test_start_uses_frozen_manifest_and_own_stop_request(self):
        process = Mock()
        process.poll.return_value = None
        with patch("aaoca_seg.panel.require_backend"), patch("aaoca_seg.panel.subprocess.Popen", return_value=process) as popen:
            code, _ = self.request("/api/start", {"case_ids": ["demo"]})
            self.assertEqual(code, 200)
            command = popen.call_args.args[0]
            self.assertIsInstance(command, list)
            self.assertNotIn("shell", popen.call_args.kwargs)
            self.assertTrue(Path(command[command.index("--manifest") + 1]).is_file())
            self.assertEqual(self.request("/api/start", {"case_ids": ["demo"]})[0], 400)
            self.assertEqual(self.request("/api/stop", {})[0], 200)
            self.assertTrue(self.controller.stop_file.exists())
            process.poll.return_value = 0
            self.assertEqual(self.request("/api/stop", {})[0], 400)

    def test_no_foreign_process_stop_or_file_api(self):
        self.assertEqual(self.request("/api/stop", {})[0], 400)
        self.assertEqual(self.request("/api/stop", {"pid": 12345})[0], 404)
        self.assertEqual(self.request("/../../input.csv")[0], 404)


if __name__ == "__main__":
    unittest.main()
