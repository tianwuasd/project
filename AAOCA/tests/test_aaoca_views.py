"""Read-only navigation and bounded log following using synthetic metadata."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aaoca_server import views
from aaoca_server.panel import Panel
from aaoca_server.state import write


class ViewTests(unittest.TestCase):
    def test_frozen_spacing_source_visible_before_dataset_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / 'output.runtime/job.json', {
                'config': {'target_spacing': [0.5, 0.5, 0.7]},
                'spacing_selection': {'method': 'synthetic-frozen-median'}
            })
            data = views.snapshot({'name': 'old', 'output': str(root / 'output')})
            self.assertIn('synthetic-frozen-median', '\n'.join(views.content(data, '2')))

    def test_missing_history_outputs_are_not_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(
                root / "panel/history.json",
                [{"name": "old", "output": str(root / "missing")}],
            )
            answers = iter(["1", "1", "d", "s", "b", "4", "b", "0"])
            messages = []
            Panel(
                root / "panel", ask=lambda _: next(answers), emit=messages.append
            ).results()
            self.assertFalse((root / "missing").exists())
            self.assertFalse((root / "missing.runtime").exists())
            self.assertTrue(
                any("暂无" in message or "未记录" in message for message in messages)
            )

    def test_completed_recipe_spacing_is_displayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / 'output/dataset.json', {'config': {'nominal_spacing_mm': [0.4, 0.5, 0.6]}})
            data = views.snapshot({'name': 'done', 'output': str(root / 'output')})
            self.assertIn('[0.4, 0.5, 0.6]', '\n'.join(views.content(data, '2')))

    def test_pages_wrap_chinese_and_snapshot_changes_only_on_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(
                root / "panel/history.json",
                [{"name": "old", "output": str(root / "output")}],
            )
            write(
                root / "output/progress.json",
                {
                    "state": "stopped",
                    "total": 80,
                    "results": [
                        {
                            "case_id": f"CASE_{i:03}",
                            "state": "completed",
                            "output": "长路径" * 80,
                        }
                        for i in range(80)
                    ],
                },
            )
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            messages = []
            answers = iter(["1", "3", "d", "n", "p", "r", "b", "0"])
            with patch.object(views, "snapshot", wraps=views.snapshot) as capture:
                Panel(
                    root / "panel", ask=lambda _: next(answers), emit=messages.append
                ).results()
            self.assertEqual(capture.call_count, 2)  # entry + explicit refresh
            self.assertEqual(
                before, {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            )
            self.assertTrue(any("CASE_000" in text for text in messages))
            for page in views.paginate(
                ["长路径" * 100, *map(str, range(50))], width=40
            ):
                self.assertLessEqual(len(page), 20)
                self.assertTrue(all(views.display_width(line) <= 40 for line in page))

    def test_log_cursor_handles_missing_rotation_truncation_and_utf8(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "worker.log"
            cursor = views.LogCursor(path)
            self.assertEqual(cursor.read(), "")
            path.write_bytes(b"first\n")
            self.assertIn("first", cursor.read())
            with path.open("ab") as f:
                f.write("中文".encode()[:2])
            self.assertEqual(cursor.read(), "")
            with path.open("ab") as f:
                f.write("中文".encode()[2:])
            self.assertEqual(cursor.read(), "中文")
            path.write_text("new and longer after truncation\n", encoding="utf-8")
            self.assertIn("new and longer", cursor.read())
            path.rename(path.with_suffix(".old"))
            path.write_text("rotated\n", encoding="utf-8")
            self.assertIn("rotated", cursor.read())
            self.assertEqual(cursor.read(), "")

    def test_static_log_tail_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "worker.log"
            path.write_text("old\n" * 50000 + "latest\n", encoding="utf-8")
            text = views.LogCursor(path).read()
            self.assertLessEqual(len(text), 65536)
            self.assertIn("latest", text)


if __name__ == "__main__":
    unittest.main()
