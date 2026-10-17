"""Exercise the real repair window with synthetic CT, never patient data."""
import json
import gc
from pathlib import Path
import tempfile
import time
import tkinter as tk
import unittest

import numpy as np
from pydicom.uid import generate_uid

from pipeline_gui import App
from test_converters import dicom


class RepairWindowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.source = self.base / 'source'
        self.source.mkdir()
        self.output = self.base / 'output'
        study, series = generate_uid(), generate_uid()
        first = dicom(self.source / 'a.dcm', np.ones((2, 3), dtype=np.int16), study=study, series=series)
        dicom(self.source / 'b.dcm', np.ones((2, 3), dtype=np.int16), position=(0, 0, 1), study=study, series=series)
        first.PixelData = b''
        first.save_as(self.source / 'empty_copy.dcm', enforce_file_format=True)
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = App(self.root)
        self.app.settings = self.base / 'settings.json'
        self.app.source.set(str(self.source))
        self.app.output.set(str(self.output))
        self.app.patients.set('')

    def tearDown(self):
        for panel in [self.app, getattr(self.app, 'repair_window', None)]:
            if panel and panel.control:
                panel.control.stop()
            if panel and panel.thread:
                panel.thread.join(10)
        try:
            self.root.destroy()
        except tk.TclError:
            pass
        self.app=None;self.root=None;panel=None
        gc.collect()  # Release Tk variables on their owning thread between real UI tests.
        self.tmp.cleanup()

    def wait(self, predicate):
        until = time.monotonic() + 15
        while not predicate():
            self.root.update()
            if time.monotonic() > until:
                self.fail('GUI timed out')
            time.sleep(.01)

    def run_panel(self, panel):
        panel.start()
        self.wait(lambda: panel.thread is not None and not panel.thread.is_alive()
                  and str(panel.start_button['state']) == 'normal')

    def snapshot(self):
        return json.loads((self.output / '项目数据快照.json').read_text(encoding='utf-8-sig'))

    def test_repair_window_recovers_copy_and_options_do_not_leak_to_main(self):
        self.assertTrue(callable(getattr(self.app, 'open_repair', None)), '独立修复入口缺失')
        self.run_panel(self.app)
        self.assertEqual(self.snapshot()['overview']['CT']['failed'], 1)
        self.app.open_repair()
        repair = self.app.repair_window
        repair.root.withdraw()
        self.app.open_repair()
        self.assertIs(self.app.repair_window, repair)
        self.assertEqual(repair.source.get(), '')  # retry catalog, no accidental full rescan
        self.assertEqual(repair.output.get(), str(self.output))
        repair.recover_copies.set(True)
        self.run_panel(repair)
        self.assertEqual(self.snapshot()['overview']['CT']['success'], 1,
                         repair.log.get('1.0','end')+str(self.snapshot()['issues']))
        self.assertIn('CT：成功 0 → 1', repair.result.get())
        self.assertIn('失败 1 → 0', repair.result.get())
        self.assertEqual(float(repair.overall['value']), 100)
        self.app.mode.set('仅更新报告（无需原资料）')
        self.run_panel(self.app)
        status = json.loads((self.output / 'project_run_status.json').read_text(encoding='utf-8'))
        self.assertEqual(status['options']['action'], 'report')
        self.assertFalse(status['options']['recover_copies'])
        self.assertFalse(status['options']['match_names'])
        self.assertFalse(status['options']['split_phases'])

    def test_close_repair_safely_stops_and_unlocks_main(self):
        self.assertTrue(callable(getattr(self.app, 'open_repair', None)), '独立修复入口缺失')
        self.app.open_repair()
        repair = self.app.repair_window
        repair.root.withdraw()
        repair.source.set(str(self.source))
        repair.start()
        repair.toggle_pause()
        self.wait(lambda: repair.status.get().startswith('已暂停'))
        self.assertEqual(str(self.app.start_button['state']), 'disabled')
        self.app.start()
        self.assertIsNone(self.app.thread)
        repair.close()
        self.wait(lambda: self.app.repair_window is None)
        self.assertEqual(str(self.app.start_button['state']), 'normal')
        state = json.loads((self.output / 'project_run_status.json').read_text(encoding='utf-8'))
        self.assertEqual(state['status'], 'interrupted')
        self.app.open_repair()
        self.assertIsNot(self.app.repair_window, repair)

    def test_main_close_stops_active_repair_before_destroy(self):
        self.assertTrue(callable(getattr(self.app, 'open_repair', None)), '独立修复入口缺失')
        self.app.open_repair()
        repair = self.app.repair_window
        repair.root.withdraw()
        repair.source.set(str(self.source))
        repair.start()
        repair.toggle_pause()
        self.wait(lambda: repair.status.get().startswith('已暂停'))
        self.app.close()
        self.wait(lambda: not repair.thread.is_alive())
        self.assertTrue(self.app.closing)
        state = json.loads((self.output / 'project_run_status.json').read_text(encoding='utf-8'))
        self.assertEqual(state['status'], 'interrupted')

    def test_retry_without_selected_repair_shows_remaining_failure(self):
        self.assertTrue(callable(getattr(self.app, 'open_repair', None)), '独立修复入口缺失')
        self.run_panel(self.app)
        self.app.open_repair()
        repair = self.app.repair_window
        repair.root.withdraw()
        self.run_panel(repair)
        self.assertEqual(self.snapshot()['overview']['CT']['failed'], 1)
        self.assertIn('失败 1 → 1', repair.result.get())
        self.assertIn('仍有问题', repair.status.get())

    def test_minimum_window_keeps_results_controls_and_footer_visible(self):
        self.app.open_repair()
        repair=self.app.repair_window
        repair.result.set('全项目报告前后对比：\nCT：成功 0 → 1；失败 1 → 0\n超声：成功 0 → 0；失败 0 → 0\nPDF：成功 0 → 0；失败 0 → 0')
        repair.root.geometry('900x760')
        self.root.update();self.root.update()
        frame=repair.root.winfo_children()[0]
        footer=next(w for w in frame.winfo_children() if 'text' in w.keys() and str(w['text']).startswith('暂停/停止'))
        self.assertTrue(footer.winfo_ismapped(), '安全停止说明被日志区域挤出窗口')
        self.assertTrue(repair.start_button.winfo_ismapped())
        self.assertGreater(repair.log.winfo_height(), 10)


if __name__ == '__main__':
    unittest.main()
