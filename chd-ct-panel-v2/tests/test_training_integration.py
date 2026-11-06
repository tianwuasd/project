"""Deployment defaults and truthful input-check progress after branch fusion."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


class TrainingIntegrationTests(unittest.TestCase):
    def test_new_completed_default_does_not_override_explicit_or_latest(self):
        from chd_stunet.config import default_prepared
        scratch=ROOT/'outputs/tests-tmp';scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            root=Path(temporary)
            old,new,explicit=[root/name for name in ('old','new','explicit')]
            for path in (old,new,explicit):
                (path/'manifests').mkdir(parents=True);(path/'manifests/imagechd_input.json').write_text('{}')
            with patch('chd_stunet.config.ROOT',root):
                self.assertEqual(default_prepared(dict(completed_preprocessing_default=str(old))),old)
                (root/'outputs').mkdir()
                (root/'outputs/latest_preprocessing.json').write_text(json.dumps(dict(output=str(new))))
                self.assertEqual(default_prepared(dict(completed_preprocessing_default=str(old))),new)
                self.assertEqual(default_prepared(dict(completed_preprocessing_default=str(old),training_prepared_root=str(explicit))),explicit)

    def test_long_input_check_reports_real_stage_and_closes_own_thread(self):
        from chd_stunet.observability.input_check import InputCheckProgress
        seen=[];heartbeat=threading.Event()
        def emit(value):
            seen.append(value)
            if '等待' in value:heartbeat.set()
        with InputCheckProgress('fold0',heartbeat=.01,emit=emit) as feedback:
            feedback.update('ct_synthetic',0,3,'影像SHA256')
            self.assertTrue(heartbeat.wait(2))
            feedback.update('ct_synthetic',1,3,'病例核验完成')
        self.assertFalse(feedback.thread.is_alive())
        self.assertTrue(any('影像SHA256' in line and '0/3' in line for line in seen))


if __name__=='__main__':unittest.main()
