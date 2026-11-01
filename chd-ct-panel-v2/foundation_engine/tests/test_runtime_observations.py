from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))
import runtime_observations as observations
from server_data import read_json


class ObservationTests(unittest.TestCase):
    def test_actual_subprocess_output_and_nonzero_status(self):
        value=observations.probe_command('toy',[sys.executable,'-c','print("actual evidence")'])
        self.assertEqual(value['status'],'passed')
        self.assertIn('actual evidence',value['stdout'])
        value=observations.probe_command('failed',[sys.executable,'-c','raise SystemExit(7)'])
        self.assertEqual(value['exit_code'],7)
        self.assertEqual(value['status'],'failed')

    def test_timeout_and_missing_tool_remain_missing_evidence(self):
        value=observations.probe_command('slow',[sys.executable,'-c','import time; time.sleep(5)'],timeout=.1)
        self.assertEqual(value['status'],'timeout')
        value=observations.probe_command('missing',['vista-no-such-tool-317'])
        self.assertEqual(value['status'],'unavailable')

    def test_redaction_and_structured_failure(self):
        secret='https://name:private@host/path password=private token=private'
        self.assertNotIn('private',observations.redact(secret))
        with tempfile.TemporaryDirectory() as tmp:
            try:raise ValueError(secret)
            except ValueError as error:record=observations.failure(tmp,'train',error,rank=1)
            saved=read_json(Path(tmp)/'diagnostics/failure.json')
            self.assertEqual(saved['stage'],'train')
            self.assertEqual(saved['context']['rank'],1)
            self.assertNotIn('private',saved['traceback'])
            self.assertEqual(record['error_type'],'ValueError')

    def test_collection_saves_observed_failures_without_filling_defaults(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(observations,'probe_command',
                side_effect=lambda name,cmd:dict(name=name,command=cmd,status='unavailable',error='missing')):
            value=observations.collect(tmp,{'results':tmp})
            missing=read_json(Path(tmp)/'diagnostics/missing-evidence.json')['items']
            self.assertEqual(len(missing),6)
            self.assertTrue(value['actual_host'])
            self.assertTrue(value['paths']['results']['exists'])
            self.assertTrue((Path(tmp)/'diagnostics/events').is_dir())


if __name__=='__main__':unittest.main()
