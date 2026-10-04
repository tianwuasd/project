import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'skills/medical-ai-research/scripts'))
from new_study import create_study


class StudyTests(unittest.TestCase):
    def test_create_and_preserve_user_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / '心脏研究'
            state = create_study('心脏研究', out)
            self.assertEqual(state['status'], 'planned')
            self.assertEqual(json.loads((out / 'state.json').read_text(encoding='utf-8'))['topic'], '心脏研究')
            artifact = out / 'research_package.md'
            artifact.write_text('用户已编辑', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                create_study('另一主题', out)
            self.assertEqual(artifact.read_text(encoding='utf-8'), '用户已编辑')


if __name__ == '__main__':
    unittest.main()
