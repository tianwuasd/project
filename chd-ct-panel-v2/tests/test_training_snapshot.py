from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


class TrainingSnapshotTests(unittest.TestCase):
    def test_frozen_code_checks_all_bytes_and_semantic_contract(self):
        from chd_stunet.training.contracts import scientific_sources
        from chd_stunet.training.code_snapshot import freeze_code
        scratch=Path(__file__).resolve().parents[1]/'outputs/tests-tmp';scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            signature=dict(producer_ast_sha256=scientific_sources())
            entry=freeze_code(Path(temporary),signature,True)
            self.assertTrue(entry.is_file())
            self.assertEqual(freeze_code(Path(temporary),signature,False),entry)
            target=entry.parent/'src/chd_stunet/training/data.py'
            target.write_bytes(target.read_bytes()+b'\n# unexpected snapshot modification\n')
            with self.assertRaisesRegex(ValueError,'checksum differs'):
                freeze_code(Path(temporary),signature,False)


if __name__=='__main__':unittest.main()
