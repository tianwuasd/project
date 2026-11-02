"""A refused resume must preserve a previous run, including its code evidence."""
from pathlib import Path
import sys
import tempfile
import unittest
import nibabel as nib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


class RecoveryTests(unittest.TestCase):
    def test_incompatible_resume_is_read_only_for_old_run(self):
        from chd_stunet.analysis_run import run_analysis
        scratch=ROOT/"outputs"/"tests-tmp"
        scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            folder=Path(temporary).resolve()
            self.assertTrue(folder.is_relative_to(scratch.resolve()))
            raw=folder/"raw";raw.mkdir()
            for kind in ("image","label"):
                nib.save(nib.Nifti1Image(np.zeros((3,3,3),dtype=np.int16),np.eye(4)),raw/f"a_{kind}.nii.gz")
            old=folder/"old";old.mkdir()
            (old/"signature.json").write_text('{"frozen":"different contract"}',encoding="utf-8")
            (old/"code_provenance.json").write_bytes(b"frozen original code evidence")
            before={p.relative_to(old).as_posix():p.read_bytes() for p in old.rglob("*") if p.is_file()}
            with self.assertRaisesRegex(ValueError,"Resume contract differs"):
                run_analysis(raw,old,resume=True)
            after={p.relative_to(old).as_posix():p.read_bytes() for p in old.rglob("*") if p.is_file()}
            self.assertEqual(after,before)


if __name__=="__main__":unittest.main()