"""No ML dependencies: reproduce the pinned parent's filename-only contract."""
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))
from checkpoint_compat import load_parent_checkpoint


class TorchFiles:
    @staticmethod
    def save(value,filename):
        with Path(filename).open('wb') as stream:pickle.dump(value,stream)

    @staticmethod
    def load(filename,**kwargs):
        with Path(filename).open('rb') as stream:return pickle.load(stream)


class CheckpointCompatibility(unittest.TestCase):
    def fixture(self):
        return dict(network_weights={'weight':2},optimizer_state={'momentum':3},
                    current_epoch=1,vista_rng_state={'python':'frozen'})

    def test_file_and_path_preserve_checkpoint_and_use_string_for_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'checkpoint.pth';TorchFiles.save(self.fixture(),path)
            before=path.read_bytes();calls=[]
            def upstream(value):
                self.assertIsInstance(value,str);calls.append(value)
                self.assertEqual(TorchFiles.load(value),self.fixture())
            for value in [str(path),path]:
                self.assertEqual(load_parent_checkpoint(upstream,value,TorchFiles),self.fixture())
            self.assertEqual(calls,[str(path),str(path)]);self.assertEqual(path.read_bytes(),before)

    def test_mapping_is_bridged_through_file_and_temporary_is_removed(self):
        paths=[];value=self.fixture()
        def upstream(filename):
            self.assertIsInstance(filename,str);paths.append(Path(filename))
            self.assertEqual(TorchFiles.load(filename),value)
        self.assertIs(load_parent_checkpoint(upstream,value,TorchFiles),value)
        self.assertFalse(paths[0].exists());self.assertFalse(paths[0].parent.exists())

    def test_parent_error_propagates_and_mapping_temporary_is_removed(self):
        paths=[]
        def upstream(filename):
            paths.append(Path(filename));raise ValueError('corrupt model checkpoint')
        with self.assertRaisesRegex(ValueError,'corrupt model'):
            load_parent_checkpoint(upstream,self.fixture(),TorchFiles)
        self.assertFalse(paths[0].exists())

    def test_invalid_input_does_not_call_parent(self):
        calls=[]
        with self.assertRaises(TypeError):load_parent_checkpoint(calls.append,[],TorchFiles)
        self.assertFalse(calls)
