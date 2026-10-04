import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
from run_experiment import validate_checkpoint


class IdentityTests(unittest.TestCase):
    def test_changed_data_or_configuration_rejects_cache(self):
        info={'identity':{'data':'abc','fleet':4},'seed':11,'monotone':True,'training_episodes':400}
        validate_checkpoint(info,info['identity'],11,True,400)
        with self.assertRaises(ValueError):
            validate_checkpoint(info,{'data':'new','fleet':4},11,True,400)
        with self.assertRaises(ValueError):
            validate_checkpoint(info,info['identity'],22,True,400)
        with self.assertRaises(ValueError):
            validate_checkpoint({},info['identity'],11,True,400)
