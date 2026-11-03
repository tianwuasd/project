from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


class ResumeEdges(unittest.TestCase):
    def test_validation_boundary_cursor_and_overflow_accounting(self):
        from chd_stunet.training.engine import initial_state,validate_state
        from chd_stunet.training.contracts import recipe
        settings=recipe(True)
        state=initial_state();validate_state(state,settings)
        state.update(phase='validation',step=0,attempts=4,updates=2,overflows=2,
                     updates_by_cohort=dict(main=1,auxiliary=1))
        validate_state(state,settings)
        state['attempts']=3
        with self.assertRaisesRegex(ValueError,'cursor mismatch'):validate_state(state,settings)

    def test_final_metadata_repair_uses_completed_checkpoint_without_gpu(self):
        from chd_stunet.training.engine import run_train,initial_state
        from chd_stunet.training.checkpoints import save_checkpoint
        from chd_stunet.observability.runlog import write_json
        scratch=Path(__file__).resolve().parents[1]/'outputs/tests-tmp';scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            root=Path(temporary);prepared=root/'prepared';prepared.mkdir();output=root/'training';output.mkdir()
            signature={'identity':'same'};write_json(output/'signature.json',signature)
            model=torch.nn.Linear(2,1);optimizer=torch.optim.SGD(model.parameters(),lr=.1)
            scaler=torch.amp.GradScaler('cuda',enabled=False)
            state=initial_state();state.update(epoch=2,attempts=8,updates=8,history=[{},{}],
                updates_by_cohort=dict(main=4,auxiliary=4))
            save_checkpoint(output/'checkpoints',model,optimizer,scaler,state,signature,torch.device('cpu'))
            dataset=dict(plan={'configurations':{'3d_fullres':{'spacing':[.75,.3,.3]}}},
                         identity={},train_ids=['a','b'],val_ids=['v'])
            with patch('chd_stunet.training.engine.load_prepared',return_value=dataset), \
                 patch('chd_stunet.training.engine.make_signature',return_value=signature), \
                 patch('chd_stunet.training.engine.freeze_code'), \
                 patch('chd_stunet.training.engine.admit_device',side_effect=AssertionError('GPU must stay unused')):
                result=run_train(prepared,output,smoke=True,train_cases=['a','b'],val_cases=['v'],resume=True)
            self.assertEqual(result['status'],'completed')
            self.assertTrue((output/'checkpoints/final.json').is_file())
            self.assertTrue((output/'summary.json').is_file())


if __name__=='__main__':unittest.main()
