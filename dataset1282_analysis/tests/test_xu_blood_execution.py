from contextlib import contextmanager
import copy
import os
import unittest

import torch

from dataset1282_analysis.xu_blood import BloodForward, _objective
from dataset1282_analysis.xu_blood_models import BloodSequence, SliceUNet


@contextmanager
def execution(mode):
    before = os.environ.get('XU_BLOOD_EXECUTION')
    os.environ['XU_BLOOD_EXECUTION'] = mode
    try:
        yield
    finally:
        if before is None:
            os.environ.pop('XU_BLOOD_EXECUTION', None)
        else:
            os.environ['XU_BLOOD_EXECUTION'] = before


class BloodExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_grouped_frozen_encoder_preserves_loss_gradient_and_adam_update(self):
        torch.manual_seed(11)
        encoder = SliceUNet('smoke').eval()
        sequence = BloodSequence(2).train()
        original = copy.deepcopy(sequence.state_dict())
        images = torch.randn(1,7,32,32)
        labels = torch.randint(0,3,(1,32,32))
        results = []
        for mode in ('legacy','auto'):
            sequence.load_state_dict(original)
            optimizer = torch.optim.Adam(sequence.parameters(),lr=2e-4)
            with execution(mode):
                forward = BloodForward(encoder,'cpu',True,1)
                scores = forward(images,sequence)
                loss = _objective(scores,labels)
                loss.backward()
                gradients = [p.grad.clone() for p in sequence.parameters()]
                optimizer.step()
                results.append((scores.detach(),loss.detach(),gradients,copy.deepcopy(sequence.state_dict())))
            sequence.zero_grad(set_to_none=True)
        for a,b in zip(results[0][:2],results[1][:2]):
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
        for a,b in zip(results[0][2],results[1][2]):
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
        for key in original:
            torch.testing.assert_close(results[0][3][key],results[1][3][key],atol=1e-6,rtol=1e-5)

    def test_development_reuses_only_identical_frames_and_clears_between_cases(self):
        torch.manual_seed(13)
        encoder = SliceUNet('smoke').eval()
        sequence = BloodSequence(2).eval()
        planes = torch.randn(9,1,32,32)
        outputs, encoded = [], []
        for mode in ('legacy','auto'):
            with execution(mode), torch.no_grad():
                forward = BloodForward(encoder,'cpu',True,1)
                predictions = []
                for centre in (3,4,5):
                    keys = [('case-a',z) for z in range(centre-3,centre+4)]
                    images = planes[centre-3:centre+4].transpose(0,1)
                    predictions.append(forward(images,sequence,keys))
                outputs.append(predictions)
                encoded.append(forward.distribution.stats['cases_by_device']['cpu'])
                self.assertLessEqual(len(forward.frame_cache),7)
                forward.clear_frames()
                self.assertEqual(len(forward.frame_cache),0)
        self.assertEqual(encoded,[21,9])
        for a,b in zip(*outputs):
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)

    def test_training_refuses_cross_window_feature_reuse(self):
        with execution('auto'):
            forward = BloodForward(SliceUNet('smoke').eval(),'cpu',True,1)
            with self.assertRaisesRegex(ValueError,'开发集'):
                forward(torch.randn(1,7,32,32),BloodSequence(2).train(),list(range(7)))

    def test_checkpoint_switch_preserves_parameter_keys_and_real_gradients(self):
        torch.manual_seed(17)
        model = BloodSequence(32).train()
        optimizer = torch.optim.Adam(model.parameters(),lr=2e-4)
        keys = tuple(model.state_dict())
        frames = [torch.randn(1,32,12,12) for _ in range(7)]
        states, grads = [], []
        for enabled in (True,False):
            optimizer.zero_grad(set_to_none=True)
            model.activation_checkpointing = enabled
            scores = model(frames)
            scores.square().mean().backward()
            states.append(scores.detach())
            grads.append([p.grad.clone() for p in model.parameters()])
            self.assertEqual(tuple(model.state_dict()),keys)
        torch.testing.assert_close(*states,atol=0,rtol=0)
        for a,b in zip(*grads):
            torch.testing.assert_close(a,b,atol=0,rtol=0)
