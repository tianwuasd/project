import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'research PyTorch environment required')
class XuTrainingTests(unittest.TestCase):
    def test_paper_shapes_without_allocating_full_weights(self):
        from dataset1282_analysis.xu_models import build_model
        from dataset1282_analysis.xu_training import learning_rate
        with torch.device('meta'),torch.inference_mode():
            for branch,side,width in [('crop64',64,64),('all128',128,48)]:
                model=build_model(branch).eval()
                self.assertEqual((model.spec.width,model.spec.levels,model.spec.batch,model.spec.epochs),
                                 (width,5,4,15))
                self.assertEqual(model(torch.empty(1,1,side,side,side)).shape,(1,11,side,side,side))
        self.assertEqual([learning_rate(i,15) for i in [0,7,8,11,12,14]],
                         [2e-4,2e-4,2e-5,2e-5,2e-6,2e-6])

    def test_paper_loss_matches_dense_reference_and_handles_unknown(self):
        from dataset1282_analysis.xu_training import branch_loss
        torch.manual_seed(6)
        x=torch.randn(2,11,3,4,5,requires_grad=True)
        y=torch.randint(0,11,(2,3,4,5))
        y[0,0,0,0]=-1
        mask=y!=-1
        p=x.softmax(1)*mask[:,None]
        t=torch.nn.functional.one_hot(y.clamp_min(0),11).movedim(-1,1)*mask[:,None]
        axes=(0,2,3,4)
        d=(2*(p*t).sum(axes)+1e-6)/(p.sum(axes)+t.sum(axes)+1e-6)
        w=x.new_tensor([1]+[2]*10)
        expected=1-(d*w).sum()/w.sum()+torch.nn.functional.cross_entropy(x,y,ignore_index=-1)
        torch.testing.assert_close(branch_loss(x,y,[[],[]]),expected)
        y[1].masked_fill_(y[1]==3,0)
        value=branch_loss(x,y,[[],[3]])
        value.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(float(x.grad[0,:,0,0,0].abs().sum()),0.)

    def test_smoke_training_resume_and_role_guards(self):
        from dataset1282_analysis.xu_training import train_branch,load_predictor
        grid=torch.arange(64**3).reshape(64,64,64)
        image=grid.float()/(64**3)
        labels=grid.remainder(11)
        def sample(name,branch):
            self.assertEqual(branch,'crop64')
            return image,labels if name=='train' else labels.masked_fill(labels==3,0)

        old_threads=torch.get_num_threads()
        self.addCleanup(torch.set_num_threads,old_threads)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as d:
            args=dict(branch='crop64',output_dir=d,profile='smoke',load_sample=sample,
                      train_cases={'train':[]},validation_cases={'dev':[3]},data_identity='test-xyz')
            summary=train_branch(**args)
            self.assertEqual(summary['validation_partial_cases'],1)
            self.assertGreater(summary['best_loss'],0)
            state=torch.load(summary['last_checkpoint'],map_location='cpu',weights_only=True)
            self.assertTrue(any(float(i['exp_avg'].abs().sum())>0 for i in state['optimizer']['state'].values()))
            self.assertEqual(summary['best_checkpoint'],train_branch(**args,resume=True)['best_checkpoint'])
            predict=load_predictor(summary['best_checkpoint'],branch='crop64',profile='smoke',data_identity='test-xyz')
            self.assertEqual(predict(image).shape,(64,64,64))
            with self.assertRaisesRegex(ValueError,'contract changed'):
                train_branch(**dict(args,data_identity='changed'),resume=True)
            with self.assertRaisesRegex(ValueError,'contract mismatch'):
                load_predictor(summary['best_checkpoint'],branch='crop128',profile='smoke',data_identity='test-xyz')
            # A later optimizer state is valid for resume but cannot replace the
            # best-scoring epoch when exporting a development prediction.
            state['epoch'] += 1
            later=Path(d)/'later.pt'
            torch.save(state,later)
            with self.assertRaisesRegex(ValueError,'validation-selected'):
                load_predictor(later,branch='crop64',profile='smoke',data_identity='test-xyz')
            with self.assertRaisesRegex(ValueError,'disjoint'):
                train_branch(**dict(args,validation_cases={'train':[]}))
            with self.assertRaisesRegex(ValueError,'requires cases'):
                train_branch(**dict(args,profile='paper'))
