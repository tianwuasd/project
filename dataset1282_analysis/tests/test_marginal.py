"""Numerical contracts; run in the research interpreter."""
import unittest
import numpy as np
try:
    import torch
except ImportError:
    raise unittest.SkipTest('Run test_marginal with the research interpreter')
from dataset1282_analysis.training.marginal import marginal_loss, MarginalBatchLoss, observed_confusion


class MarginalTests(unittest.TestCase):
    def test_full_ce_equals_reference(self):
        torch.manual_seed(4)
        x=torch.randn(2,11,3,4,requires_grad=True)
        y=torch.randint(0,11,(2,1,3,4))
        actual=marginal_loss(x,y,[[],[]],dice_weight=0)
        expected=torch.nn.functional.cross_entropy(x,y[:,0])
        torch.testing.assert_close(actual,expected)
        ga=torch.autograd.grad(actual,x,retain_graph=True)[0]
        gb=torch.autograd.grad(expected,x)[0]
        torch.testing.assert_close(ga,gb)

    def test_unknown_mass_is_not_background_negative(self):
        p=torch.tensor([.1,.2,.7]).reshape(1,3,1)
        q=torch.tensor([.7,.2,.1]).reshape(1,3,1)
        y=torch.zeros(1,1,1,dtype=torch.long)
        torch.testing.assert_close(marginal_loss(p.log(),y,[[2]]),marginal_loss(q.log(),y,[[2]]))
        self.assertGreater(marginal_loss(p.log(),y,[[]]).item(),marginal_loss(q.log(),y,[[]]).item())

    def test_ignore_and_metadata_conflict(self):
        x=torch.randn(1,3,2,requires_grad=True)
        self.assertEqual(marginal_loss(x,torch.full((1,1,2),-1),[[2]]).item(),0)
        with self.assertRaises(ValueError):
            marginal_loss(x,torch.tensor([[[2,0]]]),[[2]])
        with self.assertRaises(ValueError):
            MarginalBatchLoss().bind(['missing'],{})

    def test_deep_supervision_gradient_and_unknown_metrics(self):
        loss=MarginalBatchLoss()
        loss.bind(['a'],{'a':{'unknown_classes':[2]}})
        x=torch.randn(1,3,4,requires_grad=True)
        y=torch.tensor([[[0,1,0,1]]])
        value=loss([x,x[:,:,:2]],[y,y[:,:,:2]])
        value.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        tp,fp,fn=observed_confusion(x,y,[[2]])
        self.assertEqual(tp[1]+fp[1]+fn[1],0)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
