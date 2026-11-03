"""Independent nnU-Net reference and partial-annotation gradient checks."""
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


class TrainingLossTests(unittest.TestCase):
    def test_complete_matches_native_values_and_gradients(self):
        from chd_stunet.training.losses import PartialDiceCE
        from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
        from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
        torch.manual_seed(11)
        for batch_dice in (False,True):
            a=torch.randn(2,8,3,4,5,requires_grad=True)
            b=a.detach().clone().requires_grad_()
            target=torch.randint(0,9,(2,1,3,4,5))
            actual=PartialDiceCE(batch_dice)(a,target,torch.ones(2,7,dtype=torch.bool))
            reference=DC_and_CE_loss(dict(batch_dice=batch_dice,do_bg=False,smooth=1e-5,ddp=False),{},
                ignore_label=8,dice_class=MemoryEfficientSoftDiceLoss)(b,target)
            torch.testing.assert_close(actual,reference,atol=2e-6,rtol=2e-6)
            actual.backward();reference.backward()
            torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-6)

    def test_partial_background_is_a_probability_union(self):
        from chd_stunet.training.losses import PartialDiceCE
        p=torch.tensor([.1,.02,.03,.04,.05,.7,.02,.04],dtype=torch.float64)
        logits=p.log().reshape(1,8,1,1,1).requires_grad_()
        available=torch.tensor([[True,True,True,True,False,True,True]])
        ce,_=PartialDiceCE().components(logits,torch.zeros(1,1,1,1,1,dtype=torch.long),available)
        self.assertAlmostEqual(ce.item(),-torch.log(torch.tensor(.8)).item(),places=6)
        ce.backward()
        self.assertLess(logits.grad[0,5].item(),0,'Unannotated MYO must not receive an ordinary BG penalty')

    def test_all_ignore_and_mixed_batch_are_finite(self):
        from chd_stunet.training.losses import PartialDiceCE
        logits=torch.randn(2,8,2,3,4,requires_grad=True)
        available=torch.ones(2,7,dtype=torch.bool);available[1,4]=False
        target=torch.full((2,1,2,3,4),8)
        loss=PartialDiceCE()(logits,target,available)
        self.assertEqual(loss.item(),0)
        loss.backward();self.assertEqual(logits.grad.abs().sum().item(),0)
        target[0]=5;target[1]=0
        loss=PartialDiceCE()(logits,target,available)
        self.assertTrue(torch.isfinite(loss))

    def test_rejects_invented_myo_and_other_missing_patterns(self):
        from chd_stunet.training.losses import PartialDiceCE
        available=torch.ones(1,7,dtype=torch.bool);available[0,4]=False
        logits=torch.randn(1,8,2,2,2)
        with self.assertRaisesRegex(ValueError,'invented'):
            PartialDiceCE()(logits,torch.full((1,1,2,2,2),5),available)
        available[0,0]=False
        with self.assertRaisesRegex(ValueError,'Only complete'):
            PartialDiceCE()(logits,torch.zeros(1,1,2,2,2,dtype=torch.long),available)


if __name__=='__main__':unittest.main()
