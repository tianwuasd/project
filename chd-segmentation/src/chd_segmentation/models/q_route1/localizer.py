import torch
from torch import nn
import torch.nn.functional as F
from .network import Residual, Decoder


class Localizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Sequential(nn.Conv3d(2,16,3,padding=1,bias=False),nn.GroupNorm(8,16),nn.SiLU(),Residual(16,16))
        self.l1,self.l2,self.l3 = Residual(16,32,2),Residual(32,64,2),Residual(64,96,2)
        self.d2,self.d1,self.d0 = Decoder(96,64),Decoder(64,32),Decoder(32,16)
        self.head = nn.Conv3d(16,3,1)

    def forward(self, image, valid):
        padding = [v for n in reversed(image.shape[2:]) for v in (0,(-n)%32)]
        x = F.avg_pool3d(F.pad(image,padding,mode='replicate'),4,4)
        mask = F.avg_pool3d(F.pad(valid.float(),padding),4,4)
        return self.forward_pooled(x,mask)

    def forward_pooled(self, x, mask):
        l0 = self.l0(torch.cat((x,mask),1))
        l1 = self.l1(l0)
        l2 = self.l2(l1)
        l3 = self.l3(l2)
        logits = self.head(self.d0(self.d1(self.d2(l3,l2),l1),l0))
        return dict(logits=logits,heatmap=logits.sigmoid(),features=l2,valid=mask,
                    feature_origin=1.5,feature_stride=16)

