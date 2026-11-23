import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .network import Residual, Decoder


class OffloadedLocalizer:
    def __init__(self, model, block_bytes, boundary=None):
        if type(block_bytes) is not int or block_bytes < 1:
            raise ValueError('低显存卷积块预算需要正整数')
        self.model, self.block_bytes, self.boundary = model, block_bytes, boundary

    def convolution(self, layer, x):
        shape = x.shape[2:]
        output = tuple((n+2*p-d*(k-1)-1)//s+1 for n,p,d,k,s in
                       zip(shape,layer.padding,layer.dilation,layer.kernel_size,layer.stride))
        extent = list(output)
        def workspace(values):
            source = math.prod((v-1)*s+d*(k-1)+1 for v,s,d,k in
                               zip(values,layer.stride,layer.dilation,layer.kernel_size))
            return x.shape[0]*4*6*(source*layer.in_channels+math.prod(values)*layer.out_channels)
        while workspace(extent) > self.block_bytes and max(extent) > 1:
            axis = max(range(3),key=lambda i:extent[i])
            extent[axis] = (extent[axis]+1)//2
        device = layer.weight.device
        def compute(source):
            if self.boundary is not None:
                self.boundary()
            return F.conv3d(source.to(device),layer.weight,layer.bias,layer.stride,0,
                            layer.dilation,layer.groups).cpu()
        depths = []
        for d in range(0,output[0],extent[0]):
            heights = []
            for h in range(0,output[1],extent[1]):
                widths = []
                for w in range(0,output[2],extent[2]):
                    start = (d,h,w)
                    end = tuple(min(a+b,n) for a,b,n in zip(start,extent,output))
                    lo = tuple(a*s-p for a,s,p in zip(start,layer.stride,layer.padding))
                    hi = tuple((b-1)*s-p+di*(k-1)+1 for b,s,p,di,k in
                               zip(end,layer.stride,layer.padding,layer.dilation,layer.kernel_size))
                    source = x[(slice(None),slice(None),*(slice(max(0,a),min(n,b)) for a,b,n in zip(lo,hi,shape)))]
                    padding = [v for a,b,n in reversed(list(zip(lo,hi,shape)))
                               for v in (max(0,-a),max(0,b-n))]
                    source = F.pad(source,padding)
                    widths.append(checkpoint(compute,source,use_reentrant=False))
                heights.append(torch.cat(widths,dim=4))
            depths.append(torch.cat(heights,dim=3))
        return torch.cat(depths,dim=2)

    def call(self, module, *args):
        if isinstance(module, nn.Conv3d):
            return self.convolution(module,*args)
        if isinstance(module, nn.GroupNorm):
            # 整个病例共同计算统计量，CPU反向通过复制操作回传到原GPU参数。
            return F.group_norm(args[0],module.num_groups,
                                module.weight.cpu() if module.weight is not None else None,
                                module.bias.cpu() if module.bias is not None else None,module.eps)
        if isinstance(module, Residual):
            return F.silu(self.call(module.body,args[0])+self.call(module.skip,args[0]))
        if isinstance(module, Decoder):
            x, skip = args
            x = F.interpolate(x,size=skip.shape[2:],mode='trilinear',align_corners=False)
            return self.call(module.block,torch.cat((self.call(module.project,x),skip),1))
        if isinstance(module, nn.Sequential):
            x, = args
            for child in module:
                x = self.call(child,x)
            return x
        if isinstance(module,(nn.SiLU,nn.Identity)):
            return module(*args)
        raise TypeError('定位器低显存路径不支持的算子：'+type(module).__name__)

    def forward(self, x, valid):
        l0 = self.call(self.model.l0,torch.cat((x.cpu(),valid.cpu()),1))
        l1 = self.call(self.model.l1,l0)
        l2 = self.call(self.model.l2,l1)
        l3 = self.call(self.model.l3,l2)
        decoded = self.call(self.model.d0,self.call(self.model.d1,self.call(self.model.d2,l3,l2),l1),l0)
        return self.call(self.model.head,decoded).float()
