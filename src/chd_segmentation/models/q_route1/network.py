import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .events import endpoints, pad_edge


def projection(cin, cout):
    return nn.Sequential(nn.Conv3d(cin,cout,1,bias=False),nn.GroupNorm(8,cout),nn.SiLU())


class Residual(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.body = nn.Sequential(nn.Conv3d(cin,cout,3,stride,1,bias=False),nn.GroupNorm(8,cout),nn.SiLU(),
                                  nn.Conv3d(cout,cout,3,padding=1,bias=False),nn.GroupNorm(8,cout))
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Conv3d(cin,cout,1,stride,bias=False)

    def forward(self, x):
        return F.silu(self.body(x)+self.skip(x))


class Decoder(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.project = nn.Conv3d(cin,cout,1,bias=False)
        self.block = Residual(cout*2,cout)

    def forward(self, x, skip):
        x = self.project(F.interpolate(x,size=skip.shape[2:],mode='trilinear',align_corners=False))
        return self.block(torch.cat((x,skip),1))


class LocalNetwork(nn.Module):
    def __init__(self, relation_fields=(), checkpointing=True, method='M07'):
        super().__init__()
        self.checkpointing = checkpointing
        self.method = method
        from .comparisons import TRAINABLE_METHODS,ConditionalDecoder
        if method not in TRAINABLE_METHODS:
            raise ValueError('局部网络方法编号不合法')
        self.relation_fields = tuple(relation_fields)
        self.stem = nn.Sequential(nn.Conv3d(16,24,3,padding=1,bias=False),nn.GroupNorm(8,24),nn.SiLU())
        self.e0, self.e1, self.e2, self.e3 = Residual(24,24),Residual(24,48,2),Residual(48,96,2),Residual(96,192,2)
        self.context = projection(64,32)
        self.combine = projection(224,192)
        self.d2, self.d1, self.d0 = Decoder(192,96),Decoder(96,48),Decoder(48,24)
        self.segmentation = nn.Conv3d(24,11,1)
        nn.init.zeros_(self.segmentation.weight)
        nn.init.zeros_(self.segmentation.bias)
        self.edge = nn.Sequential(nn.Conv3d(51,24,1),nn.SiLU(),nn.Conv3d(24,13,1)) if method != 'M01' else None
        self.relation = nn.Sequential(nn.Linear(224,64),nn.SiLU(),nn.Linear(64,len(relation_fields))) if relation_fields else None
        self.conditional = ConditionalDecoder() if method == 'M03' else None

    def _call(self, module, *args):
        if not self.training or not self.checkpointing:
            return module(*args)
        if isinstance(module,Decoder):
            x,skip = args
            x = checkpoint(F.interpolate,x,size=skip.shape[2:],mode='trilinear',align_corners=False,use_reentrant=False)
            x = self._call(module.project,x)
            return self._call(module.block,torch.cat((x,skip),1))
        if isinstance(module,Residual):
            x, = args
            return checkpoint(F.silu,self._call(module.body,x)+self._call(module.skip,x),use_reentrant=False)
        if isinstance(module,nn.Sequential):
            x, = args
            for layer in module:
                x = self._call(layer,x)
            return x
        return checkpoint(module,*args,use_reentrant=False)

    def forward(self, inputs, context, editable):
        if inputs.shape[1] != 16 or any(n%8 for n in inputs.shape[2:]):
            raise ValueError('局部输入需要16通道且各边长为8的倍数')
        e0 = self._call(self.e0,self._call(self.stem,inputs))
        e1 = self._call(self.e1,e0)
        e2 = self._call(self.e2,e1)
        e3 = self._call(self.e3,e2)
        if context.shape != (inputs.shape[0],64,*e3.shape[2:]):
            raise ValueError('上下文形状不符合编码器网格')
        ctx = self._call(self.context,context.detach())
        merged = self._call(self.combine,torch.cat((e3,ctx),1))
        decoded = self._call(self.d0,self._call(self.d1,self._call(self.d2,merged,e2),e1),e0)
        base = inputs[:,1:12].detach().float().clamp_min(1e-6).log()
        z0 = base + self._call(self.segmentation,decoded).float()*editable
        z0 = z0-z0.mean(1,keepdim=True)
        # 正式128³读取框的核心固定为16:112；小尺寸仅供接口与数值验收。
        margin = inputs.shape[-1]//8
        core = tuple(slice(margin,n-margin) for n in inputs.shape[2:])
        features = decoded[(slice(None),slice(None),*core)]
        logits = []
        for direction in range(3) if self.edge is not None else ():
            left,right = endpoints(features,direction)
            onehot = features.new_zeros((inputs.shape[0],3,*left.shape[2:]))
            onehot[:,direction] = 1
            edge = self._call(self.edge,torch.cat((left,right,onehot),1)).float()
            logits.append(pad_edge(edge,direction))
        logits = torch.stack(logits,1) if logits else None
        q = logits.softmax(2) if logits is not None else None
        relation = self.relation(torch.cat((e3.mean((2,3,4)),ctx.mean((2,3,4))),1)) if self.relation is not None else None
        conditional = self._call(self.conditional,features,z0[(slice(None),slice(None),*core)],q,
                                 editable[(slice(None),slice(None),*core)]) if self.conditional is not None else None
        return dict(z0=z0,q=q,q_logits=logits,relation_logits=relation,core=core,conditional_logits=conditional)

