import torch


def core_weight(size=96, width=8, device=None):
    index = torch.arange(size,device=device,dtype=torch.float32)
    distance = torch.minimum(index,size-1-index)
    w = torch.sin(torch.pi/2*(distance/width).clamp_max(1)).square()
    return w[:,None,None]*w[None,:,None]*w[None,None,:]


class ProbabilityFusion:
    def __init__(self, base):
        if base.shape[1] != 11:
            raise ValueError('基础概率需要11通道')
        self.base = base.detach().float()
        self.total = torch.zeros_like(self.base)
        self.weights = torch.zeros_like(self.base[:,:1])

    def add(self, probability, start, weight):
        shape = probability.shape[2:]
        target,source = [],[]
        for low,size,limit in zip(start,shape,self.base.shape[2:]):
            left,right = max(0,low),min(limit,low+size)
            if right <= left:
                return
            target.append(slice(left,right))
            source.append(slice(left-low,right-low))
        dst = (slice(None),slice(None),*target)
        src = (slice(None),slice(None),*source)
        w = weight[tuple(source)][None,None]
        self.total[dst] += probability[src].float()*w
        self.weights[dst] += w

    def finish(self):
        result = ((1-self.weights).clamp_min(0)*self.base+self.total)/self.weights.clamp_min(1)
        return torch.where(self.weights == 0,self.base,result)

