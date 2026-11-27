import torch
import torch.nn.functional as F
import numpy as np


def crop_local_input(image, base, start, size=128, valid_fov=None):
    tensor = isinstance(image, torch.Tensor)
    shape = image.shape if tensor else (1, *image.shape)
    device = image.device if tensor else base.device
    if shape[0] != 1 or shape[1] != 1 or base.shape != (1,11,*shape[2:]):
        raise ValueError('单例影像和11通道概率需要处于同一网格')
    axes = [torch.arange(s,s+size,device=device) for s in start]
    raw = torch.meshgrid(*axes,indexing='ij')
    valid = torch.ones((size,)*3,dtype=torch.bool,device=device)
    coords = []
    indices = []
    for grid,n in zip(raw,shape[2:]):
        valid &= (grid >= 0) & (grid < n)
        coords.append(2*(grid+.5)/n-1)
        indices.append(grid.clamp(0,n-1))
    if valid_fov is not None:
        if valid_fov.shape != shape or valid_fov.dtype != torch.bool:
            raise ValueError('有效视野需要与影像同形状的bool数组')
        valid &= valid_fov[(0,0,*indices)]
    if tensor:
        x = image[(slice(None),slice(None),*indices)]
    else:
        x = torch.from_numpy(np.array(image[(slice(None), *(index.cpu().numpy() for index in indices))], copy=True))[None].to(device)
    probability = base[(slice(None),slice(None),*indices)].detach().clone()
    probability = probability*valid[None,None]
    probability[:,0] += ~valid
    coordinate = torch.stack(coords,0)[None]
    mask = valid[None,None]
    inputs = torch.cat((x,probability,coordinate,mask.float()),1)
    editable = mask.clone()
    margin = size//8
    core = torch.zeros_like(mask)
    core[:,:,margin:size-margin,margin:size-margin,margin:size-margin] = True
    editable &= core
    return inputs,editable


def sample_context(features, center, feature_origin=1.5, feature_stride=16, samples=16):
    axes = []
    for c,n in zip(center,features.shape[2:]):
        original = c + 16*(torch.arange(samples,device=features.device,dtype=torch.float32)-(samples-1)/2)
        index = (original-feature_origin)/feature_stride
        axes.append(2*(index+.5)/n-1)
    d,h,w = torch.meshgrid(*axes,indexing='ij')
    grid = torch.stack((w,h,d),-1)[None]
    return F.grid_sample(features.detach(),grid,mode='bilinear',padding_mode='zeros',align_corners=False)

