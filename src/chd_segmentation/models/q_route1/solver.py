from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .events import PAIRS, endpoints, pad_edge, event_probabilities, edge_masks


def energy(z, z0, q, editable, gamma=1., lambda_=0.5, active=None):
    probability = z.softmax(1)
    result = gamma/2*((z-z0).square()*editable).flatten(1).sum(1)
    masks = edge_masks(editable)
    for d in range(3):
        left,right = endpoints(probability,d)
        condition,_ = endpoints(q[:,d],d)
        weight,_ = endpoints(masks[:,d:d+1],d)
        if active is not None:
            selected = condition[:,1:] * active.to(condition)[None,:,None,None,None]
            condition = torch.cat((1-selected.sum(1,keepdim=True),selected),1)
        delta = event_probabilities(left,right,active)-condition
        result = result + lambda_/6*(delta.square()*weight).flatten(1).sum(1)
    return result


def _dense_gradient(z, z0, q, editable, gamma=1., lambda_=0.5, active=None):
    probability = z.softmax(1)
    gp = torch.zeros_like(probability)
    masks = edge_masks(editable)
    for direction in range(3):
        left,right = endpoints(probability,direction)
        condition,_ = endpoints(q[:,direction],direction)
        weight,_ = endpoints(masks[:,direction:direction+1],direction)
        if active is not None:
            selected = condition[:,1:] * active.to(condition)[None,:,None,None,None]
            condition = torch.cat((1-selected.sum(1,keepdim=True),selected),1)
        delta = event_probabilities(left,right,active)-condition
        coefficient = lambda_/3*(delta[:,1:]-delta[:,0:1])*weight
        if active is not None:
            coefficient = coefficient * active.to(coefficient)[None,:,None,None,None]
        # 使用稀疏事件矩阵一次累加全部类别，保留两端及外层计算图。
        a = F.one_hot(torch.tensor([c for c,d in PAIRS],device=z.device),11).to(z.dtype)
        b = F.one_hot(torch.tensor([d for c,d in PAIRS],device=z.device),11).to(z.dtype)
        cross = torch.tensor([c != d for c,d in PAIRS],device=z.device,dtype=z.dtype)
        left_c = torch.einsum('jc,bc...->bj...',a,left)
        left_d = torch.einsum('jc,bc...->bj...',b,left)
        right_c = torch.einsum('jc,bc...->bj...',a,right)
        right_d = torch.einsum('jc,bc...->bj...',b,right)
        scale = cross[None,:,None,None,None]
        gu = torch.einsum('jc,bj...->bc...',a,coefficient*right_d) + torch.einsum('jc,bj...->bc...',b,coefficient*right_c*scale)
        gv = torch.einsum('jc,bj...->bc...',b,coefficient*left_c) + torch.einsum('jc,bj...->bc...',a,coefficient*left_d*scale)
        gp = gp + pad_edge(gu,direction) + pad_edge(gv,direction,True)
    return (gamma*(z-z0)*editable + probability*(gp-(probability*gp).sum(1,keepdim=True)))*editable


def _window_gradient(current,anchor,condition,editable,gamma,lambda_,active,offset,length):
    gradient = _dense_gradient(current,anchor,condition,editable,gamma,lambda_,active)
    return gradient[:,:,offset:offset+length]


def analytic_gradient(z,z0,q,editable,gamma=1.,lambda_=0.5,active=None,chunk_depth=None,checkpointing=False):
    if chunk_depth is not None and (type(chunk_depth) is not int or chunk_depth <= 0):
        raise ValueError('求解器分块深度必须为正整数')
    depth = z.shape[2]
    width = depth if chunk_depth is None else chunk_depth
    chunks = []
    for start in range(0,depth,width):
        end = min(depth,start+width)
        low,high = max(0,start-1),min(depth,end+1)
        # 一层halo保留全部面邻边；每块单独重算，跨块梯度按原始输入累加。
        arguments = (z[:,:,low:high],z0[:,:,low:high],q[:,:,:,low:high],editable[:,:,low:high],
                     gamma,lambda_,active,start-low,end-start)
        result = checkpoint(_window_gradient,*arguments,use_reentrant=False) if checkpointing and z.requires_grad else _window_gradient(*arguments)
        chunks.append(result)
    return torch.cat(chunks,2) if len(chunks)>1 else chunks[0]


@dataclass
class SolveTrace:
    probability: torch.Tensor
    logits: torch.Tensor
    energy_F: torch.Tensor
    delta_z_l2: torch.Tensor
    delta_p_l2: torch.Tensor
    gradient_l2: torch.Tensor
    flip_count: torch.Tensor
    editable_count: torch.Tensor
    valid_edge_count: torch.Tensor


def solve_local(z0, q, editable, steps=3, gamma=1., lambda_=0.5, eta=1/3,
                active=None, checkpointing=False,chunk_depth=16):
    if steps < 0 or gamma <= 0 or lambda_ < 0 or not 0 < eta <= 1/(gamma+3*lambda_):
        raise ValueError('求解器配置违反下降条件')
    if z0.shape[1] != 11 or q.shape != (z0.shape[0],3,13,*z0.shape[2:]):
        raise ValueError('求解器类别或q形状错误')
    if not torch.isfinite(q).all() or (q < 0).any() or not torch.allclose(q.sum(2),torch.ones_like(q[:,:,0]),atol=2e-6,rtol=0):
        raise ValueError('q必须是合法事件概率')
    if editable.dtype != torch.bool or not editable.flatten(1).any(1).all():
        raise ValueError('编辑范围必须为非空bool掩码')
    q = q.detach()
    z = z0
    p0 = z0.detach().softmax(1)
    energies, dz, dp, flips, gradients = [], [], [], [], []
    for t in range(steps+1):
        with torch.no_grad():
            p = z.softmax(1)
            energies.append(energy(z,z0,q,editable,gamma,lambda_,active))
            dz.append(((z-z0)*editable).flatten(1).norm(dim=1))
            dp.append(((p-p0)*editable).flatten(1).norm(dim=1))
            flips.append(((p.argmax(1)!=p0.argmax(1))*editable[:,0]).flatten(1).sum(1))
        if t == steps:
            break
        gradient = analytic_gradient(z,z0,q,editable,gamma,lambda_,active,chunk_depth,checkpointing)
        gradients.append(gradient.detach().flatten(1).norm(dim=1))
        z = z - eta*gradient
    return SolveTrace(z.softmax(1),z,torch.stack(energies,1),torch.stack(dz,1),torch.stack(dp,1),
                      torch.stack(gradients,1) if gradients else z.new_empty((z.shape[0],0)),
                      torch.stack(flips,1),editable.flatten(1).sum(1),edge_masks(editable).flatten(1).sum(1))

