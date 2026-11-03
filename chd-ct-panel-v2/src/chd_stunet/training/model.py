"""Pinned STU-Net-S topology; network kernels and state names stay upstream."""
import numpy as np
from .._vendor.stunet_architecture import STUNet


def topology(spacing, patch_size):
    from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props
    _,pools,_,adjusted,_ = get_pool_and_conv_props(spacing,patch_size,4,999)
    if list(map(int,adjusted)) != list(patch_size):
        raise ValueError('Requested patch does not satisfy native topology divisibility')
    strides = [list(map(int,s)) for s in pools[1:]][:5]
    strides += [[1,1,1]]*(5-len(strides))
    # STU-Net uses six 3x3x3 kernel stages even where the nnU-Net default
    # planner suggests a 1x3x3 first stage. Weight shapes must match the author.
    scales = np.vstack([np.ones(3),1/np.cumprod(strides,axis=0)])[:5].tolist()
    return dict(strides=strides,kernels=[[3]*3]*6,deep_supervision_scales=scales)


def build_model(plan):
    return STUNet(1,8,depth=[1]*6,dims=[16,32,64,128,256,256],
                  pool_op_kernel_sizes=plan['strides'],conv_kernel_sizes=plan['kernels'],
                  enable_deep_supervision=True)
