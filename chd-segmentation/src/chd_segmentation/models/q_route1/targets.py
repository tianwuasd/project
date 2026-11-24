import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn.functional as F


def build_heatmap_targets(allowed, valid, spacing, radius=8., sigma=4., source_region=None):
    if allowed.shape[0] != 11 or allowed.dtype != bool or valid.dtype != bool:
        raise ValueError('定位监督需要11类允许集合和有效视野')
    if len(spacing) != 3 or min(spacing) <= 0:
        raise ValueError('定位距离需要可信的毫米间距')
    exact = allowed.sum(0) == 1
    shape = valid.shape
    def distance(classes):
        foreground = np.any(allowed[list(classes)],axis=0) & exact & valid
        return distance_transform_edt(~foreground,sampling=spacing) if foreground.any() else np.full(shape,np.inf)
    maps,masks = [],[]
    for classes in ((1,2,6,7),(7,),(8,)):
        known = valid.copy()
        for c in classes:
            known &= exact | ~allowed[c]
        # 扫描外区域同样按未知处理，要求完整半径邻域有可靠参考。
        distance_unknown = distance_transform_edt(np.pad(known,1),sampling=spacing)[1:-1,1:-1,1:-1]
        reliable = distance_unknown > radius
        if len(classes) == 4:
            vessel,ventricle = distance((6,7)),distance((1,2))
            target = np.exp(-(vessel**2+ventricle**2)/(2*sigma*sigma))*(vessel<=radius)*(ventricle<=radius)
        else:
            dist = distance(classes)
            target = np.exp(-dist**2/(2*sigma*sigma))*(dist<=radius)
        if source_region is not None:
            for axis,(begin,end) in enumerate(source_region):
                selection = [slice(None)]*3
                selection[axis] = slice(0,begin)
                target[tuple(selection)] = 0
                selection[axis] = slice(end,None)
                target[tuple(selection)] = 0
        maps.append(target.astype(np.float32))
        masks.append(reliable)
    target = torch.from_numpy(np.stack(maps))[None]
    mask = torch.from_numpy(np.stack(masks))[None]
    padding = [v for n in reversed(shape) for v in (0,(-n)%32)]
    target = F.max_pool3d(F.pad(target,padding),4,4)
    mask = F.avg_pool3d(F.pad(mask.float(),padding),4,4) == 1
    return target,mask


def localizer_loss(logits, target, mask):
    terms = []
    for task in range(3):
        h,g,m = logits[:,task],target[:,task],mask[:,task]
        positive,negative = (g*m).sum(),((1-g)*m).sum()
        if not m.any():
            continue
        components = []
        if positive > 0:
            components.append((F.softplus(-h)*g*m).sum()/positive)
        if negative > 0:
            components.append((F.softplus(h)*(1-g)*m).sum()/negative)
        bce = torch.stack(components).mean()
        dice = 1-(2*(h.sigmoid()*g*m).sum()+1e-6)/((h.sigmoid()*m).sum()+positive+1e-6) if positive > 0 else h.sum()*0
        terms.append(bce+dice)
    return torch.stack(terms).mean() if terms else logits.sum()*0

