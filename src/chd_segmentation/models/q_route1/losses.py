import torch
from .events import endpoints, allowed_edge_events, edge_masks


def allowed_ce(logits, allowed, valid):
    if allowed.shape != logits.shape or allowed.dtype != torch.bool:
        raise ValueError('允许类别合同形状或类型不一致')
    count = allowed.sum(1)
    valid = valid.squeeze(1) if valid.ndim == logits.ndim else valid
    if (valid & (count == 0)).any():
        raise ValueError('有效位置的允许集合为空')
    selected = valid & (count < logits.shape[1])
    if not selected.any():
        return logits.sum()*0
    # 只读取有监督位置，避免空集合的logsumexp污染梯度。
    values = logits.movedim(1,-1)[selected]
    support = allowed.movedim(1,-1)[selected]
    return -values.log_softmax(-1).masked_fill(~support,-torch.inf).logsumexp(-1).mean()


def segmentation_loss(logits, allowed, valid):
    ce = allowed_ce(logits,allowed,valid)
    probability = logits.softmax(1)
    valid = valid.squeeze(1) if valid.ndim == logits.ndim else valid
    exact = allowed.sum(1) == 1
    terms = []
    for c in range(1,11):
        positive = exact & allowed[:,c] & valid
        known = valid & (exact | ~allowed[:,c])
        if positive.any():
            p = probability[:,c]
            terms.append(1-(2*(p*positive).sum()+1e-6)/((p*known).sum()+positive.sum()+1e-6))
    return ce + (torch.stack(terms).mean() if terms else logits.sum()*0)


def event_loss(logits, allowed, editable):
    masks = edge_masks(editable)
    total = logits.sum()*0
    count = 0
    for direction in range(3):
        left,right = endpoints(allowed,direction)
        support,informative = allowed_edge_events(left,right)
        weight,_ = endpoints(masks[:,direction:direction+1],direction)
        valid = informative & weight[:,0]
        q_logits,_ = endpoints(logits[:,direction],direction)
        n = int(valid.sum())
        if n:
            total = total + allowed_ce(q_logits,support,valid)*n
            count += n
    return total/count if count else total

