import numpy as np
import torch
import torch.nn.functional as F

from .training_inputs import input_plan,sections,target_block


def localizer_statistics(logits, entry, dataset, workspace):
    stats = np.zeros((3, 6), dtype=np.float64)
    plan = input_plan(entry['label'].shape,entry['processed_spacing'],workspace.block_bytes)
    total = plan['blocks']
    for index, section in enumerate(sections(logits.shape[1:],plan['extent']), 1):
        with workspace.boundary():
            heat,mask = target_block(entry,dataset,section,plan['halo'])
            values = torch.from_numpy(np.array(logits[(slice(None), *section)], copy=True))
            probability = values.sigmoid()
            for task in range(3):
                h, g, m, p = values[task], heat[task], mask[task], probability[task]
                stats[task] += [float((g*m).double().sum()), float(((1-g)*m).double().sum()),
                                float((F.softplus(-h)*g*m).double().sum()),
                                float((F.softplus(h)*(1-g)*m).double().sum()),
                                float((p*g*m).double().sum()), float((p*m).double().sum())]
            del heat, mask, values, probability
        workspace.release(entry['label'])
        workspace.progress('q_localizer_loss', index, total)
    terms = []
    for positive, negative, bce_positive, bce_negative, intersection, predicted in stats:
        if positive+negative == 0:
            continue
        bce = ([bce_positive/positive] if positive > 0 else []) + ([bce_negative/negative] if negative > 0 else [])
        dice = 1-(2*intersection+1e-6)/(predicted+positive+1e-6) if positive > 0 else 0.
        terms.append(float(np.mean(bce))+dice)
    return float(np.mean(terms)) if terms else 0.
