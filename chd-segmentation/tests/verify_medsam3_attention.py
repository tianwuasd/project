import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import torch
from torch.nn import functional as F


def load_attention(official_root,weights_root):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
    sys.path.insert(0, official_root)
    from lora_layers import MultiheadAttentionLoRA, LoRALinear
    from sam3.model.tokenizer_ve import SimpleTokenizer
    torch.set_num_threads(2)
    weights = Path(weights_root)
    state = torch.load(weights/'sam3.pt', map_location='cpu', weights_only=True, mmap=True)
    state = state['model'] if 'model' in state else state
    base = 'backbone.language_backbone.encoder.'
    prefix = base+'transformer.resblocks.0.'
    tensors = {k.removeprefix('detector.'):v for k,v in state.items() if k.startswith('detector.')}
    token = SimpleTokenizer(bpe_path=str(Path(official_root)/'sam3/assets/bpe_simple_vocab_16e6.txt.gz'))
    ids = token(['aorta', 'pulmonary artery'], context_length=32)
    features = F.embedding(ids, tensors[base+'token_embedding.weight'])+tensors[base+'positional_embedding'][:32]
    features = F.layer_norm(features, (1024,), tensors[prefix+'ln_1.weight'], tensors[prefix+'ln_1.bias'])
    module = MultiheadAttentionLoRA(1024, 16, batch_first=True,
        in_proj_weight=tensors[prefix+'attn.in_proj_weight'],
        in_proj_bias=tensors[prefix+'attn.in_proj_bias'],
        out_proj_weight=tensors[prefix+'attn.out_proj.weight'],
        out_proj_bias=tensors[prefix+'attn.out_proj.bias'])
    medical = torch.load(weights/'best_lora_weights.pt', map_location='cpu', weights_only=True)
    for name in ('q_proj','k_proj','v_proj','out_proj'):
        projection = LoRALinear(getattr(module, name), rank=16, alpha=32, dropout=0.)
        projection.lora.load_state_dict({key:medical[prefix+'attn.'+name+'.lora.'+key]
                                        for key in ('lora_A','lora_B')}, strict=True)
        setattr(module, name, projection)
    module.eval()
    return module,features,ids == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--official-root', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--export-attention')
    args = parser.parse_args()
    started = perf_counter()
    module,features,padding = load_attention(args.official_root,args.weights)
    if args.export_attention:
        torch.save(dict(state=module.state_dict(),features=features,padding=padding,
            prompts=['aorta','pulmonary artery'],source='official_medical_lora_first_text_attention'),args.export_attention)
    from chd_segmentation.models.medsam3.adapter import lora_sdpa
    mask = torch.ones(32,32,dtype=torch.bool).triu(1)
    checks = []
    for batch_first in (True, False):
        module.batch_first = batch_first
        for shape in ('none','bool2','float2','bool3','bool4','heads3'):
            masks = {'none':None, 'bool2':mask,
                'float2':torch.zeros(32,32).masked_fill(mask, float('-inf')),
                'bool3':mask.expand(2,-1,-1), 'bool4':mask[None,None],
                'heads3':mask.expand(32,-1,-1)}
            x = (features if batch_first else features.transpose(0,1)).detach().requires_grad_(True)
            kwargs = dict(attn_mask=masks[shape], key_padding_mask=padding, need_weights=False)
            original = module(x,x,x,**kwargs)[0]
            optimized = lora_sdpa(module,x,x,x,**kwargs)[0]
            torch.testing.assert_close(optimized, original, atol=2e-5, rtol=2e-4)
            parameters = [x, *(p for p in module.parameters() if p.requires_grad)]
            a = torch.autograd.grad(original.square().mean(), parameters)
            b = torch.autograd.grad(optimized.square().mean(), parameters)
            for expected, actual in zip(a,b,strict=True):
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=5e-4)
            checks.append(dict(batch_first=batch_first, mask=shape,
                max_absolute_difference=float((optimized-original).abs().max().detach())))
    print(json.dumps(dict(state='passed', prompts=['aorta','pulmonary artery'],
        inputs='official_tokenizer_and_pretrained_first_text_layer', checks=checks,
        seconds=perf_counter()-started), allow_nan=False))


if __name__ == '__main__':
    main()
