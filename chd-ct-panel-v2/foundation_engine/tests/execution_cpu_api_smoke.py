"""Real CPU Inductor attempt and explicit fallback; never initializes CUDA."""
import argparse
import json
import os
from pathlib import Path
import sys
import types

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report-file',required=True);args=parser.parse_args()
    # Inductor may inspect accelerator capabilities even for a CPU model.
    # Hide devices before importing Torch in this CPU-only executable.
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import torch
    assert torch.cuda.device_count()==0, 'CPU probe requires no visible CUDA devices'
    from training_execution import apply_execution,preflight_compile
    torch.set_num_threads(1);torch.manual_seed(0)
    trainer=types.SimpleNamespace(network=torch.nn.Linear(4,2),is_ddp=False,
        device=torch.device('cpu'),print_to_log_file=lambda *a:print(*a,flush=True))
    optimizer=torch.optim.SGD(trainer.network.parameters(),lr=.01,momentum=.9)
    state={k:v.clone() for k,v in trainer.network.state_dict().items()}
    rng=torch.get_rng_state().clone()
    batch=dict(data=torch.ones(2,4))
    report=apply_execution(torch,{'mode':'performance','compile':True},main_threads=1)
    preflight_compile(trainer,batch,report)
    original=getattr(trainer.network,'_orig_mod',trainer.network)
    assert all(torch.equal(original.state_dict()[k],v) for k,v in state.items())
    assert torch.equal(rng,torch.get_rng_state()) and not optimizer.state
    optimizer.zero_grad();loss=trainer.network(batch['data']).square().mean()
    loss.backward();optimizer.step()
    assert torch.isfinite(loss) and optimizer.state
    assert torch.cuda.device_count()==0
    result=dict(status='passed',device='cpu',actual_inductor_attempt=True,
        compile_applied=report['compile_applied'],explicit_eager_fallback=bool(report['fallback_reason']),
        preflight_state_preserved=True,subsequent_optimizer_update=True,
        cuda_initialized=torch.cuda.is_initialized(),visible_cuda_devices=torch.cuda.device_count(),
        model_and_batch_device='cpu',execution=report)
    Path(args.report_file).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
