"""CPU-only checkpoint roundtrip using the exact pinned upstream save/load bodies.

No CT files, CUDA context, fingerprint, planning or full training are involved.
Default: inspect installed nnU-Net 2.8.1. --upstream-source accepts its official
trainer source for an isolated local regression when nnU-Net is not installed.
"""
import argparse
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import random
import sys
import tempfile
from typing import Union

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE/'05_src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-source')
    parser.add_argument('--report-file')
    args=parser.parse_args()
    import numpy as np
    import torch
    from torch._dynamo import OptimizedModule
    from checkpoint_rng import restore
    torch.set_num_threads(1)
    if args.upstream_source:
        source=Path(args.upstream_source).read_text(encoding='utf-8')
    else:
        import importlib.metadata
        if importlib.metadata.version('nnunetv2')!='2.8.1':raise ValueError('此回归针对nnU-Net 2.8.1')
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        source=inspect.getsource(nnUNetTrainer)
    namespace=dict(torch=torch,Union=Union,OptimizedModule=OptimizedModule)
    tree=ast.parse(source)
    methods={name:next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
             for name in ['save_checkpoint','load_checkpoint']}
    exec(compile(ast.Module(body=list(methods.values()),type_ignores=[]),'<nnUNet 2.8.1 checkpoint methods>','exec'),namespace)
    class Parent:
        save_checkpoint=namespace['save_checkpoint']
        load_checkpoint=namespace['load_checkpoint']
        def on_train_start(self):
            random.random();np.random.rand();torch.rand(2) # Simulated loader initialization consumes RNG.
    project_source=(CODE/'05_src/project_trainers.py').read_text(encoding='utf-8')
    trainer=next(n for n in ast.parse(project_source).body if isinstance(n,ast.ClassDef) and n.name=='BaselineNNUNet')
    selected=[n for n in trainer.body if isinstance(n,ast.FunctionDef) and n.name in {'save_checkpoint','load_checkpoint','on_train_start'}]
    isolated=ast.ClassDef(name='PatchedTrainer',bases=[ast.Name(id='Parent',ctx=ast.Load())],keywords=[],body=selected,decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[isolated],type_ignores=[]))
    namespace.update(Parent=Parent,os=__import__('os'))
    exec(compile(module,'<actual project checkpoint overrides>','exec'),namespace)
    cls=namespace['PatchedTrainer']
    class Logger:
        def get_checkpoint(self):return {'history':[1.]}
        def load_checkpoint(self,value):self.value=value
    def instance(kind=cls):
        obj=kind();obj.was_initialized=True;obj.device=torch.device('cpu')
        obj.network=torch.nn.Linear(4,2);obj.optimizer=torch.optim.SGD(obj.network.parameters(),lr=.01,momentum=.9)
        obj.logger=Logger();obj.is_ddp=False;obj.grad_scaler=None;obj.local_rank=0;obj.disable_checkpointing=False
        obj.current_epoch=0;obj.my_init_kwargs={'mode':'synthetic_cpu'};obj._best_ema=.2;obj.inference_allowed_mirroring_axes=None
        return obj
    def step(obj):
        obj.optimizer.zero_grad();loss=obj.network(torch.ones(2,4)).square().mean()
        loss.backward();obj.optimizer.step();assert torch.isfinite(loss)
    def draws():return (random.random(),float(np.random.rand()),torch.rand(3))
    def same(a,b):return a[:2]==b[:2] and torch.equal(a[2],b[2])
    results=[]
    with tempfile.TemporaryDirectory(prefix='vista_checkpoint_api_') as temp:
        path=Path(temp)/'checkpoint.pth';original=instance();step(original);original.save_checkpoint(str(path))
        saved=torch.load(path,map_location='cpu',weights_only=False);before=path.read_bytes()
        assert saved['vista_rng_state'] and saved['optimizer_state']['state']
        try:instance(Parent).load_checkpoint(saved)
        except UnboundLocalError:results.append('original_dictionary_branch_reproduced')
        else:raise AssertionError('上游方法没有复现附件中的异常，需重新核对版本')
        for label,value in [('filename',str(path)),('pathlib',path),('mapping',saved)]:
            restored=instance();restored.load_checkpoint(value)
            assert restored.current_epoch==1 and restored.optimizer.state and restored.rng_state_restored
            for name,tensor in original.network.state_dict().items():assert torch.equal(tensor,restored.network.state_dict()[name])
            for name,state in original.optimizer.state_dict()['state'].items():
                for field,tensor in state.items():assert torch.equal(tensor,restored.optimizer.state_dict()['state'][name][field])
            restore(saved['vista_rng_state']);expected=draws()
            restored.load_checkpoint(value);assert same(expected,draws())
            restored.on_train_start();assert same(expected,draws())
            step(restored);results.append(label+'_weights_optimizer_rng_and_resumed_step')
        assert path.read_bytes()==before;results.append('original_checkpoint_bytes_unchanged')
        legacy=copy.deepcopy(saved);legacy.pop('vista_rng_state');oldpath=Path(temp)/'legacy.pth';torch.save(legacy,oldpath)
        restored=instance();restored.load_checkpoint(oldpath);assert not restored.rng_state_restored
        for name,tensor in original.network.state_dict().items():assert torch.equal(tensor,restored.network.state_dict()[name])
        results.append('legacy_checkpoint_without_rng_supported')
    assert not torch.cuda.is_initialized()
    report=dict(status='passed',device='cpu',cuda_initialized=False,real_ct=False,full_gpu_preflight=False,
                torch=torch.__version__,upstream_methods_sha256={k:hashlib.sha256(ast.unparse(v).encode()).hexdigest() for k,v in methods.items()},checks=results)
    if args.report_file:Path(args.report_file).write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
