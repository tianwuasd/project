"""Small generated CPU tensors verify software wiring, never real-size readiness."""
import sys
import tempfile
import time
from pathlib import Path
from server_data import write_json


def main(path):
    record=dict(kind='software_only',device='cpu',shape=[1,1,12,12,12],status='running',
                full_size_training_ready=False)
    started=time.perf_counter()
    try:
        from runtime_setup import verify_dependencies
        record['dependencies'] = verify_dependencies()
        import torch
        from task_resources import torch_threads
        torch_threads(torch)
        model=torch.nn.Conv3d(1,8,3,padding=1)
        optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9)
        for _ in range(2):
            optimizer.zero_grad();loss=model(torch.ones(1,1,12,12,12)).square().mean()
            loss.backward();optimizer.step()
        if not torch.isfinite(loss):raise ValueError('CPU软件检查损失非有限数')
        with tempfile.TemporaryDirectory(prefix='software_smoke_') as temp:
            checkpoint=Path(temp)/'checkpoint.pt'
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict()),checkpoint)
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            restored=torch.nn.Conv3d(1,8,3,padding=1);restored.load_state_dict(saved['model'])
            assert all(torch.equal(a,b) for a,b in zip(model.parameters(),restored.parameters()))
        record.update(status='passed',checkpoint_weights_equal=True)
        print('依赖导入、资源采集与CPU软件检查通过；尚未验证正式尺寸或申请GPU。',flush=True)
    except BaseException as error:
        record.update(status='failed',error=str(error));raise
    finally:
        record['seconds']=time.perf_counter()-started;write_json(Path(path),record)


if __name__=='__main__':main(sys.argv[1])
