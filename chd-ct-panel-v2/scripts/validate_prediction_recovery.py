import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import nibabel as nib
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from chd_stunet.acceleration.preprocess_cpu import interpreter_command
from chd_stunet.acceleration.prediction import process_alive
from chd_stunet.observability.runlog import write_json
from chd_stunet.prediction.export import valid_result
from chd_stunet.prediction.model import read_json


def main():
    parser = argparse.ArgumentParser(description='真实预测监督进程异常终止与恢复验收')
    parser.add_argument('--training-run', required=True)
    parser.add_argument('--images', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--share-desktop-pid', type=int)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('验收使用独立的新目录')
    env = dict(os.environ, PYTHONUTF8='1')
    python = interpreter_command(env)

    def command(arguments):
        result = subprocess.run([python, '-u', str(ROOT/'run.py'), *arguments], env=env,
                                capture_output=True, text=True, encoding='utf-8', timeout=180)
        if result.returncode:
            raise RuntimeError(result.stdout[-3000:]+result.stderr[-6000:])
        print(result.stdout[-1500:], flush=True)

    def wait_until(predicate, seconds=600):
        deadline = time.monotonic()+seconds
        feedback = 0
        while time.monotonic() < deadline:
            failure = output/'failure.json'
            if failure.is_file():
                raise RuntimeError(read_json(failure))
            result = predicate()
            if result:
                return result
            if time.monotonic() >= feedback:
                status = output/'status.json'
                print('恢复验收：'+json.dumps(read_json(status) if status.exists() else {}, ensure_ascii=False)[:1300], flush=True)
                feedback = time.monotonic()+10
            time.sleep(.2)
        raise TimeoutError('实际恢复验收超时：'+str(output))

    extra = ['--share-desktop-pid', str(args.share_desktop_pid)] if args.share_desktop_pid else []
    command(['predict', '--smoke', '--training-run', args.training_run, '--images', *args.images,
             '--device', args.device, '--no-tta', '--max-gpus', '1', '--max-batch', '4',
             '--wait-seconds', '180', '--output', str(output), '--background', *extra])

    def ready_worker():
        for path in (output/'control').glob('gpu-*/status.json'):
            status = read_json(path)
            owner = path.parent/'owner.json'
            if status.get('stage') in ('load_model', 'prepare', 'predict') and owner.exists():
                return read_json(owner)
        return None

    first_owner = wait_until(ready_worker)
    supervisor = read_json(output/'launch.json')
    assert process_alive(supervisor)
    process = psutil.Process(supervisor['pid'])
    assert process.create_time() == supervisor['born']
    process.kill()
    psutil.wait_procs([process], timeout=10)
    assert not process_alive(supervisor)
    command(['predict', '--resume', '--output', str(output), '--background'])
    assert not process_alive(first_owner)

    def finished():
        path = output/'summary.json'
        return path.exists() and read_json(path)['status'] == 'completed'

    wait_until(finished)
    wait_until(lambda: not process_alive(read_json(output/'launch.json')), seconds=30)
    summary = read_json(output/'summary.json')
    assert summary['completed_cases'] == len(args.images)
    markers = {}
    for result in summary['results']:
        folder = Path(result['segmentation']).parent
        valid_result(folder, summary['identity'])
        segmentation = nib.load(result['segmentation'])
        reference = nib.load(result['reference_image'])
        np.testing.assert_allclose(segmentation.affine, reference.affine, rtol=1e-6, atol=1e-6)
        assert segmentation.shape == reference.shape and result['metrics'] is None
        markers[str(folder)] = (folder/'complete.json').stat().st_mtime_ns
    for path in (output/'control').glob('*/owner.json'):
        assert not process_alive(read_json(path)), str(path)
    cleanup = read_json(output/'control/orphan-cleanup.json')
    command(['predict', '--resume', '--output', str(output), '--background'])
    wait_until(lambda: not process_alive(read_json(output/'launch.json')), seconds=60)
    for folder, before in markers.items():
        assert (Path(folder)/'complete.json').stat().st_mtime_ns == before
    report = dict(passed=True, supervisor_killed=True, orphan_worker_released=True,
                  cleanup_records=len(cleanup['cleaned']), completed_cases=summary['completed_cases'],
                  completed_files_reused=True, native_geometry_verified=True,
                  scope='technical prediction recovery with real model and supplied images')
    write_json(output/'acceptance.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
