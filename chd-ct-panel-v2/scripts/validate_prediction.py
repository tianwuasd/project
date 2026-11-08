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
from chd_stunet.prediction.model import read_json, file_sha
from chd_stunet.prediction.export import valid_result
from chd_stunet.observability.runlog import write_json


def main():
    parser = argparse.ArgumentParser(description='Real old model, native CT, detached stop/resume acceptance')
    parser.add_argument('--training-run', required=True)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--max-batch', type=int, default=1)
    parser.add_argument('--share-desktop-pid', type=int)
    parser.add_argument('--wait-seconds', type=int, default=300)
    parser.add_argument('--compare', help='previous completed result.json for voxel comparison')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    started = time.monotonic()
    signature_path = Path(args.training_run)/'signature.json'
    source_before = file_sha(signature_path)
    env = dict(os.environ, PYTHONUTF8='1')
    python = interpreter_command(env)

    def command(arguments, expected=0):
        process = subprocess.run([python, '-u', str(ROOT/'run.py'), *arguments], env=env,
                                 capture_output=True, text=True, encoding='utf-8', timeout=180)
        print(process.stdout[-1800:], flush=True)
        if (process.returncode == 0) != (expected == 0):
            raise RuntimeError(process.stderr[-6000:])
        return process

    def wait_state(states, timeout=1800, stop_after_windows=False):
        deadline = time.monotonic()+timeout
        next_log = 0
        while time.monotonic() < deadline:
            path = output/'status.json'
            state = read_json(path) if path.is_file() else {}
            if time.monotonic() >= next_log:
                print('验收观察：'+json.dumps(state, ensure_ascii=False)[:1200], flush=True)
                next_log = time.monotonic()+15
            if state.get('state') == 'failed':
                raise RuntimeError((output/'launcher.log').read_text(encoding='utf-8')[-6000:])
            if stop_after_windows and any(w.get('windows_done', 0) >= 2 for w in state.get('workers', [])):
                return state
            if state.get('state') in states:
                return state
            time.sleep(1)
        raise TimeoutError('验收超时；目录保留：'+str(output))

    def wait_exit():
        launch = read_json(output/'launch.json')
        for _ in range(40):
            alive = any(p.info['pid'] == launch['pid'] and p.info['create_time'] == launch['born']
                        for p in psutil.process_iter(['pid', 'create_time']))
            if not alive:
                return
            time.sleep(.5)
        raise RuntimeError('完成状态后监督进程未退出')

    extra = ['--share-desktop-pid', str(args.share_desktop_pid)] if args.share_desktop_pid else []
    command(['predict', '--smoke', '--training-run', args.training_run, '--prepared', args.prepared,
             '--device', args.device, '--no-tta', '--max-batch', str(args.max_batch), '--max-gpus', '1',
             '--no-prefetch', '--limit', '1', '--wait-seconds', str(args.wait_seconds), '--output', str(output), '--background', *extra])
    launch = read_json(output/'launch.json')
    if not psutil.pid_exists(launch['pid']):
        raise RuntimeError('后台任务随启动命令退出')
    wait_state([], stop_after_windows=True)
    command(['predict', '--resume', '--output', str(output), '--background'], expected=1)
    command(['predict', '--stop', '--output', str(output)])
    wait_state(['stopped'])
    wait_exit()
    command(['predict', '--resume', '--output', str(output), '--background'])
    # 旧的 stopped 状态由新监督器启动后覆盖。
    wait_state(['completed'])
    wait_exit()
    summary = read_json(output/'summary.json')
    if summary['status'] != 'completed' or summary['completed_cases'] != 1:
        raise AssertionError(summary)
    result = summary['results'][0]
    folder = Path(result['segmentation']).parent
    valid_result(folder, summary['identity'])
    segmentation = nib.load(result['segmentation'])
    reference = nib.load(result['reference_image'])
    np.testing.assert_allclose(segmentation.affine, reference.affine, rtol=1e-6, atol=1e-6)
    assert segmentation.shape == reference.shape
    assert set(result['metrics']) == {'LV', 'RV', 'LA', 'RA', 'MYO', 'AO', 'PA'}
    marker_mtime = (folder/'complete.json').stat().st_mtime_ns
    command(['predict', '--resume', '--output', str(output), '--background'])
    wait_exit()
    assert (folder/'complete.json').stat().st_mtime_ns == marker_mtime
    assert file_sha(signature_path) == source_before
    differences = None
    if args.compare:
        previous = read_json(args.compare)
        differences = int(np.count_nonzero(np.asarray(nib.load(previous['segmentation']).dataobj) != np.asarray(segmentation.dataobj)))
    report = dict(passed=True, background_survived_launcher=True, duplicate_launch_refused=True,
                  interrupted_and_resumed=True, completed_result_reused=True,
                  geometry_checked=True, source_signature_unchanged=True,
                  compared_different_voxels=differences, device=args.device,
                  elapsed_seconds=time.monotonic()-started, scope='technical smoke, old short-trained model')
    write_json(output/'acceptance.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
