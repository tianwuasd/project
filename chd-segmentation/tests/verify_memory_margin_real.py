import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import threading
from time import perf_counter, sleep
import sys

import psutil


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def control(root):
    from chd_segmentation.acceleration import ResourceSession, ResourceDemand, RuntimeRequest, attach
    from chd_segmentation.acceleration.host import host_capacity
    from chd_segmentation.artifacts.store import read_json
    facts = host_capacity()
    resources = dict(cpu_fraction=.8, ram_fraction=.8, reserve_ram_bytes=256*2**20,
        memory_limit_bytes=8*2**30, max_wait=5., memory_margin_fraction=.5)
    request = RuntimeRequest('growth', 'technical', root/'resources', root/'growth', resources,
        ResourceDemand(fixed_bytes=128*2**20))
    began = perf_counter()
    with ResourceSession(request) as session:
        client = attach(session.handle)
        initial = session.capacity
        with client.reserve(initial*2, purpose='real_reservation_growth'):
            assert session.capacity > initial
            assert session.capacity <= resources['memory_limit_bytes']
            assert session.handle.state['plan_generation'] > 0
            row = read_json(root/'resources/active.json')[0]
            assert row['ram_bytes'] == session.capacity
            assert session.allocation.memory_bytes == session.capacity
            assert not session.handle.state['waiting']
            grown = session.capacity
        assert session.handle.state['reserved'] == 0
    assert read_json(root/'resources/active.json') == []
    limited = dict(resources, memory_limit_bytes=512*2**20, max_wait=.3)
    request = RuntimeRequest('limited', 'technical', root/'limits', root/'limited', limited,
        ResourceDemand(fixed_bytes=128*2**20))
    with ResourceSession(request) as session:
        client = attach(session.handle)
        initial_limited = session.capacity
        with ThreadPoolExecutor(max_workers=1) as pool:
            def finite():
                with client.reserve(2*2**30, purpose='above_real_limit'):
                    raise AssertionError('超出真实配置上限的申请不应通过')
            future = pool.submit(finite)
            while not future.done():
                sleep(.02)
            assert isinstance(future.exception(), TimeoutError)
        assert session.capacity == initial_limited and not session.handle.state['waiting']
    unlimited = dict(limited, max_wait=None)
    request = RuntimeRequest('cancel', 'technical', root/'cancel-registry', root/'cancel', unlimited,
        ResourceDemand(fixed_bytes=128*2**20))
    with ResourceSession(request) as session:
        client = attach(session.handle)
        with ThreadPoolExecutor(max_workers=1) as pool:
            def blocked():
                with client.reserve(2*2**30, purpose='cancelled_real_wait'):
                    raise AssertionError('超出额度的申请不应通过')
            future = pool.submit(blocked)
            deadline = perf_counter()+5
            while not session.handle.state['waiting'] and perf_counter()<deadline:
                sleep(.02)
            assert session.handle.state['waiting'] and not future.done()
            sleep(1.1)
            assert not future.done()
            session.cancel.set()
            while not future.done() and perf_counter()<deadline:
                sleep(.02)
            assert future.done() and isinstance(future.exception(), InterruptedError)
        assert not session.handle.state['waiting']
    return dict(initial_bytes=initial, grown_bytes=grown, limit_enforced=True, cancel_passed=True,
                finite_timeout_passed=True, seconds=perf_counter()-began, host=facts)


def training(root, settings):
    from chd_segmentation.artifacts.store import read_json, write_json
    from chd_segmentation.config import load_config
    from chd_segmentation.preprocessing.prepare import prepare
    from chd_segmentation.preprocessing.offline import preprocess
    from chd_segmentation.acceleration.planning import cpu_plan
    from chd_segmentation.training.engine import train, supply_settings
    manifest = read_json(settings['manifest'])
    cases = []
    for name, role in (('ct_1001','train'),('ct_1012','validation')):
        row = deepcopy(next(row for row in manifest['cases'] if row['case_id']==name))
        row.update(image=str(Path(settings['data_root'])/(name+'_image.nii.gz')),
            label=str(Path(settings['data_root'])/(name+'_label.nii.gz')), roles=[role])
        cases.append(row)
    manifest.update(cases=cases, dataset='imagechd', scope='technical', plan_scope='technical',
        binding_required=False, data_root=settings['data_root'])
    course = root/'course'
    write_json(course/'manifest.json',manifest)
    config = dict(model='stunet', dataset='imagechd', scope='technical', fold=0,
        manifest=str(course/'manifest.json'), pretrained=settings['pretrained'],
        work_root=str(course), run_dir=str(course/'runs/STU'), result_root=str(root/'result'),
        technical=dict(roi_xyz=[[224,256],[224,256],[71,103]],allow_role_overlap=False),
        planning=dict(mode='auto', samples_per_case=1000, max_samples=1000,target_spacing_xyz=[1.,1.,1.]),
        preprocessing=dict(root=str(root/'data'), statistics_source='technical-train'),
        training=dict(patch=[32]*3,batch=2,attempts=2,epochs=2,steps_per_epoch=1,
            validation_steps=1,save_every_attempts=1),
        resources=dict(device='cpu',cpu_threads=1,cpu_processes=2,workers=0,max_workers=1,
            prefetch=1,cache_bytes=0,max_readers=1,memory_limit_bytes=12*2**30,
            reserve_ram_bytes=256*2**20,adapt_seconds=.01,growth_seconds=.01),
        logging=dict(heartbeat_seconds=15,case_details=False))
    write_json(course/'config.json',config)
    config = load_config(course/'config.json')
    started = perf_counter()
    prepare(config,course/'source.json')
    prepared = preprocess(config,course/'source.json',course/'offline.json')
    preprocessing_seconds = perf_counter()-started
    plan = cpu_plan(config,prepared,2)
    assert plan['admitted'] and plan['ram_bytes']==math.ceil(plan['estimated_ram_bytes']*1.5)
    same = dict(plan,ram_bytes=plan['ram_bytes']+1)
    assert supply_settings(plan) != supply_settings(same)
    started = perf_counter()
    first = train(config,course/'offline.json',resume=(course/'runs/STU/latest.json').exists(),stop_after=1)
    assert first['updates']==1
    second = train(config,course/'offline.json',resume=True)
    assert second['updates']==2 and second['restored_exact']
    return dict(cases=[row['case_id'] for row in cases],roi_xyz=config['technical']['roi_xyz'],
        cpu_processes=2,updates=second['updates'],restored_exact=second['restored_exact'],
        preprocessing_seconds=preprocessing_seconds,training_seconds=perf_counter()-started,
        estimated_bytes=plan['estimated_ram_bytes'],granted_bytes=plan['ram_bytes'],
        margin_bytes=plan['margin_bytes'],first=first,second=second)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work',required=True)
    parser.add_argument('--report',required=True)
    parser.add_argument('--settings',required=True)
    parser.add_argument('--tool-root')
    parser.add_argument('--part',choices=('control','training'),required=True)
    args = parser.parse_args()
    root = Path(args.work).resolve(strict=True)
    runtime = root/'runtime'
    runtime.mkdir(exist_ok=True)
    for key in ('TEMP','TMP','TMPDIR','MPLCONFIGDIR','TORCH_HOME','HF_HOME','XDG_CACHE_HOME'):
        os.environ[key] = str(runtime)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', USE_LIBUV='0', OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    from chd_segmentation.artifacts.store import read_json, write_json
    if args.part == 'training':
        if not args.tool_root:
            raise ValueError('训练短测需要指定现有统一尝试登记工具')
        tool = Path(args.tool_root).resolve(strict=True)
        sys.path.insert(0, str(tool/'src'))
        from chd_mini.attempts import AttemptLedger
        ledger = root/'attempts.json'
        AttemptLedger(ledger).initialize('chd-memory-margin-20261118')
        os.environ.update(CHD_MINI_TOOL_ROOT=str(tool),CHD_MINI_ATTEMPT_LEDGER=str(ledger),CHD_MINI_STAGE='stu')
    stopped = threading.Event()
    observed = []
    def observe():
        process = psutil.Process()
        while not stopped.wait(.2):
            family = [process,*process.children(recursive=True)]
            observed.append(sum(p.memory_info().rss for p in family if p.is_running()))
    observer = threading.Thread(target=observe)
    observer.start()
    began = perf_counter()
    try:
        result = control(root) if args.part=='control' else training(root,read_json(args.settings))
    finally:
        stopped.set()
        observer.join()
    files = [p for p in root.rglob('*') if p.is_file()]
    size = sum(p.stat().st_size for p in files)
    assert len(files)<=100 and size<=2**30
    result.update(state='passed',part=args.part,seconds=perf_counter()-began,
        files=len(files),bytes=size,peak_sampled_group_rss_bytes=max(observed,default=0),
        peak_vram_bytes=None)
    write_json(args.report,result)
    print(json.dumps(dict(state='passed',part=args.part,seconds=result['seconds'],files=len(files),bytes=size)),flush=True)


if __name__=='__main__':
    main()
