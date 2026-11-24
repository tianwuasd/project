import os
from pathlib import Path
from time import perf_counter

from ...artifacts.store import digest, read_json, require_output, write_json
from ...observability.events import Stage, event
from ...acceleration.legacy import SessionGuard
from . import PROTOCOL
from .input import open_image, rgb_slice, slices_for, source_record
from .storage import atomic_array, export_case, mapped_image


def execute(config, rows, output, handle, gpus, dependency_record, stop_after,rank,world,coordinator,port):
    import sys
    from ...acceleration.runtime import attach
    attach(handle)
    root = Path(output)
    stream = (root/f'prediction.rank{rank}.log').open('a',encoding='utf-8')
    sys.stdout = sys.stderr = stream
    cache = root/'runtime-cache'/f'rank{rank}'
    (cache/'temporary').mkdir(parents=True,exist_ok=True)
    for key,name in {'TRITON_CACHE_DIR':'triton','CUDA_CACHE_PATH':'cuda','MPLCONFIGDIR':'matplotlib',
        'TORCH_HOME':'torch','HF_HOME':'huggingface','TEMP':'temporary','TMP':'temporary','TMPDIR':'temporary'}.items():
        os.environ[key] = str(cache/name)
    coordinator.register(rank)
    coordinator.wait(rank,0,'startup_ready')
    run(config,rows[rank::world],output,handle,[gpus[rank]] if gpus else [],dependency_record,stop_after,rank,world)


def run(config, rows, output, handle, gpus, dependency_record, stop_after,rank=0,world=1):
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu['uuid'] for gpu in gpus)
    import torch
    import psutil
    from .environment import verify_worker_environment
    from .adapter import MedSAM3Adapter
    from .distributed import claim_prediction_slice

    environment = verify_worker_environment(config['environment'])
    event(output, 'medsam3_environment_verified', **environment)
    torch.set_num_threads(config['resources']['cpu_threads']//world)
    if gpus:
        actual = str(torch.cuda.get_device_properties(0).uuid)
        actual = actual if actual.startswith('GPU-') else 'GPU-' + actual
        if actual.lower() != gpus[0]['uuid'].lower():
            raise ValueError('MedSAM3实际GPU与公共准入结果不符')
        if config.get('capacity_probe_bytes', 0):
            free, total = torch.cuda.mem_get_info()
            cap = min(config['capacity_probe_bytes'], free-max(2**30, config['resources']['vram_reserve_bytes']))
            if cap <= 0:
                raise MemoryError('首次容量检查缺少显存余量')
            torch.cuda.set_per_process_memory_fraction(cap/total)
        torch.cuda.reset_peak_memory_stats()
    root = Path(output)
    recipe = digest({key: config[key] for key in (
        'model', 'precision', 'resolution', 'gray_window', 'intensity_semantics',
        'source_mean', 'source_std', 'prompts', 'detection_threshold', 'nms_iou_threshold',
        'geometry_policy', 'spacing_policy', 'orientation_evidence', 'intensity_evidence')})
    identity = digest(dict(recipe=recipe, dependencies=dependency_record))
    started = perf_counter()
    engine = None
    new, reused = 0, 0
    completed = []
    with SessionGuard(handle) as guard, Stage(root, 'medsam3_prediction',
            heartbeat_seconds=config['heartbeat_seconds']) as progress:
        def boundary():
            with guard.admit(0, label='medsam3_slice_or_class') as ticket:
                if ticket is None:
                    raise InterruptedError('MedSAM3资源等待已取消')

        for row in rows:
            original = open_image(row)
            selected = slices_for(config, row['case_id'], original.shape[2])
            source = source_record(row, original, config)
            case_root = require_output(root / row['case_id'], root)
            case_root.mkdir(exist_ok=True)
            path = case_root / 'case.json'
            state = dict(protocol=PROTOCOL, identity=identity, source=source,
                requested_slices=selected, completed_slices={}, coverage=config['coverage'],
                transform=dict(array_axes='ijk', model_rows='j', model_columns='i', slice_axis='k',
                    resolution=1008, inverse='resize_to_j_i_then_transpose_to_i_j'),
                prompts=config['prompts'], fusion='highest_instance_score_then_lowest_class_id',
                no_detection_label=0, uncomputed_label=255)
            if path.exists():
                old = read_json(path)
                if old['identity'] != identity or old['source'] != source:
                    raise ValueError('MedSAM3来源或科学设置改变，拒绝接续')
                state['completed_slices'] = old['completed_slices']
                if not {int(k) for k in old['completed_slices']} <= set(selected):
                    raise ValueError('接续不能丢弃已完成的覆盖切片')
            write_json(path, state)
            missing = [k for k in selected if str(k) not in state['completed_slices']]
            image = original
            if missing:
                image = mapped_image(original, row['image'], case_root / 'input-cache',
                    source, guard, root)
            for index in selected:
                boundary()
                name = str(index)
                if name in state['completed_slices']:
                    if not (case_root / f'slice-{index}.npy').is_file():
                        raise FileNotFoundError('已提交的切片文件丢失')
                    reused += 1
                    event(root, 'medsam3_slice_reused', case_id=row['case_id'], slice=index)
                    continue
                if not claim_prediction_slice(handle,stop_after):
                    write_json(root/f'summary.rank{rank}.json',dict(protocol=PROTOCOL,
                        state='interrupted_at_slice_boundary',cases=completed,rank=rank,
                        new_slices=new,reused_slices=reused,seconds=perf_counter()-started))
                    return
                if engine is None:
                    boundary()
                    load_started = perf_counter()
                    engine = MedSAM3Adapter(config)
                    event(root, 'medsam3_model_loaded', seconds=perf_counter() - load_started,
                        dependencies=dependency_record)
                slice_started = perf_counter()
                rgb = rgb_slice(image, index, config)
                labels, counts = engine.predict(rgb, boundary)
                atomic_array(case_root / f'slice-{index}.npy', labels)
                state['completed_slices'][name] = dict(seconds=perf_counter() - slice_started,
                    detections=counts, file=f'slice-{index}.npy')
                write_json(path, state)
                new += 1
                event(root, 'medsam3_slice_completed', case_id=row['case_id'], slice=index,
                    **state['completed_slices'][name])
                progress.update(case_id=row['case_id'], completed_slices=new,
                    reused_slices=reused, slice=index)
                del rgb, labels
            export_case(case_root, original, state)
            completed.append(row['case_id'])
            del image, original
        if gpus:
            torch.cuda.synchronize()
        memory = psutil.Process().memory_info()
        summary = dict(protocol=PROTOCOL, state='completed', cases=completed, new_slices=new,
            rank=rank,
            reused_slices=reused, seconds=perf_counter() - started, model_initialized=engine is not None,
            rss_bytes=memory.rss, peak_rss_bytes=memory.peak_wset if os.name == 'nt' else None,
            torch_peak_vram_bytes=torch.cuda.max_memory_reserved() if gpus else None,
            coverage=config['coverage'],
            scope='technical_slices' if config['coverage'] == 'selected' else 'whole_volume',
            training_updates=0)
        write_json(root / f'summary.rank{rank}.json', summary)
