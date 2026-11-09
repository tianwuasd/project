from pathlib import Path
import math
import os
import time

import nibabel as nib
import numpy as np
from PIL import Image

from chd_stunet.acceleration.preprocess_cpu import execute_jobs
from chd_stunet.config import ROOT
from chd_stunet.observability.locking import read_slot, run_lock
from chd_stunet.observability.runlog import RunLog, write_json
from chd_stunet.preprocessing.cohort import TARGETS, frozen_roles, map_labels
from chd_stunet.preprocessing.stunet_s import corrected_affine

from . import PROTOCOL
from .contracts import check_signature, digest, file_sha, read_json, signed


class SourceUnavailable(OSError):
    pass


def read_source(source, folds):
    source = Path(source).resolve(strict=True)
    manifest = read_json(source/'manifests/imagechd_input.json')
    producer = read_json(source/'signature.json')
    check_signature(producer)
    if (producer['identity'] != manifest['identity'] or producer['method']['protocol'] != manifest['protocol']
            or read_json(source/'dataset.json')['labels'] != TARGETS):
        raise ValueError('来源预处理签名或标签映射不一致')
    rows = manifest['cases']
    all_rows = rows + manifest['excluded']
    complete = [row['case_id'] for row in all_rows if not row['missing_label_ids']]
    if len(all_rows) != 110 or len({row['case_id'] for row in all_rows}) != 110 or len(complete) != 97:
        raise ValueError('来源病例清单不完整')
    historical = frozen_roles(complete, producer['method']['split_seed'])
    groups = {'main': 0, 'auxiliary': 0}
    for row in rows:
        group = row['cohort_group']
        groups[group] += 1
        if group == 'main':
            if row['missing_label_ids'] or row['roles'] != historical[row['case_id']]:
                raise ValueError('主队列角色不一致：'+row['case_id'])
        elif row['missing_label_ids'] != [5] or set(row['roles'].values()) != {'aux_train'}:
            raise ValueError('辅助队列标签或角色不一致：'+row['case_id'])
        if not {'image', 'label'} <= row['source_hashes'].keys():
            raise ValueError('源文件摘要不完整：'+row['case_id'])
    if groups != {'main': 96, 'auxiliary': 13} or any(row['case_id'] == 'ct_1170' for row in rows):
        raise ValueError('来源须为已审核的 96 主队列与 13 辅助队列')
    if len({row['source_hashes']['image'] for row in all_rows}) != 110:
        raise ValueError('来源影像文件摘要重复')
    fold_stats = {}
    for fold in folds:
        folder = source/f'fold_{fold}'
        fingerprint = read_json(folder/'dataset_fingerprint.json')
        plan = read_json(folder/'preprocessing_plan.json')
        original = read_json(folder/'training_contract.json')
        fit = sorted(row['case_id'] for row in rows if row['cohort_group'] == 'main'
                     and row['roles'][str(fold)] == 'train')
        field = 'foreground_intensity_properties_per_channel'
        if (original['fold'] != fold or sorted(original['fit_ids']) != fit
                or original['preprocessing_contract'] != digest(dict(run=producer['identity'], plan=plan))
                or sorted(plan['fit_ids']) != fit or sorted(fingerprint['fit_ids']) != fit
                or fingerprint['fitting_roles'] != ['train'] or fingerprint[field] != plan[field]):
            raise ValueError(f'第 {fold} 折灰度参数来源不匹配主训练病例')
        stats = plan[field]['0']
        low, high = stats['percentile_00_5'], stats['percentile_99_5']
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError(f'第 {fold} 折 HU 裁剪范围无效')
        fold_stats[str(fold)] = dict(low=low, high=high, fit_ids=fit,
                                    fingerprint_sha256=file_sha(folder/'dataset_fingerprint.json'),
                                    plan_sha256=file_sha(folder/'preprocessing_plan.json'))
    return rows, fold_stats, producer['identity'], manifest.get('patient_identity_verified', False)


def canonical_image(image, spacing=None):
    if image.ndim != 3 or not np.isfinite(image.affine).all():
        raise ValueError('要求具有有效物理坐标的三维 CT')
    affine = image.affine if spacing is None else corrected_affine(image.affine, spacing)
    origin = nib.orientations.io_orientation(affine)
    ras = nib.orientations.axcodes2ornt(('R', 'A', 'S'))
    forward = nib.orientations.ornt_transform(origin, ras)
    backward = nib.orientations.ornt_transform(ras, origin)
    native_shape = tuple(image.shape)
    canonical_affine = affine @ nib.orientations.inv_ornt_aff(forward, native_shape)
    geometry = dict(native_shape=list(native_shape), affine=np.asarray(affine).tolist(),
                    canonical_affine=canonical_affine.tolist(), forward=forward.tolist(),
                    backward=backward.tolist(), spacing_zyx=nib.affines.voxel_sizes(canonical_affine)[::-1].tolist())
    return geometry


def to_axial(array, geometry):
    return nib.orientations.apply_orientation(array, np.asarray(geometry['forward'])).transpose(2, 1, 0)


def to_native(array, geometry):
    return nib.orientations.apply_orientation(array.transpose(2, 1, 0), np.asarray(geometry['backward']))


def atomic_array(path, array):
    temporary = path.with_suffix('.partial')
    with temporary.open('wb') as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_native_pair(row):
    source = {key: Path(row['pair'][key]) for key in ('image', 'label')}
    before = {key: path.stat() for key, path in source.items()}
    for key, path in source.items():
        if file_sha(path) != row['source_hashes'][key]:
            raise ValueError('源文件摘要变化：'+str(path))
    image, label = (nib.load(source[key]) for key in ('image', 'label'))
    if (list(image.shape) != row['shape_xyz'] or image.shape != label.shape
            or not np.allclose(image.affine, row['source_affine'], rtol=0, atol=1e-5)
            or not np.allclose(image.affine, label.affine, rtol=0, atol=1e-5)):
        raise ValueError('CT 与标签网格和审核记录不一致：'+row['case_id'])
    data = image.get_fdata(dtype=np.float32)
    truth = map_labels(np.asanyarray(label.dataobj))
    for key, path in source.items():
        after = path.stat()
        if (before[key].st_size, before[key].st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('源文件在读取中变化：'+str(path))
    return data, truth, canonical_image(image, row['spacing_xyz'])


def file_stamp(path):
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def assert_case_files(folder, marker, progress=None):
    for name, stamp in marker['verified_files'].items():
        path = folder/name
        if progress is not None:
            progress(path)
        if file_stamp(path) != stamp:
            raise ValueError('MedSAM2 缓存文件在校验后变化：'+str(path))


def eligible_starts(frame_classes, frames, required_objects=1):
    return [start for start in range(len(frame_classes)-frames+1)
            if len(frame_classes[start]) >= required_objects]


def case_metadata(folder, row, *, track_files=False, progress=None):
    marker_path = folder/'complete.json'
    if progress is not None:
        progress(marker_path, 0)
    marker = read_json(marker_path)
    check_signature(marker)
    if (marker['protocol'] != PROTOCOL or marker['case_id'] != row['case_id']
            or marker['source_hashes'] != row['source_hashes']
            or set(marker['files']) != {'ct.npy', 'labels.npy'}):
        raise ValueError('MedSAM2 病例完成记录与来源不一致：'+row['case_id'])
    if not eligible_starts(marker['frame_classes'], 8):
        raise ValueError('没有首帧带可见目标的正序连续 8 帧：'+row['case_id'])
    stamps = {}
    for name, expected in marker['files'].items():
        path = folder/name
        if progress is not None:
            progress(path, 0)
        before = file_stamp(path)
        if file_sha(path, progress=progress) != expected:
            raise ValueError('MedSAM2 缓存文件损坏：'+str(path))
        if file_stamp(path) != before:
            raise ValueError('MedSAM2 缓存文件在校验中变化：'+str(path))
        stamps[name] = before
    if track_files:
        # 运行时文件属性只用于检测变化，病例签名仍以文件摘要为准。
        marker['verified_files'] = stamps
    return marker


def open_prepared(root, fold, *, smoke=False, train_cases=(), val_cases=(), progress=None):
    root = Path(root)
    if progress is not None:
        progress(root/'manifest.json', 0)
    root = root.resolve(strict=True)
    manifest = read_json(root/'manifest.json')
    if progress is not None:
        progress(root/'summary.json', 0)
    summary = read_json(root/'summary.json')
    check_signature(manifest)
    if (manifest['protocol'] != PROTOCOL or summary['status'] != 'completed'
            or summary['manifest_identity'] != manifest['identity']):
        raise ValueError('要求完成并匹配的 MedSAM2 预处理目录')
    if not smoke and (not manifest['full_cohort'] or train_cases or val_cases):
        raise ValueError('正式 MedSAM2 训练需要完整审核队列')
    cases = {row['case_id']: row for row in manifest['cases']}
    roles = {cid: row['roles'][str(fold)] for cid, row in cases.items()}
    train = sorted(train_cases or [cid for cid, role in roles.items() if role in ('train', 'aux_train')])
    val = sorted(val_cases or [cid for cid, role in roles.items() if role == 'val'])
    if smoke and not train_cases:
        train = [cid for cid in train if roles[cid] == 'train'][:1] + [
            cid for cid in train if roles[cid] == 'aux_train'][:1]
    if smoke and not val_cases:
        val = val[:1]
    if (not train or not val or len(set(train+val)) != len(train+val)
            or any(roles.get(cid) not in ('train', 'aux_train') for cid in train)
            or any(roles.get(cid) != 'val' for cid in val)):
        raise ValueError('MedSAM2 训练/验证病例与固定折角色不一致')
    metadata = {}
    for cid in train+val:
        marker = case_metadata(root/'cases'/cid, cases[cid], track_files=True, progress=progress)
        if summary['completed'].get(cid) != marker['identity']:
            raise ValueError('MedSAM2 病例未在已提交清单中：'+cid)
        metadata[cid] = marker
    return root, manifest, cases, metadata, train, val, manifest['folds'][str(fold)]


def prepare_case(folder, row, gate, phase):
    if (folder/'complete.json').is_file():
        phase('check_cached_arrays')
        return case_metadata(folder, row), True
    folder.mkdir(parents=True, exist_ok=True)
    phase('read_native_ct_and_label')
    with read_slot(gate):
        try:
            data, truth, geometry = load_native_pair(row)
        except FileNotFoundError as error:
            raise SourceUnavailable(error.errno, '审核过的原始文件暂不可访问', error.filename) from error
    phase('reorient_and_audit')
    image, truth = to_axial(data, geometry), to_axial(truth, geometry)
    if image.shape[0] < 8 or not np.isfinite(image).all():
        raise ValueError('连续 8 帧不足或 CT 含非有限值：'+row['case_id'])
    observed = sorted(set(map(int, np.unique(truth))) & set(range(1, 8)))
    expected = sorted(set(range(1, 8)) - set(row['missing_label_ids']))
    if observed != expected:
        raise ValueError('MedSAM2 标签可用性和原审核结果不一致：'+row['case_id'])
    frame_classes = []
    for mask in truth:
        resized = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((512, 512), Image.Resampling.NEAREST))
        frame_classes.append(sorted(set(map(int, np.unique(resized))) & set(expected)))
    if sorted({label for values in frame_classes for label in values}) != expected:
        raise ValueError('已知类别在正式 512 网格上消失：'+row['case_id'])
    if not eligible_starts(frame_classes, 8):
        raise ValueError('没有首帧带可见目标的正序连续 8 帧：'+row['case_id'])
    phase('write_ct_and_labels')
    atomic_array(folder/'ct.npy', image)
    atomic_array(folder/'labels.npy', truth)
    result = signed(dict(protocol=PROTOCOL, case_id=row['case_id'], shape=list(image.shape),
                         geometry=geometry, observed=observed, frame_classes=frame_classes,
                         source_hashes=row['source_hashes'],
                         files={name: file_sha(folder/name) for name in ('ct.npy', 'labels.npy')}))
    phase('commit')
    write_json(folder/'complete.json', result)
    return result, False


def run_prepare(source, output, *, folds=(0, 1, 2), cases=(), resume=False, workers=None, readers=2):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError('MedSAM2 输出和旧预处理目录必须分开')
    rows, stats, source_identity, patient_verified = read_source(source, folds)
    by_id = {row['case_id']: row for row in rows}
    if len(set(cases)) != len(cases) or set(cases)-by_id.keys():
        raise ValueError('MedSAM2 病例选择重复或不属于审核队列')
    selected = [by_id[cid] for cid in (cases or sorted(by_id))]
    for row in selected:
        for name in ('image', 'label'):
            path = Path(row['pair'][name]).resolve()
            if path.is_relative_to(output) or output.is_relative_to(path.parent):
                raise ValueError('MedSAM2 输出不得写入原始数据目录')
    contract = signed(dict(protocol=PROTOCOL, source=str(source), source_identity=source_identity,
                           folds=stats, cases=selected, full_cohort=len(selected)==len(rows),
                           patient_identity_verified=patient_verified))
    with run_lock(output.with_name(output.name+'.prepare.lock')):
        if resume:
            if read_json(output/'manifest.json') != contract:
                raise ValueError('MedSAM2 预处理恢复的输入或折统计变化')
        else:
            if output.exists() and any(output.iterdir()):
                raise FileExistsError('已有 MedSAM2 预处理目录，请选择 --resume 或新路径')
            write_json(output/'manifest.json', contract)
        write_json(output/'read_gate.json', {'slots': readers})
        began = time.monotonic()
        log = RunLog(output)
        completed, failed, jobs = {}, {}, []
        reused = []
        for row in selected:
            cid = row['case_id']
            task, result = (output/'jobs'/f'{cid}-{name}.json' for name in ('task', 'result'))
            folder = output/'cases'/cid
            result.unlink(missing_ok=True)
            write_json(task, dict(row=row, result=str(result), folder=str(folder), gate=str(output/'read_gate.json')))
            voxels = math.prod(row['shape_xyz'])
            jobs.append(dict(name=cid, task=str(task), result=str(result), work_units=voxels,
                             entry=str(ROOT/'scripts/medsam2_prepare_worker.py'),
                             estimate=voxels*20+256*1024**2))

        deferred, replay, exits, outages = set(), set(), {}, {}

        def record(cid, outcome):
            if outcome['success']:
                completed[cid] = outcome['identity']
                if outcome['reused']:
                    reused.append(cid)
            elif outcome.get('source_unavailable'):
                deferred.add(cid)
                outages[cid] = outages.get(cid, 0)+1
                log.event('case.wait_io', f'{cid} 暂时无法访问已审核来源', case_id=cid,
                          attempt=outages[cid], error=outcome['error'])
            elif 'worker_pid' not in outcome and exits.get(cid, 0) == 0:
                exits[cid] = 1
                replay.add(cid)
                log.event('case.replay', f'{cid} 进程未提交结果，重新检查并处理该病例', case_id=cid)
            else:
                failed[cid] = outcome['error']

        scheduling = dict(completed=0, peak_jobs=0, recoveries=0, worker_replays=0)
        pending = jobs
        round_index = 0
        while pending:
            stage = 'medsam2_prepare' if round_index == 0 else f'medsam2_prepare_retry_{round_index}'
            observed = execute_jobs(pending, output, log, stage_name=stage, workers=workers or 4,
                                    readers=readers, completed=record)
            round_index += 1
            scheduling['peak_jobs'] = max(scheduling['peak_jobs'], observed['peak_jobs'])
            scheduling['recoveries'] += observed['recoveries']
            if not replay and deferred:
                # 当前没有其他任务可计算，等待时不占用病例工作进程。
                next_retry = time.time()+60
                for _ in range(6):
                    log.status('waiting_io', pending=sorted(deferred), completed=len(completed),
                               next_retry_at=next_retry)
                    time.sleep(10)
                replay.update(deferred)
                deferred.clear()
            pending = [job for job in jobs if job['name'] in replay]
            for job in pending:
                Path(job['result']).unlink(missing_ok=True)
                task = read_json(job['task'])
                task['probe_only'] = job['name'] in outages
                write_json(job['task'], task)
            replay.clear()
        scheduling.update(completed=len(completed)+len(failed), worker_replays=sum(exits.values()))
        state = 'needs_attention' if failed else 'completed'
        result = dict(status=state, protocol=PROTOCOL, requested=len(selected),
                      completed=completed, failed=failed, reused=sorted(reused),
                      manifest_identity=contract['identity'], scheduling=scheduling,
                      elapsed_seconds=time.monotonic()-began)
        write_json(output/'summary.json', result)
        log.status(state, completed=len(completed), requested=len(selected), failed=len(failed))
        return result
