import sys
import time
from pathlib import Path

import torch

from chd_stunet.acceleration.prediction import admit_prediction_device, desktop_identity, sharing_inventory
from chd_stunet.config import ROOT, resolve_path
from chd_stunet.observability.locking import run_lock
from chd_stunet.observability.runlog import RunLog, write_json
from chd_stunet.prediction.export import valid_result
from chd_stunet.training.fleet import eligible, query_visible
from chd_stunet.training.resources import GPUUnavailable

from . import PROTOCOL
from .checkpoints import checkpoint_pointer, coordinates, load_committed
from .contracts import check_signature, digest, file_sha, read_json, signed
from .inference import LABEL_NAMES, evaluate_case, predict_image, read_prompted_image
from .model import build, upstream_root, versions
from .preprocessing import case_metadata
from .runtime import frozen_job


ARTIFACTS = ('segmentation.nii.gz', 'prompts.json', 'result.json')
RAW_ARTIFACTS = ('prediction_axial.npy', 'max_logits.npy', *(f'mask_{label}.npy' for label in LABEL_NAMES))


def heldout_cases(args, prepared, contract):
    manifest = read_json(prepared/'manifest.json')
    summary = read_json(prepared/'summary.json')
    check_signature(manifest)
    if (manifest['protocol'] != PROTOCOL or manifest['identity'] != contract['data_identity']
            or summary['status'] != 'completed' or summary['manifest_identity'] != manifest['identity']):
        raise ValueError('MedSAM2 预测需要来源训练合同对应的已完成预处理')
    fold = str(contract['fold'])
    candidates = {row['case_id']: row for row in manifest['cases'] if row['roles'][fold] == 'test'}
    defaults = sorted(candidates)
    wanted = args.cases or (defaults[:1] if args.smoke else defaults)
    if not wanted or len(wanted) != len(set(wanted)) or set(wanted) - candidates.keys():
        raise ValueError('MedSAM2 测试病例必须属于来源训练折留出集')
    if any(cid not in summary['completed'] for cid in wanted):
        raise ValueError('MedSAM2 留出病例未在已提交清单中')
    return {cid: candidates[cid] for cid in wanted}, {cid: summary['completed'][cid] for cid in wanted}


def audit_case(prepared, row, expected):
    cid = row['case_id']
    marker = case_metadata(prepared/'cases'/cid, row, track_files=True)
    if marker['identity'] != expected:
        raise ValueError('MedSAM2 留出病例与已提交缓存不一致：'+cid)
    return marker


def select_device(requested, desktop):
    if requested == 'cpu':
        return admit_prediction_device(requested, desktop)
    cards = query_visible(ROOT/'src')
    if desktop:
        cards = sharing_inventory(cards, desktop)
    ready = eligible(cards)
    if not ready:
        raise GPUUnavailable('当前没有可用的 MedSAM2 预测 GPU')
    device, admission = admit_prediction_device(f"cuda:{ready[0]['ordinal']}", desktop)
    torch.cuda.set_per_process_memory_fraction(admission['usable_bytes']/admission['total_bytes'], device)
    return device, admission


class Progress:
    def __init__(self, output, total):
        self.log = RunLog(output)
        events = output/'logs/events.jsonl'
        if events.is_file():
            with events.open(encoding='utf-8') as stream:
                self.log.seq = sum(1 for _ in stream)
        self.total, self.completed, self.reused = total, 0, 0
        self.case_id, self.last_frame = None, 0

    def mark(self, state, stage, message, **fields):
        self.log.event('prediction.'+stage, message, **fields)
        self.log.status(state, stage=stage, completed=self.completed, total=self.total,
                        reused=self.reused, case_id=self.case_id, **fields)

    def frames(self, label, finished, depth):
        now = time.monotonic()
        if finished == 1 or finished == depth or now-self.last_frame >= 10:
            self.last_frame = now
            self.mark('running', 'propagation', f'{self.case_id} {LABEL_NAMES[label]} 传播 {finished}/{depth} 帧',
                      structure=LABEL_NAMES[label], frames_completed=finished, frames_total=depth)


def completed_case(directory, identity, artifacts):
    marker_path = directory/'complete.json'
    if not marker_path.is_file():
        return None
    marker = read_json(marker_path)
    if set(marker['files']) != set(artifacts):
        raise ValueError('MedSAM2 病例输出清单不完整：'+directory.name)
    return valid_result(directory, identity)


def run_heldout(predictor, output, prepared, rows, inputs, job, window, progress):
    results = []
    retain_raw = job.get('retain_raw', False)
    artifacts = ARTIFACTS + RAW_ARTIFACTS if retain_raw else ARTIFACTS
    for cid, row in rows.items():
        progress.case_id = cid
        directory = output/'cases'/cid
        identity = digest(dict(prediction=job['identity'], case_id=cid, input=inputs[cid]))
        record = completed_case(directory, identity, artifacts)
        if record is None:
            progress.mark('running', 'audit', f'核验 {cid} 的图像和标签缓存')
            metadata = audit_case(prepared, row, inputs[cid])
            record = evaluate_case(predictor, prepared, cid, metadata, window, directory, progress.log,
                                   retain_raw=retain_raw, export_segmentation=True, frame_progress=progress.frames)
            write_json(directory/'complete.json',
                       dict(identity=identity, files={name: file_sha(directory/name) for name in artifacts}))
        else:
            progress.reused += 1
        results.append(record)
        progress.completed = len(results)
        progress.mark('running', 'case_complete', f'{cid} 完成，整卷进度 {progress.completed}/{progress.total}',
                      mean_dice=record['mean_dice'])
    structure = {
        name: dict(mean_dice=sum(row['metrics'][name]['dice'] for row in results)/len(results),
                   missing_predictions=sum(row['metrics'][name]['missing_prediction'] for row in results))
        for name in LABEL_NAMES.values()
    }
    return dict(status='completed', scope='whole_volume', fold=job['fold'], prompt_source=job['prompt_source'],
                cases=len(results), reused_cases=progress.reused, includes_prompt_slices=True, results=results,
                mean_dice=sum(row['mean_dice'] for row in results)/len(results), by_structure=structure)


def predict(args, paths, stamp):
    training = resolve_path(args.training_run)
    print('MedSAM2 预测准备，核验来源训练记录和检查点', flush=True)
    contract = frozen_job(training)['contract']
    config = contract['config']
    if config['scope'] not in ('formal', 'smoke') or (config['scope'] == 'smoke' and not args.smoke):
        raise ValueError('MedSAM2 预测范围与训练合同不匹配')
    if versions() != contract['versions']:
        raise ValueError('MedSAM2 预测依赖与训练合同不匹配')
    if args.fold is not None and args.fold != contract['fold']:
        raise ValueError('MedSAM2 预测与来源训练折不一致')
    manual = args.prompt_source == 'manual_box'
    if args.resume and (manual or not args.output):
        raise ValueError('MedSAM2 续跑需要留出集模式和显式 --output')
    if manual and (not args.image or not args.prompts or args.cases):
        raise ValueError('人工提示需要 --image 和 --prompts')
    if manual and args.retain_raw:
        raise ValueError('--retain-raw 只用于留出集预测')
    if not manual and (args.image or args.prompts):
        raise ValueError('真值模拟提示只用于原折留出病例')

    prepared = Path(contract['prepared']).resolve()
    output = resolve_path(args.output) if args.output else (
        resolve_path(paths.get('prediction_output_root', 'outputs/prediction'))/'medsam2'/stamp)
    external_inputs = [resolve_path(args.image), resolve_path(args.prompts)] if manual else []
    if any(output.is_relative_to(path) or path.is_relative_to(output)
           for path in (training, prepared, *external_inputs)):
        raise ValueError('MedSAM2 预测输出须和输入分离')
    if manual:
        image, geometry, prompts, image_stamp = read_prompted_image(*external_inputs)
        rows, inputs = {}, dict(image_sha256=prompts['image_sha256'], prompts_identity=digest(prompts))
    else:
        rows, inputs = heldout_cases(args, prepared, contract)

    requested = args.device or 'cuda'
    pointer = checkpoint_pointer(training, args.checkpoint)
    update, epoch, phase = coordinates(pointer)
    checkpoint = dict(pointer, role=args.checkpoint, update=update, epoch=epoch, phase=phase)
    record = dict(protocol=PROTOCOL, prediction_revision=1, training_identity=contract['identity'],
                  training_scope=config['scope'], fold=contract['fold'], checkpoint=checkpoint,
                  prompt_source=args.prompt_source, case_order=list(rows), inputs=inputs, device_type=requested)
    if args.retain_raw:
        record['retain_raw'] = True
    job = signed(record)
    if args.check_only:
        upstream_root(config)
        load_committed(training, contract, pointer)
        for cid, row in rows.items():
            print(f'MedSAM2 留出集核验 {cid}', flush=True)
            audit_case(prepared, row, inputs[cid])
        return dict(status='checked', prediction=job), output

    with run_lock(output.with_name(output.name+'.predict.lock')):
        job_path = output/'prediction.json'
        if args.resume:
            if read_json(job_path) != job:
                raise ValueError('MedSAM2 续跑的检查点、输入或计算精度与原任务不一致')
        elif output.exists() and any(output.iterdir()):
            raise FileExistsError('MedSAM2 预测输出已有内容，请指定新目录或续跑')
        else:
            write_json(job_path, job)
        progress = Progress(output, len(rows) or 1)
        try:
            progress.mark('running', 'checkpoint', f'检查点 {args.checkpoint}，第 {epoch} 轮，更新 {update}')
            payload = load_committed(training, contract, pointer)
            # 已全部提交的结果可以原样汇总，无需重新申请 GPU 或加载模型。
            complete = bool(rows) and all((output/'cases'/cid/'complete.json').is_file() for cid in rows)
            predictor = None
            device_type = 'cpu'
            if not complete:
                desktop = desktop_identity(args.share_desktop_pid, args.smoke) if args.share_desktop_pid else None
                progress.mark('running', 'device', f'准备 {requested.upper()} 预测设备')
                device, admission = select_device(requested, desktop)
                device_type = device.type
                progress.mark('running', 'model', f'在 {device} 加载 MedSAM2', device=admission)
                predictor = build(config, training=False, device=device)
                predictor.load_state_dict(payload['model'], strict=True)
            del payload
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device_type == 'cuda'):
                if manual:
                    progress.case_id = external_inputs[0].name
                    summary = predict_image(predictor, image, geometry, prompts, image_stamp, contract['window'],
                                            output, progress.log, frame_progress=progress.frames)
                    progress.completed = 1
                else:
                    summary = run_heldout(predictor, output, prepared, rows, inputs, job, contract['window'], progress)
            summary.update(fold=contract['fold'], training_scope=config['scope'], checkpoint=checkpoint,
                           prediction_identity=job['identity'])
            write_json(output/'summary.json', summary)
            progress.mark('completed', 'done', f'MedSAM2 预测完成 {progress.completed}/{progress.total}')
        finally:
            error = sys.exc_info()[1]
            if error is not None:
                progress.mark('failed', 'failure', str(error), error_type=type(error).__name__)
    return summary, output
