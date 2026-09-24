"""十个正式模型冻结后，逐病例推理、恢复原图坐标并评价；支持断点继续。

目标中心只在本文件中用于最终评价。best 来自来源开发集的 EMA Dice，
final 仅用于确认训练完整；不允许根据目标分数选择模型或后处理。
"""
import gc
import math
import os
from pathlib import Path
import time
import traceback
import nibabel as nib
import numpy as np
from scipy.ndimage import label
from data_pipeline import ROOT, CODE_ROOT, FOLDS, LABELS, inverse_labels, map_labels
from prepare_nnunet import configure_paths
from server_data import digest, read_json, write_json
from metrics import binary_metrics

PROTOCOL = {'version': 1, 'checkpoint': 'checkpoint_best.pth',
            'selection': 'source development patch EMA foreground Dice',
            'primary': 'raw', 'secondary': 'per-class LCC, 26-connectivity',
            'mirroring': False, 'tile_step_size': .5, 'gaussian': True,
            'vessel_truncation': False, 'seed': 0,
            'hd95': 'max of two directed 95th percentiles; full affine mm',
            'empty': 'one-empty Dice=0 HD=inf; both-empty undefined with counts'}


def establish_protocol():
    """在训练开始前记录评价约定；续跑时不允许悄悄修改。"""
    path = ROOT / '00_admin/evaluation_protocol.json'
    if path.exists() and read_json(path) != PROTOCOL:
        raise ValueError('评价约定已改变，请使用独立实验工作区')
    write_json(path, PROTOCOL)


def require_completed(run):
    if run.get('status') != 'completed' or not run.get('formal_training'):
        raise ValueError('十个正式训练必须全部完成后才能打开目标中心评价')


def freeze_models():
    """先检查全部十个模型，再一次性保存不可变的检查点清单。"""
    establish_protocol()
    splits = read_json(ROOT / '04_data/manifests/splits_v1.json')['folds']
    code = {p.name: digest(p) for p in sorted((CODE_ROOT / '05_src').glob('*.py'))}
    models = []
    for fold in FOLDS:
        initial = []
        split = splits[fold]
        prep = ROOT / '04_data/derived/nnUNet_preprocessed' / f"Dataset{split['dataset_id']:03d}_{fold}"
        for method in ['B0', 'B1']:
            out = ROOT / '07_experiments/formal' / f'{fold}_{method}_seed0'
            run = read_json(out / 'run.json')
            require_completed(run)
            expected = dict(fold=fold, method=method, seed=0,
                            plan_sha256=digest(prep / 'Project1Plans.json'),
                            splits_sha256=digest(ROOT / '04_data/manifests/splits_v1.json'),
                            environment_sha256=digest(ROOT / '06_configs/pip_freeze_training.txt'),
                            conda_environment_sha256=digest(ROOT / '06_configs/conda_explicit_training.txt'),
                            code_sha256=code)
            if run['signature'] != expected or run['split'] != split:
                raise ValueError(f'{fold}/{method} 训练签名与当前实验不符')
            initial.append(run['initial_weight_sha256'])
            best, final = out / 'fold_0' / PROTOCOL['checkpoint'], out / 'fold_0/checkpoint_final.pth'
            models.append(dict(fold=fold, method=method, checkpoint=str(best),
                               checkpoint_sha256=digest(best), final_sha256=digest(final),
                               signature=expected, targets=split['target']))
        if not initial[0] or initial[0] != initial[1]:
            raise ValueError(f'{fold} B0/B1初始权重不同')
    frozen = dict(protocol=PROTOCOL, models=models)
    path = ROOT / '08_results/evaluation/frozen_models.json'
    if path.exists() and read_json(path) != frozen:
        raise ValueError('模型冻结后发生变化，禁止混用已有目标评价结果')
    write_json(path, frozen)
    return models, digest(path)


def largest_components(seg):
    """每类保留一个最大连通块；对B0/B1使用完全相同的规则。"""
    output = np.zeros_like(seg)
    for index in range(1, 8):
        components, count = label(seg == index, structure=np.ones((3, 3, 3)))
        if count:
            sizes = np.bincount(components.ravel())
            sizes[0] = 0
            output[components == sizes.argmax()] = index
    return output


def score_case(prediction, reference, affine):
    classes = {}
    for name, index in LABELS.items():
        if index == 0:
            continue
        score = binary_metrics(prediction == index, reference == index, affine)
        # 标准JSON不支持Infinity，用明确的字符串记录，汇总时还原，绝不丢弃失败病例。
        classes[name] = {k: ('inf' if isinstance(v, float) and math.isinf(v) else v)
                         for k, v in score.items()}
    values = [s['dice'] for s in classes.values() if s['dice'] is not None]
    return dict(classes=classes, macro_dice=float(np.mean(values)) if values else None,
                defined_classes=len(values), both_empty_classes=7-len(values),
                one_empty_classes=sum(s['status'] == 'one_empty' for s in classes.values()))


def save_prediction(path, seg, original):
    """无损保存官方编号0/205/.../850，避免uint8把标签截断。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = original.header.copy()
    header.set_data_dtype(np.uint16)
    temp = path.with_name(path.name.replace('.nii.gz', '.tmp.nii.gz'))
    nib.save(nib.Nifti1Image(inverse_labels(seg).astype(np.uint16), original.affine, header), temp)
    check = nib.load(temp)
    if check.shape != original.shape or not np.allclose(check.affine, original.affine, atol=1e-4, rtol=0):
        raise ValueError('保存的预测未恢复原图网格')
    if not np.array_equal(map_labels(np.asanyarray(check.dataobj)), seg):
        raise ValueError('预测写出后标签发生改变')
    os.replace(temp, path)


def cached_case(path, signature):
    if not path.exists():
        return False
    item = read_json(path)
    if item.get('signature') != signature or item.get('status') != 'completed':
        raise ValueError(f'病例评价记录不匹配：{path}')
    for value in item['outputs'].values():
        if digest(value['path']) != value['sha256']:
            raise ValueError(f'预测文件校验不符：{value["path"]}')
    return True


def make_predictor(model):
    """使用与短测相同的官方推理API，关闭镜像，原图滑窗推理。"""
    import torch
    from project_trainers import Project1B0, Project1B1
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    split = read_json(ROOT / '04_data/manifests/splits_v1.json')['folds'][model['fold']]
    prep = ROOT / '04_data/derived/nnUNet_preprocessed' / f"Dataset{split['dataset_id']:03d}_{model['fold']}"
    cls = Project1B0 if model['method'] == 'B0' else Project1B1
    trainer = cls(read_json(prep / 'Project1Plans.json'), '3d_fullres', 0, read_json(prep / 'dataset.json'))
    trainer.initialize()
    trainer.load_checkpoint(model['checkpoint'])
    trainer.network.eval()
    trainer.set_deep_supervision_enabled(False)
    predictor = nnUNetPredictor(tile_step_size=.5, use_gaussian=True, use_mirroring=False,
                               perform_everything_on_device=False, device=torch.device('cuda'),
                               verbose=False, allow_tqdm=False)
    predictor.manual_initialization(trainer.network, trainer.plans_manager, trainer.configuration_manager,
                                    [trainer.network.state_dict()], trainer.dataset_json, cls.__name__, None)
    return predictor, trainer.plans_manager.image_reader_writer_class()


def evaluate_model(model, frozen_sha):
    inventory = {c['case_id']: c for c in read_json(ROOT / '04_data/manifests/case_inventory.json')['cases']}
    folder = ROOT / '08_results/evaluation' / f"{model['fold']}_{model['method']}"
    predictor = reader = None
    for case_id in model['targets']:
        case = inventory[case_id]
        signature = dict(frozen_sha256=frozen_sha, image_sha256=case['image_sha256'],
                         label_sha256=case['label_sha256'], case_id=case_id)
        record = folder / f'{case_id}.json'
        if cached_case(record, signature):
            print(f'评价已完成，跳过：{model["fold"]}/{model["method"]}/{case_id}', flush=True)
            continue
        for kind in ['image', 'label']:
            if digest(case[kind]) != case[f'{kind}_sha256']:
                raise ValueError(f'{case_id} 原始{kind}发生变化')
        if predictor is None:
            predictor, reader = make_predictor(model)
        started = time.perf_counter()
        original = nib.load(case['image'])
        images, properties = reader.read_images([case['image']])
        seg = predictor.predict_single_npy_array(images, properties, None, None, False)
        # 读写器内部轴顺序与nibabel不同。先写0..7，再从磁盘读回原图坐标。
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / f'{case_id}.axes.tmp.nii.gz'
        reader.write_seg(seg, str(temporary), properties)
        restored = nib.load(temporary)
        if restored.shape != original.shape or not np.allclose(restored.affine, original.affine, atol=1e-4, rtol=0):
            raise ValueError('推理未正确恢复原图空间')
        raw = np.asarray(restored.dataobj).copy()
        inverse_labels(raw)  # 验证仅包含0..7
        del restored
        temporary.unlink()
        reference_file = nib.load(case['label'])
        if reference_file.shape != original.shape or not np.allclose(reference_file.affine, original.affine, atol=1e-4, rtol=0):
            raise ValueError('标注与影像不在相同网格')
        reference = map_labels(np.asanyarray(reference_file.dataobj))
        results, outputs = {}, {}
        for variant, prediction in [('raw', raw), ('lcc', largest_components(raw))]:
            destination = ROOT / '08_results/predictions' / f"{model['fold']}_{model['method']}" / f'{case_id}_{variant}.nii.gz'
            save_prediction(destination, prediction, original)
            outputs[variant] = dict(path=str(destination), sha256=digest(destination))
            results[variant] = score_case(prediction, reference, original.affine)
        write_json(record, dict(signature=signature, status='completed', fold=model['fold'],
                                method=model['method'], modality=case['modality'], case_id=case_id,
                                seconds=time.perf_counter()-started, outputs=outputs, results=results))
        print(f'{model["fold"]}/{model["method"]}/{case_id}: raw Dice={results["raw"]["macro_dice"]}', flush=True)
    del predictor, reader
    gc.collect()
    import torch
    torch.cuda.empty_cache()


def main():
    configure_paths()
    os.environ['nnUNet_n_proc_DA'] = '0'
    import torch
    torch.set_num_threads(2)
    state_path = ROOT / '08_results/evaluation/status.json'
    state = dict(status='checking_all_models', started=time.strftime('%F %T'))
    write_json(state_path, state)
    try:
        models, sha = freeze_models()
        state.update(status='running', frozen_sha256=sha)
        for model in models:
            state['current'] = f"{model['fold']}_{model['method']}"
            write_json(state_path, state)
            evaluate_model(model, sha)
        from evaluation_summary import summarize
        summarize(ROOT)
        state['status'] = 'completed'
    except BaseException:
        state.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        state['ended'] = time.strftime('%F %T')
        write_json(state_path, state)


if __name__ == '__main__':
    main()
