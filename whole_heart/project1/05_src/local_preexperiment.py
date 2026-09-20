"""CT 小型预实验：固定来源训练块拟合 + 从头训练5轮基线 + 固定来源开发病例看图。

所有结果仅用于工程排错及来源内探索，不能作为跨中心效果结论。
不改变正式训练器、冻结划分、原始影像及既有缓存。默认最多运行60分钟。
"""
import argparse
import csv
import gc
import importlib.metadata
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from data_pipeline import ROOT, CODE_ROOT, load_json, inverse_labels
from prepare_nnunet import configure_paths
from server_data import digest, write_json

NAMES = ['Myo', 'LA', 'LV', 'RA', 'RV', 'AO', 'PA']
COLORS = ['#000000', '#e45756', '#f2cf5b', '#4c78a8', '#b279a2', '#54a24b', '#ff9da6', '#f58518']


def dice_scores(pred, ref):
    """完整网格上的逐类Dice；双方都没有该类时写null，不虚报满分。"""
    result = {}
    for label, name in enumerate(NAMES, 1):
        p, r = pred == label, ref == label
        denominator = int(p.sum()) + int(r.sum())
        result[name] = 2 * int(np.count_nonzero(p & r)) / denominator if denominator else None
    return result


def mean_defined(values):
    available = [float(x) for x in values if x is not None and np.isfinite(x)]
    return float(np.mean(available)) if available else None


def plot_overlay(image, ref, pred, path, title):
    """切片位置只按参考标注选定；每阶段用相同规则，不按预测好坏挑图。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    # nnU-Net日志绘图会改变全局字体大小；本图恢复常规字号，避免标题互相遮挡。
    plt.rcdefaults()
    fg = np.where(ref > 0)
    centers = [int(np.median(x)) if len(x) else ref.shape[i]//2 for i, x in enumerate(fg)]
    fig, axes = plt.subplots(3, 3, figsize=(11, 10))
    cmap = ListedColormap(COLORS)
    lo, hi = np.percentile(image, [2, 98])
    if hi <= lo:
        hi = lo + 1
    for axis, center in enumerate(centers):
        gray = np.rot90(np.take(image, center, axis=axis))
        for col, label in enumerate([None, ref, pred]):
            ax = axes[axis, col]
            ax.imshow(gray, cmap='gray', vmin=lo, vmax=hi)
            if label is not None:
                cut = np.rot90(np.take(label, center, axis=axis))
                ax.imshow(np.ma.masked_equal(cut, 0), cmap=cmap, vmin=0, vmax=7, alpha=0.55, interpolation='nearest')
            ax.axis('off')
            if axis == 0:
                ax.set_title(['Image', 'Expert reference', 'Prediction'][col])
            if col == 0:
                ax.text(0, 0, f'Native axis {axis}, index {center}', color='white', fontsize=8, va='top')
    fig.suptitle(title + '\nExploratory preview; native voxel axes, not a clinical display')
    fig.legend(handles=[Patch(color=COLORS[i+1], label=name) for i, name in enumerate(NAMES)], loc='lower center', ncol=7)
    fig.tight_layout(rect=(0, .03, 1, .94))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return centers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiny-steps', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--budget-minutes', type=float, default=60)
    parser.add_argument('--check', action='store_true', help='只检查输入与指标，不创建实验、不训练')
    args = parser.parse_args()
    if not (20 <= args.tiny_steps <= 500 and 1 <= args.epochs <= 10 and 1 <= args.budget_minutes <= 90):
        parser.error('预实验限制：20..500拟合步，1..10轮基线，1..90分钟')
    configure_paths()
    os.environ['nnUNet_n_proc_DA'] = '0'
    os.environ['MPLBACKEND'] = 'Agg'
    import torch
    import nibabel as nib
    from filelock import FileLock
    from project_trainers import Project1B0
    from smoke_train import state_hash
    from nnunetv2.training.logging.nnunet_logger import MetaLogger
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
    from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    torch.set_num_threads(2)
    split = load_json(ROOT/'04_data/manifests/splits_v1.json')['folds']['ct_holdG']
    inventory = {c['case_id']: c for c in load_json(ROOT/'04_data/manifests/case_inventory.json')['cases']}
    # 各来源中心取排序后的第一个病例，选择规则与模型成绩无关。
    train_cases = [next(i for i in split['train'] if inventory[i]['group'] == g) for g in ['A ct_train', 'B ct_train']]
    val_cases = [next(i for i in split['val'] if inventory[i]['group'] == g) for g in ['A ct_train', 'B ct_train']]
    assert not set(split['target']) & set(split['train']+split['val'])
    assert all(inventory[i]['status'] == 'eligible' for i in split['train']+split['val'])
    prep = ROOT/'04_data/derived/nnUNet_preprocessed/Dataset703_ct_holdG'
    raw = ROOT/'04_data/derived/nnUNet_raw/Dataset703_ct_holdG'
    plans, dataset_json = load_json(prep/'Project1Plans.json'), load_json(prep/'dataset.json')
    assert load_json(prep/'splits_final.json') == [{'train': split['train'], 'val': split['val']}]
    assert load_json(prep/'fingerprint_provenance.json')['fit_ids'] == split['train']
    cache = prep/plans['configurations']['3d_fullres']['data_identifier']
    assert all((cache/f'{i}{suffix}').is_file() for i in split['train']+split['val'] for suffix in ['.b2nd', '_seg.b2nd', '.pkl'])
    # 验证已选原始输入，预实验不扫描或评价任何外层目标病例。
    for i in train_cases+val_cases:
        for kind in ['image', 'label']:
            assert digest(inventory[i][kind]) == inventory[i][kind+'_sha256'], f'Original changed: {i}/{kind}'
    assert dice_scores(np.ones((2,2,2)), np.ones((2,2,2)))['Myo'] == 1
    assert dice_scores(np.zeros((2,2,2)), np.ones((2,2,2)))['Myo'] == 0
    assert dice_scores(np.zeros((2,2,2)), np.zeros((2,2,2)))['Myo'] is None
    assert torch.cuda.is_available()
    print(f'INPUT CHECK PASSED: tiny={train_cases}; preview={val_cases}; target excluded', flush=True)
    if args.check:
        return
    free, total = torch.cuda.mem_get_info()
    if free < 5.5*2**30:
        raise RuntimeError(f'空闲显存不足5.5GiB：{free/2**30:.2f}GiB')
    out = ROOT/'07_experiments'/time.strftime('preexperiment_ct_B0_seed0_%Y%m%d_%H%M%S')
    lock = FileLock(str(ROOT/'07_experiments/.preexperiment.lock'), timeout=0)
    with lock:
        out.mkdir(exist_ok=False)
        class Tee:
            def __init__(self, original, file):
                self.original, self.file = original, file
            def write(self, value):
                self.original.write(value)
                self.file.write(value)
                self.file.flush()
            def flush(self):
                self.original.flush()
                self.file.flush()
        stdout, stderr = sys.stdout, sys.stderr
        log = (out/'console.log').open('w', encoding='utf-8')
        sys.stdout, sys.stderr = Tee(stdout, log), Tee(stderr, log)
        started = time.perf_counter()
        record = {'run_id': out.name, 'status': 'running', 'started_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'pid': os.getpid(), 'scope': 'CT source-only exploratory preexperiment; NOT formal comparison', 'fold': 'ct_holdG', 'method': 'B0', 'seed': 0, 'target_evaluated': False, 'split': split, 'tiny_cases': train_cases, 'fixed_preview_cases': val_cases, 'tiny_steps_requested': args.tiny_steps, 'epochs_requested': args.epochs, 'budget_minutes': args.budget_minutes, 'gpu': torch.cuda.get_device_name(), 'versions': {n: importlib.metadata.version(n) for n in ['torch', 'nnunetv2', 'numpy', 'nibabel']}, 'code_sha256': {p.name: digest(p) for p in (CODE_ROOT/'05_src').glob('*.py')}, 'plan_sha256': digest(prep/'Project1Plans.json'), 'split_sha256': digest(ROOT/'04_data/manifests/splits_v1.json'), 'tiny_probe_history': [], 'epoch_history': [], 'volume_previews': []}
        def save():
            record['elapsed_seconds'] = time.perf_counter()-started
            record['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
            write_json(out/'run.json', record)
        def budget():
            if time.perf_counter()-started > args.budget_minutes*60:
                raise TimeoutError('达到预实验运行预算；保存已有记录，不自动扩展训练')
        def new_trainer(name):
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.benchmark = False
            trainer = Project1B0(plans, '3d_fullres', 0, dataset_json)
            folder = out/name
            folder.mkdir()
            trainer.output_folder_base = str(folder)
            trainer.output_folder = str(folder/'fold_0')
            Path(trainer.output_folder).mkdir()
            trainer.log_file = str(folder/'training.log')
            trainer.logger = MetaLogger(trainer.output_folder, resume=False)
            trainer.save_every = 1
            trainer.initialize()
            trainer.do_split()  # 再次使用已有训练器核对冻结清单。
            return trainer
        def probe(trainer, batches, step, snapshots=False):
            trainer.network.eval()
            results, scores = [], []
            with torch.no_grad():
                for j, batch in enumerate(batches):
                    result = trainer.validation_step(batch)
                    denominator = 2*result['tp_hard']+result['fp_hard']+result['fn_hard']
                    values = [float(2*t/d) if d else None for t,d in zip(result['tp_hard'],denominator)]
                    scores.append(mean_defined(values))
                    results.append(float(result['loss']))
                    if snapshots:
                        with torch.autocast('cuda'):
                            logits = trainer.network(batch['data'].to('cuda'))[0]
                        pred = logits.argmax(1)[0].cpu().numpy()
                        plot_overlay(batch['data'][0,0].numpy(), batch['target'][0][0,0].numpy(), pred, out/f'tiny_{train_cases[j]}_step{step:03d}.png', f'{train_cases[j]} fixed TRAINING patch, step {step} (not validation)')
            entry = {'step': step, 'same_training_patch_loss': float(np.mean(results)), 'same_training_patch_mean_dice': mean_defined(scores)}
            record['tiny_probe_history'].append(entry)
            trainer.network.train()
            print('TINY_PROBE', entry, flush=True)
            save()
        def predict(trainer, epoch):
            trainer.network.eval()
            trainer.set_deep_supervision_enabled(False)
            folder = out/f'preview_epoch{epoch:03d}'
            folder.mkdir()
            try:
                reader = trainer.plans_manager.image_reader_writer_class()
                predictor = nnUNetPredictor(tile_step_size=.5, use_gaussian=True, use_mirroring=False, perform_everything_on_device=False, device=torch.device('cuda'), verbose=False, allow_tqdm=False)
                predictor.manual_initialization(trainer.network, trainer.plans_manager, trainer.configuration_manager, [trainer.network.state_dict()], dataset_json, 'Project1B0', None)
                for case_id in val_cases:
                    budget()
                    tick = time.perf_counter()
                    image_path = raw/'imagesTr'/f'{case_id}_0000.nii.gz'
                    images, properties = reader.read_images([str(image_path)])
                    seg = predictor.predict_single_npy_array(images, properties, None, None, False)
                    pred_path = folder/f'{case_id}_prediction.nii.gz'
                    reader.write_seg(inverse_labels(seg), str(pred_path), properties)
                    original = nib.load(image_path)
                    pred_file = nib.load(pred_path)
                    assert original.shape == pred_file.shape and np.allclose(original.affine, pred_file.affine, atol=1e-4, rtol=0)
                    # 指标直接用写回原图网格的预测文件，避免读写器轴顺序混淆。
                    from data_pipeline import map_labels
                    pred = map_labels(np.asanyarray(pred_file.dataobj))
                    ref = np.asanyarray(nib.load(raw/'labelsTr'/f'{case_id}.nii.gz').dataobj)
                    header = original.header.copy()
                    header.set_data_dtype(np.uint16)
                    nib.save(nib.Nifti1Image(inverse_labels(pred).astype(np.uint16), original.affine, header), pred_path)
                    scores = dice_scores(pred, ref)
                    figure = folder/f'{case_id}_overlay.png'
                    slices = plot_overlay(np.asanyarray(original.dataobj), ref, pred, figure, f'{case_id} SOURCE DEVELOPMENT, epoch {epoch}')
                    entry = {'epoch': epoch, 'case_id': case_id, 'dice': scores, 'macro_dice': mean_defined(scores.values()), 'seconds': time.perf_counter()-tick, 'prediction': str(pred_path), 'figure': str(figure), 'native_slice_indices': slices, 'scope': 'complete source development volume; no postprocessing, no vessel truncation; not official evaluation'}
                    record['volume_previews'].append(entry)
                    save()
                    print('VOLUME_PREVIEW', case_id, epoch, entry['macro_dice'], flush=True)
            finally:
                trainer.set_deep_supervision_enabled(True)
                trainer.network.train()
                gc.collect()
                torch.cuda.empty_cache()
        try:
            snapshot = out/'code_snapshot'
            snapshot.mkdir()
            for source in (CODE_ROOT/'05_src').glob('*.py'):
                shutil.copyfile(source, snapshot/source.name)
            (out/'预先约定.md').write_text(
                '# 本次运行前固定的设置\n\n'
                '这是此前P2工程短测与正式P3之间的探索性预实验，不更改正式方案。\n'
                f'CT holdG / B0 / seed0；固定训练块病例：{train_cases}；固定来源开发看图病例：{val_cases}。\n'
                f'先反复拟合两个来源病例各一个固定小批次，共{args.tiny_steps}步，无随机增强、无权重衰减。\n'
                f'随后重新初始化原B0，训练{args.epochs}轮，每轮250训练步、50开发小块验证步，保留250轮学习率调度。\n'
                f'完整预测时间点：第0、1、{args.epochs}轮；切片位置按参考标注中心固定选取。\n'
                '预览计算完整病例七类Dice及其简单平均；不计算HD，不截断大血管，不做后处理。\n'
                f'运行预算{args.budget_minutes}分钟，逐步检查预算；不自动延长、不运行B1或MRI、不读取目标中心影像。\n'
                '保存真实状态、失败与已完成步骤。不设达到某个Dice即可证明方法有效的门槛。\n', encoding='utf-8')
            (out/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
            save()
            print('RUN_DIRECTORY:', out, flush=True)
            record['phase'] = 'tiny_fixed_patch_fit'
            trainer = new_trainer('tiny_fit')
            record['tiny_initial_weight_sha256'] = state_hash(trainer.network)
            record['patch_size'] = list(map(int, trainer.configuration_manager.patch_size))
            record['batch_size'] = trainer.batch_size
            # 仅排错阶段关闭权重衰减与随机增强；正式预实验恢复原B0设置。
            for group in trainer.optimizer.param_groups:
                group['weight_decay'] = 0
            transforms = trainer.get_validation_transforms(trainer._get_deep_supervision_scales())
            batches = []
            for case_id in train_cases:
                dataset = nnUNetDatasetBlosc2(str(cache), [case_id])
                loader = nnUNetDataLoader(dataset, trainer.batch_size, trainer.configuration_manager.patch_size, trainer.configuration_manager.patch_size, trainer.label_manager, oversample_foreground_percent=1, transforms=transforms)
                batches.append(next(loader))
            record['tiny_scope'] = 'two fixed minibatches from two training cases; all patches foreground-sampled, no random augmentation, weight_decay=0; does not test full-case memorization/generalization'
            torch.cuda.reset_peak_memory_stats()
            probe(trainer, batches, 0, True)
            losses = []
            for step in range(1, args.tiny_steps+1):
                budget()
                value = trainer.train_step(batches[(step-1)%len(batches)])
                losses.append(float(value['loss']))
                record['tiny_completed_steps'] = step
                if step % 50 == 0 or step == args.tiny_steps:
                    record['tiny_training_losses'] = losses
                    probe(trainer, batches, step, step == args.tiny_steps)
            record['tiny_training_losses'] = losses
            record['tiny_peak_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
            record['tiny_completed_steps'] = args.tiny_steps
            trainer.save_checkpoint(str(out/'tiny_fit/fold_0/checkpoint_tiny_debug.pth'))
            record['tiny_seconds'] = time.perf_counter()-started
            save()
            del trainer, batches, loader, dataset
            gc.collect()
            torch.cuda.empty_cache()

            record['phase'] = 'five_epoch_baseline_from_scratch'
            trainer = new_trainer('baseline')
            record['baseline_initial_weight_sha256'] = state_hash(trainer.network)
            assert record['baseline_initial_weight_sha256'] == record['tiny_initial_weight_sha256']
            record['learning_rate_horizon_epochs'] = trainer.num_epochs  # 保留250轮调度，截取前5轮，不压缩学习率。
            record['train_steps_per_epoch'] = trainer.num_iterations_per_epoch
            record['val_steps_per_epoch'] = trainer.num_val_iterations_per_epoch
            trainer.on_train_start()
            predict(trainer, 0)
            torch.cuda.reset_peak_memory_stats()
            for epoch in range(args.epochs):
                budget()
                tick = time.perf_counter()
                trainer.on_epoch_start()
                trainer.on_train_epoch_start()
                train_outputs = []
                for step in range(trainer.num_iterations_per_epoch):
                    budget()
                    train_outputs.append(trainer.train_step(next(trainer.dataloader_train)))
                    if (step+1) % 25 == 0:
                        record['live_progress'] = {'epoch': epoch+1, 'step': step+1, 'recent_train_loss': float(np.mean([v['loss'] for v in train_outputs[-25:]]))}
                        save()
                        print('BASELINE_PROGRESS', record['live_progress'], flush=True)
                trainer.on_train_epoch_end(train_outputs)
                trainer.on_validation_epoch_start()
                with torch.no_grad():
                    val_outputs = []
                    for _ in range(trainer.num_val_iterations_per_epoch):
                        budget()
                        val_outputs.append(trainer.validation_step(next(trainer.dataloader_val)))
                trainer.on_validation_epoch_end(val_outputs)
                trainer.on_epoch_end()
                tp = np.sum([v['tp_hard'] for v in val_outputs], axis=0)
                denominator = 2*tp+np.sum([v['fp_hard']+v['fn_hard'] for v in val_outputs], axis=0)
                pseudo = [float(2*t/d) if d else None for t,d in zip(tp, denominator)]
                entry = {'epoch': epoch+1, 'training_loss': float(np.mean([v['loss'] for v in train_outputs])), 'validation_patch_loss': float(np.mean([v['loss'] for v in val_outputs])), 'validation_pseudo_dice_per_class': dict(zip(NAMES, pseudo)), 'validation_pseudo_dice_macro': mean_defined(pseudo), 'seconds': time.perf_counter()-tick}
                record['epoch_history'].append(entry)
                record['baseline_completed_epochs'] = epoch+1
                record['baseline_optimizer_steps'] = (epoch+1)*trainer.num_iterations_per_epoch
                save()
                if epoch+1 in {1, args.epochs}:
                    predict(trainer, epoch+1)
            trainer.current_epoch -= 1  # native保存器会+1，确保检查点记录实际完成的5轮。
            trainer.save_checkpoint(str(out/'baseline/fold_0/checkpoint_preexperiment.pth'))
            trainer.current_epoch += 1
            record['baseline_peak_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
            record['status'] = 'completed'
            record['phase'] = 'finished'
        except BaseException as error:
            record['status'] = 'budget_stopped' if isinstance(error, TimeoutError) else 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
            record['error'] = traceback.format_exc()
            raise
        finally:
            record['ended_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
            save()
            build_report(out, record)
            print('RESULT:', out/'预实验记录.md', flush=True)
            sys.stdout, sys.stderr = stdout, stderr
            log.close()


def build_report(out, record):
    """无论成功还是失败，保留可读报告与真实完成的步骤；不补造缺失指标。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcdefaults()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    probes, epochs = record['tiny_probe_history'], record['epoch_history']
    axes[0].plot([r['step'] for r in probes], [r['same_training_patch_mean_dice'] for r in probes], marker='o')
    axes[0].set(title='Fixed TRAINING patches (debug only)', xlabel='Optimizer steps', ylabel='Mean Dice', ylim=(0,1))
    for key, label in [('training_loss','Training loss'), ('validation_patch_loss','Development patch loss')]:
        axes[1].plot([r['epoch'] for r in epochs], [r[key] for r in epochs], marker='o', label=label)
    axes[1].set(title='Baseline from scratch', xlabel='Epoch', ylabel='Dice + CE loss')
    axes[1].legend()
    for case in record['fixed_preview_cases']:
        selected = [r for r in record['volume_previews'] if r['case_id'] == case]
        axes[2].plot([r['epoch'] for r in selected], [r['macro_dice'] for r in selected], marker='o', label=case)
    axes[2].set(title='Two fixed DEVELOPMENT volumes', xlabel='Epoch', ylabel='Seven-class macro Dice', ylim=(0,1))
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(out/'learning_curves.png', dpi=130)
    plt.close(fig)
    with (out/'volume_dice.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch','case_id','macro_dice']+NAMES)
        writer.writeheader()
        for r in record['volume_previews']:
            writer.writerow(dict(epoch=r['epoch'], case_id=r['case_id'], macro_dice=r['macro_dice'], **r['dice']))
    lines = ['# CT 小型预实验记录', '', '## Material Passport', '', 'Origin: academic_tools experiment-agent / local_preexperiment.py；Status: 单次实际运行记录，尚无独立重复验证。', '', f"运行：{record['run_id']}；状态：**{record['status']}**；耗时：{record.get('elapsed_seconds',0)/60:.1f} 分钟。", '', f"GPU：{record['gpu']}。仅 CT、B0、ct_holdG；外层 G 中心未推理、未评分。", '', f"来源训练共 {len(record['split']['train'])} 例，来源开发共 {len(record['split']['val'])} 例。固定看图病例：{', '.join(record['fixed_preview_cases'])}，选择规则是每个来源中心排序后的首例，不按成绩选择。", '', '## 小样本排错', '', f"在 {', '.join(record['tiny_cases'])} 各取一个固定小批次，反复学习其中的三维块；关闭随机增强及权重衰减。这是训练块拟合检查，不是完整病例拟合，也不是泛化评价。实际完成 {record.get('tiny_completed_steps',0)} 步。", '', '| 步骤 | 同一训练块损失 | 同一训练块平均Dice |', '|---:|---:|---:|']
    for r in probes:
        lines.append(f"| {r['step']} | {r['same_training_patch_loss']:.4f} | {r['same_training_patch_mean_dice']:.4f} |")
    lines += ['', '## 初步基线', '', f"重新从随机初始化训练，没有加载拟合排错权重。实际完成 {record.get('baseline_completed_epochs',0)} 轮 / {record.get('baseline_optimizer_steps',0)} 个优化步骤；学习率保留原来250轮调度，只截取前几轮。", '', '![学习曲线](learning_curves.png)', '', '| 完成轮数 | 训练损失 | 开发小块损失 | 开发小块近似Dice |', '|---:|---:|---:|---:|']
    for r in epochs:
        lines.append(f"| {r['epoch']} | {r['training_loss']:.4f} | {r['validation_patch_loss']:.4f} | {r['validation_pseudo_dice_macro']:.4f} |")
    lines += ['', '## 固定来源开发病例的完整体积分割', '', '以下为完整网格、逐类Dice的简单平均，不是体积加权官方WHS指标。没有血管截断或后处理；本次不计算HD。近似Dice与完整病例Dice不能混用。', '', '| 轮数 | 病例 | 七类平均Dice | Myo | LA | LV | RA | RV | AO | PA |', '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in record['volume_previews']:
        vals = [r['dice'][n] for n in NAMES]
        lines.append(f"| {r['epoch']} | {r['case_id']} | {r['macro_dice']:.4f} | "+' | '.join('NA' if v is None else f'{v:.4f}' for v in vals)+' |')
    for r in record['volume_previews']:
        lines += ['', f"### {r['case_id']}：第 {r['epoch']} 轮", '', f"![分割对比]({Path(r['figure']).relative_to(out).as_posix()})"]
    lines += ['', '## 解释边界', '', '- 只检查CT及两个预先选定的来源开发病例；没有评价全部开发集完整体积，没有MRI结果，没有外层跨中心结果。', '- 未比较B1、未做多随机种子，不能据此判断增强收益或最终模型性能。', '- 图像切片按参考标注中心固定选择，图像用于定位错误；三张切片不能代表整个体积，数值使用完整体积。', '- 预实验与正式实验独立保存，不能把本次已观察的开发成绩当作独立测试结果。', '- run.json 记录代码/计划/划分哈希、设备、时间、完整参数和异常；pip_freeze.txt 记录实际环境。', '']
    if 'error' in record:
        lines += ['## 异常', '', '```text', record['error'], '```']
    (out/'预实验记录.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
