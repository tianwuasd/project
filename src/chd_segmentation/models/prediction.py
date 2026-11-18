from copy import deepcopy
from pathlib import Path
import os
from time import perf_counter

import nibabel as nib
import numpy as np
import torch
import yaml

from ..artifacts.store import digest, read_json, write_json
from ..datasets.splits import select_test
from ..observability.events import event
from ..prediction.dice import supervised_dice
from ..prediction.metrics import case_data_identity
from ..prediction.restore import restore
from ..preprocessing.prepare import load_prepared


def prediction_rows(config, prepared, case_ids, test_set, test_group):
    if test_group is not None and test_set is None:
        raise ValueError('测试子组需要同时指定内部或外部测试入口')
    if test_set is not None:
        rows = select_test(prepared['manifest'], test_set, test_group, case_ids)['cases']
        if config['model'] == 'q_route1' and not config['technical']:
            groups = yaml.safe_load(Path(config['groups']).read_text(encoding='utf-8'))
            if not {row['case_id'] for row in rows} <= set(groups['D']):
                raise ValueError('Q测试病例必须属于D组')
    else:
        rows = prepared['manifest']['cases']
        if case_ids is not None:
            if not case_ids or len(set(case_ids)) != len(case_ids) or not set(case_ids) <= {r['case_id'] for r in rows}:
                raise ValueError('预测病例为空、重复或未登记')
            rows = [row for row in rows if row['case_id'] in case_ids]
    if not rows:
        raise ValueError('没有选中的预测病例')
    return rows


def predict(config, prepared_path, checkpoint, output, case_ids=None, prompts=None,
            prompt_mode='external', label_classes=None, box_seed=2024, box_max_shift=20,
            official_weights=False, test_set=None, test_group=None):
    from .execution import admitted, run_workers
    from .offline import model_storage_lease
    prepared = load_prepared(prepared_path)
    if prompt_mode not in ('external', 'label-box') or prompt_mode == 'label-box' and config['model'] != 'medsam2':
        raise ValueError('标签框模式仅适用于MedSAM2')
    if official_weights and config['model'] != 'medsam2':
        raise ValueError('直接发布权重入口仅适用于MedSAM2')
    if prompt_mode == 'label-box' and prompts is not None or prompt_mode == 'external' and label_classes is not None:
        raise ValueError('标签框与外部提示参数不能混用')
    if config['technical'] != (prepared['scope'] == 'technical'):
        raise ValueError('模型配置与准备描述的正式或技术范围不同')
    rows = prediction_rows(config, prepared, case_ids, test_set, test_group)
    names = [row['case_id'] for row in rows]
    if config.get('input_source',{}).get('path'):
        from ..data.input_binding import bind_prepared_source
        prepared = bind_prepared_source(prepared,config['input_source']['path'],case_ids=names)
    if config['model'] == 'medsam2' and prompt_mode == 'external' and (
            not prompts or any(not prompts.get(name) for name in names)):
        raise ValueError('MedSAM2普通预测需要每例的人工或图像定位器提示')
    protocol = dict(prompt_mode=prompt_mode, label_classes=label_classes, box_seed=box_seed,
                    box_max_shift=box_max_shift, official_weights=official_weights,
                    test_set=test_set, test_group=test_group)
    applied = deepcopy(config)
    applied['execution'].update(gpu_count=1, min_gpus=None, cpu_processes=1, batch_layout=None, workers=0)
    if applied['execution'].get('gpu_ids'):
        applied['execution']['gpu_ids'] = applied['execution']['gpu_ids'][:1]
    key = digest(dict(checkpoint=str(Path(checkpoint).resolve()), prepared_path=str(Path(prepared_path).resolve()),
        output=str(Path(output).resolve()), names=names, prompts=prompts, protocol=protocol))[:16]
    label = ('ImageCHD' if prepared['dataset'] == 'imagechd' else 'CHD1282')+'-'+('Q' if config['model'] == 'q_route1' else 'MedSAM2')
    applied['run_dir'] = str(Path(config['work_root'] or Path(config['run_dir']).parent)/'runs'/label/'predictions'/key)
    budget = deepcopy(prepared)
    budget['manifest']['cases'] = rows
    with model_storage_lease(applied,budget,'prediction'), admitted(applied, budget, 'prediction') as (common, allocation):
        return run_workers(dict(operation='predict', config=applied, prepared=prepared, stage='prediction',
            prepared_path=str(Path(prepared_path).resolve()), checkpoint=str(Path(checkpoint).resolve()),
            output=str(Path(output).resolve()), case_ids=names, prompts=prompts,
            prediction_options=protocol), common, allocation)


def _predict(config, prepared_path, checkpoint, output, case_ids=None, prompts=None,
             prompt_mode='external', label_classes=None, box_seed=2024, box_max_shift=20,
             official_weights=False, test_set=None, test_group=None, *, memory):
    from .workflow import resources,load_state,save_state,adapter,dataset_for,selected_prepared
    started = perf_counter()
    resources(config)
    prepared = load_prepared(prepared_path)
    selected = prediction_rows(config,prepared,case_ids,test_set,test_group)
    if config.get('input_source',{}).get('path'):
        from ..data.input_binding import bind_prepared_source
        prepared = bind_prepared_source(prepared,config['input_source']['path'],
            case_ids=[row['case_id'] for row in selected])
    root = Path(output).resolve()
    if root.is_relative_to(Path(prepared['manifest']['data_root']).resolve()):
        raise ValueError('预测输出不能位于原始数据目录内')
    root.mkdir(parents=True, exist_ok=True)
    from ..prediction.log_cursor import start_event_range,finish_event_range
    log_dir = Path(config['run_dir'])/'prediction'
    event_cursor = start_event_range(log_dir)
    run_path = root/'prediction-run.json'
    task = dict(source={key:value for key,value in prepared.get('storage',{}).items() if key!='cases'}, source_plan=prepared['plan'],
        available_cases=sorted(row['case_id'] for row in prepared['manifest']['cases']),
        selection_source=str(Path(checkpoint).resolve()), prompts=prompts,
        protocol=dict(prompt_mode=prompt_mode, label_classes=label_classes, box_seed=box_seed,
                      box_max_shift=box_max_shift, test_set=test_set, test_group=test_group))
    legacy_source_cases = {}
    if run_path.exists():
        run = read_json(run_path)
        if run['model'] != config['model'] or run['dataset'] != prepared['dataset']:
            raise ValueError('预测运行的数据集或模型不同，请使用新的输出目录')
        recorded_task = deepcopy(run.get('task'))
        if not isinstance(recorded_task,dict):
            raise ValueError('已有预测缺少实际来源记录')
        if isinstance(recorded_task.get('source'),dict):
            legacy_source_cases = recorded_task['source'].pop('cases',{})
        if recorded_task != task:
            raise ValueError('预测来源、几何、提示或模型选择不同，请使用新的输出目录')
        checkpoint = run['checkpoint']
        official_weights = run['official_weights']
    rows = prediction_rows(config, prepared, case_ids, test_set, test_group)
    progress_path = root/'completed-cases.json'
    completed = read_json(progress_path)['cases'] if progress_path.exists() else {}
    if not completed and (root/'summary.json').exists():
        completed = {item['case_id']:item for item in read_json(root/'summary.json')['cases']}
    metadata_only = run_path.exists() and (root/'summary.json').exists() and all(
        row['case_id'] in completed and Path(completed[row['case_id']]['file']).is_file() for row in rows)
    bundle = None
    if metadata_only:
        if not Path(checkpoint).is_file():
            raise FileNotFoundError('预测固定模型文件不存在')
        initialization = read_json(root/'summary.json')['initialization']
    else:
        state = load_state(checkpoint)
        if official_weights:
            bundle = adapter(config,prepared,'sequence',initialize=False)
            initialization = dict(kind='official_pretrained',path=str(checkpoint),optimizer_updates=0)
        else:
            if state['config']['model']!=config['model'] or state['scope']!=prepared['scope']:
                raise ValueError('预测模型或范围与检查点不同')
            trained = state['prepared']
            if prepared['dataset']!=trained['dataset'] or prepared['dataset_json']!=trained['dataset_json']:
                raise ValueError('预测输入的数据集语义与模型不同')
            model_config = dict(config,patch=state['config']['patch'],local_size=state['config']['local_size'],
                parents=state['config']['parents'])
            bundle = adapter(model_config,trained,state['stage'],initialize=False,parent_states=state.get('parent_models'))
            initialization = dict(kind='trained_checkpoint',identity=state['identity'])
        bundle.model.load_state_dict(state['model'],strict=True)
        if not run_path.exists():
            snapshot = root/'selected-model.pt'
            snapshot_state = {key:state[key] for key in (
                'model','config','prepared','stage','scope','identity','initialization_sha256','spec') if key in state}
            if config['model']=='q_route1':
                snapshot_state['parent_models'] = bundle.parent_models
            save_state(snapshot,snapshot_state)
            write_json(run_path,dict(model=config['model'],dataset=prepared['dataset'],checkpoint=str(snapshot),
                source_checkpoint=str(checkpoint),official_weights=official_weights,task=task,initialization=initialization))
        del state
        bundle.model.to(config['device']).eval()
    dataset_prepared = selected_prepared(prepared,[row['case_id'] for row in rows])
    label_mode = prompt_mode == 'label-box'
    dataset = None
    if not metadata_only and config['model'] == 'medsam2':
        from .cases import ModelCases
        dataset = ModelCases(dataset_prepared,config,with_label=label_mode)
    elif not metadata_only:
        dataset = dataset_for(dataset_prepared,False,dict(reserve_ram_bytes=config['execution']['reserve_ram_bytes'],
            cache_bytes=config['execution']['cache_bytes']),Path(config['run_dir'])/'prediction',config['logging'])
    results, reused = [], 0
    classes = len(prepared['dataset_json']['labels'])
    from ..observability.progress import Progress
    feedback = Progress(memory.config, 'predict', total=len(rows), unit='cases')
    for row in rows:
        mark = perf_counter()
        name = row['case_id']
        if name in completed and Path(completed[name]['file']).is_file():
            previous_source = completed[name].get('source_case',legacy_source_cases.get(name))
            if previous_source != prepared['storage']['cases'][name]:
                raise ValueError('已完成病例对应的实际预处理来源不同，请使用新的预测目录')
            reused += 1
            results.append(dict(completed[name],reused=True))
            event(log_dir, 'model_case_reused', case_id=name, file=completed[name]['file'])
            feedback.update(done=len(results),new=len(results)-reused,reused=reused,
                current_case=name,pending=len(rows)-len(results),active=0)
            feedback.snapshot(force=True)
            continue
        event(log_dir, 'model_case_started', case_id=name, model=config['model'], prompt_mode=prompt_mode)
        feedback.update(done=len(results),new=len(results)-reused,reused=reused,
            current_case=name,pending=len(rows)-len(results),active=1)
        feedback.snapshot()
        with memory.admit(0, label='prediction_case:'+name, release=dataset.clear) as ticket:
            if ticket is None:
                raise RuntimeError('预测在明确取消或停止后结束，已有输出保持原位置')
        entry = dataset.get(name)
        case_prompts, protocol = prompts, None
        truth = entry['label'] if label_mode else None
        if label_mode:
            from .medsam2.prompts import label_box_prompts
            case_prompts, protocol = label_box_prompts(entry, row.get('sha256', {}).get('label'), classes,
                label_classes, box_seed, box_max_shift)
            event(log_dir, 'label_box_generated', case_id=name, protocol=protocol)
        # 传播模型只接收图像与显式提示；参考标签只用于生成框和计算指标。
        inference_entry = dict(entry, label=None)
        inference_started = perf_counter()
        with torch.inference_mode():
            if config['model'] == 'medsam2':
                probability, diagnostics = bundle.predict(inference_entry, case_prompts, label_box=label_mode)
            else:
                probability, diagnostics = bundle.predict(inference_entry, case_prompts)
        inference_seconds = perf_counter()-inference_started
        if probability.shape!=(classes,*entry['image'].shape[1:]) or not np.isfinite(probability).all():
            raise ValueError('模型概率的类别、网格或有限性无效')
        metrics = supervised_dice(probability.argmax(0), truth, classes, entry['unknown_classes'],
            protocol['prompted_classes']) if label_mode else None
        from ..prediction.geometry import restore_source
        native = restore_source(probability, entry)
        path = root/(name+'.nii.gz')
        if path.exists() and name not in completed:
            raise FileExistsError('预测文件没有完成记录：'+str(path))
        image = nib.Nifti1Image(native, np.asarray(entry['source_affine']))
        image.header.set_xyzt_units(entry['source_units'])
        pending = root/(name+'.pending.nii.gz')
        nib.save(image, pending)
        os.replace(pending, path)
        results.append(dict(case_id=name, file=str(path), diagnostics=diagnostics,
            reused=False,
            source_case=prepared['storage']['cases'][name],
            seconds=perf_counter()-mark, inference_seconds=inference_seconds, shape=list(native.shape),
            timing=entry['timing'], roi_xyz=entry['roi_xyz'], prompt_protocol=protocol, classes=metrics,
            test_set=row.get('test_set'), test_group=row.get('test_group')))
        completed[name] = results[-1]
        write_json(progress_path, dict(cases=completed))
        event(log_dir, 'model_case_completed', case_id=name, seconds=perf_counter()-mark,
              inference_seconds=inference_seconds, shape=list(native.shape), classes=metrics)
        dataset.clear()
        del probability, native, image, entry, inference_entry, truth
        feedback.update(done=len(results),new=len(results)-reused,reused=reused,
            pending=len(rows)-len(results),active=0)
        feedback.snapshot(force=True)
    feedback.close()
    summary = dict(model=config['model'], scope=prepared['scope'], checkpoint=str(checkpoint),
        initialization=initialization, prompt_mode=prompt_mode, test_set=test_set, test_group=test_group,
        cases=results,new=len(results)-reused,reused=reused,model_compute_cases=len(results)-reused,
        seconds=perf_counter()-started)
    if label_mode:
        aggregates = {}
        for class_id in range(1, classes):
            values = [r['classes'][str(class_id)]['dice'] for r in results if str(class_id) in r['classes']]
            if values:
                aggregates[str(class_id)] = dict(cases=len(values), mean_dice=float(np.mean(values)))
        evaluation = dict(scope=prepared['scope'], metric='prompted_known_present_class_dice',
            interpretation='参考标签框提示下的分割结果；不代表自动定位效果',
            case_classes={r['case_id']: r['classes'] for r in results}, aggregate=aggregates)
        write_json(root/'evaluation.json', evaluation)
        summary['evaluation'] = evaluation
    else:
        by_id = {row['case_id']:row for row in rows}
        records = {result['case_id']:dict(file=Path(result['file']).name,
            checkpoint=summary['checkpoint'],scope=prepared['scope'],roi_xyz=result['roi_xyz'],
            data_identity=case_data_identity(prepared,by_id[result['case_id']]),
            reference_sha256=by_id[result['case_id']].get('sha256', {}).get('label')) for result in results}
        write_json(root/'predictions.json',dict(model=config['model'],prompt_mode='external',cases=records))
    event(log_dir,'model_prediction_completed',model=config['model'],cases=len(results),new=len(results)-reused,reused=reused)
    summary.update(finish_event_range(event_cursor))
    write_json(root/'summary.json', summary)
    return summary
