"""Read-only inspection. No ML import, installation, locks or last-job writes."""
import csv
import io
import json
import os
from pathlib import Path
import re
import math
import subprocess
import time
from server_data import read_json
from task_runtime import active_processes, clean, history, latest

STRUCTURES = {'Myo': '心肌', 'LA': '左房', 'LV': '左室', 'RA': '右房', 'RV': '右室', 'AO': '主动脉', 'PA': '肺动脉'}


def read_optional(path):
    try:
        return read_json(path)
    except (OSError, ValueError, TypeError):
        return None


def context(job):
    job = Path(job)
    plan = read_optional(job / 'plan.json')
    if plan:
        return plan, Path(plan['workspace']), read_optional(job / 'state.json') or {}
    # Legacy workspaces are intentionally read-only and need no migration.
    return {'name': job.name, 'workspace': str(job), 'legacy': True}, job, {}


def tail(path, size=24000):
    try:
        with Path(path).open('rb') as stream:
            stream.seek(max(0, Path(path).stat().st_size-size))
            return stream.read().decode('utf-8', errors='replace')
    except OSError:
        return '日志尚未生成或不可访问。'


def model_lines(work):
    lines = []
    for path in sorted((work / '07_experiments/formal').glob('*/run.json')):
        value = read_optional(path)
        if not value:
            lines.append(f'{path.parent.name}：记录不可读'); continue
        logs = tail(path.parent / 'training.log', 48000).splitlines()
        progress = [x for x in logs if re.search(r'epoch|loss|learning rate|pseudo dice', x, re.I)][-7:]
        lines.append(f"{path.parent.name}：{value.get('status', '未知')}，完成轮数 {value.get('completed_epochs', '见日志')}/{value.get('planned_epochs', '未记录')}")
        lines.extend('  '+clean(x) for x in progress)
        epoch, completed, seconds = None, None, []
        for line in logs:
            match = re.search(r'\bEpoch\s+(\d+)\b', line, re.I)
            if match:
                epoch = int(match.group(1))
            timing = re.search(r'Epoch time:\s*([0-9.]+)\s*s', line, re.I)
            if timing and epoch is not None:
                duration = float(timing.group(1))
                if math.isfinite(duration) and duration > 0:
                    completed = epoch + 1
                    seconds.append(duration)
        planned = value.get('planned_epochs')
        if seconds and isinstance(planned, int) and value.get('status') != 'completed':
            recent = seconds[-10:]
            remaining = max(0, planned-completed)*sum(recent)/len(recent)/60
            lines.append(f'  已提交轮次日志：{completed}/{planned}；按最近轮次估算剩余训练计算约{remaining:.1f}分钟（不含排队/预测）。')
        elif value.get('status') != 'completed':
            lines.append('  暂无足够轮次计时，无法估计剩余时间。')
    return lines or ['尚无正式模型训练记录；短测不计为正式训练。']


def prediction_lines(work):
    frozen = read_optional(work / '08_results/evaluation/frozen_models.json')
    if not frozen:
        return []
    lines = ['预测进度（最后保存的完整病例记录）：']
    for model in frozen.get('models', []):
        folder = work / '08_results/evaluation' / f"{model['fold']}_{model['method']}"
        completed, seconds = 0, []
        for case in model.get('targets', []):
            record = read_optional(folder / (case+'.json'))
            if record and record.get('status') == 'completed':
                completed += 1
                duration = record.get('seconds')
                if isinstance(duration, (int,float)) and not isinstance(duration,bool) and math.isfinite(duration) and duration > 0:
                    seconds.append(duration)
        total = len(model.get('targets', []))
        estimate = f'；按已完成病例估算尚需{(total-completed)*sum(seconds)/len(seconds)/60:.1f}分钟计算（不含排队）' if seconds and completed < total else ''
        lines.append(f"  {model['fold']}/{model['method']}：{completed}/{total}{estimate}")
    return lines


def result_lines(job, section='summary', detailed=False):
    plan, work, state = context(job)
    lines = [f"任务：{clean(plan.get('name', Path(job).name))}；状态：{state.get('status', '旧记录/未知')}"]
    summary = read_optional(work / '09_reports/final_evaluation/summary.json')
    if section == 'summary':
        if not summary:
            return lines + ['尚无最终评价摘要。不能据此推断模型分数为0。']
        lines.append('Dice为分割重叠度；按留出中心等权。raw主结果，lcc次要分析。')
        for label, value in summary.get('results', {}).items():
            for method, m in value.get('methods', {}).items():
                lines.append(f"{label}/{method}：Dice={m.get('center_equal_dice', '未记录')}；最差中心={m.get('worst_center_dice', '未记录')}")
                if detailed:
                    lines.append('  逐中心：' + json.dumps(m.get('per_center'), ensure_ascii=False))
                    lines.append('  HD/HD95：' + json.dumps({k: v for k, v in m.items() if 'hd' in k}, ensure_ascii=False))
            pair = value.get('paired')
            lines.append('  配对差异：' + (str(pair) if pair else '不适用/未记录'))
        lines.append('这里只读取已保存结果；没有重新计算或验证预测文件。')
    elif section == 'cases':
        path = work / '09_reports/final_evaluation' / ('class_metrics.csv' if detailed else 'case_metrics.csv')
        if not path.is_file():
            return lines + ['逐病例/结构指标尚未生成。']
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        lines += [f'已保存指标行数：{len(rows)}；按fold/method/variant区分，不能当独立患者数。']
        for row in rows:
            lines.append(' / '.join(str(row.get(k, '')) for k in ['fold','method','variant','case_id','structure']) +
                         f"：Dice={row.get('dice', row.get('macro_dice', '未记录'))}" +
                         (f"，HD95={row.get('hd95_mm', '未记录')}，状态={row.get('status', '')}" if detailed else ''))
    elif section == 'training':
        lines += [f"模型选择：{plan.get('methods', '旧任务未记录')}；范围：{plan.get('folds', '旧任务未记录')}"]
        lines += model_lines(work)
        splits = read_optional(work / '04_data/manifests/splits_v1.json')
        if splits and detailed:
            for name, split in splits.get('folds', {}).items():
                lines.append(f"{name}：训练{len(split.get('train', []))}，开发{len(split.get('val', []))}，目标{len(split.get('target', []))}个影像记录")
    elif section == 'files':
        lines += [f'工作区：{work}', f'任务日志：{Path(job) / "launcher.log"}',
                  f'结果报告：{work / "09_reports/final_evaluation"}']
        if detailed:
            lines += [f'来源：{plan.get("source", "未记录")}', f'源码快照：{plan.get("code", "旧任务未记录")}',
                      '预测原网格；HD和HD95单位为mm；空预测保留失败记录。']
    else:
        raise ValueError('未知结果部分')
    return lines


def query(kind, runtime, job=None, section='summary', detailed=False):
    if kind == 'history':
        entries = history(runtime)
        return '\n'.join(f"{i}. {clean(x['name'])} [{x['status']}] {x['directory']}" for i,x in enumerate(entries,1)) or '暂无命名任务。可用 --job 指定旧工作区只读查看。'
    if kind == 'gpu':
        try:
            return subprocess.check_output(['nvidia-smi'], text=True, timeout=15)
        except (OSError, subprocess.SubprocessError) as error:
            return f'显卡查询不可用：{error}'
    if kind == 'processes':
        if os.name != 'posix':
            return '进程查询仅在Linux服务器提供。'
        return subprocess.check_output(['ps', '-u', str(os.getuid()), '-o', 'pid,ppid,stat,etime,args'], text=True, timeout=15)
    job = Path(job) if job else latest(runtime)
    if not job:
        return '尚无任务；新建前不会安装环境或使用显卡。'
    plan, work, state = context(job)
    if kind == 'results':
        return '\n'.join(result_lines(job, section, detailed))
    if kind == 'models':
        return '\n'.join(model_lines(work) + prediction_lines(work))
    if kind == 'logs':
        candidates = [job / 'launcher.log', *sorted((job / 'logs').glob('*.log')),
                      *sorted((work / '07_experiments/formal').glob('*/training.log')),
                      *sorted((work / '08_results/evaluation/queue_logs').glob('*.log'))[-5:]]
        return '\n\n'.join(f'{p}\n{tail(p, 10000)}' for p in candidates if p.exists()) or '暂无日志。'
    lines = [f"任务：{clean(plan.get('name', job.name))}", f'目录：{job}', f'工作区：{work}']
    if state:
        status = state.get('status', '未知')
        living = active_processes(job)
        if living:
            status = '运行中（含活动子进程）'
        elif status in {'running','queued','stopping','waiting_gpu'}:
            status = '记录未结束，但未发现身份匹配进程；可能已中断'
        lines.append('状态：' + status)
        lines.extend(f"{name}：{value.get('status', '未知')} {value.get('error', '')}" for name,value in state.get('steps',{}).items())
    for path in [work / '00_admin/gpu_queue.json', work / '08_results/evaluation/status.json']:
        value = read_optional(path)
        if value:
            lines.append(f"{path.name}（最后落盘记录）：{value.get('status', '未知')}")
            lines.extend(f"  {v['fold']}/{v['method']}：{v.get('status', '未知')} {v.get('gpu_uuid','')}" for v in value.get('jobs', []))
    return '\n'.join(lines + prediction_lines(work))


def follow(kind, runtime, job=None, **kwargs):
    try:
        while True:
            print(query(kind, runtime, job, **kwargs), flush=True)
            print('Ctrl+C 返回查看入口；后台任务继续。', flush=True)
            time.sleep(1 if kind == 'gpu' else 5)
    except KeyboardInterrupt:
        return
