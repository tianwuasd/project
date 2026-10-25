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
from datetime import datetime
from server_data import read_json
from task_runtime import active_processes, clean, history, latest
from task_progress import STATES, step_lines, process_lines

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
    selection=read_optional(work/'00_admin/model_selection.json') or {}
    queue=read_optional(work/'00_admin/gpu_queue.json') or {}
    queued={(r['fold'],r['method']):r for r in queue.get('jobs',[])}
    keys=set(queued)|{(f,m) for f in selection.get('folds',[]) for m in selection.get('methods',[])}
    if keys:
        lines.append('模型 / 状态 / epoch / train loss / val loss / 学习率 / GPU UUID')
        for fold,method in sorted(keys):
            base=work/f'07_experiments/formal/{fold}_{method}_seed0'
            run=read_optional(base/'run.json') or {};record=queued.get((fold,method),{})
            status=run.get('status') or record.get('status') or '尚未开始'
            if status=='pending':status='等待GPU/前序调度'
            log=tail(base/'training.log',48000)
            def last(pattern):
                values=re.findall(pattern,log,re.I)
                return values[-1] if values else '—'
            epoch=last(r'\bEpoch\s+(\d+)\b')
            train=last(r'train[_ ]loss\s*[:=]?\s*(-?[0-9.eE+]+)')
            val=last(r'val[_ ]loss\s*[:=]?\s*(-?[0-9.eE+]+)')
            lr=last(r'(?:Current learning rate|learning rate|\blr)\s*[:=]\s*([0-9.eE+-]+)')
            lines.append(f'{fold}/{method} / {status} / {epoch}/{run.get("planned_epochs","—")} / {train} / {val} / {lr} / {record.get("gpu_uuid","—")}')
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
        # A resumed attempt must first complete its own timed epoch. Old logs still
        # display as history but cannot supply a fresh ETA.
        attempt=value.get('history',[])
        if len(attempt)>1:
            boundary=attempt[-1].get('started','')
            fresh=[];epoch=None;completed=None
            for line in logs:
                stamp=re.match(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)',line)
                if not stamp or stamp.group(1)<boundary:continue
                match=re.search(r'\bEpoch\s+(\d+)\b',line,re.I)
                if match:epoch=int(match.group(1))
                timing=re.search(r'Epoch time:\s*([0-9.]+)\s*s',line,re.I)
                if timing and epoch is not None:
                    duration=float(timing.group(1))
                    if math.isfinite(duration) and duration>0:fresh.append(duration);completed=epoch+1
            seconds=fresh
        if seconds and isinstance(planned, int) and value.get('status') == 'running':
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
    queue=read_optional(work/'08_results/evaluation/status.json') or {}
    if queue.get('phase'):
        phase='GPU逐病例推理' if queue['phase']=='gpu' else 'CPU原网格恢复/压缩/指标（GPU已释放）'
        jobs=queue.get('jobs',[]);done=sum(r.get('status')=='completed' for r in jobs)
        active=[r for r in jobs if r.get('status')=='running']
        lines.append(f'当前阶段：{phase}；队列 {done}/{len(jobs)}；活动病例 {len(active)}；状态 {queue.get("status","未知")}')
        samples=[r['seconds'] for r in jobs if r.get('status')=='completed' and isinstance(r.get('seconds'),(int,float)) and r['seconds']>0]
        if samples and active and queue.get('status') in {'running','cpu_tail'}:
            remaining=(len(jobs)-done)*sum(samples)/len(samples)/len(active)/60
            lines.append(f'  按本轮实际耗时估算本阶段剩余约 {remaining:.1f} 分钟（不含未知资源等待及后续阶段）')
        elif done<len(jobs):lines.append('  无本轮完成样本或仍在等待，暂不能估算剩余时间。')
    for model in frozen.get('models', []):
        folder = work / '08_results/evaluation' / f"{model['fold']}_{model['method']}"
        completed, seconds, gpu_done = 0, [], 0
        for case in model.get('targets', []):
            record = read_optional(folder / (case+'.json'))
            middle=read_optional(folder/'intermediate'/(case+'.json'))
            if middle and middle.get('status')=='completed':gpu_done+=1
            if record and record.get('status') == 'completed':
                completed += 1
                duration = record.get('seconds')
                if isinstance(duration, (int,float)) and not isinstance(duration,bool) and math.isfinite(duration) and duration > 0:
                    seconds.append(duration)
        total = len(model.get('targets', []))
        estimate = ''  # ETA above uses only the current phase's worker completions.
        lines.append(f"  {model['fold']}/{model['method']}：{completed}/{total}{estimate}")
        if gpu_done:lines.append(f'    已提交GPU中间结果：{gpu_done}/{total}；完成病例计数仍以CPU输出为准。')
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


def log_sources(job):
    job = Path(job)
    _, work, _ = context(job)
    candidates = [('0', '总日志 / 环境安装与调度', job/'launcher.log')]
    candidates += [('step:'+p.stem, '步骤 / '+p.stem, p) for p in sorted((job/'logs').glob('*.log'))]
    candidates += [('model:'+p.parent.name, '模型 / '+p.parent.name, p) for p in sorted((work/'07_experiments/formal').glob('*/training.log'))]
    for area, name in [('07_experiments/queue_logs','训练队列'), ('08_results/evaluation/queue_logs','预测队列')]:
        candidates += [(area+':'+p.stem, name+' / '+p.stem, p) for p in sorted((work/area).glob('*.log'))[-20:]]
    for area,name in [('00_admin/preflight','正式尺寸预检'),('00_admin/preparation_logs','预处理病例'),('09_reports','短测套件')]:
        candidates += [(area+':'+str(p.relative_to(work/area)),name+' / '+str(p.relative_to(work/area)),p) for p in sorted((work/area).rglob('*.log'))]
    candidates += [('smoke:'+str(p.relative_to(work/'07_experiments')),'完整尺寸短测 / '+p.parent.name+' / '+p.name,p) for root in sorted((work/'07_experiments').glob('smoke_*')) for p in root.glob('*.log')]
    return [(key, label, p) for key,label,p in candidates if p.is_file()]


class LogCursor:
    """Tail on first discovery; bounded incremental reads; handle rotation/truncation."""
    def __init__(self):
        self.positions = {}

    def read(self, path, limit=12000):
        try:
            with Path(path).open('rb') as stream:
                stat = os.fstat(stream.fileno())
                token = (stat.st_dev, stat.st_ino)
                old = self.positions.get(str(path))
                offset = old[1] if old and old[0] == token and stat.st_size >= old[1] else max(0,stat.st_size-limit)
                if old and stat.st_size == old[1] and stat.st_mtime_ns != old[2]:
                    offset = max(0,stat.st_size-limit)
                # On a burst, retain only a bounded tail instead of loading an entire file.
                skipped = stat.st_size-offset > limit
                stream.seek(max(offset,stat.st_size-limit))
                data = stream.read(limit)
                self.positions[str(path)] = (token, stream.tell(), stat.st_mtime_ns)
                return ('[新增日志过多，仅显示最近部分]\n' if skipped else '') + data.decode('utf-8', errors='replace')
        except OSError:
            return ''


def query(kind, runtime, job=None, section='summary', detailed=False, log_source='all', process_scope='task'):
    if kind == 'history':
        entries = history(runtime)
        return '\n'.join(f"{i}. {clean(x['name'])} [{x['status']}] {x['directory']}" for i,x in enumerate(entries,1)) or '暂无命名任务。可用 --job 指定旧工作区只读查看。'
    if kind == 'gpu':
        try:
            return subprocess.check_output(['nvidia-smi'], text=True, timeout=15)
        except (OSError, subprocess.SubprocessError) as error:
            return f'显卡查询不可用：{error}'
    if kind == 'processes' and process_scope == 'all':
        if os.name != 'posix':
            return '进程查询仅在Linux服务器提供。'
        return subprocess.check_output(['ps', '-u', str(os.getuid()), '-o', 'pid,ppid,stat,etime,pcpu,pmem,args'], text=True, timeout=15)
    job = Path(job) if job else latest(runtime)
    if not job:
        return '尚无任务；新建前不会安装环境或使用显卡。'
    plan, work, state = context(job)
    if kind == 'results':
        return '\n'.join(result_lines(job, section, detailed))
    if kind == 'models':
        return '\n'.join(model_lines(work) + prediction_lines(work))
    if kind == 'processes':
        return '\n'.join(process_lines(job))
    if kind == 'logs':
        candidates = [(key,label,p) for key,label,p in log_sources(job) if log_source in {'all',key}]
        return '\n\n'.join(f'{label}：{p}\n{tail(p, 10000)}' for _,label,p in candidates) or '所选日志尚未生成；可返回总日志查看安装与当前阶段。'
    lines = [f"任务：{clean(plan.get('name', job.name))}", f'目录：{job}', f'工作区：{work}']
    if state:
        status = state.get('status', '未知')
        living = active_processes(job)
        if living:
            status = '运行中（含活动子进程）' if status not in {'stopping','waiting_gpu'} else STATES[status]
        elif status in {'running','queued','stopping','waiting_gpu'}:
            status = '记录未结束，但未发现身份匹配进程；可能已中断'
        lines.append('状态：' + status)
        lines += step_lines(job, plan, work, state)
    for path in [work / '00_admin/gpu_queue.json', work / '08_results/evaluation/status.json']:
        value = read_optional(path)
        if value:
            queue_status = value.get('status', '未知')
            expected_step = 'train' if path.name == 'gpu_queue.json' else 'predict'
            try:
                step_started = datetime.fromisoformat(state['steps'][expected_step]['started']).timestamp()
                current_queue = state.get('current_step') == expected_step and path.stat().st_mtime >= step_started
            except (KeyError, ValueError, TypeError, OSError):
                current_queue = False
            label = '当前阶段最后落盘记录' if current_queue else '历史阶段记录，非本轮实时进度'
            lines.append(f"{path.name}（{label}）：{STATES.get(queue_status, queue_status)}")
            lines.extend(f"  {v['fold']}/{v['method']}：{v.get('status', '未知')} {v.get('gpu_uuid','')}" for v in value.get('jobs', []))
    if state.get('current_step') == 'train':
        lines += model_lines(work)
    return '\n'.join(lines + prediction_lines(work))


def follow(kind, runtime, job=None, **kwargs):
    cursor, last_summary = LogCursor(), 0
    try:
        while True:
            if kind == 'logs':
                current = Path(job) if job else latest(runtime)
                if time.monotonic()-last_summary >= 10:
                    print(query('status', runtime, current), flush=True)
                    last_summary = time.monotonic()
                if current:
                    for key,label,path in log_sources(current):
                        if kwargs.get('log_source','all') in {'all',key}:
                            text = cursor.read(path)
                            if text:
                                print(f'\n[{label}] {path}\n{text}', flush=True)
            else:
                print(query(kind, runtime, job, **kwargs), flush=True)
            print('Ctrl+C 返回查看入口；后台任务继续。', flush=True)
            time.sleep(1 if kind == 'gpu' else 5)
    except KeyboardInterrupt:
        return
