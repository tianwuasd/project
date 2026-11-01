"""Read-only progress from committed metadata and verified Linux processes."""
from datetime import datetime, timezone
import os
from pathlib import Path

from server_data import read_json
from task_runtime import TASKS, active_processes, clean

STATES = {'pending':'尚未执行', 'queued':'已提交', 'running':'运行中', 'completed':'已完成',
          'failed':'失败', 'interrupted':'已中断', 'stopping':'正在退出', 'waiting_gpu':'等待显卡',
          'stale':'进程已退出，记录未结束', 'draining':'停止派发，等待已运行模型收尾'}
INSTALL_PHASES = {'starting':'准备环境', 'download':'下载安装包', 'checksum':'校验安装包',
                  'miniforge':'安装 Miniforge', 'conda':'复用 Conda', 'python_env':'独立 Python 环境',
                  'pytorch':'安装 PyTorch', 'nnunet':'安装 nnU-Net', 'verify':'依赖版本核验', 'complete':'环境检查完成'}


def optional(path):
    try:
        return read_json(path)
    except (OSError, ValueError, TypeError):
        return {}


def duration(start, end=None):
    try:
        first = datetime.fromisoformat(start)
        last = datetime.fromisoformat(end) if end else datetime.now(timezone.utc)
        seconds = max(0, int((last-first).total_seconds()))
        return f'{seconds//3600}时{seconds%3600//60:02d}分{seconds%60:02d}秒'
    except (TypeError, ValueError):
        return '未记录'


def preparation_lines(work, folds=None):
    splits = optional(work / '04_data/manifests/splits_v1.json').get('folds', {})
    lines = []
    for fold in folds or splits:
        split = splits.get(fold)
        if not split:
            continue
        expected = set(split.get('train', []) + split.get('val', []))
        completed = set()
        for path in (work / '04_data/derived/nnUNet_preprocessed').glob(f'*_{fold}/server_case_completion/*/*.complete.json'):
            if optional(path).get('plan_sha256'):
                completed.add(path.name.removesuffix('.complete.json'))
        lines.append(f'预处理 {fold}：已保存完成病例 {len(expected & completed)}/{len(expected)}（来源 fit/cal，不含目标评价）；导出/指纹阶段没有病例百分比。')
    return lines


def resource_lines(work,state):
    """Current phase settings and actual observations are separate kinds of evidence."""
    work=Path(work);phase=state.get('current_step');lines=[];decision={};source=None;effective={}
    paths=[]
    if phase in {'train','smoke','assessment'}:
        kind='assessment' if phase=='assessment' else 'formal'
        paths=list((work/'00_admin/stage-decisions').glob(kind+'_*.json'))
    if phase=='assessment':
        cpu_paths=[p for p in [work/'00_admin/cpu-case-budget.json',work/'00_admin/prediction-host-budget.json'] if p.exists()]
        if cpu_paths:
            current=max(cpu_paths,key=lambda p:p.stat().st_mtime_ns)
            if not paths or current.stat().st_mtime_ns>max(p.stat().st_mtime_ns for p in paths):
                decision=optional(current);source=current;paths=[]
    if paths:
        source=max(paths,key=lambda p:p.stat().st_mtime_ns);decision=optional(source)
        effective=decision.get('effective_cpu',{});budget=decision.get('resource_snapshot',{})
        # A feedback row is current only if created after this capacity decision.
        feedback=[]
        for path in (work/'07_experiments').rglob('feedback.rank*.json'):
            if path.stat().st_mtime_ns>=source.stat().st_mtime_ns:
                events=optional(path)
                if isinstance(events,list):
                    feedback.extend((path.stat().st_mtime_ns,event) for event in events)
                elif isinstance(events,dict) and events.get('schema_version')==3:
                    feedback.append((path.stat().st_mtime_ns,events))
        for _,event in sorted(feedback,key=lambda item:item[0]):
            item=event.get('decision',{})
            if event.get('applied_settings'):effective=event['applied_settings']
            elif item.get('applied_boundary')=='replay_safe_batch':effective=item.get('settings',effective)
        if feedback:
            event=sorted(feedback,key=lambda item:item[0])[-1][1];window=event.get('window',{})
            lines.append('供数观察：最高等待比例 '+str(window.get('wait_fraction','缺失'))+'；逐rank '+str(window.get('rank_windows',[])))
        batch=decision.get('global_batch_size');reference=decision.get('reference_global_batch_size')
        if batch and reference:
            formal=decision.get('formal_budget',{})
            train=formal.get('train_iterations',250)
            validation=formal.get('validation_iterations',50)
            epochs=formal.get('epochs','未知；历史记录缺少预算')
            lines.append(f'采样：训练 {batch*train}/{reference*train}，验证 {batch*validation}/{reference*validation} patches/轮（当前/参考）；{epochs}轮×{train}训练/{validation}验证。')
    else:
        if phase=='predict':
            queue=optional(work/'08_results/evaluation/status.json');decision=queue.get('host_budget',{})
        if not decision and phase in {'prepare','predict','assessment'}:decision=optional(work/'00_admin/cpu-case-budget.json')
        effective=decision;budget=decision
        source=source or work/'00_admin/cpu-case-budget.json'
    if not decision:return ['当前阶段资源决策尚未记录；启动预算不作为实际使用量。']
    assigned=effective.get('total_cpu_slots','缺失');cpu=budget.get('cpu_budget','缺失')
    try:started=datetime.fromisoformat(state.get('steps',{}).get(phase,{}).get('started',state.get('started'))).timestamp()
    except (TypeError,ValueError):started=0
    observed=decision.get('observed_at') or (source.stat().st_mtime if source and source.exists() else 0)
    label='当前阶段记录' if state.get('status')=='running' and observed>=started else '历史阶段记录（非当前资源）'
    lines.append(f"{label}：CPU获准 {budget.get('cpu_permitted','缺失')} / 当前可用 {budget.get('cpu_available','缺失')} / 80%预算 {cpu}，已分配 {assigned}；实际CPU用量缺少所属PID时间窗测量。")
    lines.append('阶段：'+str(decision.get('stage',phase)))
    lines.append('RAM预算 '+str(budget.get('memory_budget_bytes','缺失'))+' bytes；实际峰值请见病例/rank观察，未观测项不能视作0。')
    if effective.get('rank_settings'):lines.append('当前已应用供数配置：'+str(effective['rank_settings']))
    else:lines.append('病例并行 '+str(effective.get('workers','缺失'))+'；每病例支持的叶线程 '+str(effective.get('threads_per_worker','缺失')))
    lines.append('原因：'+str(effective.get('reasons',[]))+'；缺失证据：'+str(budget.get('missing_evidence',[])))
    return lines


def step_lines(job, plan, work, state):
    steps = state.get('steps', {})
    done = sum(v.get('status') == 'completed' for v in steps.values())
    lines = [f'步骤完成：{done}/{len(steps)}（按步骤数，非总计算量）；本次运行耗时：{duration(state.get("started"), state.get("ended"))}']
    if state.get('current_step') == 'input':
        lines.append('当前前置：校验并复制所选历史输入/缓存；大文件复制期间尚无计算百分比。')
    lines += resource_lines(work,state)
    software=optional(Path(job)/'software-smoke.json')
    if software:lines.append('小尺寸CPU软件检查：'+software.get('status','未知')+'；不代表正式尺寸GPU预检通过。')
    if state.get('current_step') in {'smoke','train'}:
        for path in sorted((work/'00_admin/preflight').glob('*.state.json')):
            item=optional(path)
            lines.append(f'正式尺寸预检 {item.get("fold")}/{item.get("method")}：{item.get("status")}；GPU {item.get("gpu")}；日志 {item.get("log")}')
    if state.get('current_step') == 'prepare':
        for fold in plan.get('folds', []):
            value = optional(work / f'00_admin/preparation_progress/{fold}.json')
            if value:
                updated = value.get('updated')
                stamp = datetime.fromtimestamp(updated, timezone.utc).strftime('%m-%d %H:%M:%S UTC') if isinstance(updated, (float,int)) else '未记录'
                lines.append(f'预处理当前阶段 {fold}：{clean(value.get("phase", "未记录"))}；更新时间 {stamp}')
    install = optional(Path(job) / 'environment.json')
    if install:
        phase = INSTALL_PHASES.get(install.get('phase'), install.get('phase', '未记录'))
        lines.append(f'环境安装：{STATES.get(install.get("status"), install.get("status", "未知"))} / {phase} / {clean(install.get("message", ""))}')
        if install.get('downloaded_bytes') is not None:
            size, total = install['downloaded_bytes'], install.get('total_bytes')
            lines.append(f'  安装包下载：{size/2**20:.1f} MiB' + (f' / {total/2**20:.1f} MiB' if total else ' / 总大小未知'))
        if install.get('error'):
            lines.append('  错误：' + clean(install['error']))
        lines.append(f'  阶段更新时间：{install.get("updated", "未记录")}；详情：{Path(job)/"launcher.log"}')
    elif state.get('current_step') == 'installation':
        lines.append('环境安装：正在准备独立安装环境；下载/依赖安装尚未生成明细。')
    for number, (name, value) in enumerate(steps.items(), 1):
        status = value.get('status', 'unknown')
        lines.append(f'{number}. {TASKS.get(name, name)}：{STATES.get(status, status)}' +
                     (f'；耗时 {duration(value["started"], value.get("ended") or (state.get("ended") if status in {"failed","interrupted"} else None))}' if value.get('started') else '') +
                     (f'；{clean(value["error"])}' if value.get('error') else ''))
    if state.get('error'):
        lines.append('任务错误：' + clean(state['error']))
    lines += preparation_lines(work, plan.get('folds'))
    return lines


def process_lines(job):
    if os.name != 'posix' or not Path('/proc').is_dir():
        return ['本任务进程树需在 Linux 服务器查看；本机仅展示已保存记录。']
    roots = {r['pid']:r for r in active_processes(job)}
    if not roots:
        return ['没有身份匹配的活动任务进程。旧 PID 记录不代表仍在运行。']
    rows = {}
    ticks = os.sysconf('SC_CLK_TCK')
    pages = os.sysconf('SC_PAGE_SIZE')
    uptime = float(Path('/proc/uptime').read_text().split()[0])
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            stat = (path/'stat').read_text().rsplit(')', 1)[1].split()
            with (path/'cmdline').open('rb') as stream:
                command = stream.read(600).replace(b'\0', b' ').decode('utf-8', errors='replace')
            rows[int(path.name)] = dict(ppid=int(stat[1]), start=stat[19], status=stat[0], cpu=(int(stat[11])+int(stat[12]))/ticks,
                                       rss=int(stat[21])*pages/2**20, elapsed=max(0,int(uptime-int(stat[19])/ticks)), command=command)
        except (OSError, ValueError, IndexError):
            continue
    selected = {pid for pid,r in roots.items() if pid in rows and rows[pid]['start'] == r.get('identity',{}).get('start')}
    if not selected:
        return ['核对期间任务进程已退出；请刷新，旧 PID 不作为活动进程。']
    while True:
        added = {pid for pid,r in rows.items() if r['ppid'] in selected} - selected
        if not added:
            break
        selected.update(added)
    lines = ['本任务进程树（本用户，boot/start 身份核验根进程及其后代）：',
             'PID / PPID / 状态 / 已运行秒 / CPU累计秒 / 内存MiB / 角色与命令']
    for pid in sorted(selected):
        r = rows[pid]
        role = roots.get(pid, {}).get('role', '子进程')
        lines.append(f'{pid} / {r["ppid"]} / {r["status"]} / {r["elapsed"]} / {r["cpu"]:.1f} / {r["rss"]:.1f} / {role} {clean(r["command"])}')
    return lines
