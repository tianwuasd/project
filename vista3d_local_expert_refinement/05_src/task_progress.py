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


def step_lines(job, plan, work, state):
    steps = state.get('steps', {})
    done = sum(v.get('status') == 'completed' for v in steps.values())
    lines = [f'步骤完成：{done}/{len(steps)}（按步骤数，非总计算量）；本次运行耗时：{duration(state.get("started"), state.get("ended"))}']
    if state.get('current_step') == 'input':
        lines.append('当前前置：校验并复制所选历史输入/缓存；大文件复制期间尚无计算百分比。')
    budget=optional(Path(job)/'resource-plan.json')
    if budget:
        lines.append(f'本轮CPU共享预算：有效核数 {budget.get("cpu_slots")}，并行上限 {budget.get("parallel_workers")}，每worker线程 {budget.get("threads_per_worker")}；CPU尾部 {budget.get("cpu_tail_workers")} 进程。')
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
