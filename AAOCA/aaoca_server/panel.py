"""Chinese terminal workbench, following the CHD panel's task-first structure."""
import argparse
from datetime import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

from .state import Registry, dispatch_lock, job_status, now, read, validate_paths, write


DEFAULTS = {'input': '', 'results': '', 'manifest': '', 'scratch': '', 'spacing': 'auto',
            'cpu_limit': None, 'memory_budget_gib': None, 'workers': None,
            'compute_threads': None, 'read_workers': None, 'max_case_gib': None}
RESOURCE_FIELDS = ('cpu_limit', 'memory_budget_gib', 'workers', 'compute_threads', 'read_workers', 'max_case_gib')
STATES = {'not_submitted': '未提交', 'submitted': '已提交', 'checking': '检查中', 'running': '运行中',
          'completed': '已完成', 'completed_with_errors': '部分失败', 'stopped': '已停止',
          'failed': '失败', 'interrupted': '进程已退出，需查看日志'}


class Back(Exception):
    pass


class Exit(Exception):
    pass


def parse_setting(key, value):
    value = value.strip()
    if key in {'input', 'results', 'manifest', 'scratch'}:
        return str(Path(value).expanduser().resolve()) if value else ''
    if key == 'spacing':
        if value.lower() == 'auto':
            return 'auto'
        values = [float(v) for v in value.replace(',', ' ').split()]
        if len(values) != 3 or not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError('间距需为 auto 或三个正数，单位 mm，按 R A S 轴顺序')
        return values
    if value.lower() == 'auto':
        return None
    result = float(value) if key in {'memory_budget_gib', 'max_case_gib'} else int(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError('请输入正数或 auto')
    if key in {'workers', 'compute_threads', 'read_workers'} and result > 32:
        raise ValueError('病例/计算/读取并发数上限为 32')
    return result


def prepare(settings):
    # Load numerical dependencies only when the user asks for a header plan.
    from aaoca_preprocessing.pipeline import suggest_spacing
    from aaoca_preprocessing.autoresources import select_resources
    manifest = settings['manifest'] or None
    spacing = (suggest_spacing(settings['input'], manifest) if settings['spacing'] == 'auto'
               else {'spacing_mm': settings['spacing'], 'method': 'manual'})
    request = {key: settings[key] for key in RESOURCE_FIELDS}
    config, resources, planned = select_resources(settings['input'], spacing['spacing_mm'], request, manifest)
    return config, spacing, resources, planned


class Panel:
    def __init__(self, runtime, overrides=None, ask=input, emit=print):
        self.registry = Registry(runtime)
        self.settings = {**DEFAULTS, **read(self.registry.root / 'settings.json', {}), **(overrides or {})}
        self.ask, self.emit = ask, emit
        entries = self.registry.entries()
        self.selected = entries[0] if entries else None

    def prompt(self, message):
        answer = self.ask(message).strip()
        if answer.lower() in {'q', '0'}:
            raise Exit()
        if answer.lower() == 'b':
            raise Back()
        return answer

    def current(self):
        if not self.selected:
            raise ValueError('暂无任务，请用 n 新建任务或 h 查看历史')
        return self.selected

    def show_status(self):
        entry = self.current()
        status = job_status(entry['output'])
        self.emit(f"任务：{entry['name']} | {STATES.get(status['state'], status['state'])}")
        self.emit(f"输出：{entry['output']}\n后台进程：{status['alive']}；停止请求：{status['stop_requested']}")
        progress = status['progress']
        self.emit('病例：' + ' / '.join(f'{k}={progress.get(k, 0)}' for k in ['total', 'completed', 'reused', 'failed', 'pending']))
        if status['worker'].get('error'):
            self.emit('原因：' + status['worker']['error'])
        for case in progress.get('active', {}).values():
            self.emit('当前阶段：' + json.dumps(case, ensure_ascii=False))
        resources = status['resources'].get('selected')
        if resources:
            self.emit(f"本次实际选择：CPU {resources['cpu_limit']}，病例并发 {resources['workers']}，"
                      f"计算线程/病例 {resources['compute_threads']}，内存 {resources['memory_budget_gib']:.2f} GiB")
        for item in progress.get('errors', [])[:5]:
            self.emit('失败：' + str(item.get('error', item)))

    def history(self):
        entries = self.registry.entries()
        if not entries:
            self.emit('暂无历史任务'); return
        for i, entry in enumerate(entries, 1):
            state = job_status(entry['output'])['state']
            self.emit(f"{i}. {entry['name']} | {STATES.get(state, state)} | {entry['created_at']}")
        answer = self.prompt('选择任务编号，b 返回：')
        if not answer:
            return
        index = int(answer) - 1
        if not 0 <= index < len(entries):
            raise ValueError('编号不在列表中')
        self.selected = entries[index]
        self.show_status()

    def logs(self, follow=False):
        status = job_status(self.current()['output'])
        path = Path(status['runtime']) / 'worker.log'
        if not path.exists():
            self.emit('尚无日志'); return
        with path.open('rb') as handle:
            handle.seek(0, 2)
            length = handle.tell()
            handle.seek(max(0, length - 65536))
            if length > 65536:
                handle.readline()
            self.emit(handle.read().decode('utf-8', errors='replace'))
            if follow:
                self.emit('实时日志：Ctrl+C 返回面板，后台任务继续运行。')
                try:
                    while job_status(self.current()['output'])['alive'] is True:
                        time.sleep(.5)
                        data = handle.read()
                        if data:
                            self.emit(data.decode('utf-8', errors='replace'))
                    data = handle.read()
                    if data:
                        self.emit(data.decode('utf-8', errors='replace'))
                except KeyboardInterrupt:
                    self.emit('已返回面板。')

    def show_plan(self, settings):
        validate_paths(settings, self.registry.root)
        config, spacing, resources, planned = prepare(settings)
        self.emit('预处理计划（仅读取头信息，不处理像素）：')
        self.emit('目标间距 R/A/S：' + ' × '.join(f'{v:.6g}' for v in spacing['spacing_mm']) + ' mm')
        self.emit(f"病例 {planned['total']}，头信息无效 {planned['invalid']}，超出内存估算 {resources['oversized_cases']}")
        self.emit(f'CPU 预算 {config.cpu_limit}，病例并发 {config.workers}，计算线程/病例 {config.compute_threads}，读取并发 {config.read_workers}')
        self.emit(f'内存预算 {config.memory_budget_gib:.2f} GiB，单病例上限 {config.max_case_gib:.2f} GiB')
        self.emit(f"未压缩输出估算 {planned['output_bytes_uncompressed'] / 2**30:.2f} GiB；完整视野，HU [-1000,1000] → [-1,1]")
        self.emit('资源在后台启动时复核；间距随任务固定。当前负载只作短时估算。')
        return config, spacing, resources, planned

    def configure(self, new_name=None):
        draft = dict(self.settings)
        fields = [('input', 'HU NIfTI 输入目录'), ('results', '结果根目录'), ('spacing', '目标间距 mm'),
                  ('cpu_limit', 'CPU 总预算'), ('memory_budget_gib', '内存预算 GiB'),
                  ('manifest', '输入清单（可留空）'), ('scratch', '临时根目录（空=结果内临时区）'),
                  ('workers', '病例并发'), ('compute_threads', '每病例计算线程'),
                  ('read_workers', '读取并发'), ('max_case_gib', '单病例内存上限 GiB')]
        while True:
            self.emit('\n预处理配置' + (f' — {new_name}' if new_name else ''))
            for index, (key, label) in enumerate(fields, 1):
                value = draft[key]
                self.emit(f'{index:2}. {label}：{value if value is not None else "auto"}')
            self.emit('p 预览计划  w 保存配置' + ('  s 启动本任务' if new_name else '') + '  b 返回  q 退出')
            action = self.prompt('配置操作：').lower()
            try:
                if action == 'p':
                    self.show_plan(draft)
                elif action == 'w':
                    validate_paths(draft, self.registry.root)
                    with dispatch_lock(self.registry.root):
                        write(self.registry.root / 'settings.json', draft)
                    self.settings = dict(draft)
                    self.emit('配置已保存；没有启动任务。')
                elif action == 's' and new_name:
                    if self.launch(new_name, draft):
                        return
                elif action.isdigit() and 1 <= int(action) <= len(fields):
                    key, label = fields[int(action)-1]
                    answer = self.prompt(label + '（auto 为自动；b 返回）：')
                    draft[key] = parse_setting(key, answer)
                elif action:
                    self.emit('请选择菜单中的操作。')
            except Back:
                continue
            except (ValueError, RuntimeError, OSError, ImportError) as exc:
                self.emit('未完成：' + str(exc))

    def launch(self, name, settings):
        self.registry.ensure_idle()
        config, spacing, resources, planned = self.show_plan(settings)
        task_id = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
        output = Path(settings['results']) / task_id
        self.emit(f'将启动：{name}\n输出目录：{output}')
        if planned['invalid'] or resources['oversized_cases']:
            self.emit('存在无效或过大病例，任务会记录失败；请按需要返回修改配置。')
        if self.prompt('输入 y 确认启动，其他输入返回配置：').lower() != 'y':
            return False
        from aaoca_preprocessing import jobs
        with dispatch_lock(self.registry.root):
            self.registry.ensure_idle()
            entry = {'id': task_id, 'name': name, 'output': str(output), 'created_at': now()}
            # Persist history BEFORE spawning, so a terminal crash cannot orphan
            # a successfully submitted worker from this registry.
            self.registry.register(entry)
            self.selected = entry
            write(self.registry.root / 'settings.json', settings)
            self.settings = dict(settings)
            scratch = str(Path(settings['scratch']) / task_id) if settings['scratch'] else None
            result = jobs.start(settings['input'], output, config, settings['manifest'] or None,
                                scratch, spacing, resource_request={k: settings[k] for k in RESOURCE_FIELDS})
            write(Path(result['runtime']) / 'preview.json', {'resources': resources, 'spacing': spacing})
        self.emit('已提交后台预处理。退出面板或断开终端后继续运行；用 s 看进度、l 看日志。')
        return True

    def new(self):
        self.registry.ensure_idle()
        name = self.prompt('新任务名称（1–80 字，b 返回）：')
        if not 1 <= len(name) <= 80 or any(ord(c) < 32 for c in name):
            raise ValueError('任务名称须为 1–80 字，不含控制字符')
        self.emit('已接入：1 图像强度归一化 + 重采样。后续步骤尚未接入。')
        step = self.prompt('选择步骤 [1]，b 返回：')
        if step not in {'', '1'}:
            raise ValueError('当前仅开放预处理步骤 1')
        self.configure(name)

    def resume(self):
        entry = self.current()
        self.registry.ensure_idle()
        self.emit(f"续跑：{entry['name']}\n输出：{entry['output']}\n沿用冻结的输入和间距，自动资源重新估算。")
        if self.prompt('输入 y 确认续跑：').lower() != 'y':
            return
        from aaoca_preprocessing import jobs
        with dispatch_lock(self.registry.root):
            self.registry.ensure_idle()
            jobs.resume(entry['output'])
        self.emit('已提交续跑。')

    def stop(self):
        entry = self.current()
        if job_status(entry['output'])['alive'] is not True:
            raise ValueError('选中任务当前没有可确认的运行进程')
        self.emit(f"将请求停止：{entry['name']}\n输出：{entry['output']}")
        if self.prompt('输入 y 确认停止：').lower() != 'y':
            return
        from aaoca_preprocessing import jobs
        jobs.stop(entry['output'])
        self.emit('已请求停止。当前压缩写入可能先完成，随后可用 r 续跑。')

    def environment(self):
        self.emit(f'Python：{sys.executable}\n版本：{sys.version.split()[0]}\n面板目录：{self.registry.root}')
        for name in ('numpy', 'scipy', 'nibabel', 'psutil'):
            self.emit(f'{name}：' + ('可导入位置已找到' if importlib.util.find_spec(name) else '未安装'))
        self.emit('本模块仅用 CPU；启动需安装本项目预处理依赖。')

    def run(self):
        self.emit('AAOCA 服务器工作台 — 独立于原始资料整理入口')
        while True:
            try:
                if self.selected:
                    state = job_status(self.selected['output'])['state']
                    self.emit(f"\n当前任务：{self.selected['name']} | {STATES.get(state, state)}")
                else:
                    self.emit('\n当前任务：暂无')
                self.emit('n 新建任务  c 预处理配置  s 进度  l 日志  f 实时日志\nh 历史任务  o 结果位置  r 续跑  x 停止  e 环境  q/0 退出')
                action = self.prompt('操作 [s]：').lower() or 's'
                actions = {'n': self.new, 'c': self.configure, 's': self.show_status, 'l': self.logs,
                           'f': lambda: self.logs(True), 'h': self.history, 'r': self.resume,
                           'x': self.stop, 'e': self.environment,
                           'o': lambda: self.emit('结果目录：' + self.current()['output'])}
                if action in actions:
                    actions[action]()
                else:
                    self.emit('请选择菜单中的操作。')
            except Back:
                continue
            except (Exit, EOFError, KeyboardInterrupt):
                self.emit('已退出面板，已提交的后台任务继续运行。')
                return 0
            except (ValueError, RuntimeError, OSError, ImportError) as exc:
                self.emit('未完成：' + str(exc))


def main(argv=None):
    parser = argparse.ArgumentParser(description='AAOCA 中文服务器工作台（当前开放预处理）')
    parser.add_argument('--runtime', default=os.environ.get('AAOCA_SERVER_RUNTIME', '~/.local/state/aaoca-server'), help='面板配置与任务历史目录')
    parser.add_argument('--input', help='已转换且保留 HU 的 NIfTI 根目录')
    parser.add_argument('--results', help='独立预处理结果根目录')
    queries = parser.add_mutually_exclusive_group()
    queries.add_argument('--status', action='store_true', help='只读最近任务状态后退出')
    queries.add_argument('--logs', action='store_true', help='只读最近任务日志后退出')
    args = parser.parse_args(argv)
    try:
        overrides = {key: parse_setting(key, getattr(args, key)) for key in ('input', 'results') if getattr(args, key)}
        panel = Panel(args.runtime, overrides)
        if args.status or args.logs:
            (panel.show_status if args.status else panel.logs)()
            return 0
        return panel.run()
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print('未完成：' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
