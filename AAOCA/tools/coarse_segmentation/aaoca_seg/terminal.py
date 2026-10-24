"""Independent Chinese terminal panel; queries do not import inference packages."""
import argparse
import getpass
import json
import os
import time
from pathlib import Path

from .operations import hardware, probe
from .resources import select_devices
from .service_state import read_json, tail
from .workbench import Workbench, status


class Back(Exception):
    pass


class Exit(Exception):
    pass


class Panel:
    def __init__(self, workspace, ask=input, emit=print, overrides=None):
        self.work = Workbench(workspace, overrides)
        self.ask, self.emit = ask, emit
        entries = self.work.entries()
        self.selected = Path(entries[0]['job']) if entries else None

    def prompt(self, message, default=''):
        value = self.ask(message).strip()
        if value.lower() in {'b', 'back', '返回'}:
            raise Back
        if value.lower() in {'q', 'quit', 'exit', '退出'}:
            raise Exit
        return value or default

    def choose(self):
        entries = self.work.entries()
        if not entries:
            raise ValueError('暂无历史任务')
        for i, row in enumerate(entries, 1):
            self.emit(f"{i}. {row['name']} | {row['status']} | {row['output']}")
        index = int(self.prompt('选择任务编号，b 返回：')) - 1
        if not 0 <= index < len(entries):
            raise ValueError('任务编号不在列表中')
        return Path(entries[index]['job'])

    def current(self):
        if not self.selected:
            raise ValueError('暂无任务，请 n 新建或 h 选择历史任务')
        return self.selected

    def show(self, job=None):
        job = job or self.current()
        request = read_json(job / 'request.json')
        info = status(job)
        self.emit(f"任务：{request['name']} | {info['status']} | 后台存活：{info['active']}")
        self.emit('结果：' + request['output'])
        self.emit('来源任务：' + str(request.get('input_job') or request.get('preprocessing_source') or '手动原始 HU 清单'))
        progress = read_json(Path(request['output']) / 'status.json', {})
        self.emit(f"完成 {progress.get('finished', 0)}/{progress.get('total', len(request['cases']))}，"
                  f"失败 {progress.get('failed', 0)}，复用 {progress.get('reused', 0)}")
        for key, item in progress.get('active', {}).items():
            self.emit(f"{key}：{item['phase']}，设备 {item.get('device') or 'CPU'}")
        for row in progress.get('cases', []):
            if row.get('status') == 'failed':
                self.emit(f"失败 {row['case_id']}：{row.get('error', row.get('phase'))}，日志 {row.get('log', 'l 查看')}")
        if progress.get('waiting_gpu'):
            self.emit('等待获准 GPU 空闲；不会扩大分配范围。')
        eta = progress.get('eta_seconds')
        self.emit(f'估算剩余 {eta} 秒' if eta is not None else '剩余时间暂无法估算。')
        if info.get('error'):
            self.emit(info['error'])

    def configure(self):
        draft = dict(self.work.settings)
        fields = [('manifest', '原始 HU/mm CSV 清单'), ('results', '结果根目录'),
                  ('cpu_limit', 'CPU 总预算'), ('memory_gib', '内存 GiB'), ('workers', '总病例并发上限'),
                  ('threads', '每病例线程上限'), ('io_workers', '读写并发上限'),
                  ('gpu_count', 'GPU 并发上限'), ('min_gpu_gib', '最低空闲显存 GiB'),
                  ('preprocess_output', '预处理任务目录或结果根目录')]
        while True:
            self.emit('\n分割配置（auto 会按 affinity/cgroup、当前负载和可用内存收缩）')
            for i, (key, label) in enumerate(fields, 1):
                self.emit(f'{i}. {label}：{draft[key] if draft[key] is not None else "auto"}')
            self.emit(f"设备：{draft['mode']} {draft['devices']}\nd 设备选择  w 保存  b 返回  q 退出")
            action = self.prompt('配置操作：')
            if action == 'w':
                self.work.save(draft)
                self.emit('已保存配置。')
                return
            if action == 'd':
                self.emit(json.dumps(hardware(self.work.root), ensure_ascii=False, indent=2))
                mode = self.prompt('cpu 或 gpu：', 'cpu')
                if mode == 'cpu':
                    draft.update(mode='cpu', devices=[])
                elif mode == 'gpu':
                    info = probe(self.work.root)
                    self.emit(json.dumps(info.get('devices', []), ensure_ascii=False, indent=2))
                    values = self.prompt('输入分配给你的可见 GPU 编号，用逗号隔开（0 是有效编号）：')
                    candidates = [int(v.strip()) for v in values.split(',')]
                    devices = select_devices(info, candidates)
                    if self.prompt('确认这些 GPU 已分配给你，输入 y：') == 'y':
                        draft.update(mode='gpu', devices=devices)
                else:
                    raise ValueError('请选择 cpu 或 gpu')
            elif action.isdigit() and 1 <= int(action) <= len(fields):
                key, label = fields[int(action)-1]
                value = self.prompt(label + '（b 返回，资源可填 auto）：')
                if key in {'manifest', 'results', 'preprocess_output'}:
                    draft[key] = str(Path(value).expanduser().resolve())
                elif value == 'auto':
                    draft[key] = None
                else:
                    draft[key] = float(value) if key in {'memory_gib', 'min_gpu_gib'} else int(value)

    def steps(self, default='1,2,3'):
        self.emit('1 输入/资源预检（CPU）  2 模型推理  3 合并与质检（CPU）')
        value = self.prompt(f'步骤，多选用逗号 [{default}]：', default)
        mapping = {'1': 'plan', '2': 'infer', '3': 'merge'}
        try:
            return [mapping[v.strip()] for v in value.split(',')]
        except KeyError:
            raise ValueError('请选择 1、2、3') from None

    def new(self):
        name = self.prompt('新任务名称：')
        source = self.prompt('输入来源：p 预处理结果，0 原始 HU 清单，h 历史分割任务 [p]：', 'p')
        parent = self.choose() if source == 'h' else None
        if source not in {'p', '0', 'h'}:
            raise ValueError('请选择 p、0 或 h')
        preprocessed = self.choose_preprocessed() if source == 'p' else None
        steps = self.steps()
        subset = self.prompt('病例代号，逗号分隔；回车全部：')
        ids = [v.strip() for v in subset.split(',')] if subset else None
        self.emit('预处理输入使用同网格的未截断 HU 配套影像；分割标签将与归一化 CT 对齐。'
                  if preprocessed else '使用所选来源的 HU/mm 影像；不会把[-1,1]直接传给模型。')
        if self.prompt('输入 y 创建并提交后台任务，b 返回：') != 'y':
            return
        self.selected = self.work.create(name, steps, input_job=parent.name if parent else None,
                                         case_ids=ids, preprocessed=preprocessed)
        self.work.start(self.selected)
        self.emit('已提交；退出面板、断开 SSH 后继续运行。')

    def choose_preprocessed(self):
        from .preprocessed import candidates, load
        default = self.work.settings['preprocess_output']
        path = self.prompt(f'预处理任务目录或结果根目录 [{default}]：', default)
        if not path:
            raise ValueError('请填写已完成预处理任务的位置')
        options = candidates(path)
        if not options:
            raise ValueError('该目录中没有预处理任务')
        for index, root in enumerate(options, 1):
            progress = read_json(root / 'progress.json', {})
            state = progress.get('state', '未知') if isinstance(progress, dict) else '记录无效'
            self.emit(f"{index}. {root.name} | {state} | {root}")
        index = int(self.prompt('选择预处理任务编号 [1]：', '1')) - 1
        if not 0 <= index < len(options):
            raise ValueError('编号不在列表中')
        selected = options[index]
        rows = load(selected)
        self.emit(f'已选择 {len(rows)} 个预处理体积；执行前完整校验 HU/归一化配套影像。')
        return selected

    def resume(self):
        self.selected = self.choose()
        self.show()
        self.emit('续跑沿用冻结代码、输入和输出；完整掩膜/结果通过 SHA256 核验后复用。')
        if self.prompt('输入 y 续跑：') == 'y':
            self.work.start(self.selected)

    def rerun(self):
        parent = self.choose()
        steps = self.steps()
        if self.prompt('将另建任务和结果目录，旧结果保留；输入 y 提交：') == 'y':
            self.selected = self.work.rerun(parent.name, steps)
            self.work.start(self.selected)

    def stop(self):
        self.show()
        job = self.current()
        if self.prompt('输入 y 请求停止上面这个任务：') == 'y':
            self.emit(self.work.stop(job.name))

    def logs(self, follow=False):
        while True:
            self.emit(self.work.logs(self.current()))
            if not follow:
                return
            self.emit('每 10 秒刷新；Ctrl+C 返回，后台继续。')
            try:
                time.sleep(10)
            except KeyboardInterrupt:
                return

    def gpu(self, follow=True):
        while True:
            self.emit(json.dumps(hardware(self.work.root), ensure_ascii=False, indent=2))
            if not follow:
                return
            self.emit('每 1 秒刷新；Ctrl+C 返回，后台继续。')
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                return

    def setup(self):
        self.emit('环境操作独立于科学任务；不会启动分割。')
        action = self.prompt('1 安装独立环境  2 检查环境  3 验证许可  b 返回：')
        if action == '1':
            from .managed import ManagedController
            build = self.prompt('PyTorch 安装方案 auto/cpu/cu118/cu126 [auto]：', 'auto')
            if build not in {'auto', 'cpu', 'cu118', 'cu126'}:
                raise ValueError('安装方案无效')
            controller = ManagedController(self.work.root)
            config = {k: v for k, v in controller.config.items() if k != 'gpu_uuid'}
            config['torch_build'] = build
            controller.configure(config)
            self.work.setup('install')
        elif action == '2':
            self.work.setup('probe')
        elif action == '3':
            secret = getpass.getpass('许可码（不回显，留空使用已导入的私有配置）：')
            self.work.setup('license', {'license': secret} if secret else {})
        else:
            raise ValueError('请选择 1、2、3')
        self.emit('已提交环境操作；用 a 查看安装/许可日志。')

    def environment_logs(self):
        jobs = []
        for job in (self.work.root / 'jobs').glob('*'):
            request = read_json(job / 'request.json', {})
            if isinstance(request, dict) and request.get('action') in {'install', 'probe', 'license'}:
                jobs.append(job)
        if not jobs:
            self.emit('尚无环境操作。')
            return
        job = max(jobs, key=lambda p: p.stat().st_mtime)
        self.emit(str(status(job)))
        self.emit(tail(job / 'worker.log'))

    def results(self):
        self.selected = self.choose()
        self.show()
        request = read_json(self.selected / 'request.json')
        for pointer in Path(request['output']).glob('cases/*/latest-result.json'):
            self.emit(str(pointer.parent.name) + '：' + read_json(pointer, {}).get('directory', '未知'))
        self.emit('粗分割仍需人工复核；result.json/qc.json 记录输出及空掩膜、重叠。')

    def run(self):
        self.emit('AAOCA 独立粗分割面板 — 预处理结果 / TotalSegmentator / 11 类')
        while True:
            try:
                self.emit('\nn 新建任务  c 配置  h 历史  s 进度  o 结果\nr 续跑  k 已完成任务重跑  x 停止  l 日志  f 持续日志\ng 显卡  e 环境/许可  a 环境日志  u 更新  q/0 退出')
                action = self.prompt('操作 [s]：', 's')
                if action == '0':
                    raise Exit
                if action == 'h':
                    self.selected = self.choose()
                    self.show()
                elif action == 'u':
                    from .updater import update
                    self.emit(update(self.work))
                    return 0
                else:
                    actions = {'n': self.new, 'c': self.configure, 's': self.show, 'o': self.results,
                               'r': self.resume, 'k': self.rerun, 'x': self.stop, 'l': self.logs,
                               'f': lambda: self.logs(True), 'g': self.gpu, 'e': self.setup, 'a': self.environment_logs}
                    if action in actions:
                        actions[action]()
                    else:
                        self.emit('请选择菜单中的操作。')
            except Back:
                pass
            except (Exit, EOFError, KeyboardInterrupt):
                self.emit('已退出面板，后台任务继续运行。')
                return 0
            except (ValueError, OSError, RuntimeError) as error:
                self.emit('未完成：' + str(error))


def main(argv=None):
    parser = argparse.ArgumentParser(description='AAOCA 独立中文粗分割面板')
    parser.add_argument('--runtime', default=os.environ.get('AAOCA_SEG_RUNTIME', '~/.local/state/aaoca-seg-terminal'))
    parser.add_argument('--query', choices=['status', 'history', 'logs', 'gpu', 'setup'])
    parser.add_argument('--job', help='只读查询的历史任务 UUID')
    parser.add_argument('--profile', help='导入之前的私有服务器配置，不打印许可')
    parser.add_argument('--preprocessed', help='本次默认选择的已完成预处理任务或根目录；不启动预处理')
    parser.add_argument('--results', help='本次分割结果根目录；省略则沿用已保存配置')
    args = parser.parse_args(argv)
    try:
        overrides = {key: str(Path(value).expanduser().resolve()) for key, value in
                     [('preprocess_output', args.preprocessed), ('results', args.results)] if value}
        panel = Panel(args.runtime, overrides=overrides)
        if args.profile:
            from .managed import ManagedController
            from .server import import_profile
            panel.work.idle()
            controller = ManagedController(panel.work.root)
            import_profile(controller, args.profile)
            config = controller.config
            panel.work.save({**panel.work.settings, 'manifest': config['manifest'], 'results': config['output_root']})
        if args.job:
            panel.selected = panel.work.resolve(args.job)
        if args.query:
            actions = {'status': panel.show, 'history': lambda: print(json.dumps(panel.work.entries(), ensure_ascii=False, indent=2)),
                       'logs': panel.logs, 'gpu': lambda: panel.gpu(False), 'setup': panel.environment_logs}
            actions[args.query]()
            return 0
        return panel.run()
    except (ValueError, OSError, RuntimeError) as error:
        print('未完成：' + str(error))
        return 1
