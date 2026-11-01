"""One CHD V2 entry, with explicit scientific methods and isolated registries."""
import argparse
from pathlib import Path
import sys
import subprocess

VERSION='2.2.1'
PROJECT=Path(__file__).resolve().parents[3]


def run_stunet(arguments):
    engine=PROJECT/'foundation_engine/05_src'
    if not (engine/'v2_host_identity.py').is_file():
        raise ValueError('V2的STU-Net引擎不完整；请使用完整V2工程检出')
    sys.path.insert(0,str(engine))
    import task_panel
    try:return task_panel.main(arguments)
    except (task_panel.Quit,task_panel.Back,KeyboardInterrupt,EOFError):
        print('已退出V2面板，后台任务继续。')
        return 0


def run_multistage(arguments):
    from .panel import main
    return main(arguments)


def main(argv=None):
    arguments=list(sys.argv[1:] if argv is None else argv)
    selector=argparse.ArgumentParser(add_help=False)
    selector.add_argument('--engine',choices=['stunet','multistage'])
    chosen,remaining=selector.parse_known_args(arguments)
    if ('-h' in remaining or '--help' in remaining) and chosen.engine is None:
        print(f'CHD V2 {VERSION} integrated panel')
        print('start_server_v2.sh --engine stunet|multistage [method options]')
        print('stunet: STU-Net-S七结构微调；multistage: 原CT六网络/三折。')
        print('打开面板和查询不会安装环境或启动训练。方法帮助：--engine METHOD --help')
        return 0
    engine=chosen.engine
    if engine is None and not remaining:
        if not sys.stdin.isatty():
            print('非交互运行请指定--engine和阶段/查询，或--help。',file=sys.stderr)
            return 1
        print(f'\nCHD V2 {VERSION} | 选择分割方法')
        print('1 STU-Net-S预训练微调（默认）  2 原CT六网络/三折  q 退出')
        try:answer=input('选择 [1]：').strip().lower() or '1'
        except (KeyboardInterrupt,EOFError):return 0
        if answer in ('q','0'):return 0
        if answer not in ('1','2'):
            print('请输入1或2。',file=sys.stderr);return 1
        engine='stunet' if answer=='1' else 'multistage'
    try:
        return run_multistage(remaining) if engine=='multistage' else run_stunet(remaining)
    except (ValueError,OSError,RuntimeError,subprocess.SubprocessError) as error:
        print('V2未执行/未完成：'+str(error),file=sys.stderr)
        return 1
