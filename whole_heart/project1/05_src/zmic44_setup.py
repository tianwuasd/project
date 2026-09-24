"""zmic44 初学者入口：标准库即可运行，个人目录安装、单卡、后台日志。

服务器文档中的位置仅是建议，首次输入才绑定；不会使用旧用户目录本身。
不安装驱动、不使用 sudo、不登录网盘、不下载预训练权重。
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

CODE = Path(__file__).resolve().parents[1]
SETTINGS = CODE / '.zmic44_settings.json'
VERSION = '26.7.2-0'
INSTALLER = f'Miniforge3-{VERSION}-Linux-x86_64.sh'
URL = f'https://github.com/conda-forge/miniforge/releases/download/{VERSION}/{INSTALLER}'
SHA256 = '281b0ac7d550802efc81af633225a5e6116d29ae72f3ab4eae7168c3931a4c05'


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def personal_path(value):
    """拦住共享磁盘根目录、旧用户根目录和拥挤的系统/Home 分区。"""
    path = Path(value).expanduser().resolve()
    blocked = {'/', '/home', '/data2', '/data3', '/data4', '/data5', '/data_nas', '/data5/zhougaowei'}
    if str(path) in blocked or path == CODE or path in CODE.parents:
        raise ValueError('请选择个人子目录，不能使用共享盘根目录、旧用户根目录或代码根目录')
    if path == Path.home() or Path.home() in path.parents or str(path).startswith(('/tmp/', '/var/', '/usr/')):
        raise ValueError('此服务器系统盘/Home空间不足，请选择个人数据盘目录')
    return path


def validate_work_directory(work):
    """首次校验中断可能仅留下工作锁；该锁不等于未知实验数据。"""
    if work.exists() and not (work / '00_admin/server_config.json').exists():
        if any(p.name != '.run.lock' for p in work.iterdir()):
            raise ValueError('结果目录非空且不是本项目工作区，请使用新的专用子目录')


def environment(base):
    """只对当前子进程设缓存目录，不改系统配置或 shell 登录文件。"""
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    mapping = {'CONDA_PKGS_DIRS': 'cache/conda_pkgs', 'CONDA_ENVS_PATH': 'envs',
               'PIP_CACHE_DIR': 'cache/pip', 'XDG_CACHE_HOME': 'cache/xdg',
               'TORCH_HOME': 'cache/torch', 'HF_HOME': 'cache/huggingface',
               'MPLCONFIGDIR': 'cache/matplotlib', 'TMPDIR': 'cache/tmp'}
    for key, relative in mapping.items():
        folder = base / relative
        folder.mkdir(parents=True, exist_ok=True)
        env[key] = str(folder)
    env.update(TEMP=env['TMPDIR'], TMP=env['TMPDIR'], PYTHONUNBUFFERED='1',
               PYTHONNOUSERSITE='1', MPLBACKEND='Agg',
               WHOLE_HEART_ENV_PREFIX=str(base / 'envs/project1_py311'))
    return env


def ensure_conda(base, env):
    prefix = base / 'tools/miniforge3'
    executable = prefix / 'bin/conda'
    if executable.is_file():
        return executable
    if prefix.exists():
        raise RuntimeError(f'Conda目录不完整：{prefix}。保留现场，请核查，不自动删除。')
    download = base / 'cache' / INSTALLER
    if not download.exists():
        print('下载官方 Miniforge 安装包；若网络失败，日志会保留具体地址。', flush=True)
        partial = download.with_suffix('.partial')
        with urllib.request.urlopen(URL, timeout=120) as response, partial.open('wb') as stream:
            shutil.copyfileobj(response, stream)
        os.replace(partial, download)
    with download.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != SHA256:
        raise RuntimeError(f'安装包SHA256不符，停止执行：{download}')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['bash', str(download), '-b', '-p', str(prefix)], env=env, check=True)
    if not executable.is_file():
        raise RuntimeError('Miniforge 安装没有生成 conda')
    return executable


def select_gpu(index):
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.free,utilization.gpu',
                                      '--format=csv,noheader,nounits'], text=True)
    rows = list(csv.reader(output.splitlines(), skipinitialspace=True))
    print('GPU编号 / UUID / 名称 / 空闲MiB / 利用率：\n' + output)
    if index is None:
        index = input('请填写一张已获准使用的空闲GPU编号（不会自动占用全部8张）：').strip()
    matches = [r for r in rows if r[0] == str(index) or r[1] == str(index)]
    if len(matches) != 1:
        raise ValueError('GPU编号不存在，或填写了多张卡')
    row = matches[0]
    if float(row[3]) < 6000 or float(row[4]) > 10:
        raise RuntimeError('所选GPU显存不足或正在忙碌；请与使用者协调后选择空闲卡')
    # 仅查看进程，不停止任何任务；有计算进程时保守拒绝共享。
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                     '--format=csv,noheader,nounits'], text=True)
    if any(line.split(',')[0].strip() == row[1] for line in active.splitlines()):
        raise RuntimeError('所选GPU已有计算进程，本向导不抢占或终止其他任务')
    return row[1]


def run_worker(config_path, mode):
    """后台执行；从安装开始就持有锁，避免两个入口改同一环境/结果。"""
    import fcntl
    config = json.loads(config_path.read_text(encoding='utf-8'))
    base, work = Path(config['base']), Path(config['work'])
    env = environment(base)
    with (base / '.project1_job.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_file = base / 'job_state.json'
        state = {'status': 'running', 'pid': os.getpid(), 'mode': mode, 'started': time.strftime('%F %T')}
        write_json(state_file, state)
        try:
            # 再次检查，降低用户选择后到后台启动间被占用的风险；不是集群调度锁。
            env['CUDA_VISIBLE_DEVICES'] = select_gpu(config['gpu_uuid'])
            env['CONDA_EXE'] = str(ensure_conda(base, env))
            # 首次安装可能较久，真正开始项目入口前再检查一次。
            env['CUDA_VISIBLE_DEVICES'] = select_gpu(config['gpu_uuid'])
            command = ['bash', str(CODE / 'start_server.sh'), config['data'], '--work-dir', str(work),
                       '--mode', mode, '--scope', 'all', '--gpu-memory', '5', '--yes-train']
            subprocess.run(command, env=env, check=True)
            state['status'] = 'completed'
        except BaseException as error:
            state.update(status='failed', error=str(error))
            raise
        finally:
            state['ended'] = time.strftime('%F %T')
            write_json(state_file, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', help='Conda/缓存/后台日志所在的个人目录')
    parser.add_argument('--work-dir', help='数据准备、模型和评价输出目录；默认NAS')
    parser.add_argument('--data-dir', help='已解压的原始106例数据目录')
    parser.add_argument('--gpu', help='一张可用GPU的编号')
    parser.add_argument('--mode', choices=['check', 'smoke', 'experiment', 'evaluate', 'status'])
    parser.add_argument('--foreground', action='store_true', help='调试时前台运行；默认后台运行')
    parser.add_argument('--worker', nargs=2, metavar=('CONFIG', 'MODE'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not sys.platform.startswith('linux'):
        raise RuntimeError('此入口用于Linux服务器；Windows上只能查看 --help')
    if args.worker:
        run_worker(Path(args.worker[0]), args.worker[1])
        return
    old = json.loads(SETTINGS.read_text(encoding='utf-8')) if SETTINGS.exists() else {}
    if args.mode == 'status':
        if not old:
            print('尚未启动。先执行 bash run_zmic44.sh')
            return
        state = Path(old['base']) / 'job_state.json'
        print(state.read_text(encoding='utf-8') if state.exists() else '正在启动，查看日志。')
        print('日志：', old.get('log', '尚无日志'), '\n结果：', old['work'])
        print('以上为最后保存状态；服务器重启或强制结束后状态可能滞后。')
        return
    def answer(given, key, default, prompt):
        return given or old.get(key) or (input(f'{prompt}\n[回车默认 {default}] > ').strip() or default)
    print('默认使用你已提供的本地个人目录安装环境和运行缓存；\n'
          '结果与预处理数据放NAS。只在填写的个人工作区内创建本项目文件。')
    base = personal_path(answer(args.base_dir, 'base', '/data5/zhougaowei/zhangruichen_workspace/project1_runtime', '个人运行目录'))
    work = personal_path(answer(args.work_dir, 'work', '/data_nas/zhangruichen/project1_results', '实验结果目录（建议NAS，完整实验预留100GiB以上）'))
    data = Path(answer(args.data_dir, 'data', '/data_nas/zhangruichen/baidu_import/Wholeheart_Train_Dataset', '原始数据目录')).expanduser().resolve()
    if not data.is_dir():
        raise ValueError('原始数据目录不存在；请先完成网盘下载与解压，再填写真实目录')
    for p in [base, work]:
        if p == data or p in data.parents or data in p.parents:
            raise ValueError('运行/输出目录必须与原始数据分开，不能互相包含')
    if base == work or base in work.parents or work in base.parents:
        raise ValueError('环境运行目录和实验结果目录必须互相独立')
    validate_work_directory(work)
    base.mkdir(parents=True, exist_ok=True)
    parent = work
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(base).free < 20 * 2**30 or shutil.disk_usage(parent).free < 100 * 2**30:
        raise RuntimeError('环境盘需20GiB、结果盘需100GiB空闲空间；这些是最低检查，不是容量保证')
    gpu = select_gpu(args.gpu)
    mode = args.mode
    if mode is None:
        print('1 检查环境与数据\n2 短测CT/MRI（首次推荐）\n3 完整B0/B1实验或继续：十次训练＋评价\n4 十次训练已完成，仅继续评价')
        mode = {'1': 'check', '2': 'smoke', '3': 'experiment', '4': 'evaluate'}[input('选择 [默认2]：').strip() or '2']
    if mode in {'experiment', 'evaluate'}:
        print('完整训练包含CT 6次、MRI 4次；不加入CaberNet。旧本机核心计算估计155小时，3090/NAS耗时需实测。\n'
              '会在十次训练全部完成后评价留出中心；不要根据目标结果改方法后覆盖原实验。')
        if input('接受所选实验范围请输入 RUN：').strip() != 'RUN':
            print('已取消。')
            return
    # 验证同一用户的后台工作锁；后台会再次竞争该锁，防止并发启动。
    import fcntl
    with (base / '.project1_job.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    log_dir = base / 'logs'
    log_dir.mkdir(exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}'
    log = log_dir / f'{mode}_{stamp}.log'
    config = {'base': str(base), 'work': str(work), 'data': str(data), 'gpu_uuid': gpu, 'log': str(log)}
    config_path = log_dir / f'{mode}_{stamp}.json'
    write_json(config_path, config)
    write_json(SETTINGS, config)
    if args.foreground:
        run_worker(config_path, mode)
    else:
        # start_new_session脱离SSH会话；安装和训练全过程都重定向到持久日志。
        with log.open('w', encoding='utf-8') as stream:
            process = subprocess.Popen([sys.executable, '-u', '-B', str(Path(__file__).resolve()),
                                        '--worker', str(config_path), mode], stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(f'已提交后台任务 PID={process.pid}，尚不代表运行成功。\n日志：{log}\n'
              f'查看进度：tail -f "{log}"\n查看状态：bash run_zmic44.sh --mode status\n'
              '日志开始正常更新后可断开SSH；Ctrl+C退出tail不会停止训练。')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'停止：{error}', file=sys.stderr)
        sys.exit(1)
