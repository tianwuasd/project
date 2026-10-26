"""标准库GPU选择：显式填写数量和编号，不抢占其他计算任务。"""
import csv
import subprocess
import os


class GPUBusy(RuntimeError):
    """A permitted device is temporarily unavailable; managed queues may wait."""


class GPUCoolingDown(RuntimeError):
    """没有计算进程，但利用率采样尚未回落；仅任务交接时允许短暂重查。"""


def select_gpus(count=None, selection=None):
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,name,memory.free,utilization.gpu,memory.total',
         '--format=csv,noheader,nounits'], text=True, timeout=15)
    rows = [[cell.strip() for cell in row] for row in csv.reader(output.splitlines()) if row]
    from evaluation_gpu import allowed_ids
    visible = allowed_ids([dict(index=r[0], uuid=r[1]) for r in rows], {'allowed_gpus': None})
    print('GPU编号 / UUID / 名称 / 空闲MiB / 利用率 / 总显存MiB：\n' + output, flush=True)
    choices = str(selection).replace('，', ',').replace(',', ' ').split() if selection is not None else None
    if count is None:
        count = len(choices) if choices is not None else int(input('使用几张显卡？[回车默认1]：').strip() or '1')
    if count < 1 or count > len(rows):
        raise ValueError(f'显卡数量应为1到{len(rows)}之间')
    if choices is None:
        choices = input(f'填写{count}张已获准使用的空闲GPU编号，以逗号分隔（如0,2）：').replace('，', ',').replace(',', ' ').split()
    if len(choices) != count:
        raise ValueError(f'选择了{count}张卡，但填写了{len(choices)}个编号')
    selected, selected_rows = [], []
    for choice in choices:
        matched = [r for r in rows if choice in r[:2]]
        if len(matched) != 1:
            raise ValueError(f'GPU编号不存在：{choice}')
        row = matched[0]
        if row[1] not in visible:
            raise ValueError(f'GPU {row[0]} 不在继承的可见/分配范围内')
        if row[1] in selected:
            raise ValueError('不能重复选择同一张显卡')
        selected.append(row[1])
        selected_rows.append(row)
    active = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True, timeout=15)
    busy = {line.split(',')[0].strip() for line in active.splitlines()}
    if busy.intersection(selected):
        raise GPUBusy('所选GPU已有计算进程，不抢占或终止其他任务；请等待空闲卡')
    for row in selected_rows:
        required=6000
        if os.environ.get('WHOLE_HEART_MANAGED')=='1':
            from gpu_budget import required_free_mib
            if len(row)<6:raise ValueError('GPU查询缺少总显存，不能核对规划预算')
            required=required_free_mib(os.environ.get('WHOLE_HEART_GPU_MEMORY',5),row[5])
        if float(row[3]) < required:
            raise GPUBusy(f'GPU {row[0]} 空闲显存不足{required:g}MiB')
        if float(row[4]) > 10:
            raise GPUCoolingDown(f'GPU {row[0]} 当前利用率仍高于10%，请等待空闲')
    return selected
