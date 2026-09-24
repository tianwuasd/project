"""标准库GPU选择：显式填写数量和编号，不抢占其他计算任务。"""
import csv
import subprocess


class GPUCoolingDown(RuntimeError):
    """没有计算进程，但利用率采样尚未回落；仅任务交接时允许短暂重查。"""


def select_gpus(count=None, selection=None):
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,name,memory.free,utilization.gpu',
         '--format=csv,noheader,nounits'], text=True, timeout=15)
    rows = [[cell.strip() for cell in row] for row in csv.reader(output.splitlines()) if row]
    print('GPU编号 / UUID / 名称 / 空闲MiB / 利用率：\n' + output, flush=True)
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
        if row[1] in selected:
            raise ValueError('不能重复选择同一张显卡')
        selected.append(row[1])
        selected_rows.append(row)
    active = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True, timeout=15)
    busy = {line.split(',')[0].strip() for line in active.splitlines()}
    if busy.intersection(selected):
        raise RuntimeError('所选GPU已有计算进程，不抢占或终止其他任务；请重新选择空闲卡')
    for row in selected_rows:
        if float(row[3]) < 6000:
            raise RuntimeError(f'GPU {row[0]} 空闲显存不足6000MiB')
        if float(row[4]) > 10:
            raise GPUCoolingDown(f'GPU {row[0]} 当前利用率仍高于10%，请等待空闲')
    return selected
