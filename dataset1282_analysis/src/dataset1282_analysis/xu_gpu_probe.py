import csv
from functools import lru_cache
import io
import subprocess
from uuid import UUID


def _cuda(result):
    if int(result[0]) != 0:
        raise RuntimeError('CUDA 只读查询失败: '+str(result[0]))
    return result[1:]


@lru_cache(maxsize=1)
def visible_devices():
    from cuda.bindings import driver
    initialized = driver.cuInit(0)
    if int(initialized[0]) == 100:
        return []
    _cuda(initialized)
    count, = _cuda(driver.cuDeviceGetCount())
    rows = []
    for index in range(count):
        device, = _cuda(driver.cuDeviceGet(index))
        uuid, = _cuda(driver.cuDeviceGetUuid(device))
        name, = _cuda(driver.cuDeviceGetName(256,device))
        total, = _cuda(driver.cuDeviceTotalMem(device))
        rows.append(dict(device=f'cuda:{index}',uuid='GPU-'+str(UUID(bytes=bytes(uuid.bytes))),
                         name=name.split(b'\x00',1)[0].decode('utf-8'),total_bytes=int(total)))
    return rows


def _query(kind, columns):
    proc = subprocess.run(['nvidia-smi',f'--query-{kind}='+columns,'--format=csv,noheader,nounits'],
                          check=True,capture_output=True,text=True,encoding='utf-8',timeout=10)
    return [[part.strip() for part in row] for row in csv.reader(io.StringIO(proc.stdout)) if row]


def gpu_snapshot():
    rows = {}
    for row in _query('gpu','index,uuid,name,memory.free,memory.total,utilization.gpu'):
        if len(row) != 6:
            raise ValueError('无法解析 NVIDIA 显卡查询结果')
        index, uuid, name, free, total, utilization = row
        rows[uuid] = dict(physical_index=int(index),uuid=uuid,name=name,
                          free_bytes=int(free)*1024**2,total_bytes=int(total)*1024**2,
                          utilization=int(utilization),compute_pids=[])
    for row in _query('compute-apps','gpu_uuid,pid'):
        if len(row) != 2 or int(row[1]) <= 0 or row[0] not in rows:
            raise ValueError('无法核实 NVIDIA 计算进程的 GPU 归属')
        rows[row[0]]['compute_pids'].append(int(row[1]))
    return rows


def device_info(selection):
    names = {row['device']:row for row in visible_devices()}
    name = 'cuda:0' if str(selection) == 'cuda' else str(selection)
    if name not in names:
        raise ValueError('设备不在当前 CUDA 可见范围内: '+name)
    selected = names[name]
    inventory = gpu_snapshot()
    if selected['uuid'] not in inventory:
        raise ValueError('CUDA 与 NVIDIA 查询的显卡 UUID 不一致')
    return {**inventory[selected['uuid']],**selected}
