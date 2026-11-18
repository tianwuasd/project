import csv
import io
import subprocess
import shutil
from time import perf_counter,sleep


def process_vram_observation(uuid,pid):
    executable = shutil.which('nvidia-smi')
    if executable is None:
        return dict(bytes=None,reason='nvidia_smi_unavailable')
    with subprocess.Popen([executable,'--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
            '--format=csv,noheader,nounits'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) as process:
        deadline = perf_counter()+5
        while process.poll() is None and perf_counter()<deadline:
            sleep(.02)
        if process.poll() is None:
            process.kill()
            process.communicate()
            return dict(bytes=None,reason='nvidia_smi_query_timeout')
        stdout,_ = process.communicate()
        if process.returncode:
            return dict(bytes=None,reason='nvidia_smi_query_failed',exit_code=process.returncode)
    rows = [row for row in csv.reader(io.StringIO(stdout)) if len(row)==3
            and row[0].strip()==uuid and row[1].strip()==str(pid)]
    if len(rows)!=1:
        return dict(bytes=None,reason='process_record_missing_or_ambiguous')
    amount = rows[0][2].strip()
    if not amount.isdecimal():
        return dict(bytes=None,reason='process_memory_unobservable',reported=amount)
    return dict(bytes=int(amount)*1024**2,reason=None)


def complete_peak(measurements):
    values = [row['peak_vram_bytes'] for row in measurements]
    return max(values) if values and all(value is not None for value in values) else None
