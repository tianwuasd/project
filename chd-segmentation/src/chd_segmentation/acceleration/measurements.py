import csv
import io
import shutil

from .gpu_query import query_nvidia_smi


def process_vram_observation(uuid,pid, *, resources=None, cancel=None, log_dir=None):
    executable = shutil.which('nvidia-smi')
    if executable is None:
        return dict(bytes=None,reason='nvidia_smi_unavailable')
    stdout = query_nvidia_smi(['--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                              '--format=csv,noheader,nounits'],
                             resources=resources, cancel=cancel, log_dir=log_dir)
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
