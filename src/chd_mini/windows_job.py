import os
from pathlib import Path
import subprocess
from time import monotonic, sleep

import win32api
import win32con
import win32event
import win32job
import win32process
import msvcrt

from .contracts import write_json


def supervise(command, cwd, env, console, timeout, log_limit, ram_limit, receipt_path=None):
    job = win32job.CreateJobObject(None, '')
    limits = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
    limits['BasicLimitInformation']['LimitFlags'] = (
        win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | win32job.JOB_OBJECT_LIMIT_JOB_MEMORY)
    limits['JobMemoryLimit'] = ram_limit
    win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, limits)
    started = monotonic()
    result = dict(command=command, state='running', all_workers_exited=False)
    try:
        with Path(console).open('wb') as stream, open(os.devnull, 'rb') as source:
            output_handle = msvcrt.get_osfhandle(stream.fileno())
            input_handle = msvcrt.get_osfhandle(source.fileno())
            os.set_handle_inheritable(output_handle, True)
            os.set_handle_inheritable(input_handle, True)
            startup = win32process.STARTUPINFO()
            startup.dwFlags = win32process.STARTF_USESTDHANDLES
            startup.hStdOutput = startup.hStdError = output_handle
            startup.hStdInput = input_handle
            process, thread, pid, _ = win32process.CreateProcess(
                None, subprocess.list2cmdline(command), None, None, True,
                win32con.CREATE_SUSPENDED | win32con.CREATE_NO_WINDOW, env, str(cwd), startup)
            try:
                # 挂入Job后才恢复主线程，子进程自动继承且不允许脱离。
                win32job.AssignProcessToJobObject(job, process)
                win32process.ResumeThread(thread)
                result['pid'] = pid
                while win32event.WaitForSingleObject(process, 1000) == win32con.WAIT_TIMEOUT:
                    if monotonic() - started > timeout:
                        raise TimeoutError('步骤超过限定执行时间')
                    if Path(console).stat().st_size > log_limit:
                        raise RuntimeError('步骤输出超过报告额度')
                result['exit_code'] = win32process.GetExitCodeProcess(process)
                if result['exit_code']:
                    raise subprocess.CalledProcessError(result['exit_code'], command)
                result['state'] = 'completed'
            finally:
                win32job.TerminateJobObject(job, 1)
                # Assign失败时主进程仍为挂起状态，也必须明确终止。
                if win32process.GetExitCodeProcess(process) == win32con.STILL_ACTIVE:
                    win32api.TerminateProcess(process, 1)
                win32event.WaitForSingleObject(process, 15000)
                deadline = monotonic() + 15
                while win32job.QueryInformationJobObject(
                        job, win32job.JobObjectBasicAccountingInformation)['ActiveProcesses']:
                    if monotonic() >= deadline:
                        raise RuntimeError('Job仍有活动worker，保留现场')
                    sleep(.1)
                result.update(all_workers_exited=True, wall_seconds=monotonic()-started)
                usage = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
                result['peak_job_memory_bytes'] = usage['PeakJobMemoryUsed']
                win32api.CloseHandle(thread)
                win32api.CloseHandle(process)
    finally:
        win32api.CloseHandle(job)
        if result['state'] == 'running':
            result['state'] = 'failed'
        if receipt_path is not None:
            write_json(receipt_path, result)
    if Path(console).stat().st_size > log_limit:
        raise RuntimeError('步骤输出超过报告额度')
    return result
