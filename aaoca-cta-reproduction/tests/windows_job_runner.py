import os
import subprocess
import time

import msvcrt
import win32api
import win32con
import win32event
import win32job
import win32process


class WindowsJobRunner:
    def __init__(self, command, log, cwd, environment, memory_limit):
        self.job = win32job.CreateJobObject(None, "")
        limits = win32job.QueryInformationJobObject(self.job, win32job.JobObjectExtendedLimitInformation)
        limits["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | win32job.JOB_OBJECT_LIMIT_JOB_MEMORY
        limits["JobMemoryLimit"] = memory_limit
        win32job.SetInformationJobObject(self.job, win32job.JobObjectExtendedLimitInformation, limits)
        current = win32api.GetCurrentProcess()
        self.output = win32api.DuplicateHandle(current, msvcrt.get_osfhandle(log.fileno()), current,
                                              0, True, win32con.DUPLICATE_SAME_ACCESS)
        self.null = open(os.devnull, "rb")
        self.input = win32api.DuplicateHandle(current, msvcrt.get_osfhandle(self.null.fileno()), current,
                                             0, True, win32con.DUPLICATE_SAME_ACCESS)
        startup = win32process.STARTUPINFO()
        startup.dwFlags |= win32con.STARTF_USESTDHANDLES
        startup.hStdInput = self.input
        startup.hStdOutput = startup.hStdError = self.output
        flags = win32con.CREATE_SUSPENDED | win32con.CREATE_NEW_PROCESS_GROUP
        self.process, self.thread, self.pid, _ = win32process.CreateProcess(
            None, subprocess.list2cmdline(command), None, None, True, flags, environment, str(cwd), startup)
        win32job.AssignProcessToJobObject(self.job, self.process)
        # 先纳入Job再恢复启动，后代进程自动继承同一内存和退出保护。
        win32process.ResumeThread(self.thread)

    def poll(self):
        code = win32process.GetExitCodeProcess(self.process)
        return None if code == win32con.STILL_ACTIVE else code

    def memory(self):
        value = win32job.QueryInformationJobObject(self.job, win32job.JobObjectExtendedLimitInformation)
        return int(value["PeakJobMemoryUsed"])

    def active(self):
        return win32job.QueryInformationJobObject(self.job, win32job.JobObjectBasicAccountingInformation)["ActiveProcesses"]

    def kill(self):
        win32job.TerminateJobObject(self.job, 1)

    def wait(self):
        win32event.WaitForSingleObject(self.process, win32event.INFINITE)
        started = time.monotonic()
        while self.active():
            if time.monotonic() - started > 5:
                self.kill()
                raise RuntimeError("测试进程组未在退出后及时清空")
            time.sleep(.01)
        return self.poll()

    def close(self):
        if self.active():
            self.kill()
            self.wait()
        for handle in (self.thread, self.process, self.output, self.input, self.job):
            handle.Close()
        self.null.close()
