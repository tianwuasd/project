from task_runtime import latest, active_processes


def job_active(job):
    return bool(job and active_processes(job))
