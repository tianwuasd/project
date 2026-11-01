"""Strict, shared identity for project-owned preprocessing producers."""
from pathlib import Path
from server_data import digest

FILES=('server_prepare.py','data_pipeline.py','prepare_nnunet.py','case_execution.py','read_admission.py','case_memory.py','cpu_case_queue.py','imagechd_geometry.py','imagechd_cohort.py')


def current(directory=None):
    root=Path(directory) if directory is not None else Path(__file__).parent
    return {name:digest(root/name) for name in FILES}
