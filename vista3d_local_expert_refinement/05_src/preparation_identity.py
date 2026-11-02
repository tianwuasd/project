"""Strict, shared identity for project-owned preprocessing producers."""
from pathlib import Path
from server_data import digest

FILES=('server_prepare.py','data_pipeline.py','prepare_nnunet.py','case_execution.py','read_admission.py','case_memory.py','cpu_case_queue.py')


def current(directory=None):
    root=Path(directory) if directory is not None else Path(__file__).parent
    return {name:digest(root/name) for name in FILES}


def compatible(recorded, expected=None, directory=None):
    """Exact reviewed producer-map edges; all other cache fields stay strict."""
    import hashlib
    import json
    root=Path(directory) if directory is not None else Path(__file__).parent
    expected=current(root) if expected is None else expected
    if not isinstance(recorded,dict) or not isinstance(expected,dict):return False
    if recorded==expected:return True
    def fingerprint(mapping):
        return hashlib.sha256(json.dumps(mapping,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    try:
        audit=json.loads((root.parent/'06_configs/preparation_code_compatibility.json').read_text(encoding='utf-8'))
        if audit.get('schema_version')!=1 or audit.get('scope')!='preprocessing-preserving-scheduling-upgrade':return False
        edges=audit['transitions'];target=fingerprint(expected);pending=[fingerprint(recorded)];seen=set()
        if not isinstance(edges,list):return False
        while pending:
            source=pending.pop()
            if source in seen:continue
            seen.add(source)
            for edge in edges:
                if edge['source']!=source:continue
                destination=edge['target']
                if destination==target:return True
                pending.append(destination)
    except (OSError,ValueError,KeyError,TypeError):return False
    return False
