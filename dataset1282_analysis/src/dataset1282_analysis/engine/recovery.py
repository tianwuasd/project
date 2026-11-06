from pathlib import Path
from ..contracts import atomic_json,read_json,sha256_file


def commit_result(path,result):
    path=Path(path)
    atomic_json(path,result)
    atomic_json(path.with_suffix('.complete.json'),{'result_sha256':sha256_file(path),'contract':result['contract']})


def reusable_result(path,contract):
    path=Path(path); marker=path.with_suffix('.complete.json')
    try:
        evidence=read_json(marker)
        if evidence['contract']!=contract or evidence['result_sha256']!=sha256_file(path): return False
        result=read_json(path)
        if result['contract']!=contract or result['status']!='complete': return False
        if set(result.get('source_sha256',{}))!={'image','label'}: return False
        return all(sha256_file(result['input_paths'][kind])==value for kind,value in result['source_sha256'].items())
    except (OSError,ValueError,KeyError): return False
