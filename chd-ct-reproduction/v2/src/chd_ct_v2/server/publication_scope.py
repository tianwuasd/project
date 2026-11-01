"""One CHD project publication scope, with two isolated runtime versions."""
from pathlib import PurePosixPath

PROJECT_ROOT='chd-ct-reproduction'
CANONICAL=PROJECT_ROOT+'/v2'
LEGACY='chd-ct-panel-v2'
PROJECTS=(CANONICAL,LEGACY)


def relative(name):
    if '\\' in name or any(p in {'.','..'} for p in name.split('/')):
        return None
    for prefix in (*PROJECTS,PROJECT_ROOT):
        if name.startswith(prefix+'/'):
            return PurePosixPath(name).parts[len(PurePosixPath(prefix).parts):]
    return None


def permits_update(name,subtree):
    if relative(name) is None:return False
    prefix=PROJECT_ROOT if subtree==CANONICAL else subtree
    return name.startswith(prefix+'/')
