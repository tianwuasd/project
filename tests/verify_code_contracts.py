import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT/'src/chd_segmentation'


def git_bytes(revision, path):
    return subprocess.check_output(['git', '-C', str(ROOT), 'show', revision+':'+path])


def package_snapshot(revision=None):
    if revision is None:
        names = [p.relative_to(PACKAGE).as_posix() for p in sorted(PACKAGE.rglob('*.py'))]
        content = {name:(PACKAGE/name).read_bytes() for name in names}
    else:
        names = subprocess.check_output(['git', '-C', str(ROOT), 'ls-tree', '-r', '--name-only',
            revision, '--', 'src/chd_segmentation']).decode().splitlines()
        content = {name.removeprefix('src/chd_segmentation/'):git_bytes(revision, name)
                   for name in names if name.endswith('.py')}
    aggregate = hashlib.sha256()
    for name, value in sorted(content.items()):
        aggregate.update(name.encode())
        aggregate.update(value)
    return dict(code_identity=aggregate.hexdigest(),
        files={name:hashlib.sha256(value).hexdigest() for name, value in sorted(content.items())})


def run(args):
    current = package_snapshot()
    upstream = package_snapshot('7b50f5f7f64f9b3bd47b43fed6ea8e173dc649a8')
    frozen = package_snapshot('69289532a3da2c94bc72769f5726ca260f6980a7')
    if args.inspect:
        print(json.dumps(dict(current=current, upstream=upstream, frozen=frozen)))
        return
    sys.path.insert(0, str(ROOT/'src'))
    from chd_segmentation.artifacts.store import digest, read_json
    from chd_segmentation.compatibility import compatible_code
    assert compatible_code(upstream['code_identity'], current['code_identity'])
    assert not compatible_code('0'*64, current['code_identity'])
    audit = read_json(PACKAGE/'models/code_compatibility.json')
    target = digest(current['files'])
    assert any(edge['target']==target and edge['source_files']==frozen['files'] and
               set(edge['models'])=={'q_route1', 'medsam2'} for edge in audit['edges'])
    assert all(edge['source_files']!={'unknown':'0'*64} for edge in audit['edges'])
    print(json.dumps(dict(state='passed', upstream=upstream['code_identity'],
        frozen=frozen['code_identity'], current=current['code_identity'], model_target=target,
        unknown_rejected=True, CT_voxels_read=0, optimizer_updates=0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inspect', action='store_true')
    run(parser.parse_args())
