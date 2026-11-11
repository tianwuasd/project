import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import uuid

from .contracts import sha256_file
from .resources import snapshot
from .storage_paths import assert_output_location


def _checked_path(value):
    path = Path(value).expanduser().absolute()
    if '..' in path.parts:
        raise ValueError(f'Unsafe parent traversal: {path}')
    for component in (path, *path.parents):
        if component.is_symlink() or (component.exists() and
                getattr(component.lstat(), 'st_file_attributes', 0) & 0x400):
            raise ValueError(f'Linked workspace path is forbidden: {component}')
    return path.resolve()


class ShortWorkspace:
    def __init__(self, output_root, protected_paths=(), prefix='short'):
        assert_output_location(output_root)
        self.output_root = _checked_path(output_root)
        self.code_root = Path(__file__).resolve().parents[2]
        if self.output_root == Path(self.output_root.anchor):
            raise ValueError('A filesystem root cannot hold a short test')
        for value in (self.code_root, *protected_paths):
            protected = _checked_path(value)
            if (protected == self.output_root or protected in self.output_root.parents
                    or self.output_root in protected.parents):
                raise ValueError(f'Short-test workspace overlaps protected input: {protected}')
        if not prefix or not all(c.isalnum() or c in '-_' for c in prefix):
            raise ValueError('Unsafe short-test prefix')
        self.run_root = self.output_root / (prefix + '-' + time.strftime('%y%m%d%H%M%S') + '-' + uuid.uuid4().hex[:8])
        self.logs = self.run_root
        self.scratch = self.run_root / '.scratch'
        self.started = time.monotonic()

    def record(self, event, **fields):
        with (self.logs / 'acceptance.log').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                                         event=event, **fields), ensure_ascii=False,
                                    allow_nan=False, default=str) + '\n')

    def __enter__(self):
        self.run_root.mkdir(parents=True, exist_ok=False)
        self.scratch.mkdir()
        self._scratch_identity = self.scratch.stat().st_ino
        entered=False
        try:
            self.record('start', python=sys.executable, argv=sys.argv,
                        code_version={str(path.relative_to(self.code_root)): sha256_file(path)
                                      for folder in ('src', 'scripts')
                                      for path in sorted((self.code_root / folder).rglob('*.py'))},
                        capacity=snapshot(),disk_free_bytes=shutil.disk_usage(self.run_root).free)
            entered=True
        finally:
            if not entered:
                shutil.rmtree(self.scratch)
        return self

    def retain_json_reports(self):
        names = {'acceptance.json', 'report.json', 'checks.json', 'numerical.json',
                 'recovered.json', 'synthetic-validation.json', 'complete.json',
                 'summary.json', 'resource_plan.json', 'provenance.json', 'request.json', 'status.json'}
        for path in sorted(self.scratch.rglob('*.json')):
            if (path.name in names or path.name in {'capacity.json','result.json','inference.json'}
                    or path.name.startswith('rank-') or path.name.endswith('-admission.json')) and not path.is_symlink():
                self.record('report', path=str(path.relative_to(self.scratch)),
                            contents=json.loads(path.read_text(encoding='utf-8-sig')))
        for path in sorted(self.scratch.rglob('*')):
            if path.suffix in ('.log', '.jsonl') and path.is_file() and not path.is_symlink():
                target=self.logs/('diagnostic-'+str(path.relative_to(self.scratch)).replace('\\','__').replace('/','__')+'.log')
                shutil.copyfile(path,target)

    def __exit__(self, exc_type, exc, tb):
        try:
            self.retain_json_reports()
        finally:
            current_type,current_error,current_tb=sys.exc_info()
            try:
                # 仅移除本次创建且身份未变化的目录；外部输入从未成为清理目标。
                if (_checked_path(self.scratch) != self.scratch or
                        self.scratch.stat().st_ino != self._scratch_identity):
                    raise ValueError('Owned short-test scratch identity changed')
                shutil.rmtree(self.scratch)
            finally:
                failure=sys.exc_info()
                if failure[0] is None:
                    failure=(current_type,current_error,current_tb) if current_type else (exc_type,exc,tb)
                self.record('finish',passed=failure[0] is None,seconds=time.monotonic()-self.started,
                            error=None if failure[0] is None else ''.join(traceback.format_exception(*failure)))
        return False
