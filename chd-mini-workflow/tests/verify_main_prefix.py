import ast
import importlib.abc
import importlib.util
from pathlib import Path
import subprocess
import sys
import tomllib

sys.dont_write_bytecode = True
REPO = Path(sys.argv[1]).resolve(strict=True)
COMMIT = sys.argv[2]
PREFIX = 'chd-mini-workflow'


def git(*args):
    return subprocess.check_output(['git','-C',str(REPO),*args])


def source(path):
    return git('show',f'{COMMIT}:{path}').decode('utf-8')


FILES = set(git('ls-tree','-r','--name-only',COMMIT).decode('utf-8').splitlines())
PACKAGES = {'chd_mini': PREFIX, 'chd_segmentation': 'chd-segmentation'}


class GitSource(importlib.abc.SourceLoader):
    def __init__(self, filename):
        self.filename = filename

    def get_filename(self, fullname):
        return str(REPO / self.filename)

    def get_data(self, path):
        return git('show',f'{COMMIT}:{Path(path).relative_to(REPO).as_posix()}')

    def get_code(self, fullname):
        return compile(source(self.filename),self.get_filename(fullname),'exec')


class GitPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        package = fullname.split('.')[0]
        if package not in PACKAGES:
            return None
        project = PACKAGES[package]
        if path is None and str(REPO/project/'src') not in sys.path:
            return None
        stem = project+'/src/'+fullname.replace('.','/')
        candidates = (stem+'/__init__.py',stem+'.py')
        for candidate in candidates:
            if candidate in FILES:
                loader = GitSource(candidate)
                return importlib.util.spec_from_file_location(fullname,loader.get_filename(fullname),loader=loader)
        raise ModuleNotFoundError(fullname)


def main():
    if len(sys.argv) > 3:
        action = sys.argv[3]
        sys.meta_path.insert(0,GitPackages())
        if action == 'imports':
            for package, project in PACKAGES.items():
                sys.path.insert(0,str(REPO/project/'src'))
                module = __import__(package+'.cli',fromlist=['main'])
                assert module.__file__ == str(REPO/project/'src'/package/'cli.py')
            print('相邻工具和生产CLI真实模块导入通过')
            return
        entry = PREFIX+'/run.py'
        sys.argv = [str(REPO/entry),*(['review','--help'] if action == 'review' else ['--help'])]
        exec(compile(source(entry),str(REPO/entry),'exec'),{'__name__':'__main__','__file__':str(REPO/entry)})
        return
    files = sorted(p for p in FILES if p.startswith(PREFIX+'/'))
    assert files and all(not any(part in ('.runs','__pycache__') for part in Path(p).parts) for p in files)
    assert not any(p.endswith('.local.json') for p in files)
    assert tomllib.loads(source(PREFIX+'/pyproject.toml'))['project']['name'] == PREFIX
    for path in files:
        if path.endswith('.py'):
            ast.parse(source(path),filename=path)
    for action in ('help','review','imports'):
        result = subprocess.run([sys.executable,'-B',__file__,str(REPO),COMMIT,action],check=True,capture_output=True,text=True,encoding='utf-8')
        assert ('--handoff' in result.stdout if action == 'review' else
                'review' in result.stdout if action == 'help' else '导入通过' in result.stdout)
        print(action+': 通过')
    print(f'Git候选提交实际源码检查通过：{len(files)}文件；未检出副本，未写入字节码或运行产物')


if __name__ == '__main__':
    main()
