import ast
import os
from pathlib import Path
import subprocess
import sys


def main():
    path = Path(__file__).resolve().parents[1]/'src/chd_mini/geometry_review.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='source_geometry')
    calls = [n for n in ast.walk(function) if isinstance(n,ast.Call)]
    resolve = next(n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='resolve_preprocessed')
    assert set(k.arg for k in resolve.keywords) == {'dataset','case_ids','storage'}
    storage = next(k.value for k in resolve.keywords if k.arg=='storage')
    assert isinstance(storage,ast.Name) and storage.id=='storage'
    opening = next(n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='open')
    assert any(k.arg=='with_label' and isinstance(k.value,ast.Constant) and k.value.value is False
               for k in opening.keywords)
    reads = [n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='read_json']
    assert len(reads)==1
    assert 'catalog.json' not in ast.unparse(function)
    print('STATIC_SOURCE_BINDING_OK: explicit storage; production open without labels; no direct catalog selection')
    if len(sys.argv) > 1:
        root, commit = sys.argv[1:3]
        assert len(commit)==40 and all(c in '0123456789abcdef' for c in commit)
        source = subprocess.check_output(['git','-C',root,'show',
            commit+':src/chd_segmentation/preprocessing/inputs.py'],text=True,encoding='utf-8',
            env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'})
        production = ast.parse(source)
        resolver = next(n for n in production.body if isinstance(n,ast.FunctionDef) and n.name=='resolve_preprocessed')
        assert [a.arg for a in resolver.args.kwonlyargs] == ['dataset','case_ids','storage']
        offline = next(n for n in production.body if isinstance(n,ast.ClassDef) and n.name=='OfflineDataset')
        opened = next(n for n in offline.body if isinstance(n,ast.FunctionDef) and n.name=='open')
        fields = {k.arg for n in ast.walk(opened) if isinstance(n,ast.Call)
                  and isinstance(n.func,ast.Attribute) and n.func.attr=='update' for k in n.keywords}
        assert {'source_plan','axis_order','processed_spacing','source_directory'} <= fields
        assert any(isinstance(n,ast.keyword) and n.arg=='mmap_mode'
                   and isinstance(n.value,ast.Constant) and n.value.value=='r' for n in ast.walk(opened))
        print('FIXED_PRODUCTION_API_AST_OK',commit)


if __name__ == '__main__':
    main()
