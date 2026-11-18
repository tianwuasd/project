import ast
from pathlib import Path


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


if __name__ == '__main__':
    main()
