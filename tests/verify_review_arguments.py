import argparse
import ast
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from chd_mini.review import prediction_args, evaluation_args


def production_parser(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    wait = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'wait_seconds')
    body, collecting = [], False
    for node in main.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'parser' for t in node.targets):
            collecting = True
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'args' for t in node.targets):
            break
        if collecting:
            body.append(node)
    main.body = body + [ast.Return(value=ast.Name(id='parser', ctx=ast.Load()))]
    selected = ast.fix_missing_locations(ast.Module(body=[wait, main], type_ignores=[]))
    # 执行生产文件原有argparse注册代码，仅返回解析器；不进入业务分派。
    namespace = dict(argparse=argparse, __version__='parameter-check')
    exec(compile(selected, str(path), 'exec'), namespace)
    return namespace['main']()


def main():
    root = Path(sys.argv[1]).resolve(strict=True)
    parsers = dict(stu=production_parser(root/'src/chd_segmentation/cli.py'),
                   models=production_parser(root/'src/chd_segmentation/models_cli.py'))
    # 字符串仅用于参数解析，不读取配置、输入或模型，也不构造运行回执。
    item = dict(config='CONFIG', prepared='SHARED_PREPARED', evaluation_config='EVALUATION_CONFIG',
                prompt_mode='external', prompts='PROMPTS')
    for model in ('stu', 'q', 'med'):
        for cases in (['ct_1001'], ['ct_1001', 'ct_1012']):
            args = prediction_args(model, item, 'BEST_FILE', 'OUTPUT')
            args += [arg for case in cases for arg in ('--case', case)]
            parsed = parsers['stu' if model == 'stu' else 'models'].parse_args(args if model == 'stu' else args[1:])
            assert parsed.case == cases
            assert parsed.checkpoint == ('best' if model == 'stu' else 'BEST_FILE')
            assert (parsed.preprocessed_source if model == 'stu' else parsed.prepared) == item['prepared']
        evaluation = parsers['stu'].parse_args(evaluation_args(item,'PREDICTIONS','METRICS.json'))
        assert evaluation.command == 'evaluate' and evaluation.predictions == 'PREDICTIONS'
        assert evaluation.prepared == item['prepared'] and evaluation.config == item['evaluation_config']
    print('PRODUCTION_ARGPARSE_OK: 6 prediction + 3 evaluation parameter sets; zero model execution')


if __name__ == '__main__':
    main()
