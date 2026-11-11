from ..contracts import digest
from ..xu_resume_compat import source_compatible


# 已审查的旧短测版本；完整源码映射和当前执行映射同时固定。
LEGACY = {
    ('xu-model-run-1','3ba5d6706798e9ba1f363e7ee51535e35f668734bd8bfa8824843d009d73129f'): 'e1aa831',
    ('xu-model-run-2','6fd418b6974ced498f4461094865d9fd9bb71582a0667c05bae9e9ed94bf6811'): '7ab17d9',
}
CURRENT_TECHNICAL = '50c1204cf674b59a1cc25961f64c00f1f7745995d63878d8761981140977173e'


def source_schema(record, current):
    if record['protocol'] == 'xu-model-run-4' and source_compatible(record['implementation'],current):
        return 'xu-stage-2'
    key = record['protocol'],digest(record['implementation'])
    if (record['scope'] == 'technical_subset' and key in LEGACY
            and digest(current) == CURRENT_TECHNICAL):
        return 'xu-stage-1-alias' if record['protocol']=='xu-model-run-1' else 'xu-stage-1'
    raise ValueError('Training implementation has no reviewed prediction compatibility mapping')


def receipt_specs(stage, contract_id, schema):
    if schema == 'xu-stage-2':
        from ..xu_pipeline import _stage_spec
        return {_stage_spec(stage,contract_id)}
    if schema not in ('xu-stage-1','xu-stage-1-alias'):
        raise ValueError('Unknown training receipt schema')
    argv = list(map(str,stage.argv))
    def spec(values):
        return digest(dict(contract=contract_id,name=stage.name,argv=values,
                           required=list(stage.required),skip=stage.skip_reason))
    values = {spec(argv)}
    # e1aa831 保存规范设备身份，但收据保留原命令的 cuda 别名。
    if schema=='xu-stage-1-alias' and '--device' in argv:
        index = argv.index('--device')+1
        if argv[index]=='cuda:0':
            argv[index] = 'cuda'
            values.add(spec(argv))
    return values
