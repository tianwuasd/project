from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from chd_mini.review import review_models


def main():
    # 仅检查交接声明，不提供模型路径，不执行计算或生成生产回执。
    handoff = dict(review_mode='partial', available_models=['stu','q'], models=dict(stu={},q={}),
                   unavailable_models=dict(med='控制检查：缺少训练模型'))
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == 'missing_reason':
            handoff['unavailable_models'] = dict(med='')
        elif mode == 'implicit_skip':
            handoff.pop('review_mode')
        elif mode == 'missing_declaration':
            handoff['unavailable_models'] = {}
        elif mode == 'fake_med':
            handoff['models']['med'] = {}
        review_models(handoff)
        return
    assert review_models(handoff) == ['stu','q']
    complete = dict(models=dict(stu={},q={},med={}))
    assert review_models(complete) == ['stu','q','med']
    for mode in ('missing_reason','implicit_skip','missing_declaration','fake_med'):
        result = subprocess.run([sys.executable,'-B',__file__,mode],capture_output=True)
        assert result.returncode != 0
    print('PARTIAL_SCOPE_OK: explicit STU/Q; complete goal retained; four invalid declarations rejected; zero model execution')


if __name__ == '__main__':
    main()
