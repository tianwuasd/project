from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from chd_mini.contracts import git
from chd_mini.version_review import model_versions


def main():
    engine, source, final = sys.argv[1:4]
    actual = sorted(p for p in git(engine,'diff','--no-renames','--name-only','-z',source,final,'--').split('\0') if p)
    if len(sys.argv) > 4:
        mode = sys.argv[4]
        review = dict(production_commit=final, approved_changed_files=actual,
                      secretary_review_reference='控制检查：仅验证真实Git比较，不批准模型交接')
        if mode == 'mismatch':
            review['approved_changed_files'] = actual+['not-an-approved-real-file']
        item = dict(training_commits=[source], version_reviews={} if mode == 'unreviewed' else {source:review})
        model_versions(engine,'control-only',item,final)
        return
    same = model_versions(engine,'control-only',dict(training_commits=[source]),source)
    assert same[0]['training_commit'] == source and not same[0]['changed_files']
    for mode in ('exact','mismatch','unreviewed'):
        child = subprocess.run([sys.executable,'-B',__file__,engine,source,final,mode], capture_output=True)
        assert (child.returncode == 0) == (mode == 'exact')
    print('REAL_GIT_VERSION_CHECK_OK: same SHA; exact list; mismatch rejected; missing review rejected')
    print('actual_changed_files', actual)


if __name__ == '__main__':
    main()
