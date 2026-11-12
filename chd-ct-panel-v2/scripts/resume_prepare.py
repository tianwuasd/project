import json
from pathlib import Path
import shutil
import sys
import tempfile

# 该进程只导入已核验的冻结生产者，来源签名和缓存合同继续使用原记录。
code = Path(sys.argv[1]).resolve(strict=True)
options = json.loads(sys.argv[2])
output = Path(options['output']).resolve(strict=True)
# 生产者会重新写出相同的源码快照，独立副本保证复制的源和目标不同。
with tempfile.TemporaryDirectory(prefix='.prepare-resume-', dir=output.parent) as temporary:
    frozen = Path(temporary)/'code'
    shutil.copytree(code, frozen)
    sys.path.insert(0, str(frozen/'src'))
    from chd_stunet.preprocessing.run import run_prepare
    result = run_prepare(**options)
    print(json.dumps(result, ensure_ascii=False), flush=True)
