"""One-time small download for older panels; no checkout merge or frozen edits."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time


def bootstrap(code, runtime):
    code,runtime=Path(code).expanduser().resolve(),Path(runtime).expanduser().resolve()
    if not (code/'05_src/task_update.py').is_file() or not runtime.is_dir():
        raise ValueError('请指定已有VISTA代码目录和原runtime；不会新建另一套运行目录')
    sys.path.insert(0,str(code/'05_src'))
    import task_update as old
    from task_runtime import runtime_lock
    from panel_publication import network
    with runtime_lock(runtime):pass  # Refuse a still-running old update or worker.
    output=runtime/'upgrade_bootstrap'/str(time.time_ns())
    cache=output/'bootstrap.git';cache.mkdir(parents=True)
    with network.use(network.load(runtime)):
        route=network.routes(network.current())[0]
        old.git(cache,'init','--bare')
        if old.git(cache,'ls-remote','--get-url',route).stdout!=route:
            raise ValueError('官方URL被Git配置重写，未发送认证信息')
        old.git(cache,'config','remote.origin.url',route)
        old.git(cache,'config','remote.origin.promisor','true')
        old.git(cache,'config','remote.origin.partialclonefilter','blob:none')
        known=old.git(cache,'ls-remote',route,'refs/heads/main',timeout=30).stdout.split()[0]
        print('一次性引导：只获取版本目录信息和新版更新器文件，原Git检出不合并。',flush=True)
        old.git(cache,'fetch','--no-tags','--progress','--filter=blob:none','origin','refs/heads/main',
                timeout=180,progress_log=output/'metadata.log')
        commit=old.git(cache,'rev-parse','FETCH_HEAD').stdout
        if commit!=known:raise ValueError('引导期间官方main变化，请重新执行')
        relative='vista3d_local_expert_refinement/05_src/task_update.py'
        row=old.git(cache,'ls-tree',commit,'--',relative).stdout
        metadata,name=row.split('\t',1);mode,kind,oid=metadata.split()
        if name!=relative or kind!='blob' or mode not in {'100644','100755'} or not re.fullmatch('[0-9a-f]{40}',oid):
            raise ValueError('新版更新器对象清单不符')
        old.git(cache,'fetch','--no-tags','--no-write-fetch-head','--progress','origin',oid,
                timeout=180,progress_log=output/'updater-object.log')
        data=old.git(cache,'cat-file','blob',oid,binary=True,timeout=30).stdout
        module=output/'task_update.py';module.write_bytes(data)
        receipt=dict(commit=commit,git_blob=oid,sha256=hashlib.sha256(data).hexdigest(),
                     original_code=str(code),runtime=str(runtime),checkout_changed=False)
        (output/'receipt.json').write_bytes((json.dumps(receipt,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
        namespace={'__name__':'vista_verified_update_bootstrap','__file__':str(module)}
        exec(compile(data,str(module),'exec'),namespace)
        namespace['update'](runtime)
    print('小下载升级完成。重新打开原面板，核对版本；没有自动恢复或启动任务。',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code',required=True,type=Path)
    parser.add_argument('--runtime',required=True,type=Path)
    args=parser.parse_args()
    bootstrap(args.code,args.runtime)
