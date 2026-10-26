# 旧更新器的一次性小下载升级

适用于现有VISTA 0.3.3—0.3.6的旧入口，包括全量下载和缺失promisor对象检查失败。新版本为0.3.7。先Ctrl+C取消旧更新并退出旧面板，确认该原runtime没有正在执行的任务；下面会检查运行锁，忙时拒绝，不杀进程。

在服务器一次复制执行整个代码块。沿用原runtime/network.json内已有的官方认证、代理和密钥路径；不需要提供密钥内容。它先只取版本/目录信息和新更新器这一个文件，再由新更新器按需下载运行文件。原Git检出不merge、不改其他项目，不修改旧受管源码/清单，不安装环境、不恢复训练。

```bash
cd /data5/zhougaowei/zhangruichen_workspace/chd_ct_code/project/vista3d_local_expert_refinement
python3 - --code "$PWD" --runtime /data5/zhougaowei/zhangruichen_workspace/vista3d_runtime <<'PY'
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
PY
bash start_server.sh --runtime /data5/zhougaowei/zhangruichen_workspace/vista3d_runtime
```

看到面板v0.3.7后正常使用u；以后不必重复这段引导。新建task4仍需完成预览并输入RUN，确认“已提交”和唯一目录，再用s/h查询。旧任务的r继续使用其冻结源码，不能把更新当作原地修复旧快照。

引导的日志与来源摘要在原runtime/upgrade_bootstrap/<编号>，常规更新备份与fetch.log在update_backups/<编号>。引导阶段沿用旧Git助手，版本读取30秒、元信息和单文件下载各180秒；新更新器下载阶段总限900秒并有120秒无可见进度界限。网络失败保留原版本，先查具体阶段，不反复让旧u下载整个仓库，也不删除缓存或重配已匹配密钥。

新发布只包括运行代码和配置；文献、研究数据库、其他项目与历史文件内容不主动下载。仍需要少量提交/目录元信息来检查版本和快进关系。旧缓存和release不会因本次升级自动瘦身或被删除。实现对照见[更新链路审查](UPDATE_CHAIN_AUDIT.md)，验证边界见[JSON](CODE_ONLY_UPDATE_VALIDATION.json)。

若已运行引导但在lazy fetching disabled处失败，沿用同一终端的f/r，只需重跑`python3 "$f" --code "$PWD" --runtime "$r" && bash start_server.sh --runtime "$r"`。本次只修本地对象枚举，详见[修复说明](PROMISOR_INVENTORY_FIX.md)。
