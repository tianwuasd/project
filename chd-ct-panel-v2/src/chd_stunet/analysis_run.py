"""Weighted bounded case execution; no source-data writes, GPU, or training imports."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import ast
import hashlib
import shutil
from pathlib import Path
import importlib.metadata
import json
import multiprocessing as mp
import os
import platform
import sys
import time
import traceback

import nibabel as nib
import numpy as np
import psutil

from .acceleration.cpu import estimate_case_bytes,live_budget
from .data.analysis import VERSION,analyze_pair,sha256_file
from .data.discovery import discover_pairs,descriptor_evidence
from .observability.runlog import RunLog,write_json

def semantic_source_hash(path):
    """Hash behavior-bearing AST; ignore comments and leading docstrings."""
    tree=ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        body=getattr(node,"body",None)
        if (isinstance(body,list) and body and isinstance(body[0],ast.Expr)
                and isinstance(body[0].value,ast.Constant) and isinstance(body[0].value.value,str)):
            node.body=body[1:]
    return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()


READ_GUARD=None
THREAD_CONTROLLER=None


def initialize_worker(semaphore):
    global READ_GUARD,THREAD_CONTROLLER
    READ_GUARD=semaphore
    from threadpoolctl import threadpool_limits
    THREAD_CONTROLLER=threadpool_limits(limits=1)


def worker(task):
    pair,extra,snapshot,components,selected=task
    try:
        row=analyze_pair(pair["image"],pair["label"],case_id=pair["case_id"],
            expected_hashes=extra.get("expected_hashes"),author_spacing=extra.get("author_spacing"),
            components=components,selected_components=selected,snapshot_path=snapshot,read_guard=READ_GUARD)
        row.update(previous_status=extra.get("previous_status"),roles=extra.get("roles",{}),worker_pid=os.getpid())
        return row
    except Exception as error:
        return dict(case_id=pair["case_id"],analysis_success=False,error=str(error),
                    traceback=traceback.format_exc(),worker_pid=os.getpid())


def run_analysis(raw_root,output,*,descriptor=None,workers=None,cap=4,readers=2,
                 components="selected",component_cases=(),snapshot_cases=(),limit=None,
                 selected_cases=(),resume=False):
    """Analyze discovered pairs with bounded weighted RAM and separate I/O slots."""
    raw_root,output=Path(raw_root).resolve(strict=True),Path(output).resolve()
    if not raw_root.is_dir():
        raise ValueError("Raw root is not a directory")
    if output==raw_root or output.is_relative_to(raw_root) or raw_root.is_relative_to(output):
        raise ValueError("Analysis output and original data must be separate")
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("Output is not empty; choose a fresh directory or explicit --resume")
    output.mkdir(parents=True,exist_ok=True)

    discovery=discover_pairs(raw_root)
    pairs=discovery["pairs"]
    if selected_cases:
        wanted=set(selected_cases)
        missing=wanted-{p["case_id"] for p in pairs}
        if missing: raise ValueError("Unknown selected cases: "+str(sorted(missing)))
        pairs=[p for p in pairs if p["case_id"] in wanted]
    if limit is not None: pairs=pairs[:limit]
    if not pairs: raise ValueError("No image/label pairs found")
    evidence,provenance=descriptor_evidence(descriptor,raw_root)
    package=Path(__file__).resolve().parent
    semantic={name:semantic_source_hash(package/name) for name in ("data/analysis.py","data/discovery.py")}

    signature=dict(version=VERSION,semantic_producers=semantic,raw_root=str(raw_root),case_ids=[p["case_id"] for p in pairs],
        descriptor_sha256=provenance.get("sha256"),components=components,
        component_cases=sorted(component_cases),snapshot_cases=sorted(snapshot_cases))
    marker=output/"signature.json"
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8"))!=signature:
        raise ValueError("Resume contract differs; preserve this run and choose a new output")
    # Validate old identities before writing logs, status, or code evidence.
    cached={}
    if resume:
        for pair in pairs:
            previous=output/"cases"/(pair["case_id"]+".json")
            if not previous.is_file(): continue
            try: old=json.loads(previous.read_text(encoding="utf-8"))
            except (OSError,ValueError): continue
            if old.get("analysis_success"):
                if (sha256_file(pair["image"])!=old.get("image_sha256")
                        or sha256_file(pair["label"])!=old.get("label_sha256")):
                    raise ValueError("Source changed since this run; preserve it and choose a new output")
                cached[pair["case_id"]]=dict(old,analysis_reused_this_attempt=True)
        if marker.is_file():
            history=output/"history"/("before_resume_"+str(time.time_ns()))
            history.mkdir(parents=True)
            for name in ("code_provenance.json","resources.json","environment.json","summary.json",
                         "report.md","resolved_config.json","status.json"):
                source=output/name
                if source.is_file(): shutil.copy2(source,history/name)
            if (output/"code").is_dir():
                shutil.copytree(output/"code",history/"code")
    log=RunLog(output); log.event("checking","开始本地只读分析；原始数据不改写")
    log.status("checking")
    source_map={}
    for source in [*package.rglob("*.py"),package.parents[1]/"configs/analysis.json"]:
        relative=source.relative_to(package.parents[1])
        target=output/"code"/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        source_map[relative.as_posix()]=sha256_file(target)
    write_json(output/"code_provenance.json",dict(source_sha256=source_map,semantic_producers=semantic))
    write_json(marker,signature)
    write_json(output/"discovery.json",discovery)
    write_json(output/"descriptor_evidence.json",provenance)
    rows=[];pending=[]
    for pair in pairs:
        estimate=256*1024**2
        try: estimate=estimate_case_bytes(int(np.prod(nib.load(pair["image"]).shape)))
        except Exception: pass
        if pair["case_id"] in cached:
            rows.append(cached[pair["case_id"]])
            log.event("reuse","复用同合同且源SHA未变："+pair["case_id"])
            continue
        pending.append((pair,estimate))
    budget=live_budget(min((n for _,n in pending),default=1),workers,cap)
    budget.update(read_concurrency=min(readers,max(1,budget["max_workers"])),components_policy=components,
        limitation="Host Windows admission only; Linux cgroup/CUDA not tested")
    write_json(output/"resources.json",budget)
    if pending and budget["max_workers"]<1:
        log.status("blocked",reason="Insufficient host memory under reserve policy")
        raise MemoryError("Insufficient host RAM after reserve")
    write_json(output/"environment.json",dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
        packages={n:importlib.metadata.version(n) for n in ("numpy","scipy","nibabel","psutil","matplotlib","threadpoolctl")}))
    write_json(output/"resolved_config.json",dict(signature,output=str(output),
        workers_requested=workers or "auto",workers_applied=budget["max_workers"]))
    log.event("admission",f"配对{len(pairs)}项；最大计算并发{budget['max_workers']}，读取并发{budget['read_concurrency']}，每进程1线程")
    started=time.perf_counter();ctx=mp.get_context("spawn")
    semaphore=ctx.BoundedSemaphore(budget["read_concurrency"])
    active={};peak_rss=0;last_heartbeat=0;peak_inflight=0;producing_pids=set()
    # Preserve caller's environment; changes apply only to this already-isolated task.
    names=("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")
    prior={name:os.environ.get(name) for name in names}
    for name in names: os.environ[name]="1"
    try:
        with ProcessPoolExecutor(max_workers=max(1,budget["max_workers"]),mp_context=ctx,
                                 initializer=initialize_worker,initargs=(semaphore,)) as pool:
            while pending or active:
                reserved=sum(n for _,n in active.values())
                admitted=False
                for item in list(pending):
                    pair,estimate=item
                    headroom=psutil.virtual_memory().available-budget["reserve_bytes"]
                    if (len(active)<budget["max_workers"] and reserved+estimate<=budget["memory_budget_bytes"]
                            and estimate<=headroom):
                        cid=pair["case_id"]
                        snapshot=str(output/"previews"/(cid+".npz")) if cid in snapshot_cases else None
                        task=(pair,evidence.get(cid,{}),snapshot,
                              components=="all" or (components=="selected" and cid in component_cases),components=="selected")
                        future=pool.submit(worker,task);active[future]=item;pending.remove(item)
                        reserved+=estimate;admitted=True;peak_inflight=max(peak_inflight,len(active))
                        log.event("case_started","开始："+cid,estimated_bytes=estimate)
                if not active:
                    if pending:
                        log.status("blocked",reason="No pending case fits current RAM budget")
                        raise MemoryError("No pending case fits current RAM budget")
                    break
                done,_=wait(active,timeout=.5,return_when=FIRST_COMPLETED)
                for future in done:
                    pair,_=active.pop(future)
                    try: row=future.result()
                    except Exception as error:
                        row=dict(case_id=pair["case_id"],analysis_success=False,error=str(error),traceback=traceback.format_exc())
                    rows.append(row);producing_pids.add(row.get("worker_pid"))
                    write_json(output/"cases"/(pair["case_id"]+".json"),row)
                    if row.get("traceback"):
                        (output/"logs"/(pair["case_id"]+".error.log")).write_text(row["traceback"],encoding="utf-8")
                    log.event("case_completed",f"完成{len(rows)}/{len(pairs)}：{pair['case_id']}；"
                        +("已分析" if row.get("analysis_success") else "失败"),
                        flags=row.get("flags",[]),case_seconds=row.get("elapsed_seconds"))
                if time.perf_counter()-last_heartbeat>=5:
                    process=psutil.Process()
                    members=[process,*process.children(recursive=True)]
                    rss=sum(p.memory_info().rss for p in members if p.is_running())
                    peak_rss=max(peak_rss,rss)
                    log.status("analyzing",completed=len(rows),total=len(pairs),
                        active=[pair["case_id"] for pair,_ in active.values()],
                        queued=len(pending),rss_bytes=rss,available_bytes=psutil.virtual_memory().available)
                    log.event("heartbeat",f"进度{len(rows)}/{len(pairs)}；活动{len(active)}；"
                        f"本组RSS约{rss/1024**3:.2f}GiB",active=len(active))
                    last_heartbeat=time.perf_counter()
    except BaseException as error:
        log.status("cancelled" if isinstance(error,KeyboardInterrupt) else "failed",reason=str(error),completed=len(rows))
        raise
    finally:
        for name,value in prior.items():
            if value is None: os.environ.pop(name,None)
            else: os.environ[name]=value
    rows.sort(key=lambda r:r["case_id"])
    budget.update(peak_sampled_group_rss_bytes=peak_rss,peak_inflight=peak_inflight,
        producing_worker_pids=sorted(p for p in producing_pids if p),reused_pairs=len(cached),analysis_wall_seconds=time.perf_counter()-started)
    write_json(output/"resources.json",budget)
    log.status("reporting",completed=len(rows),total=len(pairs))
    from .reporting import build_report
    summary=build_report(rows,output,discovery,provenance,budget)
    state="completed" if summary["failed_pairs"]==0 else "completed_with_errors"
    log.status(state,completed=len(rows),total=len(pairs),report=str(output/"report.md"))
    log.event("completed","分析完成；报告："+str(output/"report.md"))
    return summary