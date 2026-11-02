"""Dataset-level CSV, figures, and a Chinese report from native read-only evidence."""
from collections import Counter, defaultdict
import csv
from pathlib import Path

import numpy as np

from .data.analysis import LABELS
from .observability.runlog import write_json


def csv_table(path, rows):
    if not rows:
        Path(path).write_text("",encoding="utf-8-sig");return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def repeated(rows, key):
    groups=defaultdict(list)
    for row in rows:
        if row.get(key): groups[row[key]].append(row["case_id"])
    return [ids for ids in groups.values() if len(ids)>1]


def ranges(values):
    values=np.asarray(values,dtype=float)
    return dict(min=float(values.min()),median=float(np.median(values)),max=float(values.max())) if values.size else None


def make_figures(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    assets=output/"assets";assets.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":10,"figure.dpi":130})
    names=list(LABELS.values())
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    axes[0].bar(names,[sum(r["structures"][n]["voxels"]>0 for r in rows) for n in names])
    axes[0].set(title="Annotation presence",ylabel="Image/label pairs",ylim=(0,max(1,len(rows))*1.12))
    for index,name in enumerate(names):
        n=sum(r["structures"][name]["voxels"]>0 for r in rows)
        axes[0].text(index,n,str(n),ha="center",va="bottom")
    axes[1].hist([r["shape"][2] for r in rows],bins=min(16,max(1,len(rows))),edgecolor="white")
    axes[1].set(title="Native array axis-2 length",xlabel="Voxels",ylabel="Pairs")
    for i,n in enumerate(names):
        vals=[r["structures"][n]["voxels"] for r in rows if r["structures"][n]["voxels"]]
        axes[2].scatter([i]*len(vals),vals,s=10,alpha=.45)
    axes[2].set(xticks=range(7),xticklabels=names,yscale="log",title="Native label sizes",ylabel="Voxels (log)")
    fig.suptitle(f"ImageCHD native-data audit: {len(rows)} pairs; no source edits")
    fig.tight_layout();fig.savefig(assets/"01_dataset_overview.png");plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for i,n in enumerate(names):
        for row in rows:
            stats=row["structures"][n];frac=stats["mode_fraction"]
            if frac is not None and stats["voxels"]:
                flagged=("dominant_class_constant_intensity:"+n) in row["flags"]
                axes[0].scatter(i,frac,s=28 if flagged else 9,c="crimson" if flagged else "steelblue",alpha=.7)
                if flagged:
                    axes[0].annotate(row["case_id"],(i,frac),xytext=(4,3),textcoords="offset points",fontsize=7)
            if stats["std"] is not None:
                axes[1].scatter(i,stats["std"],s=10,alpha=.4)
    axes[0].axhline(.5,color="orange",linestyle="--",label="screening threshold")
    for ax in axes: ax.set(xticks=range(7),xticklabels=names)
    axes[0].set(title="Most frequent raw value within each class",ylabel="Exact mode fraction",ylim=(-.02,1.05))
    axes[0].legend(fontsize=8)
    axes[1].set(title="Raw within-class standard deviation",ylabel="Raw intensity units")
    fig.tight_layout();fig.savefig(assets/"02_intensity_screening.png");plt.close(fig)

    author=[r for r in rows if r["author_spacing_mm"] is not None]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    if author:
        axes[0].scatter([r["author_spacing_mm"][0] for r in author],[r["author_spacing_mm"][2] for r in author],s=18,alpha=.6)
        axes[0].set(xlabel="Author reported axis-0 spacing (mm)",ylabel="Author reported axis-2 spacing (mm)")
    axes[0].set_title("Reported spacing; physical mapping not re-certified")
    for axis in range(3):
        axes[1].plot([r["header_spacing"][axis] for r in rows],label=f"Header axis {axis}",alpha=.65)
    axes[1].set(title="Native header numeric spacing",xlabel="Case order",ylabel="Header value (check native unit)")
    axes[1].legend()
    fig.tight_layout();fig.savefig(assets/"03_spacing_evidence.png");plt.close(fig)

    colors=["#000000","#ff5c5c","#4aa8ff","#ffc857","#71d27d","#d787ff","#f08a38","#31c9ce","#a0a0a0"]
    cmap=ListedColormap(colors)
    previews=[]
    for row in rows:
        snapshot=row.get("snapshot")
        if not snapshot or not Path(snapshot).is_file(): continue
        with np.load(snapshot) as data:
            fig,axes=plt.subplots(3,2,figsize=(8,11))
            limits=row["intensity"].get("percentiles") or {}
            vmin,vmax=limits.get("p01"),limits.get("p99")
            for axis in range(3):
                image=data[f"image{axis}"].T
                labels=data[f"label{axis}"].T
                clipped=np.where((labels>=0)&(labels<=7),labels,8)
                for col in range(2):
                    ax=axes[axis,col];ax.imshow(image,cmap="gray",origin="lower",vmin=vmin,vmax=vmax)
                    ax.set_title(f"Array axis {axis}, index {int(data[f'index{axis}'])}"+(" / source labels" if col else " / raw image"))
                    ax.axis("off")
                overlay=np.ma.masked_where(clipped==0,clipped)
                axes[axis,1].imshow(overlay,cmap=cmap,vmin=0,vmax=8,alpha=.40,origin="lower",interpolation="nearest")
            fig.suptitle(row["case_id"]+" | Native voxel views; orientation not independently certified\n"
                +"Display p01-p99 in raw units; no intensity repair",fontsize=10)
            fig.legend(handles=[Patch(color=colors[i],label=LABELS[i]) for i in LABELS],loc="lower center",ncol=7)
            fig.tight_layout(rect=(0,.04,1,.94))
            name=f"case_{row['case_id']}.png"
            fig.savefig(assets/name);plt.close(fig);previews.append(name)
    return previews


def build_report(all_rows, output, discovery, provenance, resources):
    output=Path(output)
    rows=[r for r in all_rows if r.get("analysis_success")]
    failed=[dict(case_id=r["case_id"],error=r.get("error","source changed")) for r in all_rows if not r.get("analysis_success")]
    flag_counts=Counter(f for r in rows for f in r["flags"])
    missing=[dict(case_id=r["case_id"],missing=r["missing_structures"]) for r in rows if r["missing_structures"]]
    intensity_flags=[dict(case_id=r["case_id"],structure=name,mode=s["mode"],mode_fraction=s["mode_fraction"],
                          voxels=s["voxels"],std=s["std"])
        for r in rows for name,s in r["structures"].items()
        if ("dominant_class_constant_intensity:"+name) in r["flags"]]
    summary=dict(discovered_pairs=len(discovery["pairs"]),selected_pairs=len(all_rows),
        analyzed_pairs=len(rows),failed_pairs=len(failed),reused_pairs=resources.get("reused_pairs",0),missing_labels=discovery["missing_labels"],
        orphan_labels=discovery["orphan_labels"],complete_seven_structure_pairs=sum(not r["missing_structures"] for r in rows),
        incomplete_annotation_pairs=len(missing),missing_annotations=missing,failed=failed,
        native_unit_counts=dict(Counter(r["header_units"][0] for r in rows)),
        native_spacing_counts=dict(Counter(str(r["header_spacing"]) for r in rows)),
        image_dtype_counts=dict(Counter(r["image_dtype"] for r in rows)),
        header_grid_mismatch_pairs=sum(not r["grid_match"] for r in rows),
        nonfinite_image_pairs=sum(r["intensity"]["nonfinite_voxels"]>0 for r in rows),
        invalid_label_pairs=sum(r["invalid_label_voxels"]>0 for r in rows),
        extra_label_cases=[dict(case_id=r["case_id"],label_ids=r["extra_label_ids"]) for r in rows if r["extra_label_ids"]],
        constant_class_screening=intensity_flags,flag_counts=dict(flag_counts),
        duplicate_compressed_image_groups=repeated(rows,"image_sha256"),
        duplicate_voxel_groups=repeated(rows,"image_voxel_sha256"),
        source_unchanged_pairs=sum(r["source_unchanged"] for r in rows),
        descriptor_mismatch_pairs=sum(any(f.startswith("descriptor_") for f in r["flags"]) for r in rows),
        components_checked_pairs=sum(r["components_checked"] for r in rows),
        native_shape_axis_ranges={str(a):ranges([r["shape"][a] for r in rows]) for a in range(3)},
        author_reported_spacing_ranges={str(a):ranges([r["author_spacing_mm"][a] for r in rows if r["author_spacing_mm"]]) for a in range(3)},
        physical_geometry_verified=False,screening_threshold=dict(mode_fraction=.5,min_class_voxels=1000),
        percentile_policy="exact at <=200000 voxels; otherwise fixed-seed uniform sample; counts/moments exact",
        analysis_wall_seconds=resources.get("analysis_wall_seconds"),
        scope="Read-only native image/label pairs; no cohort exclusions, preprocessing, repairs, training or prediction")
    write_json(output/"summary.json",summary)
    case_rows=[]
    class_rows=[]
    for row in rows:
        case_rows.append(dict(case_id=row["case_id"],shape="x".join(map(str,row["shape"])),
            header_spacing=str(row["header_spacing"]),native_unit=row["header_units"][0],
            author_spacing_mm=str(row["author_spacing_mm"]),grid_match=row["grid_match"],
            missing_structures=";".join(row["missing_structures"]),extra_label_ids=str(row["extra_label_ids"]),
            raw_min=row["intensity"]["min"],raw_max=row["intensity"]["max"],raw_mean=row["intensity"]["mean"],
            raw_std=row["intensity"]["std"],nonfinite_voxels=row["intensity"]["nonfinite_voxels"],
            foreground_bbox_axis_coverage=str(row["foreground_bbox_axis_coverage"]),flags=";".join(row["flags"]),
            source_unchanged=row["source_unchanged"],image_sha256=row["image_sha256"],label_sha256=row["label_sha256"],
            components_checked=row["components_checked"],elapsed_seconds=row["elapsed_seconds"]))
        for name,stats in row["structures"].items():
            class_rows.append(dict(case_id=row["case_id"],structure=name,**{k:stats[k] for k in (
                "voxels","author_volume_ml","mean","std","min","max","mode","mode_fraction","mode_method",
                "component_count","largest_component_fraction","touches_boundary")},bbox=str(stats["bbox"])))
    csv_table(output/"cases.csv",case_rows)
    csv_table(output/"structures.csv",class_rows)
    csv_table(output/"intensity_flags.csv",intensity_flags)
    csv_table(output/"missing_annotations.csv",[dict(case_id=r["case_id"],missing=";".join(r["missing"])) for r in missing])
    csv_table(output/"source_integrity.csv",[dict(case_id=r["case_id"],
        image_before=r["image_sha256"],image_after=r["sha256_after"]["image"],
        label_before=r["label_sha256"],label_after=r["sha256_after"]["label"],unchanged=r["source_unchanged"]) for r in rows])
    previews=make_figures(rows,output) if rows else []
    lines=["# ImageCHD 本机数据基本情况分析","",
        "本报告来自本轮原始文件只读分析。数量单位为影像/标签文件对，未认证为患者人数。",
        "没有修改源文件、自动排除样本、修复灰度/标签、重采样或训练模型。","",
        f"- 本地发现：{summary['discovered_pairs']}对；本轮选择{len(all_rows)}对，成功分析{len(rows)}对，失败{len(failed)}对。",
        f"- 七结构标注均存在：{summary['complete_seven_structure_pairs']}对；缺目标结构：{len(missing)}对。",
        f"- 原生网格不匹配：{summary['header_grid_mismatch_pairs']}对；影像含非有限值：{summary['nonfinite_image_pairs']}对；非法标签：{summary['invalid_label_pairs']}对。",
        f"- 源文件前后SHA一致：{summary['source_unchanged_pairs']}/{len(rows)}对；描述文件来源不一致：{summary['descriptor_mismatch_pairs']}对。",
        f"- 压缩文件重复组：{len(summary['duplicate_compressed_image_groups'])}；同shape/dtype解码像素重复组：{len(summary['duplicate_voxel_groups'])}。",
        f"- 本次直接复用既有结果：{resources.get('reused_pairs',0)}对；原执行证据在history中保留（如发生续跑）。",
        "","## 1. 数据形状和标注","",
        "![数据概况](assets/01_dataset_overview.png)","",
        "| 结构 | 有标注文件对 | 缺标注文件对 |","|---|---:|---:|"]
    for name in LABELS.values():
        n=sum(r["structures"][name]["voxels"]>0 for r in rows)
        lines.append(f"| {name} | {n} | {len(rows)-n} |")
    if missing:
        lines += ["","缺标注清单（只标记，本轮不执行排除）：",""]
        lines += [f"- {r['case_id']}：{', '.join(r['missing'])}" for r in missing]
        lines += ["","缺少标注不等于该结构在解剖上不存在；是否排除、补标或使用部分监督留待讨论。"]
    lines += ["","## 2. 灰度与纹理筛查","",
        "![灰度筛查](assets/02_intensity_screening.png)","",
        "筛查条件：某参考结构内至少1000体素，且一个精确原始灰度值占比≥50%。这是工程复核标记，不是诊断或自动排除规则。",
        "原始灰度未认证为真实HU；不根据最小值/最大值猜测偏移。全体计数、均值和标准差为完整数据统计；大区域百分位使用固定种子最多20万体素采样。","",
        "| 文件对 | 结构 | 最常见原始值 | 占比 | 结构体素数 |","|---|---|---:|---:|---:|"]
    for row in intensity_flags:
        lines.append(f"| {row['case_id']} | {row['structure']} | {row['mode']:g} | {row['mode_fraction']:.2%} | {row['voxels']} |")
    if not intensity_flags: lines.append("| 无达到当前筛查阈值的项 | — | — | — | — |")
    lines += ["","大片常值区域若来自源图，不会因Z-score或重采样而恢复真实纹理。本轮保留这些样本和统计，后续再决定复核原件、标记或敏感性分析。","",
        "## 3. 空间头与作者间距证据","",
        "![空间信息](assets/03_spacing_evidence.png)","",
        f"- 原生空间单位统计：{summary['native_unit_counts']}。",
        f"- 原生头spacing统计：{summary['native_spacing_counts']}。",
        f"- 作者表摘要与冻结描述一致：{provenance.get('spreadsheet_matches_descriptor',False)}。",
        f"- 作者报告间距范围（mm）：{summary['author_reported_spacing_ranges']}。",
        "",
        "作者表间距与原生头分别展示。作者间距不等于重新认证了数组轴、方向/原点或扫描仪坐标；未改头。structures.csv中的author_volume_ml仅按作者报告spacing计算的暂定体积，不能当已认证物理体积。",
        "","## 4. 连通区域与原生参考视图","",
        f"本轮对{summary['components_checked_pairs']}对计算6邻域连通区域；其余项未计算，不记为正常或通过。数量与最大区域占比见structures.csv。",
        "分支血管、切片断开或视野边界可能产生多个区域；本轮不删除小区域或只保留最大区域。边界框来自参考标签，不能作为模型定位框表现的证据。",""]
    for name in previews:
        lines += [f"![{name}]({('assets/'+name)})",""]
    lines += ["## 5. 本机执行与加速","",
        f"- 最大计算并发：{resources['max_workers']}；实际峰值在途：{resources.get('peak_inflight',0)}；I/O准入：{resources['read_concurrency']}；每worker内层线程1。",
        f"- 实际产出worker PID：{resources.get('producing_worker_pids',[])}。",
        f"- 只读统计墙钟：{resources.get('analysis_wall_seconds',0):.2f}秒（图表写出另计）；采样本组RSS峰值：{resources.get('peak_sampled_group_rss_bytes',0)/1024**3:.2f}GiB。",
        "- 使用当前CPU/RAM80%预算、有界逐例派发、独立读取准入、每例一次解码、精确整数直方图和仅失败项续跑。源SHA前后复核额外读取压缩文件，但不重复解码完整体积。",
        "- 上述是实际应用的执行措施，未做串行/并行对照，不声称特定加速倍数。未申请GPU，未验证Linux/cgroup/CUDA/NCCL。",
        "","## 6. 结果文件与下一步","",
        "- cases.csv：逐例基本统计和全部筛查标记。",
        "- structures.csv：逐例逐结构大小、灰度、边界与可选连通域统计。",
        "- missing_annotations.csv、intensity_flags.csv：需讨论的问题清单。",
        "- source_integrity.csv：源影像/标签前后摘要。",
        "- cases/*.json：完整逐例证据；resources.json、environment.json：实际资源和环境。",
        "- logs/run.log、logs/events.jsonl、status.json：运行日志、结构化事件和阶段状态。",
        "",
        "下一步先讨论缺标注、源灰度异常和物理尺度证据，再确定预处理策略。本轮没有依据预测分数选择或剔除样本。"]
    if failed:
        lines += ["","## 分析失败项",""]+[f"- {r['case_id']}：{r['error']}" for r in failed]
    (output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return summary