"""入口：python analyze.py --input ../Wholeheart_Train_Dataset。

代码阅读顺序：main → medical.inspect_case → charts → reporting。
默认做完整检查；--limit 只用于试运行，报告会注明未扫描全部数据。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import platform
import sys
import traceback

import matplotlib
import nibabel
import numpy

from medical import discover, inspect_case, issue
from charts import setup_font, group_figures, case_preview
from reporting import export_tables, build_report, json_safe


def prepare_output(root, output):
    """防止把结果写进数据集。已有非空目录也不覆盖，以保留每次运行。"""
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError("输出目录不能位于原始数据目录内")
    if output.exists() and any(output.iterdir()):
        raise ValueError("输出目录已存在且非空，请使用新的目录名")
    output.mkdir(parents=True, exist_ok=True)
    return output


def inventory(root):
    """运行前后对比尺寸和纳秒时间戳，检测意外修改（不是内容哈希的替代）。"""
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Whole Heart 初学者数据探索：只读原件，生成中文报告")
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parent.parent / "Wholeheart_Train_Dataset")
    parser.add_argument("--output", type=Path, help="独立结果目录；默认在工具目录 results 下创建时间戳文件夹")
    parser.add_argument("--limit", type=int, help="仅处理前 N 例，用于快速试运行；默认全部")
    parser.add_argument("--samples", type=int, default=200000, help="每例强度分位数最多抽样体素数（默认 200000）")
    parser.add_argument("--skip-hash", action="store_true", help="跳过完整文件字节哈希；不能报告文件重复检查")
    parser.add_argument("--no-previews", action="store_true", help="只生成统计图，不生成病例三方向预览")
    parser.add_argument("--max-issue-previews", type=int, default=8, help="问题病例的额外预览数量上限")
    args = parser.parse_args(argv)
    if args.samples < 100 or (args.limit is not None and args.limit < 1) or args.max_issue_previews < 0:
        parser.error("samples 至少 100，limit 至少 1，max-issue-previews 不能为负")
    root = args.input.resolve()
    if not root.is_dir():
        parser.error(f"数据目录不存在：{root}")
    pairs, discoveries = discover(root)
    if not pairs:
        parser.error("没有找到 *_image.nii.gz，请指向训练数据目录")
    before = inventory(root)
    count_all = len(pairs)
    pairs = pairs[:args.limit] if args.limit else pairs
    output = prepare_output(root, args.output or Path(__file__).parent / "results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    font = setup_font()
    records = []
    for number, pair in enumerate(pairs, 1):
        print(f"[{number}/{len(pairs)}] {pair['group']} / {pair['case']}", flush=True)
        try:
            record, _ = inspect_case(pair, args.samples, not args.skip_hash)
        except Exception as error:
            record = dict(pair, status="failed", error=str(error), issues=[issue("read_failure", str(error), "错误")])
            with (output / "错误详情.log").open("a", encoding="utf-8") as log:
                log.write(f"\n{pair}\n{traceback.format_exc()}\n")
        records.append(record)
        # 每例完成即记录检查结果，长时间扫描意外中止也能保留已完成的工作。
        with (output / "逐例检查记录.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False) + "\n")
    if not args.skip_hash:
        for key in ["image_sha256", "label_sha256"]:
            buckets = defaultdict(list)
            for r in records:
                if key in r:
                    buckets[r[key]].append(r)
            for duplicate in buckets.values():
                if len(duplicate) > 1:
                    names = [f"{r['group']}/{r['case']}" for r in duplicate]
                    detail = f"{key} 完全相同的文件：{names}"
                    if key == "image_sha256":
                        detail += "；实验划分时需避免同一影像跨集合"
                        if len({r.get('label_sha256') for r in duplicate}) > 1:
                            detail += "；对应标签文件哈希不同（可能为标注或文件头差异），请核实"
                    for r in duplicate:
                        r["issues"].append(issue("duplicate_file", detail))
    print("开始生成统计图与预览……", flush=True)
    figures = group_figures(records, output)
    previews, preview_errors = [], []
    good = [r for r in records if r["status"] == "ok"]
    if not args.no_previews:
        representatives = []
        for group in sorted({r["group"] for r in good}):
            rs = [r for r in good if r["group"] == group]
            # 优先选择没有错误级空间问题的病例；按文件名确定选择，可重复。
            representatives.append(next((r for r in rs if not any(p["severity"] == "错误" for p in r["issues"])), rs[0]))
        extra = [r for r in good if r not in representatives and r["issues"]][:args.max_issue_previews]
        for number, r in enumerate(representatives + extra, 1):
            print(f"预览 {r['case']}", flush=True)
            try:
                path = case_preview(r, output, number)
                kind = "代表病例" if r in representatives else "需核实病例"
                previews.append((f"{kind}：{r['group']} / {r['case']}", path, (r['group'], r['case'])))
            except Exception as error:
                preview_errors.append({"path": r["image"], **issue("preview_failure", str(error))})
    after = inventory(root)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(), "input": str(root), "output": str(output),
        "discovered_cases": count_all, "processed_cases": len(records), "partial_run": len(pairs) != count_all,
        "input_file_count": len(before), "input_bytes": sum(v[0] for v in before.values()),
        "input_size_and_mtime_unchanged": before == after,
        "fully_read_cases": len(good), "intensity_sample_limit": args.samples,
        "sampling": "每例按现有内存顺序固定步长均匀抽样；包括背景；分位数为估计值",
        "duplicate_hash_enabled": not args.skip_hash, "font": font,
        "versions": {"python": platform.python_version(), "numpy": numpy.__version__, "nibabel": nibabel.__version__, "matplotlib": matplotlib.__version__},
        "preview_count": len(previews), "preview_errors": preview_errors,
        "figure_index": figures, "preview_index": previews,
    }
    if before != after:
        discoveries.append({"path": str(root), **issue("input_changed", "运行期间输入的尺寸/时间戳发生变化，请核对其他程序活动", "错误")})
    if manifest["partial_run"]:
        discoveries.append({"path": str(root), **issue("partial_run", f"仅扫描 {len(pairs)}/{count_all} 例，不能代表全数据集", "说明")})
    issues = export_tables(records, discoveries + preview_errors, output)
    (output / "完整统计.json").write_text(json.dumps(json_safe(records), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output / "运行信息.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    build_report(records, issues, figures, previews, manifest, output)
    print(f"完成：成功 {len(good)}/{len(records)}；报告：{output / '数据说明书.html'}", flush=True)
    # 数据问题如 421 记录在报告；只有读取失败/运行期间输入变化导致非零退出码。
    return 0 if len(good) == len(records) and before == after else 2


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
