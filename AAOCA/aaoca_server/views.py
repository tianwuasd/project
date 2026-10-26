"""Read-only metadata snapshots, terminal pages and resilient bounded log reads."""

import codecs
import json
import unicodedata
from pathlib import Path

from .state import job_status, read


def display_width(text):
    return sum(
        0
        if unicodedata.combining(c)
        else 2
        if unicodedata.east_asian_width(c) in "WF"
        else 1
        for c in text
    )


def paginate(lines, width=80):
    width = max(20, width)
    wrapped = []
    for text in lines:
        for line in str(text).splitlines() or [""]:
            current, size = "", 0
            for c in line:
                if unicodedata.category(c).startswith("C"):
                    c = " "  # Do not execute terminal controls from filenames/log metadata.
                amount = display_width(c)
                if size + amount > width:
                    wrapped.append(current)
                    current, size = "", 0
                current += c
                size += amount
            wrapped.append(current)
    return [wrapped[i : i + 20] for i in range(0, len(wrapped), 20)] or [["暂无记录"]]


def snapshot(entry):
    output = Path(entry["output"])
    status = job_status(output)
    runtime = Path(status["runtime"])
    return dict(
        entry=dict(entry),
        status=status,
        dataset=read(output / "dataset.json", {}),
        job=read(runtime / "job.json", {}),
        cases=read(output / "manifest.json", status["progress"].get("results", [])),
    )


def content(data, section, detail=False):
    entry, status = data["entry"], data["status"]
    progress, job, dataset = status["progress"], data["job"], data["dataset"]
    if section == "1":
        from .panel import STATES
        from .progress import lines

        result = [
            f"任务：{entry['name']} | {STATES.get(status['state'], status['state'])}",
            f"进程存活：{status['alive']}；停止请求：{status['stop_requested']}",
        ]
        if status["state"] in {"submitted", "checking"}:
            result += ["本次正在启动/校验，旧进度尚待复核。"]
        else:
            result += lines(progress)
        result += [
            "预处理完成数表示文件处理结果；不代表分割或诊断质量。",
            "完整产物可供独立粗分割步骤选择；本面板不启动分割。",
        ]
        if detail:
            result += [
                "工作进程记录：",
                *json.dumps(
                    status["worker"], ensure_ascii=False, indent=2
                ).splitlines(),
            ]
        return result
    if section == "2":
        config = dataset.get("config") or job.get("config") or {}
        spacing = dataset.get("spacing_selection") or job.get("spacing_selection") or {}
        result = [
            "当前模块规则：HU 截断 [-1000,1000] 后除以 1000 → [-1,1]；另存未截断 HU。",
            "当前空间规则：完整视野，目标物理网格。历史实际方法以冻结配置为准。",
            "目标间距 mm：" + str(config.get("nominal_spacing_mm", config.get("target_spacing", "未记录"))),
            "间距来源：" + str(spacing.get("method", "未记录")),
            "实际资源："
            + json.dumps(status["resources"].get("selected", "未记录"), ensure_ascii=False),
        ]
        if detail:
            for title, value in [
                ("冻结配置", config),
                ("间距选择", spacing),
                ("资源依据与执行预算", status["resources"]),
            ]:
                result += [
                    title,
                    *json.dumps(value, ensure_ascii=False, indent=2).splitlines(),
                ]
        return result
    if section == "3":
        cases = data["cases"]
        errors = progress.get("errors", [])
        result = [
            f"已记录病例条目 {len(cases)}；失败条目 {len(errors)}；缺失记录不按成功计。"
        ]
        if not cases and not errors:
            return result + ["暂无病例记录；尚未运行、检查中或旧记录未提供清单。"]
        selected = cases if detail else cases[:5]
        for row in selected:
            result.append(
                f"{row.get('case_id', '未知病例')} | {row.get('state', '未记录')}"
            )
            if detail:
                result += [json.dumps(row, ensure_ascii=False)]
        if not detail and len(cases) > 5:
            result.append(f"另有 {len(cases) - 5} 条，按 d 查看详情并翻页。")
        for row in errors if detail else errors[:5]:
            result.append("失败：" + json.dumps(row, ensure_ascii=False))
        return result
    if section == "4":
        output, runtime = Path(entry["output"]), Path(status["runtime"])
        result = [
            "原始输入：" + str(job.get("input", "未记录；不从新建配置推测历史来源")),
            "结果目录：" + str(output),
            "运行记录：" + str(runtime),
            "原任务代码：" + str(entry.get("code_root", "旧记录未保存")),
            "原任务 Python：" + str(entry.get("python", "旧记录未保存")),
            "归一化影像：cases/<病例>/image.nii.gz",
            "同网格 HU：cases/<病例>/image_hu.nii.gz",
        ]
        if detail:
            result += [
                str(output / name)
                for name in ("dataset.json", "manifest.json", "progress.json")
            ]
            result += [
                str(runtime / name)
                for name in ("job.json", "inputs.json", "resources.json", "worker.log")
            ]
            result += [
                "以上是记录/约定路径；查看不会验证影像哈希。续跑与缓存复用仍执行完整验证。"
            ]
        return result
    raise ValueError("未知结果栏目")


class LogCursor:
    """Follow by filename with a short anchor to detect truncate-and-regrow."""

    def __init__(self, path):
        self.path = Path(path)
        self.identity = None
        self.offset = 0
        self.anchor = b""
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.pending = False

    def read(self):
        try:
            handle = self.path.open("rb")
        except FileNotFoundError:
            self.pending = False
            return ""
        with handle:
            import os

            stat = os.fstat(handle.fileno())
            identity = (stat.st_dev, stat.st_ino)
            reset = self.identity != identity or stat.st_size < self.offset
            if not reset and self.anchor:
                handle.seek(self.offset - len(self.anchor))
                reset = handle.read(len(self.anchor)) != self.anchor
            if reset:
                self.decoder.reset()
                self.offset = max(0, stat.st_size - 65536)
                handle.seek(self.offset)
                if self.offset:
                    handle.readline(65536)
                    self.offset = handle.tell()
            handle.seek(self.offset)
            chunk = handle.read(65536)
            self.offset = handle.tell()
            handle.seek(max(0, self.offset - 64))
            self.anchor = handle.read(min(self.offset, 64))
            self.identity = identity
            self.pending = self.offset < stat.st_size
        return self.decoder.decode(chunk)
