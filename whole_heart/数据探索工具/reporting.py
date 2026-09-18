"""输出可查询的数据表和面向初学者的中文报告。"""
import base64
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from medical import LABELS


def json_safe(value):
    """JSON 没有 NaN/Inf；异常文件头中的非有限数值保存为 null。"""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_csv(path, rows):
    """使用 UTF-8 BOM，方便 Windows Excel 正确显示中文。"""
    if not rows:
        path.write_text("无记录\n", encoding="utf-8-sig")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def export_tables(records, discovery_issues, output):
    cases, labels, issues = [], [], list(discovery_issues)
    groups = defaultdict(list)
    for r in records:
        common = {"病例": r["case"], "数据组": r["group"], "模态": r["modality"]}
        row = dict(common, 状态=r["status"], 影像路径=r["image"], 标签路径=r["label"])
        for problem in r.get("issues", []):
            issues.append(dict(common, **problem))
        if r["status"] == "ok":
            im, lm, intensity = r["image_meta"], r["label_meta"], r["intensity"]
            groups[r["group"]].append(r)
            for n in range(3):
                row[f"第{n + 1}轴体素数"] = im["shape"][n]
                row[f"第{n + 1}轴间距_mm"] = im["spacing_mm"][n] if im["spacing_mm"] else None
                row[f"第{n + 1}轴覆盖_mm"] = im["fov_mm"][n] if im["fov_mm"] else None
            row.update(原始存储类型=im["storage_dtype"], 文件缩放斜率=im["slope"], 文件缩放截距=im["intercept"],
                       方向代码=im["orientation"], 空间单位=im["space_unit"],
                       影像最小值=intensity["min"], 影像最大值=intensity["max"],
                       抽样P1=intensity["p01"], 抽样中位数=intensity["median"], 抽样P99=intensity["p99"],
                       强度抽样数=intensity["sample_count"], 影像非有限体素数=intensity["nonfinite_voxels"],
                       标签体素体积_ml=lm["voxel_ml"], 体积计算依据=lm["volume_basis"],
                       问题="；".join(p["detail"] for p in r["issues"]),
                       影像SHA256=r.get("image_sha256"), 标签SHA256=r.get("label_sha256"))
            for l in r["labels"]:
                labels.append(dict(common, 标签值=l["value"], 中文结构=l["name"], 体素数=l["voxels"],
                                   标注体积_ml=l["volume_ml"], 体积计算依据=lm["volume_basis"], 占整图比例=l["image_fraction"],
                                   占七类目标比例=l["target_fraction"]))
        else:
            row["错误说明"] = r.get("error", "未知错误")
        cases.append(row)
    group_rows = []
    for group, rs in groups.items():
        row = {"数据组": group, "模态": rs[0]["modality"], "成功读取组数": len(rs),
               "有需核实问题的组数": sum(bool(r["issues"]) for r in rs)}
        for axis in range(3):
            vals = [r["image_meta"]["spacing_mm"][axis] for r in rs if r["image_meta"]["spacing_mm"]]
            row[f"第{axis + 1}轴间距中位数_mm"] = float(np.median(vals)) if vals else None
        group_rows.append(row)
    write_csv(output / "病例信息.csv", cases)
    write_csv(output / "标签体积.csv", labels)
    write_csv(output / "问题清单.csv", issues)
    write_csv(output / "分组概览.csv", group_rows)
    return issues


class Report:
    """同时构建 Markdown 和无需外部图片文件的 HTML，避免额外渲染依赖。"""
    def __init__(self, output):
        self.output = output
        self.md, self.html = [], []

    def heading(self, text, level=2):
        self.md.extend(["", "#" * level + " " + text, ""])
        self.html.append(f"<h{level}>{html.escape(text)}</h{level}>")

    def paragraph(self, text):
        self.md.extend(["", text, ""])
        self.html.append("<p>" + html.escape(text).replace("\n", "<br>") + "</p>")

    def table(self, headers, rows):
        def cell(value):
            return "—" if value is None else str(value)
        self.md.extend(["", "| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"])
        self.html.append("<div class='table'><table><thead><tr>" + "".join("<th>" + html.escape(h) + "</th>" for h in headers) + "</tr></thead><tbody>")
        for row in rows:
            self.md.append("| " + " | ".join(cell(v).replace("|", "\\|").replace("\n", " ") for v in row) + " |")
            self.html.append("<tr>" + "".join("<td>" + html.escape(cell(v)) + "</td>" for v in row) + "</tr>")
        self.html.append("</tbody></table></div>")
        self.md.append("")

    def image(self, title, relative):
        path = self.output / relative
        self.md.extend(["", f"![{title}]({relative})", ""])
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        self.html.append(f"<figure><img alt='{html.escape(title)}' src='data:image/png;base64,{encoded}'><figcaption>{html.escape(title)}</figcaption></figure>")

    def save(self):
        (self.output / "数据说明书.md").write_text("\n".join(self.md).strip() + "\n", encoding="utf-8")
        style = "body{font-family:'Microsoft YaHei',sans-serif;max-width:1100px;margin:auto;padding:32px;color:#22334b;background:#f8fafc;line-height:1.8}h1,h2{color:#143b60}h2{margin-top:40px;border-bottom:2px solid #dae4ef}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border:1px solid #d8e0e8;text-align:left}th{background:#e8eff7}.table{overflow:auto}img{max-width:100%;height:auto}figure{margin:25px 0;background:white;padding:12px}figcaption{color:#526579;font-size:13px}"
        page = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Whole Heart 数据说明书</title><style>" + style + "</style><body>" + "\n".join(self.html) + "</body></html>"
        (self.output / "数据说明书.html").write_text(page, encoding="utf-8")


def build_report(records, issues, figures, previews, manifest, output):
    good = [r for r in records if r["status"] == "ok"]
    report = Report(output)
    report.heading("Whole Heart 数据说明书", 1)
    report.paragraph(f"运行时间：{manifest['created_at']}。数据目录：{manifest['input']}。本次扫描 {len(records)} 组影像记录，成功完整读取 {len(good)} 组，失败 {len(records) - len(good)} 组。")
    report.paragraph("这是一份数据探索报告。病例数指影像记录数，不等于已验证的独立患者人数；没有根据影像推测年龄、疾病或扫描日期。")
    report.heading("1. 先认识数据：图像是题目，标签是参考答案")
    report.paragraph("每个 image.nii.gz 是一份三维 CT 或 MRI；同名 label.nii.gz 为各位置的结构编号。三维图像由体素构成，多张切片共同组成一例影像，不能把切片数当患者数。")
    groups = sorted({r["group"] for r in records})
    report.table(["数据组", "模态（目录识别）", "发现影像", "成功配对读取"], [[g, next(r['modality'] for r in records if r['group'] == g), sum(r['group'] == g for r in records), sum(r['group'] == g for r in good)] for g in groups])
    report.paragraph(f"输入文件共 {manifest['input_file_count']} 个，占用 {manifest['input_bytes'] / 1024**3:.2f} GiB。未匹配文件、读取失败及其他问题详见问题清单。")
    if figures:
        report.image(*figures[0])
    report.heading("2. 一例三维影像怎样理解")
    report.table(["字段", "通俗含义"], [["数组尺寸", "每个方向有多少体素，如 512×512×300"], ["体素间距", "相邻体素中心之间的实际距离；不等同于扫描仪真正的空间分辨率"], ["覆盖长度", "尺寸×间距，描述沿数组各轴的名义覆盖范围"], ["原点和方向", "影像在物理空间中的位置与朝向；数组相同并不保证空间一致"], ["数据类型与缩放", "原始整数可能需按 slope/intercept 换算，本报告使用换算后的值"]])
    report.paragraph("预览图从接近 RAS 的三个方向显示（可能仍带原扫描倾斜，并非统一重采样的标准解剖平面）；按物理间距保持比例。三列从左到右是近似矢状面、冠状面、横断面；三行从上到下是原图、标签、叠加。只改变显示窗口，不改数据。")
    for title, relative, key in previews:
        r = next(r for r in good if (r["group"], r["case"]) == key)
        m = r["image_meta"]
        report.heading(title, 3)
        report.paragraph(f"尺寸：{m['shape']}；间距（mm）：{m['spacing_mm']}；原始方向：{m['orientation']}；存储类型：{m['storage_dtype']}；缩放：×{m['slope']} + {m['intercept']}。")
        report.image(title, relative)
    report.heading("3. 七种结构与体积")
    report.table(["标签值", "结构名称"], [[v, name] for v, name in LABELS.items()])
    report.paragraph("体积（mL）= 标签体素数 × 每个体素的体积（mm³）÷1000。体素体积优先由标签的有效空间矩阵计算（无剪切时等于三个物理边长的乘积）；矩阵无效时使用间距推算的名义值，并在数据表标出依据。空间单位未知时不输出毫升。背景比例的分母是整份影像；目标内比例的分母仅含七种官方结构，不含非预期标签。")
    report.paragraph("这些是当前文件头和人工标注定义下的名义体积，不是临床测量结论；存在空间元数据问题的病例应先核实。血管标注长度和扫描覆盖范围也会影响比较。")
    for title, relative in figures:
        if title in ("七种结构体积", "背景与类别比例"):
            report.image(title, relative)
    report.heading("4. 不同中心与模态有什么差异")
    report.paragraph("以下描述观察到的差异，不推断原因。三个数组轴在不同组中未必对应相同身体方向。CT/MRI 的强度分开显示；MRI 数值通常不能与 CT 数值直接比较。强度分位数为均匀抽样估计，包含背景与扫描范围差异；min/max 和 NaN/Inf 检查覆盖全部体素。")
    for title, relative in figures:
        if title in ("体素间距", "尺寸与覆盖范围", "影像灰度分布"):
            report.image(title, relative)
    report.heading("5. 需要核实的问题")
    counts = Counter(p['severity'] for p in issues)
    report.paragraph(f"问题记录共 {len(issues)} 条，分类计数：{dict(counts)}。这是问题条数，不是异常患者数。同一病例可能产生多条记录。")
    report.table(["病例或路径", "级别", "检查项", "说明"], [[p.get('病例', p.get('path', '')), p['severity'], p['code'], p['detail']] for p in issues])
    if not issues:
        report.paragraph("本轮未发现所实现检查能检出的异常，不等于标注医学准确性已经确认。")
    report.paragraph("脚本不会把 421 改成 420，不会修复文件头，也不会删除离群病例。完全相同的 SHA256 只能说明文件字节相同；文件不同不能证明患者不同。")
    report.heading("6. 怎样使用这些输出")
    report.table(["文件", "内容"], [["病例信息.csv", "每例尺寸、间距、强度、文件哈希与问题"], ["标签体积.csv", "每例每类的体素数、体积和比例（含缺失类别的零值）"], ["分组概览.csv", "组别数量与采样间距摘要"], ["问题清单.csv", "缺失文件、异常标签与空间问题"], ["完整统计.json", "所有字段、空间矩阵与详细问题"], ["运行信息.json", "参数、依赖版本、抽样规则、读取前后只读检查"], ["figures/", "统计图与代表/问题病例 PNG"]])
    report.paragraph("先读代表病例，再对照类别比例和中心差异；最后查看问题清单。根据这一步的结果，之后再设计预处理和训练，不需要立刻做纹理特征提取。")
    report.paragraph("类别定义来源：https://zmic.org.cn/care_2026/track_wholeheart/ 。本报告不执行网络请求，不读取临床身份信息。")
    report.save()
