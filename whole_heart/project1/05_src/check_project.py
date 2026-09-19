"""检查方案文件与病例清单；不读取影像体素、不修改原始数据。

用法：在项目 Conda 环境运行 python 05_src/check_project.py。
这是准备阶段的完整性检查，不是模型、分割指标或 GPU 的测试。
"""

from collections import Counter
from pathlib import Path
import hashlib
import json
import re


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    # utf-8-sig 同时兼容 Windows 写出的带 BOM UTF-8 文件。
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    # 分块计算文件校验值，避免一次加载整个文件。
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    inventory = read_json(ROOT / "04_data/manifests/case_inventory.json")
    cases = inventory["cases"]
    quarantine = read_json(ROOT / "04_data/manifests/quarantine.json")
    expected = {"Case2009", "Case2017", "Case3010", "Case5006", "Case5015"}
    assert len(cases) == 106 and len({c["case_id"] for c in cases}) == 106
    assert {c["case_id"] for c in quarantine} == expected
    assert {c["case_id"] for c in cases if c["status"] == "quarantined"} == expected
    assert all(c["status"] in {"eligible", "quarantined"} for c in cases)
    eligible = [c for c in cases if c["status"] == "eligible"]
    counts = Counter(c["modality"] for c in eligible)
    assert counts == {"CT": 58, "MRI": 43}, counts
    for case in cases:
        # 仅检查路径存在；此处不重复计算大型原始影像的哈希。
        assert Path(case["image"]).is_file(), case["case_id"]
        assert Path(case["label"]).is_file(), case["case_id"]
    assert sha256(Path(inventory["source_audit"])) == inventory["source_sha256"]
    for entry in read_json(ROOT / "03_workflow/registry.json"):
        assert sha256(ROOT / entry["snapshot"]) == entry["sha256"], entry

    # 检查本项目撰写文档的相对链接。第三方快照保留原上下文，不检查其外部依赖。
    links_checked = 0
    for doc in ROOT.rglob("*.md"):
        relative = doc.relative_to(ROOT).as_posix()
        if "/skill_snapshots/" in relative or "/agent_roles/" in relative:
            continue
        for target in re.findall(r"\]\(([^)]+)\)", doc.read_text(encoding="utf-8-sig")):
            if ":" in target or target.startswith("#"):
                continue
            target = target.split("#", 1)[0]
            assert (doc.parent / target).exists(), (relative, target)
            links_checked += 1

    pdfs = []
    for path in sorted((ROOT / "02_materials/pdf").glob("*.pdf")):
        with path.open("rb") as stream:
            assert stream.read(5) == b"%PDF-", path
        pdfs.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)})
    assert len(pdfs) == 2
    report = {
        "status": "passed",
        "total_cases": len(cases),
        "quarantined": sorted(expected),
        "eligible_by_modality": dict(counts),
        "eligible_by_group": dict(Counter(c["group"] for c in eligible)),
        "relative_links_checked": links_checked,
        "pdfs": pdfs,
        "scope": "manifest, audit hash, skill snapshots, project links, PDF headers; no model test",
    }
    output = ROOT / "00_admin/project_check.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
