import argparse
import json
from pathlib import Path
import re
import sys
from time import perf_counter

from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aaoca_reproduction.labels import COLUMNS, CERTAINTY_COLUMNS, _table, reviewed_labels


def main():
    parser = argparse.ArgumentParser(description="两名真实患者的原文研究标签内存映射，零文件产物")
    parser.add_argument("--workbook", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--aaoca-root")
    args = parser.parse_args()
    started = perf_counter()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    aliases = {str(alias): case["patient_key"] for case in manifest["cases"] if case["dataset"] == "aaoca"
               for alias in (case["patient_key"], *case["aliases"])}
    workbook = load_workbook(args.workbook, read_only=True, data_only=False)
    selected = {}
    try:
        values = workbook["患者标签"].iter_rows(values_only=True)
        headers = next(values)
        patients = {str(row[0]): dict(zip(headers, row)) for row in values}
        values = workbook["逐条证据"].iter_rows(values_only=True)
        headers = next(values)
        patterns = (("右冠状动脉起自左冠窦", 0), ("右冠状动脉起源于左冠窦", 0),
                    ("左冠状动脉异常起源于右冠窦", 1), ("左冠起源于右冠窦", 1))
        for values_row in values:
            row = dict(zip(headers, values_row))
            key = str(row["患者主编号"])
            canonical = aliases.get(key)
            context = re.sub(r"\s+", "", str(row["原文上下文"]))
            if canonical is None or canonical in selected or row["原文确定程度"] != "原文陈述待复核":
                continue
            for phrase, origin in patterns:
                clauses = [clause for clause in re.split(r"[，,。；;]", context) if phrase in clause]
                if any(not re.search(r"疑|可能|考虑|不除外|待排|否认|未见|无冠|[?？]", clause) for clause in clauses):
                    patient = patients[key]
                    selected[canonical] = {
                        "patient_key": key, "aaoca_present": "1", "origin_left": str(origin), "high_risk": "unknown",
                        "source_type": "document_explicit", "aaoca_certainty": "explicit",
                        "origin_certainty": "explicit", "risk_certainty": "unknown",
                        "reference": json.dumps([{field: row[field] for field in
                            ("来源文书或文本", "页码", "片段日期", "提取方式", "原文确定程度", "原文上下文")}],
                            ensure_ascii=False), "review_status": patient["人工复核状态"], "reviewer": "",
                        "ct_matched": "unknown", "ct_match_status": patient["就诊与CT对应"]}
                    break
            if len(selected) == 2:
                break
    finally:
        workbook.close()
    assert len(selected) == 2, "未找到两名满足所选明确句式的真实冻结患者，不扩大测试范围"
    headers = [*COLUMNS, "source_type", *CERTAINTY_COLUMNS.values(), "ct_match_status"]
    rows = _table(headers, [[row[key] for key in headers] for row in selected.values()])
    research = reviewed_labels(rows, manifest, "document_explicit", aaoca_root=args.aaoca_root)
    confirmed = reviewed_labels(rows, manifest, "confirmed", aaoca_root=args.aaoca_root)
    for patient, item in research.items():
        assert item["research_labels"]["aaoca_present"] == 1
        assert item["research_labels"]["origin_left"] == int(selected[patient]["origin_left"])
        assert item["research_labels"]["high_risk"] is None
        assert all(value is None for value in item["labels"].values())
        assert not item["ct_matched"] and item["review_status"] == selected[patient]["review_status"]
        assert item["ct_match_status"] == selected[patient]["ct_match_status"]
        assert not item["ct_match_declared"] and item["ct_binding_state"] == "not_declared_matched"
        if args.aaoca_root:
            case = next(case for case in manifest["cases"] if case["patient_key"] == patient)
            assert item["selected_ct_image"] == Path(case["image"]).resolve().relative_to(
                Path(args.aaoca_root).resolve()).as_posix()
        assert item["reviewer"] == "" and item["reference"] == selected[patient]["reference"]
        assert all(value is None for value in confirmed[patient]["labels"].values())
    print(json.dumps({"state": "passed", "patients": 2, "raw_document_positive": 2,
                      "ct_matched": 0, "selected_training_labels": 0, "unknown_risk": 2,
                      "seconds": perf_counter() - started, "files_created": 0, "artifact_bytes": 0,
                      "model_operations": 0, "training_updates": 0, "memory_peak": "not_measured",
                      "scope": "two actual document assertions projected in memory; pending English artifact not tested"}))


if __name__ == "__main__":
    main()
