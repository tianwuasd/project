"""Discover file pairs and optionally cross-check an existing frozen descriptor."""
from pathlib import Path
import json
import math
from .analysis import sha256_file


def discover_pairs(root):
    root=Path(root)
    images={p.name.removesuffix("_image.nii.gz"):p for p in root.glob("*_image.nii.gz")}
    labels={p.name.removesuffix("_label.nii.gz"):p for p in root.glob("*_label.nii.gz")}
    return dict(pairs=[dict(case_id=cid,image=str(images[cid]),label=str(labels[cid]))
                       for cid in sorted(images.keys()&labels.keys())],
                missing_labels=sorted(images.keys()-labels.keys()),
                orphan_labels=sorted(labels.keys()-images.keys()),
                image_files=len(images),label_files=len(labels))


def descriptor_evidence(path, root):
    """Descriptor is provenance evidence, never the discovery/cohort authority.

    Author spacing is accepted as a reported value only after its spreadsheet
    hash is checked. It does not certify array-axis or scanner coordinates.
    """
    if not path:
        return {},dict(provided=False)
    path=Path(path)
    record=json.loads(path.read_text(encoding="utf-8-sig"))
    excel=Path(root)/"imagechd_dataset_image_info.xlsx"
    provenance=record.get("source_provenance",{})
    expected=provenance.get("spacing_excel_sha256")
    if expected is None:
        expected=provenance.get(excel.name)
    excel_sha=sha256_file(excel) if excel.is_file() else None
    spacing_verified=bool(expected and excel_sha==expected)
    cases={}
    for row in record.get("cases",[]):
        cid=row["case_id"]
        if cid in cases:
            raise ValueError("Duplicate case ID in descriptor: "+cid)
        spacing=row.get("spacing_mm")
        valid=bool(spacing and len(spacing)==3 and all(
            isinstance(x,(int,float)) and math.isfinite(x) and x>0 for x in spacing))
        cases[cid]=dict(expected_hashes={k:row.get(k+"_sha256") for k in ("image","label")},
                       author_spacing=spacing if valid and spacing_verified else None,
                       previous_status=row.get("status"),roles=row.get("roles",{}))
    return cases,dict(provided=True,path=str(path),sha256=sha256_file(path),
        spreadsheet_path=str(excel),spreadsheet_sha256=excel_sha,
        spreadsheet_matches_descriptor=spacing_verified,
        spacing_interpretation="author-reported only; physical array mapping not newly authenticated")