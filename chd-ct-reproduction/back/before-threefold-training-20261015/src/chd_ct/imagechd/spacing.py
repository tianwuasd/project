"""ImageCHD Excel spacing and reversible, centre-aligned physical resampling."""

import posixpath
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy import ndimage as ndi

EXCEL_NAME = "imagechd_dataset_image_info.xlsx"
RESAMPLED_FORMAT = "imagechd7-v3"
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def validate_spacing(values):
    spacing = np.asarray(values, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("spacing 必须为三个有限正数，单位 mm")
    return spacing


def read_spacing_excel(path):
    """Read the author's headerless key/value rows; idx + 1000 is the case ID."""
    result = {}
    with ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = [
                "".join(x.itertext())
                for x in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("m:si", NS)
            ]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheet = workbook.find("m:sheets/m:sheet", NS)
        key = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = next(r.attrib["Target"] for r in rels if r.attrib["Id"] == key)
        name = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
        root = ET.fromstring(archive.read(name))
        for row in root.findall("m:sheetData/m:row", NS):
            values = {}
            for cell in row.findall("m:c", NS):
                col = re.match(r"[A-Z]+", cell.attrib["r"]).group()
                # Read only identifiers and geometry, not personal metadata.
                if col not in {"A", "B", "I", "J", "K", "L", "M"}:
                    continue
                if cell.find("m:f", NS) is not None:
                    raise ValueError("影像间距表需要数值，不能使用公式单元格")
                value = cell.findtext("m:v", default="", namespaces=NS)
                if cell.attrib.get("t") == "s" and value:
                    value = strings[int(value)]
                elif cell.attrib.get("t") == "inlineStr":
                    value = "".join(cell.find("m:is", NS).itertext())
                values[col] = value.strip()
            if not any(values.values()):
                continue
            number = int(row.attrib["r"])
            if any(
                values.get(c) != v
                for c, v in (("A", "idx"), ("I", "PixelSpacing"), ("L", "calculate_z_thick"))
            ):
                raise ValueError(f"影像间距表第 {number} 行字段不符合 ImageCHD 格式")
            try:
                index = float(values["B"])
                if not np.isfinite(index) or not index.is_integer() or index < 0:
                    raise ValueError()
                spacing = validate_spacing([float(values[c]) for c in ("J", "K", "M")])
            except (KeyError, ValueError) as error:
                raise ValueError(f"影像间距表第 {number} 行编号或 spacing 无效") from error
            case = f"ct_{1000 + int(index)}"
            if case in result:
                raise ValueError(f"影像间距表病例重复：{case}")
            result[case] = {"spacing_mm": spacing.tolist(), "excel_row": number}
    if not result:
        raise ValueError("影像间距表为空")
    return result


def find_spacing_excel(image_path, explicit=None, no_resample=False):
    if no_resample:
        if explicit:
            raise ValueError("--no-resample 与 --spacing-excel 不能同时使用")
        return None
    path = Path(explicit).expanduser().resolve() if explicit else Path(image_path).parent / EXCEL_NAME
    if explicit and not path.is_file():
        raise ValueError(f"影像间距表不存在：{path}")
    return path if path.is_file() else None


def resampling_grid(shape, source_spacing, target_spacing):
    source = validate_spacing(source_spacing)
    target = validate_spacing(target_spacing)
    shape = np.asarray(shape, dtype=int)
    # Rounded field of view, with the volume centre unchanged. Edge voxels are
    # extended by nearest boundary mode when rounding needs partial support.
    output_shape = np.maximum(2, np.floor(shape * source / target + 0.5).astype(int))
    transform = np.eye(4)
    transform[:3, :3] = np.diag(target / source)
    transform[:3, 3] = (shape - 1) / 2 - (target / source) * (output_shape - 1) / 2
    return output_shape.tolist(), transform


def sample_grid(array, shape, output_to_input, labels=False):
    transform = np.asarray(output_to_input, dtype=float)
    return ndi.affine_transform(
        array,
        transform[:3, :3],
        offset=transform[:3, 3],
        output_shape=tuple(shape),
        order=0 if labels else 1,
        mode="nearest",
        prefilter=False,
    )
