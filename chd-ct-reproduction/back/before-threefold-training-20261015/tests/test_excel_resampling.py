"""Physical grid alignment, label preservation, native restoration and recovery."""

import argparse
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile

import nibabel as nib
import numpy as np
import pytest

from chd_ct.imagechd.common import load_case, read_prepared
from chd_ct.imagechd.predict import restore_native
from chd_ct.imagechd.preprocess import prepare
from chd_ct.imagechd.spacing import EXCEL_NAME, read_spacing_excel
from chd_ct.quickstart.commands import add_task_arguments, build_command
from chd_ct.recovery import file_hash


def write_excel(path, entries):
    rows = []
    for row_number, (index, spacing) in enumerate(entries, 1):
        values = dict(
            A="idx",
            B=index,
            I=" PixelSpacing",
            J=spacing[0],
            K=spacing[1],
            L=" calculate_z_thick",
            M=spacing[2],
        )
        cells = []
        for col, value in values.items():
            if isinstance(value, str):
                cells.append(f'<c r="{col}{row_number}" t="inlineStr"><is><t>{escape(value)}</t></is></c>')
            else:
                cells.append(f'<c r="{col}{row_number}"><v>{value}</v></c>')
        rows.append(f'<row r="{row_number}">' + "".join(cells) + "</row>")
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "xl/workbook.xml",
            """<workbook
          xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
          <sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>""",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            """<Relationships
          xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
          <Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>""",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            "<sheetData>" + "".join(rows) + "</sheetData></worksheet>",
        )


@pytest.fixture
def excel_data(tmp_path):
    root = tmp_path / "ImageCHD_dataset"
    root.mkdir()
    shape = (12, 10, 8)
    grid = np.indices(shape)
    labels = np.zeros(shape, np.uint8)
    for label in range(1, 8):
        labels[label, 1:9, 1:7] = label
    labels[0, 0, 0] = 14
    for index in range(1, 4):
        # Deliberately unrelated header spacing; the Excel values must win.
        affine = np.diag([5.0, 5.0, 5.0, 1.0])
        if index == 2:
            affine = np.array(
                [[0.0, 0.0, -5.0, 20.0], [5.0, 0.0, 0.0, -3.0], [0.0, 5.0, 0.0, 2.0], [0.0, 0.0, 0.0, 1.0]]
            )
        image = (grid[0] * 100 + grid[1] * 10 + grid[2] + index * grid[2] ** 2).astype(np.float32)
        for kind, array in (("image", image), ("label", labels)):
            nib.save(nib.Nifti1Image(array, affine), root / f"ct_{1000 + index}_{kind}.nii.gz")
    write_excel(root / EXCEL_NAME, [(1, (2, 1, 1)), (2, (1, 2, 3)), (3, (1, 1, 1))])
    return root


def test_excel_resampling_and_exact_upsampled_label_restoration(excel_data, tmp_path):
    output = tmp_path / "cache"
    data = prepare(excel_data, output)
    assert data["format"] == "imagechd7-v3" and data["resolution"] == "resampled"
    assert data["resampling"]["excel_sha256"] == file_hash(excel_data / EXCEL_NAME)
    directory, parsed = read_prepared(output)
    for row in parsed["cases"]:
        image, labels = load_case(directory, row)
        assert image.dtype == np.float32 and labels.dtype == np.uint8
        assert set(np.unique(labels)) == {*range(8), 255}
        original = nib.load(excel_data / f"{row['case_id']}_label.nii.gz")
        expected = np.asanyarray(original.dataobj)
        expected = np.where(expected > 7, 255, expected)
        restored = restore_native(labels, row)
        np.testing.assert_array_equal(np.asanyarray(restored.dataobj), expected)
        np.testing.assert_array_equal(restored.affine, original.affine)
        assert restored.header.get_xyzt_units() == original.header.get_xyzt_units()
    first, second = data["cases"][:2]
    assert first["cache_shape"] == [24, 10, 8]
    assert second["canonical_shape"] == [8, 12, 10]
    assert second["resampling"]["canonical_spacing_mm"] == [3, 1, 2]
    assert second["cache_shape"] == [24, 12, 20]
    # Independently check linear interpolation at an interior half-voxel position.
    original = nib.load(excel_data / "ct_1001_image.nii.gz").get_fdata()
    low, high = first["intensity"]["low"], first["intensity"]["high"]
    normalized = (np.clip(original, low, high) - low) / (high - low)
    cached, _ = load_case(directory, first)
    np.testing.assert_allclose(
        cached[5, 4, 3], 0.75 * normalized[2, 4, 3] + 0.25 * normalized[3, 4, 3], rtol=1e-6
    )


def test_resampling_resume_rejects_excel_and_target_changes(excel_data, tmp_path):
    output = tmp_path / "cache"
    prepare(excel_data, output)
    digest = file_hash(output / "dataset.json")
    prepare(excel_data, output, resume=True)
    assert file_hash(output / "dataset.json") == digest
    with pytest.raises(ValueError, match="改变"):
        prepare(excel_data, output, resume=True, target_spacing=(0.5, 0.5, 0.5))
    write_excel(excel_data / EXCEL_NAME, [(1, (1, 1, 1)), (2, (1, 2, 3)), (3, (1, 1, 1))])
    with pytest.raises(ValueError, match="改变"):
        prepare(excel_data, output, resume=True)
    assert file_hash(output / "dataset.json") == digest


def test_old_behavior_and_single_image_prediction(excel_data, tmp_path):
    legacy = prepare(excel_data, tmp_path / "legacy", no_resample=True)
    assert legacy["format"] == "imagechd7-v2" and legacy["resolution"] == "native"
    assert "resampling" not in legacy
    assert "resampling" not in legacy["cases"][0]
    data = prepare(excel_data / "ct_1001_image.nii.gz", tmp_path / "prediction", for_prediction=True, size=8)
    row = data["cases"][0]
    assert row["case_id"] == "ct_1001"
    assert row["cache_shape"] == [8, 8, 8]
    assert row["resampling"]["resampled_shape"] == [24, 10, 8]
    image, _ = load_case(tmp_path / "prediction", row, labels=False)
    restored = restore_native(np.ones(image.shape, np.uint8), row)
    assert restored.shape == (12, 10, 8)
    assert np.all(np.asanyarray(restored.dataobj) == 1)


@pytest.mark.parametrize(
    "entries, error",
    [
        ([(1, (0, 1, 1))], "spacing 无效"),
        ([(1, (1, 1, float("nan")))], "spacing 无效"),
        ([(1, (1, 1, 1)), (1, (1, 1, 1))], "重复"),
    ],
)
def test_invalid_spacing_table(tmp_path, entries, error):
    path = tmp_path / EXCEL_NAME
    write_excel(path, entries)
    with pytest.raises(ValueError, match=error):
        read_spacing_excel(path)


def test_missing_case_and_invalid_target_fail(excel_data, tmp_path):
    with pytest.raises(ValueError, match="三个有限正数"):
        prepare(excel_data, tmp_path / "bad_target", target_spacing=(1, 0, 1))
    write_excel(excel_data / EXCEL_NAME, [(1, (1, 1, 1))])
    with pytest.raises(ValueError, match="Excel 缺少病例间距"):
        prepare(excel_data, tmp_path / "missing")


def test_launcher_forwards_spacing_options(tmp_path):
    from chd_ct.server.settings import task_configuration

    parser = argparse.ArgumentParser()
    add_task_arguments(parser)
    args = parser.parse_args(
        [
            "--task",
            "preprocess",
            "--dataset",
            str(tmp_path),
            "--spacing-excel",
            "geometry.xlsx",
            "--target-spacing",
            ".5",
            ".5",
            "1",
        ]
    )
    command = build_command(args, tmp_path, "python", tmp_path / "output", "cpu")
    position = command.index("--target-spacing")
    assert command[position + 1 : position + 4] == ["0.5", "0.5", "1.0"]
    assert command[command.index("--spacing-excel") + 1] == "geometry.xlsx"
    config = task_configuration(args)
    assert config["spacing_excel"] == "geometry.xlsx" and config["target_spacing"] == [0.5, 0.5, 1.0]
    args.spacing_excel = None
    args.no_resample = True
    assert "--no-resample" in build_command(args, Path.cwd(), "python", tmp_path, "cpu")
