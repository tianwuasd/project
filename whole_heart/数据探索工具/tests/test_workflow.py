"""小型端到端测试：真正运行扫描、绘图和报告，不访问真实医学数据。"""
import csv
import json
from pathlib import Path
import tempfile
import unittest
import shutil

import nibabel as nib
import numpy as np

from analyze import main, inventory
from medical import LABELS


class WorkflowTests(unittest.TestCase):
    def test_complete_report_and_readonly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / "input", root / "output"
            for group, case in [("A ct_train", "TinyCT"), ("E mr_train", "TinyMR")]:
                folder = source / group
                folder.mkdir(parents=True)
                shape = (16, 20, 12)
                a = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
                label = np.zeros(shape, dtype=np.int16)
                for i, value in enumerate(v for v in LABELS if v):
                    label[i * 2:i * 2 + 2, 5:15, 3:9] = value
                if case == "TinyMR":
                    label[8, 10, 6] = 421
                for kind, data in [("image", a), ("label", label)]:
                    image = nib.Nifti1Image(data, np.diag([1., 2., 3., 1.]))
                    image.header.set_xyzt_units("mm")
                    nib.save(image, folder / f"{case}_{kind}.nii.gz")
            # 人为制造相同影像、不同标签，检查批次重复提醒，不把它当独立样本。
            shutil.copyfile(source / "A ct_train/TinyCT_image.nii.gz",
                            source / "E mr_train/TinyMR_image.nii.gz")
            before = inventory(source)
            result = main(["--input", str(source), "--output", str(output), "--samples", "100"])
            self.assertEqual(result, 0)
            self.assertEqual(before, inventory(source))
            stats = json.loads((output / "完整统计.json").read_text(encoding="utf-8"))
            self.assertEqual(len(stats), 2)
            unknown = [l for r in stats for l in r["labels"] if l["value"] == 421]
            self.assertEqual(unknown[0]["voxels"], 1)
            duplicates = [p for r in stats for p in r["issues"] if p["code"] == "duplicate_file"]
            self.assertEqual(len(duplicates), 2)
            self.assertTrue(all("对应标签文件哈希不同" in p["detail"] for p in duplicates))
            page = (output / "数据说明书.html").read_text(encoding="utf-8")
            self.assertIn("data:image/png;base64,", page)
            self.assertIn("非官方取值", page)
            markdown = (output / "数据说明书.md").read_text(encoding="utf-8")
            self.assertIn("| 标签值 | 结构名称 |\n| --- | --- |\n| 0 |", markdown)
            self.assertTrue((output / "病例信息.csv").read_bytes().startswith(b"\xef\xbb\xbf"))
            with (output / "病例信息.csv").open(encoding="utf-8-sig", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)
            self.assertGreaterEqual(len(list((output / "figures").glob("*.png"))), 8)


if __name__ == "__main__":
    unittest.main()
