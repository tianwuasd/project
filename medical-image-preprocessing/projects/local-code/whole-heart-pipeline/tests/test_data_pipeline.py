"""验证研究中可能造成错误结论的数据风险，不测试模型效果。"""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "05_src"))
from data_pipeline import OFFICIAL, ROOT, load_json, make_splits, map_labels, inverse_labels


class DataTests(unittest.TestCase):
    def test_labels_roundtrip(self):
        a = np.resize(OFFICIAL, (4, 5, 6))
        np.testing.assert_array_equal(inverse_labels(map_labels(a)), a)

    def test_unknown_label_rejected(self):
        for value in [421, 420.5, np.nan, np.inf, -1]:
            with self.assertRaises(ValueError):
                map_labels(np.array([0, value]))

    def test_split_no_leakage_or_quarantine(self):
        cases = load_json(ROOT / "04_data/manifests/case_inventory.json")["cases"]
        a = make_splits(cases)
        self.assertEqual(a, make_splits(list(reversed(cases))))
        excluded = {c["case_id"] for c in cases if c["status"] != "eligible"}
        lookup = {c["case_id"]: c for c in cases}
        for fold, split in a.items():
            sets = [set(split[x]) for x in ["train", "val", "target"]]
            for i in range(3):
                self.assertFalse(sets[i] & excluded)
                for j in range(i):
                    self.assertFalse(sets[i] & sets[j])
            self.assertEqual(len(set.union(*sets)), 58 if split["modality"] == "CT" else 43)
            source_groups = {lookup[i]["group"] for i in split["train"] + split["val"]}
            self.assertFalse(source_groups & {lookup[i]["group"] for i in split["target"]})
            self.assertEqual(split["train"], split["fingerprint_fit_ids"])
        duplicate = dict(cases[0], case_id="duplicate")
        with self.assertRaises(ValueError):
            make_splits(cases + [duplicate])


if __name__ == "__main__":
    unittest.main()
