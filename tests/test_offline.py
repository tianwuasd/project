from copy import deepcopy
import unittest

from chd_mini.offline import verify_selected_statistics


class StatisticsContract(unittest.TestCase):
    def test_only_selected_training_patients_supply_statistics(self):
        rows = [{"case_id": name, "patient": name, "roles": [role], "sha256": {"image": name, "label": name},
                 "spacing_xyz": [1, 1, 1], "geometry_policy": "shared_index_excel", "unknown_classes": []}
                for name, role in (("a", "train"), ("b", "train"), ("c", "validation"), ("d", "test"))]
        prepared = {"manifest": {"cases": rows}, "plan": {"fit_ids": ["a", "b"],
                    "generation": {"request": {"training_cases": deepcopy(rows[:2])}}}}
        self.assertEqual(verify_selected_statistics(prepared), ["a", "b"])
        broken = deepcopy(prepared)
        broken["plan"]["generation"]["request"]["training_cases"].append(rows[2])
        with self.assertRaises(ValueError):
            verify_selected_statistics(broken)
        broken = deepcopy(prepared)
        broken["plan"]["generation"]["request"]["training_cases"][0]["ignore_label_ids"] = [255]
        with self.assertRaises(ValueError):
            verify_selected_statistics(broken)
        broken = deepcopy(prepared)
        broken["manifest"]["cases"][3]["patient"] = "a"
        with self.assertRaises(ValueError):
            verify_selected_statistics(broken)


if __name__ == "__main__":
    unittest.main()
