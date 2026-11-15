import copy
import unittest

from chd_mini.contracts import select_cases, validate_settings


class Contracts(unittest.TestCase):
    def test_bounded_settings(self):
        cfg = {"epochs": 2, "steps_per_epoch": 2, "batch": 4, "profile": "technical", "device": "cpu"}
        self.assertEqual(validate_settings(cfg)["epochs"], 2)
        for key, value in (("epochs", 200), ("steps_per_epoch", 250), ("batch", 1), ("epochs", True)):
            broken = {**cfg, key: value}
            with self.assertRaises(ValueError):
                validate_settings(broken)

    def test_selection_keeps_roles_and_seed(self):
        rows = [{"case_id": str(i), "patient": str(i), "roles": [role], "cohort": "main"}
                for i, role in enumerate(["train", "train", "train", "validation", "test"])]
        selection = select_cases(rows, 20261115)
        self.assertEqual(selection, select_cases(list(reversed(rows)), 20261115))
        self.assertEqual([r["roles"] for r in selection], [["train"], ["train"], ["validation"], ["test"]])
        self.assertEqual(len({r["patient"] for r in selection}), 4)
        broken = copy.deepcopy(rows)
        broken[3]["patient"] = broken[0]["patient"]
        with self.assertRaises(ValueError):
            select_cases(broken, 20261115)
        broken[3].pop("patient")
        with self.assertRaises(ValueError):
            select_cases(broken, 20261115)


if __name__ == "__main__":
    unittest.main()
