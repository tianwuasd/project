"""Guard the NSW experimental benchmark, and tie it to the benchmark task.

The point of `demo-notebooks/nsw-lalonde-1986/` is that the +$1,794 experimental
effect, which `benchmark/tasks/lalonde-recovery.toml` grades candidates against,
stopped being a hand-transcribed constant and became a value derived from the
randomized data.

That only holds if the two stay pinned to each other, which is what
`TestBenchmarkConstantIsDerived` does: editing the task's constant without the
data, or letting the derivation drift, fails here rather than silently changing
what every candidate is graded against.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import unittest

from _helpers import ROOT, load_module

nsw = load_module("demo-notebooks/nsw-lalonde-1986/replicate_nsw.py", "aers_nsw")
toml_compat = load_module("scripts/toml_compat.py", "aers_toml_nsw")

TASK = ROOT / "benchmark" / "tasks" / "lalonde-recovery.toml"

# Recorded in data/PROVENANCE.md at vendoring time.
EXPECTED_SHA256 = {
    "nswre74_treated.txt": "e7b742fe0ff07a0f45e129b4ff108bb9611cd83d53604732c48a8a0a3e20eda3",
    "nswre74_control.txt": "a1364cea459d953dc691a667d99194b4ad335d6d550354fe23a5d2dc58d729b5",
}


class TestVendoredData(unittest.TestCase):
    def test_files_match_the_recorded_hashes(self):
        # A replication whose input can change unnoticed is not a replication.
        for name, expected in EXPECTED_SHA256.items():
            path = nsw.HERE / "data" / name
            with self.subTest(file=name):
                self.assertTrue(path.exists(), f"{name} is not vendored")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(digest, expected, "vendored data changed")

    def test_provenance_records_the_same_hashes(self):
        provenance = (nsw.HERE / "data" / "PROVENANCE.md").read_text(encoding="utf-8")
        for name, expected in EXPECTED_SHA256.items():
            with self.subTest(file=name):
                self.assertIn(expected, provenance, f"{name} hash missing from PROVENANCE.md")

    def test_parser_rejects_a_malformed_row(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.txt"
            path.write_text("1.0 2.0 3.0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                nsw.load_nsw(path)


class TestExperimentalBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.got = nsw.replicate()

    def test_sample_sizes(self):
        self.assertEqual(self.got["n_treated"], 185)
        self.assertEqual(self.got["n_control"], 260)
        self.assertEqual(self.got["n_psid_control"], 429)

    def test_experimental_effect_is_the_published_benchmark(self):
        self.assertAlmostEqual(self.got["experimental_att"], 1794.34, places=2)

    def test_randomization_left_pre_treatment_earnings_balanced(self):
        # What licenses reading the raw difference as the causal effect.
        self.assertLess(abs(self.got["experimental_re74_gap"]), 150.0)

    def test_the_psid_comparison_group_is_visibly_imbalanced(self):
        # And is so *before* the outcome is consulted.
        self.assertGreater(abs(self.got["observational_re74_gap"]), 1000.0)

    def test_the_observational_estimate_flips_the_sign(self):
        self.assertLess(self.got["observational_att"], 0)
        self.assertGreater(self.got["experimental_att"], 0)
        self.assertAlmostEqual(self.got["observational_att"], -635.03, places=2)

    def test_both_estimates_use_the_same_treated_men(self):
        # Otherwise the comparison is between two different treatment effects
        # and says nothing about selection bias.
        self.assertTrue(nsw.same_treated_arm())


class TestBenchmarkConstantIsDerived(unittest.TestCase):
    """The lalonde task's literature constant must equal the derived value."""

    def test_task_constant_matches_the_randomized_data(self):
        with TASK.open("rb") as fh:
            task = toml_compat.load(fh)
        constant = task["experimental_att"]
        derived = nsw.replicate()["experimental_att"]
        self.assertAlmostEqual(
            constant,
            derived,
            delta=1.0,
            msg=(
                "benchmark/tasks/lalonde-recovery.toml cites an experimental benchmark "
                "that the vendored NSW experimental arms no longer produce — one of the "
                "two moved"
            ),
        )

    def test_the_task_still_uses_the_psid_composite_not_the_experiment(self):
        # The task is an *observational* stress test; pointing it at the
        # randomized controls would make it trivially passable.
        with TASK.open("rb") as fh:
            task = toml_compat.load(fh)
        self.assertEqual(task["data"], "demo-notebooks/_lalonde_data.csv")


class TestSelfGating(unittest.TestCase):
    def test_main_exits_zero_and_estimates_json_is_current(self):
        path = nsw.OUT
        before = path.read_text(encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            code = nsw.main()
        self.assertEqual(code, 0)
        self.assertEqual(
            path.read_text(encoding="utf-8"), before, "estimates.json is stale; re-run"
        )


if __name__ == "__main__":
    unittest.main()
