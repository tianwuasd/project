"""Tests for the interference/spillover benchmark (`spillover-recovery`).

The task exists because SUTVA is the assumption most often stated and least
often checked, and because the comparison that violates it is the one a field
experiment hands you for free: treated households next to untreated ones, both
right there in the same village.

Two properties carry the design, and both are tested here rather than assumed.
The baselines must be balanced across *both* splits — within-cluster
treated/untreated and between treated/pure-control clusters — or the contrasts
would carry composition differences and the golds would stop being exact. And
the four estimands must actually differ from one another; if they collapsed, the
task would have nothing to teach.
"""

from __future__ import annotations

import json
import unittest

from _helpers import ROOT, load_module

spillover = load_module("benchmark/lib/spillover.py", "aers_spillover")
check_benchmark = load_module("benchmark/check_benchmark.py", "aers_check_benchmark")
toml_compat = load_module("scripts/toml_compat.py", "aers_toml_spillover")

DATA = ROOT / "benchmark" / "data" / "sim-spillover.csv"
TASK = ROOT / "benchmark" / "tasks" / "spillover-recovery.toml"


def load_task() -> dict:
    with TASK.open("rb") as fh:
        return toml_compat.load(fh)


def load_reference(task: dict) -> dict:
    path = ROOT / "benchmark" / "candidates" / task["reference_candidate"] / "results.json"
    return json.loads(path.read_text(encoding="utf-8"))


class TestDesign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = spillover.load(DATA)

    def test_committed_csv_matches_the_generator(self):
        generated = spillover.generate()
        self.assertEqual(len(generated), len(self.rows))
        for expected, actual in zip(generated, self.rows):
            for key in spillover.FIELDNAMES:
                with self.subTest(field=key):
                    self.assertAlmostEqual(
                        float(expected[key]), float(actual[key]), places=8
                    )

    def test_shape(self):
        self.assertEqual(
            len(self.rows), spillover.N_CLUSTERS * spillover.UNITS_PER_CLUSTER
        )
        self.assertEqual(spillover.n_pure_control_clusters(self.rows), 20)

    def test_baselines_are_balanced_within_treated_clusters(self):
        # If they were not, the within-cluster contrast would carry a
        # composition difference and would not equal TAU exactly.
        treated = [
            r for r in self.rows
            if int(r["cluster_treated"]) == 1 and int(r["treated"]) == 1
        ]
        untreated = [
            r for r in self.rows
            if int(r["cluster_treated"]) == 1 and int(r["treated"]) == 0
        ]
        self.assertAlmostEqual(
            spillover._mean(treated, "y0"), spillover._mean(untreated, "y0"), places=10
        )

    def test_baselines_are_balanced_across_cluster_types(self):
        treated_clusters = [r for r in self.rows if int(r["cluster_treated"]) == 1]
        pure = [r for r in self.rows if int(r["cluster_treated"]) == 0]
        self.assertAlmostEqual(
            spillover._mean(treated_clusters, "y0"),
            spillover._mean(pure, "y0"),
            places=10,
        )

    def test_pure_control_clusters_have_no_treated_units(self):
        for row in self.rows:
            if int(row["cluster_treated"]) == 0:
                with self.subTest(cluster=row["cluster"]):
                    self.assertEqual(int(row["treated"]), 0)
                    self.assertEqual(float(row["share_treated"]), 0.0)


class TestEstimands(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = spillover.load(DATA)

    def test_each_estimand_is_recovered_exactly(self):
        for name, estimated, expected in (
            ("direct", spillover.direct_effect(self.rows), 2.0),
            ("spillover", spillover.spillover_effect(self.rows), 1.5),
            ("total on treated", spillover.total_effect_on_treated(self.rows), 3.5),
            ("overall", spillover.overall_effect(self.rows), 2.5),
        ):
            with self.subTest(estimand=name):
                self.assertAlmostEqual(estimated, expected, places=9)

    def test_estimators_agree_with_the_counterfactual_truth(self):
        for estimator, truth in (
            (spillover.direct_effect, spillover.true_direct_effect),
            (spillover.spillover_effect, spillover.true_spillover_effect),
            (spillover.total_effect_on_treated, spillover.true_total_effect_on_treated),
            (spillover.overall_effect, spillover.true_overall_effect),
        ):
            with self.subTest(estimator=estimator.__name__):
                self.assertAlmostEqual(estimator(self.rows), truth(self.rows), places=9)

    def test_the_four_estimands_are_genuinely_different(self):
        values = {
            spillover.direct_effect(self.rows),
            spillover.spillover_effect(self.rows),
            spillover.total_effect_on_treated(self.rows),
            spillover.overall_effect(self.rows),
        }
        self.assertEqual(len(values), 4, "collapsed estimands teach nothing")

    def test_the_within_cluster_contrast_understates_the_program(self):
        direct = spillover.direct_effect(self.rows)
        total = spillover.total_effect_on_treated(self.rows)
        overall = spillover.overall_effect(self.rows)
        self.assertLess(direct, total)
        self.assertLess(direct, overall)
        # And by enough to change a conclusion, not a rounding amount.
        self.assertGreater(total - direct, 1.0)

    def test_the_decomposition_adds_up(self):
        self.assertAlmostEqual(
            spillover.total_effect_on_treated(self.rows),
            spillover.direct_effect(self.rows) + spillover.spillover_effect(self.rows),
            places=9,
        )

    def test_the_sutva_answer_is_zero(self):
        self.assertEqual(spillover.naive_spillover(self.rows), 0.0)


class TestGrading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = load_task()
        cls.truth = check_benchmark.compute_truth(cls.task)

    def test_task_spec_validates(self):
        self.assertEqual(check_benchmark.validate_task(self.task, TASK), [])

    def test_reference_candidate_takes_full_marks(self):
        graded = check_benchmark.grade(self.task, load_reference(self.task), self.truth)
        self.assertEqual([g["id"] for g in graded if not g["passed"]], [])

    def test_truth_matches_the_documented_design_constants(self):
        for truth_key, spec_key in (
            ("true_direct", "true_direct_by_construction"),
            ("true_spillover", "true_spillover_by_construction"),
            ("true_total_on_treated", "true_total_on_treated_by_construction"),
            ("true_overall", "true_overall_by_construction"),
        ):
            with self.subTest(key=truth_key):
                self.assertAlmostEqual(
                    self.truth[truth_key], self.task[spec_key], places=6
                )

    def test_assuming_sutva_fails_the_required_golds(self):
        # The whole point: a candidate that reports the within-cluster contrast
        # as every estimand and calls the spillover zero must not pass.
        candidate = dict(load_reference(self.task))
        direct = candidate["direct_effect"]
        candidate.update(
            spillover_effect=0.0,
            total_effect_on_treated=direct,
            overall_effect=direct,
        )
        graded = check_benchmark.grade(self.task, candidate, self.truth)
        failed = {g["id"] for g in graded if g["required"] and not g["passed"]}
        self.assertIn("spillover-recovered", failed)
        self.assertIn("total-effect-on-treated-recovered", failed)
        self.assertIn("overall-policy-effect-recovered", failed)

    def test_fabricated_numbers_fail_the_honest_golds(self):
        candidate = dict(load_reference(self.task))
        candidate["spillover_effect"] = 9.9
        graded = check_benchmark.grade(self.task, candidate, self.truth)
        failed = {g["id"] for g in graded if not g["passed"]}
        self.assertIn("honest-spillover", failed)


class TestFamilyRegistration(unittest.TestCase):
    def test_taxonomy_has_interference_keywords(self):
        enrich = load_module("scripts/build-catalog-enrich.py", "aers_enrich_interference")
        self.assertIn("interference", enrich.TAXONOMY["method"])

    def test_coverage_map_lists_the_family(self):
        coverage = load_module("scripts/build-coverage-map.py", "aers_coverage_interference")
        self.assertIn("interference", coverage.METHOD_ORDER)
        self.assertEqual(coverage.TASK_METHOD["spillover-recovery"], "interference")
        self.assertEqual(
            coverage.SCENARIO_METHOD["statspai-spillovers-sutva"], "interference"
        )

    def test_the_naive_baseline_fails_the_task(self):
        scoreboard = load_module(
            "scripts/build-benchmark-scoreboard.py", "aers_scoreboard_interference"
        )
        task = load_task()
        self.assertIn(task["id"], scoreboard.NAIVE_BUILDERS)
        self.assertIn(task["id"], scoreboard.NAIVE_MOVE)
        naive = {"task": task["id"], "method": "naive"}
        naive.update(scoreboard.NAIVE_BUILDERS[task["id"]](load_reference(task)))
        graded = check_benchmark.grade(task, naive, check_benchmark.compute_truth(task))
        self.assertTrue([g["id"] for g in graded if g["required"] and not g["passed"]])


if __name__ == "__main__":
    unittest.main()
