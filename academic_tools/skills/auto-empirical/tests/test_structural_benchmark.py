"""Tests for the structural demand benchmark (`structural-demand-recovery`).

The task's whole claim is that its golds are *exact*, not asymptotic: the
dataset is built so the sample moment conditions hold to machine precision, so
just-identified 2SLS returns the design parameter itself rather than an
estimate near it. These tests pin that claim, the direction of the OLS bias
(toward zero, which is the direction that flatters pricing power), and the two
downstream steps a structural pipeline owes on top of a coefficient — an
elasticity and an inverted marginal cost.
"""

from __future__ import annotations

import json
import unittest

from _helpers import ROOT, load_module

structural = load_module("benchmark/lib/structural.py", "aers_structural")
check_benchmark = load_module("benchmark/check_benchmark.py", "aers_check_benchmark")
toml_compat = load_module("scripts/toml_compat.py", "aers_structural_toml")

DATA = ROOT / "benchmark" / "data" / "sim-structural.csv"
TASK = ROOT / "benchmark" / "tasks" / "structural-demand-recovery.toml"


def load_task() -> dict:
    with TASK.open("rb") as fh:
        return toml_compat.load(fh)


def load_reference(task: dict) -> dict:
    path = ROOT / "benchmark" / "candidates" / task["reference_candidate"] / "results.json"
    return json.loads(path.read_text(encoding="utf-8"))


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = structural.load(DATA)

    def test_committed_csv_matches_the_generator(self):
        # The dataset is a generated artifact like everything else in the repo;
        # a hand-edit would silently move every gold.
        generated = structural.generate()
        self.assertEqual(len(generated), len(self.rows))
        for expected, actual in zip(generated, self.rows):
            for key in structural.FIELDNAMES:
                with self.subTest(row=expected["market"], field=key):
                    self.assertAlmostEqual(
                        float(expected[key]), float(actual[key]), places=10
                    )

    def test_shape(self):
        self.assertEqual(len(self.rows), structural.N_MARKETS * structural.N_PRODUCTS)

    def test_shares_leave_room_for_an_outside_good(self):
        # Without a meaningful outside share the Berry inversion is degenerate.
        for market in range(1, structural.N_MARKETS + 1):
            rows = [r for r in self.rows if int(r["market"]) == market]
            inside = sum(float(r["share"]) for r in rows)
            outside = float(rows[0]["outside_share"])
            with self.subTest(market=market):
                self.assertAlmostEqual(inside + outside, 1.0, places=9)
                self.assertGreater(outside, 0.01)

    def test_the_hidden_shock_is_orthogonal_to_the_instruments(self):
        # This is what makes the golds exact rather than approximate.
        xi = [float(r["xi"]) for r in self.rows]
        for name in ("x", "w"):
            moment = sum(float(r[name]) * x for r, x in zip(self.rows, xi))
            with self.subTest(instrument=name):
                self.assertAlmostEqual(moment, 0.0, places=8)
        self.assertAlmostEqual(sum(xi), 0.0, places=8)

    def test_price_is_correlated_with_the_hidden_shock(self):
        # ... and this is what makes OLS wrong. Both must hold for the task to
        # be testing anything.
        xi = [float(r["xi"]) for r in self.rows]
        price = [float(r["price"]) for r in self.rows]
        mean_p = sum(price) / len(price)
        covariance = sum((p - mean_p) * x for p, x in zip(price, xi)) / len(price)
        self.assertGreater(covariance, 0.01)


class TestEstimators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = structural.load(DATA)

    def test_iv_recovers_the_design_parameter_exactly(self):
        self.assertAlmostEqual(structural.iv_alpha(self.rows), structural.ALPHA, places=6)

    def test_oracle_regression_recovers_the_same_truth(self):
        self.assertAlmostEqual(
            structural.oracle_alpha(self.rows), structural.ALPHA, places=6
        )

    def test_ols_is_biased_toward_zero(self):
        # Direction matters: the bias says demand is *less* price-sensitive
        # than it is, which is the flattering direction for a pricing study.
        ols = structural.ols_alpha(self.rows)
        self.assertLess(ols, structural.ALPHA)
        self.assertGreater(structural.ALPHA - ols, 0.2)

    def test_the_instrument_is_strong(self):
        self.assertGreater(structural.first_stage_f(self.rows), 10.0)

    def test_elasticity_is_not_the_coefficient(self):
        alpha = structural.iv_alpha(self.rows)
        elasticity = structural.mean_own_elasticity(self.rows, alpha)
        self.assertLess(elasticity, -1.0)
        # The units error is large enough to be a different conclusion, not a
        # rounding difference.
        self.assertGreater(abs(elasticity - structural.naive_elasticity(self.rows)), 1.0)

    def test_elasticity_formula_uses_one_minus_share(self):
        alpha = structural.ALPHA
        for row, value in zip(
            self.rows, structural.own_price_elasticities(self.rows, alpha)
        ):
            expected = -alpha * float(row["price"]) * (1.0 - float(row["share"]))
            self.assertAlmostEqual(value, expected, places=10)

    def test_marginal_cost_is_below_price(self):
        alpha = structural.iv_alpha(self.rows)
        for row, mc in zip(self.rows, structural.marginal_costs(self.rows, alpha)):
            with self.subTest(market=row["market"], product=row["product"]):
                self.assertLess(mc, float(row["price"]), "a positive markup is implied")
                self.assertGreater(mc, 0.0)

    def test_biased_demand_propagates_into_the_cost_estimate(self):
        true_mc = structural.mean_marginal_cost(self.rows, structural.ALPHA)
        naive_mc = structural.naive_marginal_cost(self.rows)
        self.assertLess(naive_mc, true_mc, "understated alpha overstates the markup")
        self.assertGreater(true_mc - naive_mc, 0.1)


class TestGrading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = load_task()
        cls.truth = check_benchmark.compute_truth(cls.task)

    def _reference(self) -> dict:
        return load_reference(self.task)

    def test_task_spec_validates(self):
        self.assertEqual(check_benchmark.validate_task(self.task, TASK), [])

    def test_reference_candidate_takes_full_marks(self):
        graded = check_benchmark.grade(self.task, self._reference(), self.truth)
        failures = [g["id"] for g in graded if not g["passed"]]
        self.assertEqual(failures, [])

    def test_truth_matches_the_documented_design_constants(self):
        self.assertAlmostEqual(
            self.truth["true_alpha"], self.task["true_alpha_by_construction"], places=5
        )
        self.assertAlmostEqual(
            self.truth["true_mean_elasticity"],
            self.task["true_mean_elasticity_by_construction"],
            places=3,
        )
        self.assertAlmostEqual(
            self.truth["true_mean_marginal_cost"],
            self.task["true_mean_marginal_cost_by_construction"],
            places=3,
        )

    def test_quoting_the_coefficient_as_an_elasticity_fails(self):
        candidate = dict(self._reference())
        candidate["mean_own_elasticity"] = -candidate["iv_alpha"]
        graded = check_benchmark.grade(self.task, candidate, self.truth)
        failed = {g["id"] for g in graded if not g["passed"]}
        self.assertIn("elasticity-recovered", failed)

    def test_skipping_the_instrument_fails(self):
        candidate = dict(self._reference())
        candidate["iv_alpha"] = candidate["ols_alpha"]
        candidate["first_stage_F"] = 0.0
        graded = check_benchmark.grade(self.task, candidate, self.truth)
        failed = {g["id"] for g in graded if not g["passed"]}
        self.assertIn("alpha-recovered", failed)
        self.assertIn("instrument-relevant", failed)

    def test_reading_price_as_marginal_cost_fails(self):
        candidate = dict(self._reference())
        rows = structural.load(DATA)
        candidate["mean_marginal_cost"] = sum(
            float(r["price"]) for r in rows
        ) / len(rows)
        graded = check_benchmark.grade(self.task, candidate, self.truth)
        failed = {g["id"] for g in graded if not g["passed"]}
        self.assertIn("marginal-cost-inverted", failed)

    def test_fabricated_numbers_fail_the_honest_golds(self):
        candidate = dict(self._reference())
        candidate["ols_alpha"] = candidate["iv_alpha"]  # pretend OLS was fine
        graded = check_benchmark.grade(self.task, candidate, self.truth)
        failed = {g["id"] for g in graded if not g["passed"]}
        self.assertIn("honest-ols-alpha", failed)
        self.assertIn("ols-price-coefficient-biased", failed)


class TestFamilyRegistration(unittest.TestCase):
    """A new method family is only real once every registry knows about it."""

    def test_taxonomy_has_structural_keywords(self):
        enrich = load_module("scripts/build-catalog-enrich.py", "aers_enrich_structural")
        self.assertIn("structural", enrich.TAXONOMY["method"])

    def test_coverage_map_lists_the_family(self):
        coverage = load_module("scripts/build-coverage-map.py", "aers_coverage_structural")
        self.assertIn("structural", coverage.METHOD_ORDER)
        self.assertEqual(coverage.TASK_METHOD["structural-demand-recovery"], "structural")
        self.assertEqual(coverage.SCENARIO_METHOD["statspai-structural-demand"], "structural")

    def test_scoreboard_has_a_naive_baseline_for_the_task(self):
        scoreboard = load_module(
            "scripts/build-benchmark-scoreboard.py", "aers_scoreboard_structural"
        )
        self.assertIn("structural-demand-recovery", scoreboard.NAIVE_BUILDERS)
        self.assertIn("structural-demand-recovery", scoreboard.NAIVE_MOVE)

    def test_the_naive_baseline_actually_fails_the_task(self):
        # A baseline that passes would make the scoreboard's headline gap a lie.
        scoreboard = load_module(
            "scripts/build-benchmark-scoreboard.py", "aers_scoreboard_structural2"
        )
        task = load_task()
        reference = load_reference(task)
        naive = {"task": task["id"], "method": "naive"}
        naive.update(scoreboard.NAIVE_BUILDERS[task["id"]](reference))
        graded = check_benchmark.grade(task, naive, check_benchmark.compute_truth(task))
        required_failures = [g["id"] for g in graded if g["required"] and not g["passed"]]
        self.assertTrue(required_failures, "the naive move must fail required golds")


if __name__ == "__main__":
    unittest.main()
