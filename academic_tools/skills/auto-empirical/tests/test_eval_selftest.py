"""Tests for the eval-harness discrimination self-test.

The self-test is what converts "41 scenarios" — a number anyone can inflate by
writing rubrics that match everything — into "9 scenarios proven to separate a
correct answer from a plausibly wrong one". That claim is only worth making if
the mechanism actually fails when a rubric stops discriminating, so most of what
follows constructs rubrics that are broken in each of the two possible
directions and requires the self-test to say so.

History: this started as a hand-rolled test over three flagship scenarios. Its
logic moved into `run_evals.py --selftest` so contributors get the same bar
without writing a bespoke test each time, and what remains here tests the
mechanism rather than three particular scenarios.
"""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from _helpers import ROOT, load_module

run_evals = load_module("eval-harness/run_evals.py", "aers_run_evals_selftest")

FIXTURES = ROOT / "eval-harness" / "fixtures"


def scenario(**overrides) -> dict:
    """A minimal well-formed scenario with one required regex item."""
    base = {
        "id": "synthetic-probe",
        "skill": "skills/00-Full-empirical-analysis-skill_StatsPAI",
        "title": "probe",
        "category": "causal-identification",
        "severity": "high",
        "rubric": [
            {
                "id": "names-the-trap",
                "check": "regex_any",
                "required": True,
                "weight": 1,
                "description": "names it",
                "patterns": ["(?i)endogenous"],
            }
        ],
    }
    base.update(overrides)
    return base


class SelftestHarness(unittest.TestCase):
    """Run selftest_scenario against fixtures written into a temp directory."""

    def run_probe(self, spec: dict, good: str, bad: str) -> list[str]:
        with tempfile.TemporaryDirectory() as tmp:
            original = run_evals.FIXTURE_DIR
            try:
                run_evals.FIXTURE_DIR = Path(tmp)
                directory = Path(tmp) / spec["id"]
                directory.mkdir()
                (directory / run_evals.FIXTURE_PASS).write_text(good, encoding="utf-8")
                (directory / run_evals.FIXTURE_FAIL).write_text(bad, encoding="utf-8")
                return run_evals.selftest_scenario(spec)
            finally:
                run_evals.FIXTURE_DIR = original


class TestMechanism(SelftestHarness):
    def test_a_discriminating_rubric_passes(self):
        problems = self.run_probe(
            scenario(),
            good="Price is endogenous here, so OLS is biased.",
            bad="Just regress the shares on price; the coefficient is the elasticity.",
        )
        self.assertEqual(problems, [])

    def test_a_rubric_the_wrong_answer_sails_through_is_rejected(self):
        # The classic way to inflate coverage: a pattern that matches prose.
        spec = scenario()
        spec["rubric"][0]["patterns"] = ["(?i)the"]
        problems = self.run_probe(
            spec,
            good="The price is endogenous here, so OLS is biased.",
            bad="Just regress the shares on price; the coefficient is the elasticity.",
        )
        # The loose pattern matches both answers, so only the discrimination
        # complaint should fire — not the "fails on the correct answer" one.
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("does not discriminate", problems[0])

    def test_a_rubric_no_correct_answer_can_satisfy_is_rejected(self):
        spec = scenario()
        spec["rubric"][0]["patterns"] = ["(?i)zzz-impossible-token"]
        problems = self.run_probe(
            spec,
            good="Price is endogenous here, so OLS is biased.",
            bad="Just regress the shares on price.",
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("fails on the correct answer", problems[0])

    def test_a_negative_item_counts_as_discrimination(self):
        # regex_none items ("must not say this") are the natural way to catch a
        # wrong answer, so they have to count.
        spec = scenario()
        spec["rubric"] = [
            {
                "id": "no-endorsement",
                "check": "regex_none",
                "required": True,
                "weight": 1,
                "description": "must not endorse it",
                "patterns": ["(?i)price can be treated as exogenous"],
            }
        ]
        problems = self.run_probe(
            spec,
            good="Price is endogenous and must be instrumented.",
            bad="Price can be treated as exogenous in this market.",
        )
        self.assertEqual(problems, [])

    def test_only_optional_items_failing_is_not_enough(self):
        # If the wrong answer only trips non-required items, the scenario does
        # not actually gate anything.
        spec = scenario()
        spec["rubric"] = [
            {
                "id": "required-and-loose",
                "check": "regex_any",
                "required": True,
                "weight": 1,
                "description": "loose",
                "patterns": ["(?i)price"],
            },
            {
                "id": "optional-and-tight",
                "check": "regex_any",
                "required": False,
                "weight": 1,
                "description": "tight",
                "patterns": ["(?i)endogenous"],
            },
        ]
        problems = self.run_probe(
            spec,
            good="Price is endogenous here.",
            bad="The price coefficient is the elasticity.",
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("does not discriminate", problems[0])

    def test_a_scenario_with_no_auto_items_is_rejected(self):
        spec = scenario()
        spec["rubric"] = [
            {
                "id": "judgement-call",
                "check": "manual",
                "required": True,
                "weight": 1,
                "description": "manual",
                "guidance": "judge it",
            }
        ]
        problems = self.run_probe(spec, good="anything", bad="anything")
        self.assertEqual(len(problems), 1)
        self.assertIn("no auto-checkable rubric items", problems[0])

    def test_missing_fixtures_are_reported_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = run_evals.FIXTURE_DIR
            try:
                run_evals.FIXTURE_DIR = Path(tmp)
                (Path(tmp) / "synthetic-probe").mkdir()
                problems = run_evals.selftest_scenario(scenario())
            finally:
                run_evals.FIXTURE_DIR = original
        self.assertEqual(len(problems), 2)
        self.assertTrue(any(run_evals.FIXTURE_PASS in p for p in problems))
        self.assertTrue(any(run_evals.FIXTURE_FAIL in p for p in problems))


class TestCommittedFixtures(unittest.TestCase):
    def test_the_committed_scenarios_pass_their_own_selftest(self):
        problems, checked = run_evals.selftest_scenarios(run_evals.load_scenarios())
        self.assertEqual(problems, [])
        self.assertGreaterEqual(checked, 9)

    def test_every_critical_scenario_ships_fixtures(self):
        # The rule the self-test enforces, asserted directly so the intent is
        # readable without reverse-engineering the error message.
        for s in run_evals.load_scenarios():
            if s.get("severity") in run_evals.SEVERITIES_REQUIRING_FIXTURES:
                with self.subTest(scenario=s["id"]):
                    self.assertTrue(
                        run_evals.has_fixtures(s["id"]),
                        f"{s['id']} is {s['severity']} but has no fixture pair",
                    )

    def test_no_orphan_fixture_directories(self):
        self.assertEqual(run_evals.orphan_fixture_dirs(run_evals.load_scenarios()), [])

    def test_fixtures_are_not_trivially_short(self):
        # A two-line "fail.md" tests the regex, not the scenario.
        for directory in sorted(p for p in FIXTURES.iterdir() if p.is_dir()):
            for name in (run_evals.FIXTURE_PASS, run_evals.FIXTURE_FAIL):
                path = directory / name
                with self.subTest(fixture=f"{directory.name}/{name}"):
                    self.assertGreater(
                        len(path.read_text(encoding="utf-8").split()),
                        60,
                        "a fixture this short is not a plausible answer",
                    )


class TestCliWiring(unittest.TestCase):
    def _run(self, argv):
        buf, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            code = run_evals.main(argv)
        return code, buf.getvalue() + err.getvalue()

    def test_selftest_flag_reports_and_exits_zero(self):
        code, out = self._run(["--selftest"])
        self.assertEqual(code, 0, out)
        self.assertIn("Discrimination self-test passed", out)

    def test_min_fixtures_gate_fails_when_unmet(self):
        code, out = self._run(["--min-fixtures", "9999"])
        self.assertEqual(code, 1)
        self.assertIn("--min-fixtures", out)

    def test_gates_are_wired_into_make_and_ci(self):
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn("--selftest", makefile)
        self.assertIn("--min-fixtures", makefile)
        workflow = (ROOT / ".github" / "workflows" / "quality-evals.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("--selftest", workflow)
        self.assertIn("--min-fixtures", workflow)


if __name__ == "__main__":
    unittest.main()
