"""Guard the Card (1995) end-to-end replication anchors.

The demo at demo-notebooks/card-1995-iv/ reproduces the paper's headline
returns-to-schooling comparison from the vendored NLSYM extract. These tests pin
the computed values so a regression in the sample restriction, the control set,
the closed-form solver, or — most easily broken and least visible — the 2SLS
variance cannot ship silently.

The 2SLS standard error gets the most attention here on purpose. It is the one
number in the replication that a plausible refactor can quietly change by ~3%
(swapping the structural residuals for the second stage's own), which is exactly
the size of error that survives review.
"""

from __future__ import annotations

import unittest

from _helpers import ROOT, load_module

card1995 = load_module("demo-notebooks/card-1995-iv/replicate_card1995.py", "aers_card1995")


class TestCard1995(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.got = card1995.replicate(card1995.load(card1995.DATA))

    def test_sample_restriction_gives_the_published_n(self):
        # Men with an observed 1976 wage: 3,010 of the extract's rows.
        self.assertEqual(self.got["n"], 3010)

    def test_ols_return_matches_the_paper(self):
        self.assertAlmostEqual(self.got["ols_return"], 0.0747, places=4)
        self.assertAlmostEqual(self.got["ols_se"], 0.0035, places=4)

    def test_first_stage_matches_the_paper(self):
        self.assertAlmostEqual(self.got["first_stage_coef"], 0.3199, places=4)
        self.assertAlmostEqual(self.got["first_stage_se"], 0.0879, places=4)
        self.assertAlmostEqual(self.got["first_stage_F"], 13.26, places=1)

    def test_iv_return_matches_the_paper(self):
        self.assertAlmostEqual(self.got["iv_return"], 0.1315, places=4)
        self.assertAlmostEqual(self.got["iv_se"], 0.0550, places=4)

    def test_the_finding_is_that_iv_exceeds_ols(self):
        # The paper's whole point. A run that hits both point estimates but
        # loses this comparison has not replicated the argument.
        self.assertGreater(self.got["iv_return"], self.got["ols_return"])
        self.assertAlmostEqual(self.got["iv_minus_ols"], 0.0568, places=4)

    def test_the_naive_2sls_standard_error_is_reported_and_wrong(self):
        # Reading the second stage's own OLS standard error off the output is
        # the classic manual-2SLS error. The demo computes it so the size of
        # the mistake is visible; it must stay distinguishable from the correct
        # one, and it must be the larger of the two here.
        naive = self.got["iv_se_naive_second_stage"]
        correct = self.got["iv_se"]
        self.assertAlmostEqual(naive, 0.0565, places=4)
        self.assertGreater(naive, correct)
        self.assertGreater(abs(naive - correct), 1e-4)


class TestReplicationIsSelfGating(unittest.TestCase):
    """The script must fail loudly, not print a warning, when it misses."""

    def test_every_published_anchor_is_checked_by_the_script(self):
        got = card1995.replicate(card1995.load(card1995.DATA))
        for key, (published, tol) in card1995.PUBLISHED.items():
            with self.subTest(anchor=key):
                self.assertIn(key, got, "anchor is declared but never computed")
                if key == "n":
                    self.assertEqual(int(got[key]), int(published))
                else:
                    self.assertLessEqual(abs(got[key] - published), tol)

    def test_main_exits_zero_on_the_committed_data(self):
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(card1995.main(), 0)

    def test_committed_estimates_json_matches_a_fresh_run(self):
        import contextlib
        import io
        import json

        path = ROOT / "demo-notebooks" / "card-1995-iv" / "estimates.json"
        before = path.read_text(encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            card1995.main()
        after = path.read_text(encoding="utf-8")
        self.assertEqual(before, after, "estimates.json is stale; re-run the script")
        payload = json.loads(after)
        self.assertEqual(payload["candidate"], "aers-card1995-replication")
        self.assertIn("iv_return", payload["coefficients"])
        self.assertIn("se", payload["coefficients"]["iv_return"])


if __name__ == "__main__":
    unittest.main()
