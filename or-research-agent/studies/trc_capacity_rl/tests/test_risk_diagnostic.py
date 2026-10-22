import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'development'))
from diagnose_policy_risk import summarize_risks


class RiskDiagnosticTests(unittest.TestCase):
    def test_unresolved_records_are_preserved_in_summary(self):
        rows = [dict(status='exact', risk=.02, risk_upper=.02000001),
                dict(status='state_limit', risk=None, risk_upper=1.)]
        result = summarize_risks(rows, alpha=.01)
        self.assertEqual(result['records'], 2)
        self.assertEqual(result['unresolved_count'], 1)
        self.assertEqual(result['mean_risk_exact_only'], .02)
        self.assertEqual(result['exact_records_over_limit'], 1)
        self.assertEqual(result['records_not_certified_within_limit'], 2)
        self.assertAlmostEqual(result['mean_risk_upper_all_records'], .510000005)

    def test_all_unresolved_is_not_reported_as_zero(self):
        result = summarize_risks([dict(status='state_limit', risk=None, risk_upper=1.)])
        self.assertIsNone(result['mean_risk_exact_only'])
        self.assertEqual(result['unresolved_count'], 1)
        self.assertEqual(result['mean_risk_upper_all_records'], 1.)


if __name__ == '__main__':
    unittest.main()
