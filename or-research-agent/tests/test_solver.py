import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / 'skill' / 'scripts'))
from solve_demo import exact, greedy, validate_solution, demo_instance


class SolverTests(unittest.TestCase):
    def test_empty_demand_has_zero_cost(self):
        data = {'orders': [], 'patterns': [], 'capacities': {'resource': 0}}
        self.assertEqual(exact(data)['objective'], 0)
        self.assertEqual(exact(data)['status'], 'optimal')
        self.assertEqual(greedy(data)['objective'], 0)
        self.assertTrue(validate_solution(data, [])['feasible'])

    def test_known_optimum_and_independent_capacity_check(self):
        data = demo_instance()
        result = exact(data)
        self.assertEqual(result['status'], 'optimal')
        self.assertEqual(result['objective'], 11)
        self.assertTrue(validate_solution(data, result['selected'])['feasible'])
        self.assertEqual(greedy(data)['objective'], 12)
        self.assertFalse(validate_solution(data, ['A_air', 'B_air', 'C_ground'])['feasible'])

    def test_infeasible_and_missing_order(self):
        data = demo_instance()
        data['patterns'] = [p for p in data['patterns'] if p['order'] != 'C']
        self.assertEqual(exact(data)['status'], 'infeasible')
        self.assertFalse(validate_solution(data, ['A_ground', 'B_air'])['feasible'])

    def test_negative_resource_usage_rejected(self):
        data = demo_instance()
        data['patterns'][0]['usage']['air_t1'] = -1
        with self.assertRaises(ValueError):
            exact(data)


if __name__ == '__main__':
    unittest.main()
