import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
from prepare_data import infer_base


class DataTests(unittest.TestCase):
    def test_missing_gps_does_not_poison_depot(self):
        rows=[{'accept_gps_lng':'nan','accept_gps_lat':'nan'},
              {'accept_gps_lng':'126.5','accept_gps_lat':'43.8'},
              {'accept_gps_lng':'126.7','accept_gps_lat':'44.0'}]
        self.assertEqual(infer_base(rows),(126.6,43.9))
