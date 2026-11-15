from pathlib import Path
import unittest

from chd_mini.simulation import load_environment


class SimulationEnvironment(unittest.TestCase):
    def test_existing_profile_and_path_interface(self):
        environment = load_environment(Path("D:/projects/test"))
        self.assertEqual(environment["profile"]["gpu"]["count"], 8)
        self.assertFalse(environment["profile"]["live_verified"])
        server = environment["profile"]["paths"]["code"]
        state = {"mappings": {server: "D:/projects/project/chd-segmentation"}}
        resolved = environment["module"].resolve_path(server + "/run.py", state)
        self.assertEqual(Path(resolved), Path("D:/projects/project/chd-segmentation/run.py"))
        with self.assertRaises(ValueError):
            environment["module"].resolve_path(server + "/../escape", state)
        self.assertEqual(Path(environment["module"].__file__).resolve(), Path("D:/projects/test/simulator.py"))


if __name__ == "__main__":
    unittest.main()
