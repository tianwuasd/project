"""Tests for requirements.txt and the interpreter contract it depends on.

`requirements.txt` is the pinned scientific stack for the two Paper-WorkFlow
checkers that do real work: `check_demo_execution.py`, which executes
`did_demo.ipynb`, and `check_defense_deck.py`, which builds the Stage 9 defence
deck via python-pptx. Everything else in the repo is stdlib-only, and the CI
matrix deliberately includes Python 3.9 for that reason.

Those two facts interact. The `linearmodels` floor was raised to 7.0 because
the 5.x wheels are built against the NumPy 1.x ABI and print a crash warning
under the NumPy 2.x this file allows — but 7.0 requires Python >= 3.10. That is
only safe while every workflow which installs this file runs on 3.10+, and
while the 3.9 leg installs nothing. This test pins both halves so a future
workflow edit cannot quietly make the constraint unsatisfiable on 3.9.
"""

from __future__ import annotations

import re
import unittest

from _helpers import ROOT

REQUIREMENTS = ROOT / "requirements.txt"
WORKFLOWS = ROOT / ".github" / "workflows"

# Packages whose published wheels require a newer interpreter than the repo's
# CI floor, mapped to the minimum they need.
INTERPRETER_FLOORS = {"linearmodels": (3, 10)}


def _requirement_lines() -> list[str]:
    return [
        line.strip()
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _workflow_text(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def _installers() -> list[str]:
    """Workflow files that `pip install -r requirements.txt`."""
    return sorted(
        path.name
        for path in WORKFLOWS.glob("*.yml")
        if "requirements.txt" in path.read_text(encoding="utf-8")
    )


def _pinned_versions(text: str) -> list[tuple[int, int]]:
    out = []
    for match in re.finditer(r'python-version:\s*"([0-9]+)\.([0-9]+)"', text):
        out.append((int(match.group(1)), int(match.group(2))))
    return out


class TestRequirementsShape(unittest.TestCase):
    def test_every_requirement_is_bounded_on_both_sides(self):
        # An unbounded upper edge lets a major release land in CI unannounced;
        # an unbounded lower edge makes "known to work" meaningless.
        for line in _requirement_lines():
            with self.subTest(requirement=line):
                self.assertIn(">=", line, "needs a lower bound")
                self.assertIn("<", line, "needs an upper bound")

    def test_the_expected_packages_are_present(self):
        names = {re.split(r"[<>=]", line)[0] for line in _requirement_lines()}
        self.assertEqual(
            names,
            {
                "numpy",
                "pandas",
                "matplotlib",
                "statsmodels",
                "linearmodels",
                "python-pptx",
            },
            "requirements.txt is the demo gate's stack; changing its membership "
            "should be a deliberate, documented change",
        )


class TestInterpreterContract(unittest.TestCase):
    def test_installers_run_on_an_interpreter_the_pins_support(self):
        installers = _installers()
        self.assertTrue(installers, "no workflow installs requirements.txt any more")
        needed = max(INTERPRETER_FLOORS.values())
        for name in installers:
            versions = _pinned_versions(_workflow_text(name))
            with self.subTest(workflow=name):
                self.assertTrue(versions, f"{name} installs requirements.txt but pins no Python")
                self.assertGreaterEqual(
                    min(versions),
                    needed,
                    f"{name} would install a requirement needing Python "
                    f"{needed[0]}.{needed[1]}+",
                )

    def test_the_python_39_matrix_leg_installs_nothing(self):
        # The 3.9 leg exists to prove the repo's own tooling is stdlib-only.
        # The moment it pip-installs this file, the linearmodels floor breaks it.
        quality = _workflow_text("quality-evals.yml")
        self.assertIn('"3.9"', quality, "the stdlib-only floor is no longer tested")
        self.assertNotIn("requirements.txt", quality)


class TestNumpyAbiFloor(unittest.TestCase):
    def test_linearmodels_floor_stays_past_the_numpy2_abi_break(self):
        line = next(
            line for line in _requirement_lines() if line.startswith("linearmodels")
        )
        match = re.search(r">=([0-9]+)", line)
        self.assertIsNotNone(match)
        self.assertGreaterEqual(
            int(match.group(1)),
            7,
            "linearmodels 5.x is compiled against the NumPy 1.x ABI and warns on "
            "import under the numpy>=2 this file allows",
        )

    def test_numpy_upper_bound_still_admits_2x(self):
        line = next(line for line in _requirement_lines() if line.startswith("numpy"))
        self.assertIn("<3", line)


if __name__ == "__main__":
    unittest.main()
