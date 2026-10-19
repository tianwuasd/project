"""Tests for the environment doctor (`make doctor`).

The doctor exists to turn late, misleading gate failures into an early, honest
one-screen report. These tests pin the two properties that matter: every
non-passing row carries a remediation command, and the script's exit code
reflects blocking-vs-advisory correctly.
"""

from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

from _helpers import ROOT, load_module

doctor = load_module("scripts/doctor.py", "aers_doctor")


class TestReport(unittest.TestCase):
    def test_render_aligns_and_marks_each_status(self):
        report = doctor.Report()
        report.add(doctor.OK, "python", "3.12.0")
        report.add(doctor.FAIL, "sci-stack", "missing numpy", "make setup")
        rendered = report.render()
        self.assertIn("[ ok ] python", rendered)
        self.assertIn("[FAIL] sci-stack", rendered)
        # Names are padded to a common width so details line up.
        self.assertIn("python     3.12.0", rendered)

    def test_fixes_are_rendered_for_failures_and_warnings(self):
        report = doctor.Report()
        report.add(doctor.FAIL, "submodule", "missing", "git submodule update --init")
        report.add(doctor.WARN, "venv", "not active", "source .venv/bin/activate")
        report.add(doctor.OK, "git", "/usr/bin/git", "should not appear")
        fixes = report.render_fixes()
        self.assertIn("git submodule update --init", fixes)
        self.assertIn("source .venv/bin/activate", fixes)
        self.assertNotIn("should not appear", fixes)

    def test_no_fixes_renders_empty(self):
        report = doctor.Report()
        report.add(doctor.OK, "python", "3.12.0")
        self.assertEqual(report.render_fixes(), "")


class TestChecksAreActionable(unittest.TestCase):
    """Every blocking row must tell the reader how to unblock themselves."""

    def test_missing_scientific_stack_names_the_packages_and_the_fix(self):
        report = doctor.Report()
        with mock.patch.object(doctor, "_module_available", return_value=False):
            doctor.check_scientific_stack(report)
        (status, name, detail, fix), = report.rows
        self.assertEqual(status, doctor.FAIL)
        self.assertEqual(name, "sci-stack")
        for package in doctor.SCIENTIFIC_STACK:
            self.assertIn(package, detail)
        self.assertIn("make setup", fix)

    def test_present_scientific_stack_passes(self):
        report = doctor.Report()
        with mock.patch.object(doctor, "_module_available", return_value=True):
            doctor.check_scientific_stack(report)
        self.assertEqual(report.rows[0][0], doctor.OK)
        self.assertEqual(report.failures, [])

    def test_missing_submodule_points_at_git_submodule_update(self):
        report = doctor.Report()
        with mock.patch.object(doctor, "SUBMODULE_SENTINEL", ROOT / "does-not-exist"):
            doctor.check_submodule(report)
        (status, _name, _detail, fix), = report.rows
        self.assertEqual(status, doctor.FAIL)
        self.assertIn("git submodule update --init --recursive", fix)

    def test_every_failing_or_warning_row_has_a_fix(self):
        # Guards against adding a check that reports a problem but leaves the
        # reader with nothing to run. `check_generated_artifacts` is the one
        # exception path (a missing generator is a broken checkout, not a
        # user-fixable state), so it is skipped here.
        report = doctor.build_report(skip_slow=True)
        for status, name, detail, fix in report.rows:
            if status in (doctor.FAIL, doctor.WARN):
                with self.subTest(check=name):
                    self.assertTrue(fix, f"{name} reports {detail!r} with no remediation")


class TestExitCodes(unittest.TestCase):
    def _run_with(self, rows, argv):
        report = doctor.Report()
        for row in rows:
            report.add(*row)
        with mock.patch.object(doctor, "build_report", return_value=report):
            # main() is a reporting CLI; swallow its screen output here.
            with contextlib.redirect_stdout(io.StringIO()):
                return doctor.main(argv)

    def test_clean_environment_exits_zero(self):
        self.assertEqual(self._run_with([(doctor.OK, "python", "3.12.0")], ["--quick"]), 0)

    def test_failure_exits_one(self):
        rows = [(doctor.FAIL, "sci-stack", "missing numpy", "make setup")]
        self.assertEqual(self._run_with(rows, ["--quick"]), 1)

    def test_warning_is_advisory_by_default_and_blocking_under_strict(self):
        rows = [(doctor.WARN, "venv", "not active", "source .venv/bin/activate")]
        self.assertEqual(self._run_with(rows, ["--quick"]), 0)
        self.assertEqual(self._run_with(rows, ["--quick", "--strict"]), 1)


class TestMakefileWiring(unittest.TestCase):
    def test_makefile_exposes_setup_and_doctor(self):
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn("\nsetup:\n", makefile)
        self.assertIn("\ndoctor:\n", makefile)
        self.assertIn("scripts/doctor.py", makefile)
        phony = next(
            line for line in makefile.splitlines() if line.startswith(".PHONY:")
        )
        self.assertIn("setup", phony.split())
        self.assertIn("doctor", phony.split())

    def test_setup_refuses_to_layer_venvs_from_different_interpreters(self):
        """`python3 -m venv` over a venv built by another interpreter breaks it.

        The result fails with "No module named encodings" — a message that
        names nothing useful and surfaces well after the command that caused
        it. The target has to refuse up front rather than produce it.
        """
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        setup = makefile.split("\nsetup:", 1)[1].split("\n\n", 1)[0]
        self.assertIn("sys.version_info", setup, "no interpreter check")
        self.assertIn("rm -rf .venv", setup, "no remediation offered")
        self.assertIn("encodings", setup, "does not name the symptom it prevents")
        # And it must not silently destroy an existing environment.
        self.assertNotIn("--clear", setup)
        self.assertNotIn("rm -rf .venv &&\n", setup)

    def test_paper_workflow_gate_preflights_the_scientific_stack(self):
        # The misleading "RIGOR.md is STALE" failure is the exact thing this
        # preflight replaces; keep the guard wired in.
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        gate = makefile.split("paper-workflow-check:", 1)[1].split("\n\n", 1)[0]
        self.assertIn("import numpy", gate)
        self.assertIn("make setup", gate)


if __name__ == "__main__":
    unittest.main()
