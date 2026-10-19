"""Tests for the ``aers-score`` CLI.

Two properties carry the whole design and are pinned here:

1. **The CLI is a front end, not a fork.** Grading the committed reference
   candidates through ``aers-score`` must reproduce exactly what CI's
   ``check_benchmark.py`` run produces — same golds, same weights, same
   verdicts. If someone adds a grader branch to the checker and the CLI drifts,
   these tests fail.
2. **Nothing an outsider can hand us is trusted.** Scaffolds, half-finished
   candidates, unknown task ids and malformed JSON all have to produce a useful
   message rather than a traceback or a silently flattering score.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from _helpers import ROOT

import sys

sys.path.insert(0, str(ROOT))
from aers_score import cli, exam as exam_mod  # noqa: E402


def open_exam() -> exam_mod.Exam:
    return exam_mod.Exam(ROOT, "test")


def run_cli(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
        code = cli.main(argv)
    return code, buf.getvalue() + err.getvalue()


class TestExamResolution(unittest.TestCase):
    def test_repo_root_looks_like_a_checkout(self):
        self.assertTrue(exam_mod.looks_like_checkout(ROOT))

    def test_explicit_repo_wins(self):
        root, source = exam_mod.find_repo(ROOT)
        self.assertEqual(root, ROOT)
        self.assertEqual(source, "--repo")

    def test_an_explicit_repo_without_an_exam_is_an_error_not_a_fallback(self):
        # Silently grading against a different checkout than the one the user
        # named would produce a real-looking score for the wrong exam.
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(exam_mod.ExamNotFound) as ctx:
                exam_mod.find_repo(tmp)
        message = str(ctx.exception)
        self.assertIn("--repo", message)
        self.assertIn("no benchmark/ exam", message)

    def test_no_checkout_anywhere_explains_how_to_get_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            deep = Path(tmp) / "a" / "b"
            deep.mkdir(parents=True)
            # Neither cwd nor the install location may resolve, so both are
            # redirected at the empty tree.
            with mock.patch.object(exam_mod.Path, "cwd", staticmethod(lambda: deep)), \
                 mock.patch.object(exam_mod, "_INSTALL_ROOT", deep):
                with self.assertRaises(exam_mod.ExamNotFound) as ctx:
                    exam_mod.find_repo(None)
        message = str(ctx.exception)
        self.assertIn("git clone", message)
        self.assertIn(exam_mod.ENV_VAR, message)

    def test_every_task_spec_loads_and_validates(self):
        tasks = open_exam().tasks()
        on_disk = {p.stem for p in (ROOT / "benchmark" / "tasks").glob("*.toml")}
        self.assertEqual(set(tasks), on_disk)


class TestReproducesTheReferenceScore(unittest.TestCase):
    """The reference pipeline must score 17/17 through the CLI, as it does in CI."""

    @classmethod
    def setUpClass(cls):
        cls.exam = open_exam()

    def test_each_reference_candidate_passes_every_required_gold(self):
        for task_id, spec in self.exam.tasks().items():
            with self.subTest(task=task_id):
                candidate = self.exam.reference_candidate(task_id)
                self.assertIsNotNone(
                    candidate, f"{task_id} has no committed reference candidate"
                )
                card = self.exam.grade_candidate(task_id, candidate)
                self.assertTrue(card["graded"], card["problems"])
                self.assertEqual(card["required_failures"], [])
                self.assertEqual(
                    card["earned"], card["possible"], "reference must take full marks"
                )

    def test_cli_grade_reproduces_a_clean_sweep(self):
        with tempfile.TemporaryDirectory() as tmp:
            for task_id in self.exam.tasks():
                payload = self.exam.reference_candidate(task_id)
                (Path(tmp) / f"{task_id}.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            code, out = run_cli(["--repo", str(ROOT), "grade", tmp, "--strict"])
        self.assertEqual(code, 0, out)
        n = len(self.exam.tasks())
        self.assertIn(f"{n}/{n} tasks with every required gold passing", out)


class TestScaffolding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exam = open_exam()

    def test_scaffold_covers_every_field_the_graders_read(self):
        # A scaffold that omits a graded field sends the user into a required
        # failure with no clue which key was missing.
        for task_id in self.exam.tasks():
            with self.subTest(task=task_id):
                scaffold = cli._scaffold(self.exam, task_id)
                numeric, maps = self.exam.candidate_fields(task_id)
                for field in numeric + maps:
                    self.assertIn(field, scaffold)

    def test_scaffold_advertises_no_field_the_checker_ignores(self):
        checker = self.exam.checker
        for task_id in self.exam.tasks():
            with self.subTest(task=task_id):
                scaffold = cli._scaffold(self.exam, task_id)
                declared = set(checker.CANDIDATE_NUMERIC_FIELDS.get(task_id, ()))
                declared |= set(checker.CANDIDATE_NUMERIC_MAP_FIELDS.get(task_id, ()))
                spec = self.exam.task(task_id)
                for gold in spec.get("gold", []):
                    for key in ("field", "near_field", "far_field"):
                        if isinstance(gold.get(key), str):
                            declared.add(gold[key])
                extra = set(scaffold) - declared - {"task", "method", "_readme"}
                self.assertEqual(extra, set(), "scaffold invents fields")

    def test_init_then_grade_reports_unfilled_rather_than_type_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _ = run_cli(
                ["--repo", str(ROOT), "init", tmp, "--task", "rdd-recovery"]
            )
            self.assertEqual(code, 0)
            code, out = run_cli(["--repo", str(ROOT), "grade", tmp])
        self.assertEqual(code, 0, out)
        self.assertIn("still unfilled", out)
        self.assertNotIn("must be numeric", out)

    def test_init_does_not_clobber_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "rdd-recovery.json"
            run_cli(["--repo", str(ROOT), "init", tmp, "--task", "rdd-recovery"])
            target.write_text('{"task": "rdd-recovery", "local_att": 3.0}', encoding="utf-8")
            run_cli(["--repo", str(ROOT), "init", tmp, "--task", "rdd-recovery"])
            self.assertIn("local_att", target.read_text(encoding="utf-8"))
            run_cli(["--repo", str(ROOT), "init", tmp, "--task", "rdd-recovery", "--force"])
            self.assertIn("null", target.read_text(encoding="utf-8"))


class TestStripUnfilled(unittest.TestCase):
    def test_nulls_become_unfilled_and_underscores_are_dropped(self):
        payload, unfilled = exam_mod.strip_unfilled(
            {"task": "t", "a": 1.0, "b": None, "_readme": "note"}
        )
        self.assertEqual(payload, {"task": "t", "a": 1.0})
        self.assertEqual(unfilled, ["b"])

    def test_zero_is_a_real_answer_not_an_unfilled_field(self):
        payload, unfilled = exam_mod.strip_unfilled({"task": "t", "effect": 0.0})
        self.assertEqual(payload, {"task": "t", "effect": 0.0})
        self.assertEqual(unfilled, [])


class TestBadInput(unittest.TestCase):
    def test_malformed_json_names_the_file_and_the_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "broken.json").write_text("{not json", encoding="utf-8")
            with self.assertRaises(exam_mod.ExamError) as ctx:
                exam_mod.load_candidate_dir(Path(tmp))
        self.assertIn("broken.json", str(ctx.exception))

    def test_candidate_without_a_task_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "x.json").write_text('{"local_att": 3.0}', encoding="utf-8")
            with self.assertRaises(exam_mod.ExamError) as ctx:
                exam_mod.load_candidate_dir(Path(tmp))
        self.assertIn("task", str(ctx.exception))

    def test_two_files_claiming_one_task_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("a.json", "b.json"):
                (Path(tmp) / name).write_text('{"task": "rdd-recovery"}', encoding="utf-8")
            with self.assertRaises(exam_mod.ExamError) as ctx:
                exam_mod.load_candidate_dir(Path(tmp))
        self.assertIn("rdd-recovery", str(ctx.exception))

    def test_unknown_task_id_is_reported_not_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "x.json").write_text('{"task": "no-such-task"}', encoding="utf-8")
            code, out = run_cli(["--repo", str(ROOT), "grade", tmp])
        self.assertEqual(code, 1)
        self.assertIn("unknown benchmark task", out)

    def test_submission_json_is_not_mistaken_for_a_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "submission.json").write_text("{}", encoding="utf-8")
            (Path(tmp) / "rdd-recovery.json").write_text(
                '{"task": "rdd-recovery"}', encoding="utf-8"
            )
            loaded = exam_mod.load_candidate_dir(Path(tmp))
        self.assertEqual(set(loaded), {"rdd-recovery"})


class TestFabricationStillFails(unittest.TestCase):
    """The CLI must not become a softer grader than the checker."""

    def test_fabricated_numbers_fail_the_honest_cross_check(self):
        exam = open_exam()
        candidate = dict(exam.reference_candidate("rdd-recovery"))
        candidate["naive_jump"] = 3.0  # claim the naive move was fine after all
        card = exam.grade_candidate("rdd-recovery", candidate)
        self.assertIn("honest-reported-numbers", card["required_failures"])

    def test_strict_grade_exits_nonzero_on_a_required_failure(self):
        exam = open_exam()
        with tempfile.TemporaryDirectory() as tmp:
            candidate = dict(exam.reference_candidate("rdd-recovery"))
            candidate["local_att"] = 99.0
            (Path(tmp) / "rdd-recovery.json").write_text(
                json.dumps(candidate), encoding="utf-8"
            )
            lenient, _ = run_cli(["--repo", str(ROOT), "grade", tmp])
            strict, _ = run_cli(["--repo", str(ROOT), "grade", tmp, "--strict"])
        self.assertEqual(lenient, 0, "falling into a trap is a result, not a tooling error")
        self.assertEqual(strict, 1)


class TestSubmit(unittest.TestCase):
    def _reference_dir(self, tmp: str) -> str:
        exam = open_exam()
        for task_id in exam.tasks():
            (Path(tmp) / f"{task_id}.json").write_text(
                json.dumps(exam.reference_candidate(task_id)), encoding="utf-8"
            )
        return tmp

    def test_submission_is_well_formed(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._reference_dir(tmp)
            code, out = run_cli(
                ["--repo", str(ROOT), "submit", tmp, "--agent", "reference-pipeline"]
            )
            self.assertEqual(code, 0, out)
            payload = json.loads((Path(tmp) / "submission.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], cli.SUBMISSION_SCHEMA)
        self.assertEqual(payload["agent"], "reference-pipeline")
        n = len(open_exam().tasks())
        self.assertEqual(payload["summary"]["tasks_attempted"], n)
        self.assertEqual(payload["summary"]["tasks_all_required_passing"], n)
        self.assertEqual(set(payload["tasks"]), set(open_exam().tasks()))

    def test_submit_refuses_an_unfinished_scaffold(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_cli(["--repo", str(ROOT), "init", tmp, "--task", "rdd-recovery"])
            code, out = run_cli(["--repo", str(ROOT), "submit", tmp, "--agent", "half-done"])
        self.assertEqual(code, 2)
        self.assertIn("unfilled", out)
        self.assertFalse((Path(tmp) / "submission.json").exists())


class TestJsonOutput(unittest.TestCase):
    def test_every_reporting_subcommand_emits_parseable_json(self):
        for argv in (
            ["--repo", str(ROOT), "--json", "where"],
            ["--repo", str(ROOT), "--json", "tasks"],
            ["--repo", str(ROOT), "--json", "describe", "rdd-recovery"],
        ):
            with self.subTest(argv=argv):
                code, out = run_cli(argv)
                self.assertEqual(code, 0, out)
                json.loads(out)


class TestPackaging(unittest.TestCase):
    def test_pyproject_exposes_the_console_script_and_no_dependencies(self):
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('aers-score = "aers_score.cli:main"', text)
        self.assertIn("dependencies = []", text)
        self.assertIn('requires-python = ">=3.9"', text)

    def test_python_compat_target_covers_the_package(self):
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn("aers_score/*.py", makefile)


if __name__ == "__main__":
    unittest.main()
