"""Tests for the external benchmark scoreboard.

The board makes one strong promise: *the numbers published are our regrade, not
the submitter's claim.* That promise is only worth anything if the cross-check
actually fires, so most of these tests are attacks — a submission that inflates
its own score, one that ships no candidates to regrade, one that claims a task
it did not attempt, one whose candidate numbers were fabricated outright.

The ranking rules are pinned too: `SCOREBOARD_RULES.md` promises that attempting
fewer tasks can never improve a position, and that is a property of the sort key,
not of prose.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from _helpers import ROOT, load_module

board = load_module("scripts/build-external-scoreboard.py", "aers_build_external_scoreboard")

EXAMPLE = ROOT / "benchmark" / "external" / "example-agent"


class _SandboxedBoard:
    """Run the generator against a temporary benchmark/external/ tree."""

    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self._original = board.EXTERNAL_DIR
        board.EXTERNAL_DIR = self.dir

    def add(self, slug: str, *, from_example: bool = True) -> Path:
        target = self.dir / slug
        if from_example:
            shutil.copytree(EXAMPLE, target)
        else:
            (target / "candidates").mkdir(parents=True)
        return target

    def load(self):
        return board.load_submissions()

    def close(self):
        board.EXTERNAL_DIR = self._original
        self._tmp.cleanup()


class BoardTestCase(unittest.TestCase):
    def setUp(self):
        self.board = _SandboxedBoard()
        self.addCleanup(self.board.close)

    @staticmethod
    def _patch_submission(directory: Path, **changes):
        path = directory / "submission.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(changes)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload

    @staticmethod
    def _patch_candidate(directory: Path, task_id: str, **changes):
        path = directory / "candidates" / f"{task_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(changes)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class TestCommittedExample(unittest.TestCase):
    def test_the_committed_example_regrades_and_cross_checks(self):
        entries = board.load_submissions()
        slugs = {e["slug"] for e in entries}
        self.assertIn("example-agent", slugs)

    def test_the_example_demonstrates_a_partial_score(self):
        # A board whose only worked example is perfect teaches the wrong thing.
        entry = next(e for e in board.load_submissions() if e["slug"] == "example-agent")
        failing = [t for t, s in entry["scored"].items() if s["required_failures"]]
        passing = [t for t, s in entry["scored"].items() if not s["required_failures"]]
        self.assertTrue(failing, "the example must fail at least one task")
        self.assertTrue(passing, "the example must pass at least one task")

    def test_committed_page_matches_a_fresh_regrade(self):
        entries = board.load_submissions()
        rendered = board.render(entries, len(board.load_tasks()))
        committed = (ROOT / "docs" / "EXTERNAL_SCOREBOARD.md").read_text(encoding="utf-8")
        self.assertEqual(committed, rendered, "run `make catalog`")


class TestClaimsAreNeverTrusted(BoardTestCase):
    def test_an_inflated_summary_fails_the_build(self):
        directory = self.board.add("liar")
        payload = json.loads((directory / "submission.json").read_text(encoding="utf-8"))
        tasks = payload["tasks"]
        target = next(t for t, v in tasks.items() if v["required_failures"])
        tasks[target]["earned"] = tasks[target]["possible"]
        tasks[target]["required_failures"] = []
        self._patch_submission(directory, tasks=tasks)

        with self.assertRaises(board.SubmissionError) as ctx:
            self.board.load()
        message = str(ctx.exception)
        self.assertIn(target, message)
        self.assertIn("regrade says", message)

    def test_fabricated_candidate_numbers_fail_the_honest_golds(self):
        # Rewriting the candidate so the *claim* becomes true is the other half
        # of the attack. The graders recompute from the CSV, so it still fails.
        directory = self.board.add("fabricator")
        self._patch_candidate(directory, "rdd-recovery", local_att=3.0, naive_jump=3.0)
        with self.assertRaises(board.SubmissionError) as ctx:
            self.board.load()
        self.assertIn("rdd-recovery", str(ctx.exception))

    def test_claiming_a_task_with_no_candidate_file_fails(self):
        directory = self.board.add("phantom")
        payload = json.loads((directory / "submission.json").read_text(encoding="utf-8"))
        payload["tasks"]["card-iv-recovery"] = {
            "earned": 14,
            "possible": 14,
            "required_failures": [],
            "optional_failures": [],
        }
        self._patch_submission(directory, tasks=payload["tasks"])
        with self.assertRaises(board.SubmissionError) as ctx:
            self.board.load()
        self.assertIn("card-iv-recovery", str(ctx.exception))
        self.assertIn("no candidate file", str(ctx.exception))

    def test_shipping_a_candidate_the_summary_omits_fails(self):
        directory = self.board.add("silent")
        payload = json.loads((directory / "submission.json").read_text(encoding="utf-8"))
        dropped = sorted(payload["tasks"])[0]
        del payload["tasks"][dropped]
        self._patch_submission(directory, tasks=payload["tasks"])
        with self.assertRaises(board.SubmissionError) as ctx:
            self.board.load()
        self.assertIn(dropped, str(ctx.exception))

    def test_a_submission_without_candidates_is_refused(self):
        directory = self.board.add("claims-only")
        shutil.rmtree(directory / "candidates")
        with self.assertRaises(board.SubmissionError) as ctx:
            self.board.load()
        self.assertIn("regrade", str(ctx.exception))


class TestMetadataValidation(BoardTestCase):
    def test_wrong_schema_points_at_the_tool(self):
        directory = self.board.add("old-schema")
        self._patch_submission(directory, schema="aers-external-scoreboard/0")
        with self.assertRaises(board.SubmissionError) as ctx:
            self.board.load()
        self.assertIn("aers-score submit", str(ctx.exception))

    def test_unknown_origin_is_refused(self):
        directory = self.board.add("weird-origin")
        self._patch_submission(directory, origin="totally-legit")
        with self.assertRaises(board.SubmissionError) as ctx:
            self.board.load()
        self.assertIn("origin", str(ctx.exception))

    def test_missing_submission_json_names_the_tool(self):
        directory = self.board.add("no-metadata")
        (directory / "submission.json").unlink()
        with self.assertRaises(board.SubmissionError) as ctx:
            self.board.load()
        self.assertIn("aers-score submit", str(ctx.exception))

    def test_an_unusable_slug_is_refused(self):
        self.board.add("Not A Slug")
        with self.assertRaises(board.SubmissionError) as ctx:
            self.board.load()
        self.assertIn("lowercase", str(ctx.exception))


class TestRanking(unittest.TestCase):
    """`SCOREBOARD_RULES.md` §4: fewer attempts can never improve a position."""

    N = 17

    @staticmethod
    def _entry(agent, clean, dirty=0, points_per=10):
        scored = {}
        for i in range(clean):
            scored[f"clean-{i}"] = {
                "earned": points_per,
                "possible": points_per,
                "required_passed": 3,
                "required_total": 3,
                "required_failures": [],
            }
        for i in range(dirty):
            scored[f"dirty-{i}"] = {
                "earned": 0,
                "possible": points_per,
                "required_passed": 0,
                "required_total": 3,
                "required_failures": ["x"],
            }
        return {"agent": agent, "origin": "external", "url": "", "version": "",
                "notes": "", "slug": agent, "scored": scored}

    def test_more_clean_tasks_ranks_higher(self):
        few = self._entry("few", clean=2)
        many = self._entry("many", clean=5)
        order = sorted([few, many], key=lambda e: board._rank_key(e, self.N))
        self.assertEqual([e["agent"] for e in order], ["many", "few"])

    def test_skipping_the_hard_tasks_does_not_beat_attempting_them(self):
        # Same three clean tasks; the second agent also tried two and failed.
        cherry = self._entry("cherry-picker", clean=3)
        honest = self._entry("tried-everything", clean=3, dirty=2)
        order = sorted([cherry, honest], key=lambda e: board._rank_key(e, self.N))
        self.assertEqual(
            [e["agent"] for e in order],
            ["tried-everything", "cherry-picker"],
            "attempting more tasks must never cost a position",
        )

    def test_ties_break_on_points_then_coverage(self):
        low = self._entry("low-points", clean=3, points_per=5)
        high = self._entry("high-points", clean=3, points_per=9)
        order = sorted([low, high], key=lambda e: board._rank_key(e, self.N))
        self.assertEqual([e["agent"] for e in order], ["high-points", "low-points"])


class TestRendering(unittest.TestCase):
    def test_empty_board_says_so_instead_of_rendering_a_blank_table(self):
        rendered = board.render([], 17)
        self.assertIn("No third-party submissions yet", rendered)
        self.assertNotIn("| 1 |", rendered)

    def test_examples_are_not_ranked(self):
        entries = board.load_submissions()
        rendered = board.render(entries, len(board.load_tasks()))
        ranked_section = rendered.split("## Ranked entries", 1)[1].split("##", 1)[0]
        self.assertNotIn("example-agent", ranked_section)

    def test_the_page_states_that_scores_are_recomputed(self):
        rendered = board.render(board.load_submissions(), 17)
        self.assertIn("never displayed", rendered)
        self.assertIn("recomputed", rendered)


class TestDocsWiring(unittest.TestCase):
    def test_rules_document_exists_and_is_linked_from_the_board(self):
        rules = ROOT / "docs" / "SCOREBOARD_RULES.md"
        self.assertTrue(rules.exists())
        rendered = board.render(board.load_submissions(), 17)
        self.assertIn("SCOREBOARD_RULES.md", rendered)

    def test_makefile_builds_and_checks_the_board(self):
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn("build-external-scoreboard.py\n", makefile)
        self.assertIn("build-external-scoreboard.py --check", makefile)


if __name__ == "__main__":
    unittest.main()
