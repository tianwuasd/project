"""Tests for the five-minute tour (`make quickstart`).

The tour is the first thing a newcomer runs, so a wrong number there is the
worst place to have one. It had two: it counted the markdown table's separator
row as a method family (reporting 19 instead of 18, with `---` in the sample
list), and it labelled the *total* family count as the *closed-coverage* count,
which are different numbers with different meanings.

Both came from the same shortcut — parsing a generated table by eye rather than
by its shape — so these tests check the parse against the generated document it
reads, not against a hardcoded expectation.
"""

from __future__ import annotations

import re
import unittest

from _helpers import ROOT, load_module

quickstart = load_module("scripts/quickstart.py", "aers_quickstart")
coverage_map = load_module("scripts/build-coverage-map.py", "aers_coverage_for_quickstart")


class TestMethodFamilyParse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.families, cls.closed = quickstart._method_families()

    def test_no_table_separator_leaks_in_as_a_family(self):
        for family in self.families:
            with self.subTest(family=family):
                self.assertFalse(
                    re.fullmatch(r"[-:\s]+", family),
                    "a markdown alignment row was parsed as a method family",
                )
                self.assertTrue(family.strip())

    def test_families_match_the_generator(self):
        expected = [coverage_map.METHOD_LABEL[m] for m in coverage_map.METHOD_ORDER]
        self.assertEqual(self.families, expected)

    def test_closed_count_is_a_subset_of_the_tracked_count(self):
        self.assertGreater(self.closed, 0)
        self.assertLessEqual(self.closed, len(self.families))

    def test_closed_count_matches_the_covered_rows(self):
        rendered = (ROOT / "docs" / "RIGOR_COVERAGE.md").read_text(encoding="utf-8")
        table = rendered.split("| Method family", 1)[1].split("\n\nNotes:", 1)[0]
        covered = sum(
            1 for line in table.splitlines() if line.strip().endswith("| covered |")
        )
        self.assertEqual(self.closed, covered)


class TestReportedNumbers(unittest.TestCase):
    def test_snapshot_exposes_both_counts_separately(self):
        rigor = quickstart._rigor_snapshot()
        self.assertIn("n_families", rigor)
        self.assertIn("n_closed", rigor)
        self.assertEqual(rigor["n_families"], len(rigor["method_families"]))

    def test_committed_markdown_report_is_current(self):
        # It is a generated artifact; a stale one misinforms the first reader.
        report = ROOT / "docs" / "QUICKSTART_REPORT.md"
        rigor = quickstart._rigor_snapshot()
        text = report.read_text(encoding="utf-8")
        self.assertIn(f"**{rigor['n_families']}** method families tracked", text)
        self.assertIn(f"**{rigor['n_closed']}** with closed coverage", text)

    def test_no_hardcoded_catalog_totals_in_the_docstring(self):
        # The docstring used to claim "1,150 vendored skills across 69
        # collections"; both were wrong by the time anyone read them.
        docstring = quickstart.__doc__ or ""
        self.assertFalse(
            re.search(r"\b\d,\d{3}\b", docstring),
            "a count in the docstring is a count nobody watches",
        )


if __name__ == "__main__":
    unittest.main()


class TestWhereToStart(unittest.TestCase):
    """The on-ramp must not point newcomers at a deprecated file."""

    def setUp(self):
        self.entry_points = quickstart.build_report()["entry_points"]

    def test_every_named_entry_point_exists(self):
        named = re.findall(r"[\w./-]+\.(?:md|html)", " ".join(self.entry_points))
        self.assertTrue(named, "no entry points parsed")
        for candidate in named:
            with self.subTest(entry=candidate):
                self.assertTrue((ROOT / candidate).exists(), f"{candidate} does not exist")

    def test_does_not_point_at_the_deprecated_chinese_stub(self):
        joined = " ".join(self.entry_points)
        self.assertNotIn("README-zh-CN.md", joined)
        stub = (ROOT / "README-zh-CN.md").read_text(encoding="utf-8")
        self.assertIn("DEPRECATED", stub, "the stub stopped being deprecated?")
