"""Tests for the methodological-rigor coverage map generator.

The map's whole value is that it *classifies* every eval scenario and every
benchmark task into a method family (or an explicit cross-cutting / process
bucket). When a new scenario or task lands without a classification entry it
silently falls into the "Unclassified" tail of the rendered page — which reads
like a to-do note rather than the coverage regression it actually is. These
tests make that regression fail the suite instead.

History: `bunching-recovery` and `statspai-bunching` shipped a full rigor pair
(eval + benchmark) but `bunching` was never added to ``METHOD_ORDER``, so the
whole family rendered under "Unclassified" and the footer under-counted the
covered families by one.
"""

from __future__ import annotations

import unittest

from _helpers import ROOT, load_module

coverage_map = load_module("scripts/build-coverage-map.py", "aers_build_coverage_map")


class TestClassificationIsTotal(unittest.TestCase):
    """Every committed scenario/task must resolve to a bucket, not the tail."""

    def test_every_scenario_is_classified(self):
        unclassified = []
        for scenario in coverage_map.load_scenarios():
            sid = scenario["id"]
            if sid in coverage_map.SCENARIO_METHOD:
                tag = coverage_map.SCENARIO_METHOD[sid]
                if tag != "*" and tag not in coverage_map.METHOD_ORDER:
                    unclassified.append(f"{sid} -> unknown family {tag!r}")
            elif scenario["category"] not in coverage_map.PROCESS_CATEGORIES:
                unclassified.append(f"{sid} (category {scenario['category']!r})")
        self.assertEqual(
            unclassified,
            [],
            "classify these in scripts/build-coverage-map.py (SCENARIO_METHOD) "
            "or give them a process category",
        )

    def test_every_benchmark_task_is_classified(self):
        unclassified = []
        for task in coverage_map.load_tasks():
            tid = task["id"]
            if tid not in coverage_map.TASK_METHOD:
                unclassified.append(f"{tid} (missing from TASK_METHOD)")
                continue
            tag = coverage_map.TASK_METHOD[tid]
            if tag != "*" and tag not in coverage_map.METHOD_ORDER:
                unclassified.append(f"{tid} -> unknown family {tag!r}")
        self.assertEqual(
            unclassified, [], "classify these in scripts/build-coverage-map.py (TASK_METHOD)"
        )

    def test_rendered_map_has_no_unclassified_section(self):
        rendered = coverage_map.render()
        self.assertNotIn("## Unclassified", rendered)


class TestFamilyTableIsWellFormed(unittest.TestCase):
    def test_every_ordered_family_has_a_label(self):
        missing = [m for m in coverage_map.METHOD_ORDER if m not in coverage_map.METHOD_LABEL]
        self.assertEqual(missing, [], "add a METHOD_LABEL entry")

    def test_labels_do_not_go_unused(self):
        # A label with no row is dead config; it also hides a family that was
        # meant to be listed (the bunching regression).
        unused = [m for m in coverage_map.METHOD_LABEL if m not in coverage_map.METHOD_ORDER]
        self.assertEqual(unused, [], "these labelled families are missing from METHOD_ORDER")

    def test_related_notes_reference_real_families(self):
        stray = [m for m in coverage_map.RELATED_NOTE if m not in coverage_map.METHOD_ORDER]
        self.assertEqual(stray, [], "RELATED_NOTE keys must be rows in the table")

    def test_committed_page_matches_the_generator(self):
        committed = (ROOT / "docs" / "RIGOR_COVERAGE.md").read_text(encoding="utf-8")
        self.assertEqual(committed, coverage_map.render(), "run `make catalog`")


if __name__ == "__main__":
    unittest.main()
