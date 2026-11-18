"""
test_semantic_targets.py
Task 40.2: semantic target generation, vocabulary pinning and label provenance.

The derivation fixtures are shared with the TypeScript testbed vocabulary: each expected ID is
what `edge-aui-framework/src/testbed/config/semanticOutcomes.ts` derives for the same event.
"""

import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from semantic_targets import (
    RECORDED_ONLY_OUTCOMES,
    TARGET_NAME_TO_ID,
    TARGET_VOCABULARY,
    TARGET_VOCABULARY_VERSION,
    derive_semantic_outcome,
    derive_semantic_outcomes,
    find_source_event,
    is_target_outcome,
    resolve_target_label,
    select_target_record,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "semantic_outcome_fixtures.json"


class TestSemanticVocabulary(unittest.TestCase):
    def test_target_vocabulary_has_the_approved_fourteen_ids(self):
        self.assertEqual(len(TARGET_VOCABULARY), 14)
        self.assertEqual(len(set(TARGET_VOCABULARY.values())), 14)
        self.assertEqual(TARGET_NAME_TO_ID["NO_OUTCOME"], 13)
        self.assertEqual(TARGET_NAME_TO_ID["navigate_overview"], 0)
        self.assertEqual(TARGET_VOCABULARY_VERSION, "testbed-semantic-target/1.0.0")

    def test_inspect_help_is_recorded_but_not_a_target(self):
        self.assertIn("inspect_help", RECORDED_ONLY_OUTCOMES)
        self.assertFalse(is_target_outcome("inspect_help"))
        self.assertNotIn("inspect_help", TARGET_NAME_TO_ID)
        self.assertTrue(is_target_outcome("apply_filter"))
        self.assertFalse(is_target_outcome("not_an_outcome"))


class TestSemanticDerivationParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = json.loads(FIXTURES.read_text(encoding="utf-8"))

    def test_every_shared_fixture_matches_the_typescript_rules(self):
        for case in self.fixtures["cases"]:
            with self.subTest(case=case["name"]):
                self.assertEqual(
                    derive_semantic_outcome(case["event"]),
                    case["expected"],
                    case["name"],
                )

    def test_backtracking_is_route_level_and_ignores_non_navigation(self):
        events = [
            {"type": "click", "componentId": "results-table"},
            {"type": "navigation", "route": "Reports"},
            {"type": "navigation", "route": "Analytics"},
            {"type": "navigation", "route": "Reports"},
            {"type": "navigation", "route": "Reports"},
        ]
        derived = derive_semantic_outcomes(events, starting_route="Overview")
        self.assertEqual(derived, [None, None, None, "backtrack", None])


class TestLookaheadSelection(unittest.TestCase):
    def test_only_records_inside_the_horizon_are_candidates(self):
        records = [
            {"sourceEventTimestamp": 1000.0, "taskOutcome": "select_region"},
            {"sourceEventTimestamp": 1600.0, "taskOutcome": "apply_filter"},
            {"sourceEventTimestamp": 2600.0, "taskOutcome": "export_report"},
        ]
        selected = select_target_record(records, anchor_end_ms=1000.0, lookahead_min_ms=500.0, lookahead_max_ms=1500.0)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["taskOutcome"], "apply_filter")

    def test_earliest_semantic_act_wins(self):
        records = [
            {"sourceEventTimestamp": 1800.0, "taskOutcome": "export_report"},
            {"sourceEventTimestamp": 1600.0, "taskOutcome": "select_report"},
        ]
        selected = select_target_record(records, 1000.0, 500.0, 1500.0)
        self.assertEqual(selected["taskOutcome"], "select_report")

    def test_tied_timestamps_use_the_declared_priority(self):
        records = [
            {"sourceEventTimestamp": 1700.0, "taskOutcome": "navigate_reports"},
            {"sourceEventTimestamp": 1700.0, "taskOutcome": "apply_filter"},
        ]
        selected = select_target_record(records, 1000.0, 500.0, 1500.0)
        self.assertEqual(selected["taskOutcome"], "apply_filter")

    def test_no_record_in_horizon_returns_none(self):
        records = [{"sourceEventTimestamp": 900.0, "taskOutcome": "select_region"}]
        self.assertIsNone(select_target_record(records, 1000.0, 500.0, 1500.0))


class TestTargetLabelResolution(unittest.TestCase):
    def test_recorded_semantic_outcome_wins(self):
        record = {"taskOutcome": "apply_filter", "taskOutcomeSource": "application_rule"}
        outcome, source = resolve_target_label(record, has_semantic_contract=True)
        self.assertEqual((outcome, source), ("apply_filter", "trace_semantic_outcome"))

    def test_absent_semantic_field_in_a_semantic_trace_is_no_outcome(self):
        outcome, source = resolve_target_label({}, has_semantic_contract=True)
        self.assertEqual((outcome, source), ("NO_OUTCOME", "trace_no_outcome"))

    def test_legacy_trace_re_derives_from_the_source_event(self):
        record = {"sourceEventTimestamp": 1200.0, "componentId": "btn-export"}
        source_event = {"timestamp": 1200.0, "type": "click", "componentId": "btn-export"}
        outcome, source = resolve_target_label(
            record, has_semantic_contract=False, source_event=source_event
        )
        self.assertEqual((outcome, source), ("export_report", "re_derived"))

    def test_legacy_trace_without_a_matching_event_is_no_outcome(self):
        outcome, source = resolve_target_label({}, has_semantic_contract=False, source_event=None)
        self.assertEqual((outcome, source), ("NO_OUTCOME", "re_derived_no_match"))

    def test_unknown_recorded_outcome_fails_loudly(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_target_label({"taskOutcome": "telepathy"}, has_semantic_contract=True)
        self.assertIn("telepathy", str(ctx.exception))

    def test_find_source_event_matches_timestamp_and_component(self):
        events = [
            {"timestamp": 1100.0, "componentId": "a"},
            {"timestamp": 1200.0, "componentId": "btn-export"},
        ]
        found = find_source_event({"sourceEventTimestamp": 1200.0, "componentId": "btn-export"}, events)
        self.assertIsNotNone(found)
        self.assertEqual(found["timestamp"], 1200.0)


if __name__ == "__main__":
    unittest.main()
