"""
test_participant_patterns.py
Participant-derived Fast Gate pattern candidates (open decision #9 / Task 50.2).

Candidates are evidence for 50.2, not the gate configuration: 50.1 correctness is still open,
so these sequences are candidate patterns with observed support, to be validated as declared
patterns rather than copied into the runtime.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from participant_patterns import (
    EXCLUDED_SYMBOLS,
    ParticipantPatternError,
    collapse_symbols,
    derive_candidates,
    load_sessions,
    render_markdown,
    run_analysis,
)


def _trace(session_id: str, symbols, provenance: str = "participant", task: str = "TD1") -> dict:
    base = 1_790_000_000_000.0
    return {
        "schemaVersion": "1.4.0",
        "session": {"sessionId": session_id, "provenance": provenance, "participantId": None},
        "task": {"currentTaskId": task, "status": "Completed"},
        "metadata": {"provenance": provenance},
        "macroInteractions": [
            {"timestamp": base + index * 100.0, "symbol": symbol, "sessionId": session_id}
            for index, symbol in enumerate(symbols)
        ],
    }


class TestParticipantPatterns(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, name: str, trace: dict) -> Path:
        path = self.root / f"experiment-trace-{name}.json"
        path.write_text(json.dumps(trace))
        return path

    def test_collapses_repeated_events_from_one_act(self):
        self.assertEqual(
            collapse_symbols(["OPEN_REPORT", "OPEN_REPORT", "OPEN_REPORT", "EXPORT_REPORT"]),
            ["OPEN_REPORT", "EXPORT_REPORT"],
        )

    def test_counts_session_support_not_event_frequency(self):
        self._write("a", _trace("a", ["NAV_ANALYTICS", "OPEN_FILTERS", "APPLY_FILTER"]))
        self._write("b", _trace("b", ["NAV_ANALYTICS", "OPEN_FILTERS", "APPLY_FILTER"]))
        self._write("c", _trace("c", ["NAV_ANALYTICS", "OPEN_FILTERS", "APPLY_FILTER"]))
        sessions = load_sessions(self.root)
        candidates = derive_candidates(sessions, min_support=3)
        by_pattern = {tuple(c["pattern"]): c for c in candidates}
        self.assertEqual(
            by_pattern[("NAV_ANALYTICS", "OPEN_FILTERS", "APPLY_FILTER")]["support_sessions"], 3
        )
        # Every contiguous sub-pattern with enough support is also present.
        self.assertEqual(by_pattern[("NAV_ANALYTICS", "OPEN_FILTERS")]["support_sessions"], 3)

    def test_requires_support_across_sessions(self):
        self._write("a", _trace("a", ["NAV_ANALYTICS", "OPEN_FILTERS"]))
        self._write("b", _trace("b", ["NAV_ANALYTICS", "OPEN_FILTERS"]))
        sessions = load_sessions(self.root)
        self.assertEqual(derive_candidates(sessions, min_support=3), [])

    def test_excludes_symbols_that_are_unreachable_or_misleading(self):
        self.assertTrue(
            {"TABLE_ROW_SELECT", "TABLE_SORT", "IDLE_DWELL"} <= EXCLUDED_SYMBOLS
        )
        for index in range(3):
            self._write(
                f"s{index}",
                _trace(f"s{index}", ["TABLE_ROW_SELECT", "NAV_ANALYTICS", "OPEN_FILTERS"]),
            )
        sessions = load_sessions(self.root)
        candidates = derive_candidates(sessions, min_support=3)
        patterns = [tuple(c["pattern"]) for c in candidates]
        self.assertNotIn(("TABLE_ROW_SELECT", "NAV_ANALYTICS"), patterns)
        self.assertIn(("NAV_ANALYTICS", "OPEN_FILTERS"), patterns)

    def test_refuses_scripted_traces(self):
        self._write("a", _trace("a", ["NAV_ANALYTICS"], provenance="scripted"))
        with self.assertRaises(ParticipantPatternError):
            load_sessions(self.root)

    def test_tracks_task_coverage_per_candidate(self):
        self._write("a", _trace("a", ["NAV_ANALYTICS", "OPEN_FILTERS"], task="TD1"))
        self._write("b", _trace("b", ["NAV_ANALYTICS", "OPEN_FILTERS"], task="TD2"))
        self._write("c", _trace("c", ["NAV_ANALYTICS", "OPEN_FILTERS"], task="TD2"))
        candidates = derive_candidates(load_sessions(self.root), min_support=2)
        entry = next(c for c in candidates if tuple(c["pattern"]) == ("NAV_ANALYTICS", "OPEN_FILTERS"))
        self.assertEqual(entry["tasks"], ["TD1", "TD2"])
        self.assertEqual(entry["support_sessions"], 3)

    def test_report_renders_participant_scope_and_caveats(self):
        self._write("a", _trace("a", ["NAV_ANALYTICS", "OPEN_FILTERS"]))
        self._write("b", _trace("b", ["NAV_ANALYTICS", "OPEN_FILTERS"]))
        report = run_analysis(self.root, min_support=2)
        markdown = render_markdown(report)
        self.assertIn("Participant-derived Fast Gate pattern candidates", markdown)
        self.assertIn("provisional", markdown.lower())
        self.assertIn("50.1", markdown)
        self.assertIn("NAV_ANALYTICS", markdown)


class TestRealParticipantEvidence(unittest.TestCase):
    def test_report_from_verified_participant_traces(self):
        verified = Path(__file__).resolve().parent.parent / ".data" / "raw" / "participant-verified"
        if not verified.is_dir():
            self.skipTest("verified participant traces not present")
        report = run_analysis(verified, min_support=3)
        self.assertEqual(report["provenance"], "participant")
        self.assertGreater(report["session_count"], 0)
        self.assertGreater(len(report["candidates"]), 0)


if __name__ == "__main__":
    unittest.main()
