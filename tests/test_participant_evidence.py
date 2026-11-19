"""
test_participant_evidence.py
Stage participant traces for analysis only after the framework trace verifier passes.

A participant capture that fails attribution is still a record, but it is not analysis input.
The staging tool copies the verified subset, records every pass/fail with the verifier detail,
and refuses scripted traces: participant evidence must never be pooled with engineering runs.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from participant_evidence import (
    ParticipantEvidenceError,
    EVIDENCE_SCHEMA_VERSION,
    stage_verified,
    verify_traces,
)

FRAMEWORK_ROOT = Path(__file__).resolve().parent.parent.parent / "edge-aui-framework"
VERIFIER = FRAMEWORK_ROOT / "scripts" / "verify-trace.mjs"


def _trace(session_id: str, provenance: str = "participant", task: str = "TD1") -> dict:
    return {
        "schemaVersion": "1.4.0",
        "session": {"sessionId": session_id, "provenance": provenance, "participantId": None},
        "task": {"currentTaskId": task, "status": "Completed"},
        "metadata": {"provenance": provenance},
        "behaviourEvents": [],
        "microTensors": [],
    }


class TestStageVerified(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.source = self.root / "participant"
        self.output = self.root / "verified"
        self.source.mkdir()
        for session_id in ("session-a", "session-b", "session-c"):
            (self.source / f"experiment-trace-{session_id}.json").write_text(
                json.dumps(_trace(session_id))
            )

    def tearDown(self):
        self._tmp.cleanup()

    def test_copies_only_verified_traces_and_records_the_rest(self):
        manifest = stage_verified(
            self.source,
            self.output,
            verification={
                "session-a": {"verified": True, "detail": "15/15 checks"},
                "session-b": {"verified": False, "detail": "prediction_window_attribution"},
            },
        )
        copied = sorted(path.name for path in self.output.glob("experiment-trace-*.json"))
        self.assertEqual(copied, ["experiment-trace-session-a.json"])
        self.assertEqual(manifest["counts"]["verified"], 1)
        # A session with no verification result fails closed.
        self.assertEqual(manifest["counts"]["excluded"], 2)
        sessions = {entry["session_id"]: entry for entry in manifest["sessions"]}
        self.assertEqual(sessions["session-b"]["status"], "excluded")
        self.assertIn("prediction_window_attribution", sessions["session-b"]["detail"])
        self.assertEqual(sessions["session-c"]["status"], "excluded")
        self.assertIn("no verification result", sessions["session-c"]["detail"])

    def test_refuses_scripted_traces(self):
        (self.source / "experiment-trace-session-d.json").write_text(
            json.dumps(_trace("session-d", provenance="scripted"))
        )
        with self.assertRaises(ParticipantEvidenceError):
            stage_verified(self.source, self.output, verification={})

    def test_manifest_pins_the_evidence_schema_and_provenance(self):
        manifest = stage_verified(
            self.source, self.output,
            verification={"session-a": {"verified": True, "detail": "ok"}},
        )
        self.assertEqual(manifest["evidence_schema_version"], EVIDENCE_SCHEMA_VERSION)
        self.assertEqual(manifest["provenance"], "participant")
        self.assertTrue((self.output / "verification-manifest.json").is_file())

    def test_refuses_to_write_into_the_source(self):
        with self.assertRaises(ParticipantEvidenceError):
            stage_verified(self.source, self.source, verification={})


class TestVerifierDetailFormatting(unittest.TestCase):
    def test_uses_the_failure_line_and_its_detail(self):
        from participant_evidence import _detail_from_output

        output = "[✓ PASS] a\n[✗ FAIL] prediction_window_attribution\n  window 398 missing\n"
        self.assertEqual(
            _detail_from_output(output),
            "[✗ FAIL] prediction_window_attribution — window 398 missing",
        )

    def test_falls_back_to_the_first_line_without_a_failure(self):
        from participant_evidence import _detail_from_output

        self.assertEqual(_detail_from_output("all good\n"), "all good")
        self.assertEqual(_detail_from_output(""), "verifier produced no detail")


class TestVerifyTracesWithFakeVerifier(unittest.TestCase):
    def test_maps_verifier_exit_code_and_detail_to_each_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "fake-verify.mjs"
            script.write_text(
                "const p = process.argv[2];\n"
                "if (p.includes('good')) { console.log('verified'); process.exit(0); }\n"
                "console.error('[✗ FAIL] prediction_window_attribution');\n"
                "process.exit(1);\n"
            )
            good = root / "experiment-trace-good.json"
            bad = root / "experiment-trace-bad.json"
            good.write_text(json.dumps(_trace("good")))
            bad.write_text(json.dumps(_trace("bad")))
            results = verify_traces([good, bad], verifier_script=script, framework_root=root)
        self.assertTrue(results["good"]["verified"])
        self.assertFalse(results["bad"]["verified"])
        self.assertIn("prediction_window_attribution", results["bad"]["detail"])


if __name__ == "__main__":
    unittest.main()
