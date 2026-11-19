"""
test_provenance_retrofit.py
Local, audited provenance retrofit for mislabelled participant captures (ADR-018 / ADR-030).

The database is never modified. The retrofit works on an exported copy and records, per
session, the original declaration, the corrected declaration, the authority for the change
and the file hashes. Provenance is never inferred from behaviour.
"""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from provenance_retrofit import (
    ProvenanceRetrofitError,
    RETROFIT_SCHEMA_VERSION,
    retrofit_provenance,
)


def _trace(session_id: str, provenance: str = "scripted", metadata_provenance: str = "scripted") -> dict:
    return {
        "schemaVersion": "1.4.0",
        "exportedAt": "2026-11-19T00:00:00.000Z",
        "session": {
            "sessionId": session_id,
            "experimentId": "exp",
            "conditionId": "adaptive",
            "provenance": provenance,
            "participantId": None,
        },
        "task": {"currentTaskId": "TD1", "status": "Completed"},
        "metadata": {"durationMs": 1000, "provenance": metadata_provenance},
        "behaviourEvents": [],
        "microTensors": [],
        "outcomes": [],
        "predictions": [],
        "policyDecisions": [],
        "interventions": [],
        "macroInteractions": [],
        "taskEvents": [],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestProvenanceRetrofit(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.source = self.root / "source"
        self.output = self.root / "output"
        self.source.mkdir()
        (self.source / "experiment-trace-a.json").write_text(json.dumps(_trace("session-a")))
        (self.source / "experiment-trace-b.json").write_text(json.dumps(_trace("session-b")))
        (self.source / "export-manifest.json").write_text(json.dumps({"exportedAt": "x"}))

    def tearDown(self):
        self._tmp.cleanup()

    def test_corrects_both_declarations_and_leaves_the_source_untouched(self):
        originals = {
            path.name: _sha256(path) for path in self.source.glob("*.json")
        }
        manifest = retrofit_provenance(
            self.source, self.output,
            provenance="participant",
            authority="researcher instruction: collection-mode oversight",
        )

        for name in ("experiment-trace-a.json", "experiment-trace-b.json"):
            corrected = json.loads((self.output / name).read_text())
            self.assertEqual(corrected["session"]["provenance"], "participant")
            self.assertEqual(corrected["metadata"]["provenance"], "participant")
            # The retrofit declares provenance; it must not invent an identifier.
            self.assertIsNone(corrected["session"]["participantId"])
            self.assertEqual(_sha256(self.source / name), originals[name])

        self.assertEqual(manifest["retrofit_schema_version"], RETROFIT_SCHEMA_VERSION)
        self.assertEqual(manifest["target_provenance"], "participant")
        self.assertEqual(manifest["counts"]["corrected"], 2)
        self.assertEqual(manifest["counts"]["skipped"], 0)

    def test_manifest_records_original_labels_hashes_and_authority(self):
        manifest = retrofit_provenance(
            self.source, self.output,
            provenance="participant",
            authority="researcher instruction 2026-11-19",
        )
        sessions = {entry["session_id"]: entry for entry in manifest["sessions"]}
        entry = sessions["session-a"]
        self.assertEqual(entry["original_session_provenance"], "scripted")
        self.assertEqual(entry["original_metadata_provenance"], "scripted")
        self.assertEqual(entry["corrected_provenance"], "participant")
        self.assertEqual(entry["status"], "corrected")
        self.assertRegex(entry["source_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(entry["output_sha256"], r"^[0-9a-f]{64}$")
        self.assertIn("researcher instruction", manifest["authority"])

    def test_already_corrected_sessions_are_skipped_not_rewritten(self):
        (self.source / "experiment-trace-c.json").write_text(
            json.dumps(_trace("session-c", "participant", "participant"))
        )
        manifest = retrofit_provenance(
            self.source, self.output, provenance="participant", authority="test"
        )
        sessions = {entry["session_id"]: entry for entry in manifest["sessions"]}
        self.assertEqual(sessions["session-c"]["status"], "already_corrected")
        self.assertEqual(manifest["counts"]["skipped"], 1)
        self.assertEqual(manifest["counts"]["corrected"], 2)

    def test_requires_an_authority(self):
        with self.assertRaises(ProvenanceRetrofitError):
            retrofit_provenance(self.source, self.output, provenance="participant", authority="")

    def test_refuses_to_write_into_the_source_directory(self):
        with self.assertRaises(ProvenanceRetrofitError):
            retrofit_provenance(
                self.source, self.source, provenance="participant", authority="test"
            )

    def test_refuses_an_unknown_target_provenance(self):
        with self.assertRaises(ProvenanceRetrofitError):
            retrofit_provenance(
                self.source, self.output, provenance="inferred", authority="test"
            )

    def test_refuses_an_unknown_original_provenance(self):
        (self.source / "experiment-trace-d.json").write_text(
            json.dumps(_trace("session-d", "synthetic", "synthetic"))
        )
        with self.assertRaises(ProvenanceRetrofitError):
            retrofit_provenance(
                self.source, self.output, provenance="participant", authority="test"
            )

    def test_refuses_to_overwrite_an_existing_output_without_force(self):
        retrofit_provenance(self.source, self.output, provenance="participant", authority="one")
        with self.assertRaises(ProvenanceRetrofitError):
            retrofit_provenance(self.source, self.output, provenance="participant", authority="two")

    def test_force_rewrites_the_output(self):
        retrofit_provenance(self.source, self.output, provenance="participant", authority="one")
        manifest = retrofit_provenance(
            self.source, self.output, provenance="participant", authority="two", force=True
        )
        self.assertEqual(manifest["authority"], "two")
        self.assertTrue((self.output / "retrofit-manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
