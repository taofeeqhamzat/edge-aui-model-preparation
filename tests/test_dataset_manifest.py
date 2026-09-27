"""
test_dataset_manifest.py
Unit tests verifying Task 11.2: Provenance-Versioned Dataset Manifest and Validators.

Enforces:
1. Valid manifests pass comprehensive validation without error.
2. The validator MUST FAIL (raise ManifestValidationError) when provenance is incomplete or violated:
   - Missing required provenance fields (Brief §8: all 9 fields must be present and resolvable).
   - Empty/blank provenance fields.
   - Non-existent sourceEventIds not in source traces.
   - Non-disjoint / overlapping session splits.
   - Label policy version mismatch.
   - Feature schema or preprocessing version mismatch.
   - Class weight leakage across splits.
3. Standalone CLI validation via python3 -m src.data --validate-manifest.
"""

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data_manager import (
    validate_dataset_manifest,
    create_dataset_manifest,
    ManifestValidationError,
    PROVENANCE_REQUIRED_FIELDS,
    PREPROCESSING_VERSION,
    FEATURE_SCHEMA_VERSION,
    LABEL_POLICY_VERSION,
)
from target_dataset import (
    build_target_dataset_from_traces,
    split_target_dataset,
)


class TestDatasetManifest(unittest.TestCase):
    """Test suite covering Task 11.2 manifest generation and strict validation."""

    @classmethod
    def setUpClass(cls):
        cls.proj_root = Path(__file__).resolve().parent.parent
        cls.manifest_path = cls.proj_root / ".data" / "processed" / "v1.0.0" / "manifest.json"
        cls.trace_dir = cls.proj_root / ".data" / "raw" / "scripted"

        if not cls.manifest_path.is_file():
            # If not built yet, build it using pipeline functions
            rows, _ = build_target_dataset_from_traces(trace_dir=cls.trace_dir)
            split_rows, summary = split_target_dataset(rows, random_seed=42)
            create_dataset_manifest(
                rows=split_rows,
                split_summary=summary,
                trace_dir=cls.trace_dir,
                output_path=cls.manifest_path,
                dataset_version="v1.0.0",
            )

        with open(cls.manifest_path, "r", encoding="utf-8") as f:
            cls.canonical_manifest = json.load(f)

    def test_valid_manifest_passes_validation(self):
        """Assert that the production manifest passes all validation checks cleanly."""
        res = validate_dataset_manifest(
            self.manifest_path,
            trace_dir=self.trace_dir,
            check_events=True,
        )
        self.assertTrue(res["valid"])
        self.assertEqual(res["total_examples"], 289)
        self.assertEqual(res["total_sessions"], 18)
        self.assertEqual(res["dataset_version"], "v1.0.0")

    def test_validator_fails_missing_header_field(self):
        """Assert validator fails when a required top-level header is missing."""
        broken = copy.deepcopy(self.canonical_manifest)
        del broken["claim_boundary"]

        with self.assertRaises(ManifestValidationError) as ctx:
            validate_dataset_manifest(broken, trace_dir=self.trace_dir)
        self.assertIn("claim_boundary", str(ctx.exception))

    def test_validator_fails_missing_provenance_field_in_example(self):
        """Assert validator fails if any of the 9 required provenance fields is missing in an example."""
        for field in PROVENANCE_REQUIRED_FIELDS:
            broken = copy.deepcopy(self.canonical_manifest)
            del broken["examples"][0][field]

            with self.assertRaises(ManifestValidationError) as ctx:
                validate_dataset_manifest(broken, trace_dir=self.trace_dir)
            self.assertIn(field, str(ctx.exception))

    def test_validator_fails_empty_provenance_field(self):
        """Assert validator fails if any required string provenance field is empty/blank."""
        broken = copy.deepcopy(self.canonical_manifest)
        broken["examples"][5]["session_id"] = "   "

        with self.assertRaises(ManifestValidationError) as ctx:
            validate_dataset_manifest(broken, trace_dir=self.trace_dir)
        self.assertIn("session_id", str(ctx.exception))

    def test_validator_fails_nonexistent_source_event_id(self):
        """Assert validator fails when an example references an event not in the source trace."""
        broken = copy.deepcopy(self.canonical_manifest)
        sid = broken["examples"][0]["session_id"]
        # Inject an event ID with index 99999 that does not exist in the source trace
        broken["examples"][0]["source_event_ids"] = [f"{sid}:ev_99999"]

        with self.assertRaises(ManifestValidationError) as ctx:
            validate_dataset_manifest(broken, trace_dir=self.trace_dir, check_events=True)
        self.assertIn("ev_99999", str(ctx.exception))
        self.assertIn("non-existent", str(ctx.exception).lower())

    def test_validator_fails_nondisjoint_session_splits(self):
        """Assert validator fails when a session appears in multiple split partitions."""
        broken = copy.deepcopy(self.canonical_manifest)
        leaked_session = broken["split_summary"]["sessions"]["train"][0]

        # Put train session into validation partition as well
        broken["split_summary"]["sessions"]["val"].append(leaked_session)

        with self.assertRaises(ManifestValidationError) as ctx:
            validate_dataset_manifest(broken, trace_dir=self.trace_dir)
        self.assertIn("Non-disjoint splits", str(ctx.exception))

    def test_validator_fails_code_version_mismatch(self):
        """Assert validator fails when code versions in manifest do not match active codebase."""
        # 1. Preprocessing version mismatch
        broken1 = copy.deepcopy(self.canonical_manifest)
        broken1["code_versions"]["preprocessing_version"] = "99.0.0"
        with self.assertRaises(ManifestValidationError) as ctx:
            validate_dataset_manifest(broken1, trace_dir=self.trace_dir)
        self.assertIn("Preprocessing version mismatch", str(ctx.exception))

        # 2. Feature schema version mismatch
        broken2 = copy.deepcopy(self.canonical_manifest)
        broken2["code_versions"]["feature_schema_version"] = "99.0.0"
        with self.assertRaises(ManifestValidationError) as ctx:
            validate_dataset_manifest(broken2, trace_dir=self.trace_dir)
        self.assertIn("Feature schema version mismatch", str(ctx.exception))

    def test_validator_fails_label_policy_version_mismatch(self):
        """Assert validator fails when label policy version does not match active code."""
        broken = copy.deepcopy(self.canonical_manifest)
        broken["code_versions"]["label_policy_version"] = "99.0.0"
        with self.assertRaises(ManifestValidationError) as ctx:
            validate_dataset_manifest(broken, trace_dir=self.trace_dir)
        self.assertIn("Label policy version mismatch", str(ctx.exception))

    def test_validator_fails_class_weights_leakage(self):
        """Assert validator fails if class weights were not derived strictly on training partition."""
        broken = copy.deepcopy(self.canonical_manifest)
        broken["class_weights"]["derived_on"] = "full_dataset"

        with self.assertRaises(ManifestValidationError) as ctx:
            validate_dataset_manifest(broken, trace_dir=self.trace_dir)
        self.assertIn("Class weights provenance invalid", str(ctx.exception))

    def test_cli_validate_manifest_command(self):
        """Assert python3 -m src.data --validate-manifest runs and succeeds on valid manifest."""
        import subprocess

        # Success case
        cmd = [
            sys.executable,
            "-m",
            "src.data",
            "--validate-manifest",
            str(self.manifest_path),
            "--trace-dir",
            str(self.trace_dir),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.proj_root))
        self.assertEqual(proc.returncode, 0, f"Valid manifest CLI validation failed: {proc.stderr}")
        self.assertIn("Manifest is strictly valid", proc.stdout)

        # Failure case with deliberately corrupted temporary manifest
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            broken = copy.deepcopy(self.canonical_manifest)
            del broken["claim_boundary"]
            json.dump(broken, tf)
            broken_path = tf.name

        try:
            cmd_fail = [
                sys.executable,
                "-m",
                "src.data",
                "--validate-manifest",
                broken_path,
                "--trace-dir",
                str(self.trace_dir),
            ]
            proc_fail = subprocess.run(cmd_fail, capture_output=True, text=True, cwd=str(self.proj_root))
            self.assertNotEqual(proc_fail.returncode, 0, "Corrupted manifest must exit non-zero")
            self.assertIn("MANIFEST VALIDATION FAILED", proc_fail.stderr + proc_fail.stdout)
        finally:
            if os.path.exists(broken_path):
                os.unlink(broken_path)


if __name__ == "__main__":
    unittest.main()
