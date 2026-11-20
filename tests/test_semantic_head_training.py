"""
test_semantic_head_training.py
Task 40.4: semantic head contract, no-future-leakage, metric availability and smoke training.

The training fixture is synthetic and carries the 1.4.0 semantic fields, so this module does
not depend on the participant export. It proves the pipeline, not participant performance.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data_manager import create_dataset_manifest
from semantic_targets import TARGET_NAME_TO_ID, TARGET_VOCABULARY
from target_dataset import (
    build_target_dataset_from_traces,
    save_target_dataset_parquet,
    split_target_dataset,
)
from training import EdgeAUIGRU, TargetSemanticHead, SEMANTIC_NUM_CLASSES
from semantic_training import evaluate_semantic_dataset, train_semantic_model

VOCABULARY = [TARGET_VOCABULARY[i] for i in sorted(TARGET_VOCABULARY)]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FOUNDATION = PROJECT_ROOT / "models" / "foundational_gru.pth"
LOOKAHEAD = (500.0, 1500.0)


def _outcome(outcome_name, timestamp, window_id):
    return {
        "timestamp": timestamp,
        "windowId": window_id,
        "sourceEventTimestamp": timestamp,
        "sourceEventType": "click",
        "componentId": "filter-accordion-region",
        "taskOutcome": outcome_name,
        "taskOutcomeVocabularyVersion": "testbed-semantic/1.0.0",
        "taskOutcomeSource": "application_rule",
        "taskOutcomeRuleId": outcome_name,
    }


def _trace(session_id, outcome_names, task_id="TD1", n_windows=12):
    """A 1.4.0 trace where every anchor has a labelled act 800 ms after its window."""
    windows = [
        {
            "windowId": index,
            "windowStart": 1000.0 + index * 500.0,
            "windowEnd": 1500.0 + index * 500.0,
            "values": [0.0] * 18,
            "eventCount": 1,
        }
        for index in range(n_windows)
    ]
    events = [
        {
            "timestamp": 1010.0 + index * 500.0,
            "type": "mousemove",
            "x": 0.5,
            "y": 0.5,
            "componentId": f"component-{index}",
            "route": "Analytics",
            "taskId": task_id,
            "viewport": {"width": 1920, "height": 1080},
            "document": {"width": 1920, "height": 1080},
        }
        for index in range(n_windows)
    ]
    outcomes = [
        _outcome(outcome_names[index % len(outcome_names)],
                 windows[index]["windowEnd"] + 800.0, index)
        for index in range(n_windows)
    ]
    return {
        "schemaVersion": "1.4.0",
        "exportedAt": "2026-11-20T00:00:00.000Z",
        "session": {
            "sessionId": session_id,
            "experimentId": "synthetic-test",
            "conditionId": "baseline",
            "provenance": "participant",
            "participantId": None,
        },
        "task": {
            "currentTaskId": task_id,
            "status": "Completed",
            "definitionVersion": "TD-v2",
        },
        "metadata": {
            "clock": "epoch_ms",
            "taskDefinitionVersion": "TD-v2",
            "semanticOutcomeVocabularyVersion": "testbed-semantic/1.0.0",
            "provenance": "participant",
        },
        "behaviourEvents": events,
        "microTensors": windows,
        "outcomes": outcomes,
        "predictions": [],
        "policyDecisions": [],
        "interventions": [],
        "macroInteractions": [],
        "taskEvents": [],
    }


def _write_traces(directory: Path, outcome_names, sessions=8) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(sessions):
        trace = _trace(f"synthetic-{index}", outcome_names, task_id=f"TD{index % 4 + 1}")
        (directory / f"experiment-trace-synthetic-{index}.json").write_text(
            json.dumps(trace), encoding="utf-8"
        )


def _build_versioned_dataset(root: Path, trace_dir: Path, version: str) -> Path:
    rows, stats = build_target_dataset_from_traces(
        trace_dir=trace_dir, target="semantic", seq_len=8,
        lookahead_min_ms=LOOKAHEAD[0], lookahead_max_ms=LOOKAHEAD[1],
    )
    assert rows, "synthetic fixture produced no rows"
    _, split = split_target_dataset(
        rows, random_seed=42, vocabulary=VOCABULARY,
        label_key="target_task_outcome", id_key="target_task_outcome_id",
        label_name="semantic", build_stats=stats,
    )
    version_dir = root / "processed" / version
    version_dir.mkdir(parents=True, exist_ok=True)
    save_target_dataset_parquet(
        rows, version_dir / "target_semantic_dataset.parquet", seq_len=8,
        split_summary=split,
    )
    create_dataset_manifest(
        rows=rows, split_summary=split, trace_dir=trace_dir,
        output_path=version_dir / "manifest.json", dataset_version=version,
    )
    return version_dir


class TestSemanticHeadContract(unittest.TestCase):
    def test_head_output_dimension_is_the_approved_vocabulary(self):
        self.assertEqual(SEMANTIC_NUM_CLASSES, 14)
        self.assertEqual(len(VOCABULARY), 14)
        head = TargetSemanticHead(hidden_dim=32)
        model = EdgeAUIGRU(input_dim=18, hidden_dim=32, num_layers=2,
                           num_classes=SEMANTIC_NUM_CLASSES, head=head)
        logits = model(torch.randn(3, 12, 18), context=torch.randn(3, 6))
        self.assertEqual(tuple(logits.shape), (3, 14))

    def test_context_is_mandatory(self):
        head = TargetSemanticHead(hidden_dim=32)
        model = EdgeAUIGRU(input_dim=18, hidden_dim=32, num_layers=2,
                           num_classes=SEMANTIC_NUM_CLASSES, head=head)
        with self.assertRaises(ValueError):
            model(torch.randn(1, 8, 18))

    def test_head_refuses_a_non_target_dimension(self):
        with self.assertRaises(ValueError):
            TargetSemanticHead(num_classes=15)


class TestSemanticDatasetLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.trace_dir = cls.root / "traces"
        _write_traces(
            cls.trace_dir, ["apply_filter", "select_region", "open_filter_section", "NO_OUTCOME"]
        )
        cls.version_dir = _build_versioned_dataset(cls.root, cls.trace_dir, "synthetic-v1")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_loader_returns_semantic_ids_and_provenance_metadata(self):
        from target_dataset import load_semantic_dataset
        dataset = load_semantic_dataset(parquet_path=self.version_dir / "target_semantic_dataset.parquet")
        self.assertGreater(len(dataset), 0)
        self.assertEqual(dataset.sequences.shape[1:], (8, 18))
        self.assertEqual(dataset.contexts.shape[1], 6)
        for target in np.unique(dataset.targets):
            self.assertLess(int(target), 14)
        metadata = dataset.metadata[0]
        for key in ("session_id", "task_id", "target_task_outcome", "split"):
            self.assertIn(key, metadata)

    def test_labels_are_causal_and_inside_the_declared_horizon(self):
        from target_dataset import load_semantic_dataset
        dataset = load_semantic_dataset(parquet_path=self.version_dir / "target_semantic_dataset.parquet")
        observed = [m["lead_time_ms"] for m in dataset.metadata if m.get("lead_time_ms") is not None]
        self.assertGreater(len(observed), 0)
        for lead in observed:
            self.assertGreaterEqual(float(lead), LOOKAHEAD[0])
            self.assertLessEqual(float(lead), LOOKAHEAD[1])

    def test_future_events_cannot_change_prediction_time_inputs(self):
        """Changing only events after the anchor must not change the model inputs."""
        early = [
            _trace("sess-a", ["apply_filter"]),
            _trace("sess-b", ["select_region"]),
        ]
        late = [
            _trace("sess-a", ["NO_OUTCOME"]),
            _trace("sess-b", ["apply_filter"]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            first_dir = Path(tmp) / "early"
            second_dir = Path(tmp) / "late"
            first_dir.mkdir()
            second_dir.mkdir()
            for index, trace in enumerate(early):
                (first_dir / f"t{index}.json").write_text(json.dumps(trace), encoding="utf-8")
            for index, trace in enumerate(late):
                (second_dir / f"t{index}.json").write_text(json.dumps(trace), encoding="utf-8")
            first_rows, _ = build_target_dataset_from_traces(
                trace_dir=first_dir, target="semantic", seq_len=8,
                lookahead_min_ms=LOOKAHEAD[0], lookahead_max_ms=LOOKAHEAD[1],
            )
            second_rows, _ = build_target_dataset_from_traces(
                trace_dir=second_dir, target="semantic", seq_len=8,
                lookahead_min_ms=LOOKAHEAD[0], lookahead_max_ms=LOOKAHEAD[1],
            )

        first_by_key = {(r["session_id"], r["anchor_window_id"]): r for r in first_rows}
        second_by_key = {(r["session_id"], r["anchor_window_id"]): r for r in second_rows}
        shared = sorted(set(first_by_key) & set(second_by_key))
        self.assertGreater(len(shared), 0)
        changed_labels = 0
        for key in shared:
            self.assertEqual(first_by_key[key]["sequence"], second_by_key[key]["sequence"])
            self.assertEqual(first_by_key[key]["context_vector"], second_by_key[key]["context_vector"])
            if first_by_key[key]["target_task_outcome_id"] != second_by_key[key]["target_task_outcome_id"]:
                changed_labels += 1
        self.assertGreater(changed_labels, 0, "fixture must change at least one future label")


class TestSemanticEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.trace_dir = cls.root / "traces"
        _write_traces(
            cls.trace_dir, ["apply_filter", "select_region", "open_filter_section", "NO_OUTCOME"]
        )
        cls.version_dir = _build_versioned_dataset(cls.root, cls.trace_dir, "synthetic-eval")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_metrics_are_available_and_absent_classes_are_named(self):
        from target_dataset import load_semantic_dataset
        dataset = load_semantic_dataset(parquet_path=self.version_dir / "target_semantic_dataset.parquet")
        model = EdgeAUIGRU(
            input_dim=18, hidden_dim=16, num_layers=1, num_classes=SEMANTIC_NUM_CLASSES,
            head=TargetSemanticHead(hidden_dim=16, num_classes=SEMANTIC_NUM_CLASSES),
        )
        result = evaluate_semantic_dataset(
            model, dataset, dataset.targets, split="test", device="cpu", verbose=False
        )
        for key in (
            "accuracy", "macro_f1", "macro_f1_all", "weighted_f1", "per_class",
            "confusion_matrix", "majority_baseline", "task_level", "lead_time",
            "calibration", "class_distribution", "absent_classes",
        ):
            self.assertIn(key, result)
        self.assertEqual(len(result["confusion_matrix"]), 14)
        self.assertEqual(len(result["confusion_matrix"][0]), 14)
        self.assertIsNotNone(result["calibration"]["ece"])
        for name, metrics in result["per_class"].items():
            if metrics["support"] == 0:
                self.assertIn(name, result["absent_classes"])


class TestSemanticTrainingSmoke(unittest.TestCase):
    def test_one_epoch_writes_a_14_class_checkpoint_and_config(self):
        if not FOUNDATION.is_file():
            raise unittest.SkipTest(f"Foundation checkpoint missing: {FOUNDATION}")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_dir = root / "traces"
            _write_traces(
                trace_dir,
                ["apply_filter", "select_region", "open_filter_section", "NO_OUTCOME"],
            )
            version_dir = _build_versioned_dataset(root, trace_dir, "synthetic-train")
            output_dir = root / "models"
            result = train_semantic_model(
                experiment="e1",
                dataset_version="synthetic-train",
                dataset_dir=str(version_dir),
                foundation_checkpoint=str(FOUNDATION),
                epochs=1,
                batch_size=8,
                device="cpu",
                output_dir=str(output_dir),
                verbose=False,
            )
            self.assertTrue(Path(result["checkpoint_path"]).is_file())
            self.assertEqual(result["val_evaluation"]["n_samples"] > 0, True)
            config = json.loads(Path(result["training_config_path"]).read_text())
            self.assertEqual(config["head_class_count"], 14)
            self.assertEqual(config["target_vocabulary"], VOCABULARY)
            self.assertEqual(config["class_weights_source"], "train_partition_only")
            self.assertEqual(len(config["class_weights"]), 14)
            state = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=True)
            self.assertEqual(state["head.fc.2.weight"].shape[0], 14)


if __name__ == "__main__":
    unittest.main()
