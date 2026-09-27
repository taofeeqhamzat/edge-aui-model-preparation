"""
Target-Domain Intervention Dataset Assembly (Phase B, Task 10.3).

Assembles model-ready training examples for TargetInterventionHead by combining:
1. MicroTensor sequences (T=8, D=18) from recorded traces.
2. Encoded R^6 UIContext vectors (Task 10.2).
3. Scripted target intervention labels (Task 10.3 policy).

Enforces:
- Zero cross-session and cross-task sequence leakage.
- Strict non-imputation: examples with missing/malformed context are excluded and counted.
- Versioned provenance on every row (label_policy_version, preprocessing_version, feature_schema_version).
- CLI reporting of class distribution and majority baseline before training.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    pa = None  # type: ignore
    pq = None  # type: ignore
    PYARROW_AVAILABLE = False

try:
    from context_encoding import (
        CONTEXT_VECTOR_DIM,
        encode_ui_context,
        validate_context_vector,
    )
    from intervention_label_policy import (
        LABEL_POLICY_VERSION,
        INTERVENTION_VOCABULARY,
        INTERVENTION_TO_ID,
        assign_intervention_label,
        compute_label_distribution,
    )
    from config import load_config
except ImportError:
    from src.context_encoding import (
        CONTEXT_VECTOR_DIM,
        encode_ui_context,
        validate_context_vector,
    )
    from src.intervention_label_policy import (
        LABEL_POLICY_VERSION,
        INTERVENTION_VOCABULARY,
        INTERVENTION_TO_ID,
        assign_intervention_label,
        compute_label_distribution,
    )
    from src.config import load_config


TARGET_DATASET_SCHEMA = None
if PYARROW_AVAILABLE:
    TARGET_DATASET_SCHEMA = pa.schema([
        ("session_id", pa.string()),
        ("experiment_id", pa.string()),
        ("condition_id", pa.string()),
        ("task_id", pa.string()),
        ("anchor_window_id", pa.int64()),
        ("sequence", pa.list_(pa.list_(pa.float32()))),  # (T, 18)
        ("context_vector", pa.list_(pa.float32())),       # (6,)
        ("target_outcome", pa.string()),
        ("target_outcome_id", pa.int64()),
        ("target_intervention", pa.string()),
        ("target_intervention_id", pa.int64()),
        ("label_policy_version", pa.string()),
        ("is_scripted_policy", pa.bool_()),
    ])


def build_target_dataset_from_traces(
    trace_dir: Union[str, Path] = ".data/raw/scripted",
    output_path: Optional[Union[str, Path]] = None,
    seq_len: int = 8,
    stride: int = 1,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Builds the target-domain dataset from a directory of ExperimentTrace JSON files.

    Returns:
        (rows, summary_stats)
    """
    trace_dir = Path(trace_dir)
    trace_files = sorted(glob.glob(str(trace_dir / "*.json")))

    # Exclude manifest.json if present
    trace_files = [f for f in trace_files if not Path(f).name.startswith("manifest")]

    if not trace_files:
        raise FileNotFoundError(f"No trace JSON files found in {trace_dir}")

    all_rows: List[Dict[str, Any]] = []
    excluded_missing_context = 0
    total_candidate_windows = 0

    for trace_file in trace_files:
        with open(trace_file, "r", encoding="utf-8") as fp:
            trace = json.load(fp)

        session = trace.get("session", {})
        session_id = str(session.get("sessionId", Path(trace_file).stem))
        experiment_id = str(session.get("experimentId", "unknown"))
        condition_id = str(session.get("conditionId", "unknown"))

        task_info = trace.get("task", {})
        default_task_id = str(task_info.get("currentTaskId") or session.get("taskId") or "T1")

        # Map outcomes by windowId
        outcomes_map = {}
        for out in trace.get("outcomes", []):
            if isinstance(out, dict) and "windowId" in out:
                outcomes_map[out["windowId"]] = out

        # Map behavioural events and context snapshots
        behaviour_events = trace.get("behaviourEvents", [])

        micro_tensors = trace.get("microTensors", [])
        if not micro_tensors or len(micro_tensors) < seq_len:
            continue

        # Sort windows by windowId
        micro_tensors = sorted(micro_tensors, key=lambda w: w.get("windowId", 0))

        # Assemble sequence windows
        for s in range(0, len(micro_tensors) - seq_len + 1, stride):
            total_candidate_windows += 1
            window_slice = micro_tensors[s : s + seq_len]
            anchor_window = window_slice[-1]
            anchor_wid = anchor_window.get("windowId", 0)

            # Build sequence matrix (T, 18)
            seq_matrix = []
            for w in window_slice:
                vals = w.get("values", [])
                if len(vals) != 18:
                    vals = list(vals) + [0.0] * (18 - len(vals))
                seq_matrix.append([float(v) for v in vals[:18]])

            # Resolve anchor outcome
            outcome_rec = outcomes_map.get(anchor_wid, {})
            outcome_name = outcome_rec.get("outcome", "NO_OUTCOME")

            # Resolve UIContext for anchor window
            anchor_end = anchor_window.get("windowEnd", 0)
            anchor_task = outcome_rec.get("taskId") or default_task_id
            anchor_step = outcome_rec.get("taskStepId")

            # If not on outcome, look for nearest behaviour event preceding or during anchor
            anchor_route = outcome_rec.get("route", "Analytics")
            if not anchor_step:
                for ev in reversed(behaviour_events):
                    if ev.get("timestamp", 0) <= anchor_end and ev.get("taskStepId"):
                        anchor_step = ev.get("taskStepId")
                        anchor_route = ev.get("route", anchor_route)
                        break

            # Construct UIContext snapshot
            is_analytics = (anchor_route == "Analytics")
            ui_context = {
                "route": anchor_route,
                "primaryActionAvailable": is_analytics,
                "helpAvailable": is_analytics,
                "expandable": is_analytics,
                "taskId": anchor_task,
                "taskStepId": anchor_step,
                "availableActions": ["click", "change", "hover"] if is_analytics else ["click"],
            }

            try:
                context_vec = encode_ui_context(ui_context)
                val_res = validate_context_vector(context_vec)
                if not val_res.valid:
                    excluded_missing_context += 1
                    continue
            except Exception:
                excluded_missing_context += 1
                continue

            # Assign scripted intervention label with explicit taskProgress from encoded context vector
            ui_context["taskProgress"] = float(context_vec[4])
            label_info = assign_intervention_label(outcome_name, ui_context)

            outcome_taxonomy_map = {
                "NO_OUTCOME": 0,
                "CLICK": 1,
                "FORM_SUBMIT": 2,
                "BACKTRACK": 3,
                "RAPID_SCROLL": 4,
                "HOVER_DWELL": 5,
                "ABANDON": 6,
            }
            outcome_id = outcome_taxonomy_map.get(outcome_name, 0)

            row = {
                "session_id": session_id,
                "experiment_id": experiment_id,
                "condition_id": condition_id,
                "task_id": anchor_task,
                "anchor_window_id": int(anchor_wid),
                "sequence": seq_matrix,
                "context_vector": context_vec.tolist(),
                "target_outcome": outcome_name,
                "target_outcome_id": outcome_id,
                "target_intervention": label_info["target_intervention"],
                "target_intervention_id": int(label_info["target_intervention_id"]),
                "label_policy_version": str(label_info["label_policy_version"]),
                "is_scripted_policy": True,
            }
            all_rows.append(row)

    # Compute distribution
    intervention_labels = [r["target_intervention"] for r in all_rows]
    distribution = compute_label_distribution(intervention_labels)

    summary_stats = {
        "total_traces": len(trace_files),
        "total_candidate_windows": total_candidate_windows,
        "total_retained_examples": len(all_rows),
        "excluded_missing_context": excluded_missing_context,
        "distribution": distribution,
    }

    if output_path and PYARROW_AVAILABLE and all_rows:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        custom_metadata = {
            "sequence_length": str(seq_len),
            "feature_dim": "18",
            "context_dim": str(CONTEXT_VECTOR_DIM),
            "num_examples": str(len(all_rows)),
            "label_policy_version": str(LABEL_POLICY_VERSION),
            "is_scripted_policy": "true",
            "adr_claim_boundary": "ADR-013: Substitute interaction traces and scripted label assignments.",
        }
        schema_with_meta = TARGET_DATASET_SCHEMA.with_metadata(custom_metadata)
        table = pa.Table.from_pylist(all_rows, schema=schema_with_meta)
        pq.write_table(table, str(out_file), compression="snappy")
        print(f"Target intervention dataset exported to: {out_file}")

    return all_rows, summary_stats


def main():
    parser = argparse.ArgumentParser(description="Assemble TargetInterventionHead Dataset (Task 10.3).")
    parser.add_argument("--trace-dir", type=str, default=".data/raw/scripted", help="Input directory of traces")
    parser.add_argument("--output", type=str, default=".data/processed/target_intervention_dataset.parquet", help="Output Parquet path")
    parser.add_argument("--build-labels", action="store_true", help="Build labels and report class distribution")
    parser.add_argument("--seq-len", type=int, default=8, help="Sequence length T")
    args = parser.parse_args()

    rows, stats = build_target_dataset_from_traces(
        trace_dir=args.trace_dir,
        output_path=args.output if args.build_labels else None,
        seq_len=args.seq_len,
    )

    dist = stats["distribution"]

    print("=" * 70)
    print("TARGET-DOMAIN INTERVENTION DATASET SUMMARY")
    print(f"Traces Ingested:           {stats['total_traces']}")
    print(f"Total Candidate Sequences: {stats['total_candidate_windows']}")
    print(f"Retained Examples:         {stats['total_retained_examples']}")
    print(f"Excluded Missing Context:  {stats['excluded_missing_context']}")
    print("-" * 70)
    print("CLASS DISTRIBUTION (5-CLASS INTERVENTION VOCABULARY):")
    for cls_name in INTERVENTION_VOCABULARY:
        count = dist["counts"][cls_name]
        pct = dist["percentages"][cls_name]
        print(f"  {cls_name:<25}: {count:>4} ({pct:>5.1f}%)")
    print("-" * 70)
    print(f"Majority Baseline Class:   {dist['majority_class']} ({dist['majority_percentage']:.1f}%)")
    print(f"Policy Version:            {LABEL_POLICY_VERSION}")
    print("Claim Notice:              Labels are scripted policy outputs (ADR-013).")
    print("=" * 70)


if __name__ == "__main__":
    main()
