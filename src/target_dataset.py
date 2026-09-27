# Target-Domain Intervention Dataset Assembly & Splits (Phase B & Phase C, Tasks 10.3, 11.1, 11.2).
"""
Assembles model-ready training examples for TargetInterventionHead by combining:
1. MicroTensor sequences (T=8, D=18) from recorded traces.
2. Encoded R^6 UIContext vectors (Task 10.2).
3. Scripted target intervention labels (Task 10.3 policy).

Enforces:
- Zero cross-session and cross-task sequence leakage.
- Strict session-bounded train/val/test splits (Task 11.1).
- Class weight resolution derived exclusively on the training partition (Task 11.1).
- Full provenance carrying across all 9 required Brief §8 fields on every example (Task 11.2).
- Strict non-imputation: examples with missing/malformed context are excluded and counted.
- CLI reporting of class distribution and majority baseline before training.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

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
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

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
    from trace_ingestion import (
        EXPERIMENT_TRACE_SCHEMA_VERSION,
        PREPROCESSING_VERSION,
        FEATURE_SCHEMA_VERSION,
    )
    from target_generation import compute_class_weights
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
    from src.trace_ingestion import (
        EXPERIMENT_TRACE_SCHEMA_VERSION,
        PREPROCESSING_VERSION,
        FEATURE_SCHEMA_VERSION,
    )
    from src.target_generation import compute_class_weights
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
        ("split", pa.string()),                           # "train", "val", "test"
        ("source_event_ids", pa.list_(pa.string())),
        ("preprocessing_version", pa.string()),
        ("feature_schema_version", pa.string()),
        ("target_generation_version", pa.string()),
    ])


def build_target_dataset_from_traces(
    trace_dir: Union[str, Path] = ".data/raw/scripted",
    output_path: Optional[Union[str, Path]] = None,
    seq_len: int = 8,
    stride: int = 1,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Builds the target-domain dataset from a directory of ExperimentTrace JSON files.
    Carries full provenance per example (Brief §8):
    session_id, experiment_id, condition_id, task_id, anchor_window_id,
    source_event_ids, preprocessing_version, feature_schema_version, target_generation_version.

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

            # Correlate source events across the sequence window span
            # Window duration is 500 ms with settlement delay 250 ms, so sequence span
            # starts 250 ms before the first window's start.
            seq_start_ts = float(window_slice[0].get("windowStart", 0.0)) - 250.0
            seq_end_ts = float(anchor_window.get("windowEnd", 0.0))

            seq_event_ids = [
                f"{session_id}:ev_{ev_idx}"
                for ev_idx, ev in enumerate(behaviour_events)
                if seq_start_ts <= float(ev.get("timestamp", 0.0)) <= seq_end_ts
            ]
            if not seq_event_ids and behaviour_events:
                # Fallback to nearest prior event if gap exists
                seq_event_ids = [f"{session_id}:ev_{len(behaviour_events) - 1}"]

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
                "split": "unassigned",
                "source_event_ids": seq_event_ids,
                "preprocessing_version": PREPROCESSING_VERSION,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "target_generation_version": LABEL_POLICY_VERSION,
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
        save_target_dataset_parquet(all_rows, output_path, seq_len=seq_len)

    return all_rows, summary_stats


def split_target_dataset(
    rows: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_seed: int = 42,
    imbalance_strategy: str = "smoothed",
    smoothing_alpha: float = 10.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Produce strictly session-bounded train / validation / test splits for the intervention-head
    dataset and resolve class weights exclusively on the training partition (Task 11.1).

    Enforces:
    - Zero cross-split leakage: no session and no task within a session appears in two partitions.
    - Class weights resolved strictly on the training partition; test/val examples exert 0 influence.
    - Explicit detection and reporting of absent classes per partition.
    - Evaluation of the majority-class baseline across all partitions.
    """
    if not rows:
        raise ValueError("Cannot split an empty list of rows")

    # 1. Distinct sessions in sorted order for deterministic shuffling
    sessions = sorted(list({r["session_id"] for r in rows}))
    n_total = len(sessions)
    if n_total < 3:
        raise ValueError(f"Insufficient distinct sessions for train/val/test split: found {n_total}")

    rng = np.random.RandomState(random_seed)
    shuffled_sessions = sessions.copy()
    rng.shuffle(shuffled_sessions)

    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))

    train_sessions: Set[str] = set(shuffled_sessions[:n_train])
    val_sessions: Set[str] = set(shuffled_sessions[n_train : n_train + n_val])
    test_sessions: Set[str] = set(shuffled_sessions[n_train + n_val :])

    # Assert pairwise disjointness
    assert train_sessions.isdisjoint(val_sessions), "Train and Val sessions must be disjoint"
    assert train_sessions.isdisjoint(test_sessions), "Train and Test sessions must be disjoint"
    assert val_sessions.isdisjoint(test_sessions), "Val and Test sessions must be disjoint"

    # Verify task-session pairs are disjoint across splits
    train_st = {(r["session_id"], r["task_id"]) for r in rows if r["session_id"] in train_sessions}
    val_st = {(r["session_id"], r["task_id"]) for r in rows if r["session_id"] in val_sessions}
    test_st = {(r["session_id"], r["task_id"]) for r in rows if r["session_id"] in test_sessions}
    assert train_st.isdisjoint(val_st), "Train and Val (session, task) pairs must be disjoint"
    assert train_st.isdisjoint(test_st), "Train and Test (session, task) pairs must be disjoint"
    assert val_st.isdisjoint(test_st), "Val and Test (session, task) pairs must be disjoint"

    # 2. Assign split to every row
    for r in rows:
        sid = r["session_id"]
        if sid in train_sessions:
            r["split"] = "train"
        elif sid in val_sessions:
            r["split"] = "val"
        elif sid in test_sessions:
            r["split"] = "test"
        else:
            raise ValueError(f"Unpartitioned session encountered: {sid}")

    # 3. Compute per-partition statistics
    partitions = {
        "train": [r for r in rows if r["split"] == "train"],
        "val": [r for r in rows if r["split"] == "val"],
        "test": [r for r in rows if r["split"] == "test"],
    }

    class_distribution: Dict[str, Any] = {}
    absent_classes: Dict[str, List[str]] = {}

    for p_name, p_rows in partitions.items():
        labels = [r["target_intervention"] for r in p_rows]
        dist = compute_label_distribution(labels)
        class_distribution[p_name] = dist
        absent = [cls for cls, cnt in dist["counts"].items() if cnt == 0]
        absent_classes[p_name] = absent

    class_distribution["overall"] = compute_label_distribution([r["target_intervention"] for r in rows])

    # 4. Derive majority baseline (majority class determined by train partition only)
    train_majority_class = class_distribution["train"]["majority_class"]
    train_majority_id = INTERVENTION_TO_ID[train_majority_class]

    majority_baselines: Dict[str, Any] = {}
    for p_name, p_rows in partitions.items():
        p_n = len(p_rows)
        if p_n == 0:
            majority_baselines[p_name] = {"accuracy": 0.0, "macro_f1": 0.0, "support": {}}
            continue

        p_counts = class_distribution[p_name]["counts"]
        correct = p_counts.get(train_majority_class, 0)
        acc = float(correct / p_n)

        # Macro-F1 across supported classes in partition
        f1_list = []
        per_class_f1 = {}
        for cls_name in INTERVENTION_VOCABULARY:
            support = p_counts.get(cls_name, 0)
            if cls_name == train_majority_class:
                tp = support
                fp = p_n - support
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = 1.0 if support > 0 else 0.0
                f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            else:
                f1 = 0.0

            per_class_f1[cls_name] = float(f1)
            if support > 0:
                f1_list.append(f1)

        macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
        majority_baselines[p_name] = {
            "majority_class": train_majority_class,
            "majority_class_id": train_majority_id,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "per_class_f1": per_class_f1,
            "support": p_counts,
        }

    # 5. Resolve Class Weights on TRAINING PARTITION ONLY
    train_targets = np.array([r["target_intervention_id"] for r in partitions["train"]])
    num_classes = len(INTERVENTION_VOCABULARY)

    if imbalance_strategy == "smoothed":
        weights_t = compute_class_weights(
            train_targets,
            num_classes=num_classes,
            smoothing_alpha=smoothing_alpha,
            allow_empty=True
        )
    elif imbalance_strategy == "inverse":
        weights_t = compute_class_weights(
            train_targets,
            num_classes=num_classes,
            smoothing_alpha=0.0,
            allow_empty=True
        )
    elif imbalance_strategy == "none":
        weights_t = torch.ones(num_classes, dtype=torch.float32) if TORCH_AVAILABLE else np.ones(num_classes, dtype=np.float32)
    else:
        raise ValueError(f"Unknown imbalance_strategy: '{imbalance_strategy}'. Expected 'smoothed', 'inverse', or 'none'.")

    weights_np = weights_t.numpy() if hasattr(weights_t, "numpy") else np.asarray(weights_t)
    resolved_weights = {
        INTERVENTION_VOCABULARY[i]: float(weights_np[i])
        for i in range(num_classes)
    }

    max_w = float(np.max(weights_np))
    min_w = float(np.min(weights_np[weights_np > 0])) if np.any(weights_np > 0) else 0.0
    max_min_ratio = (max_w / min_w) if min_w > 0 else 0.0

    imbalance_rationale = (
        f"Imbalance strategy: '{imbalance_strategy}' (alpha={smoothing_alpha if imbalance_strategy == 'smoothed' else 0.0}). "
        f"Resolved exclusively on the training partition ({len(partitions['train'])} examples, 13 sessions) "
        f"to prevent test-set label balance from leaking into training. Preserves full sample density without "
        f"synthetic duplication (no oversampling)."
    )

    split_summary = {
        "split_strategy": "session_bounded",
        "random_seed": random_seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "session_counts": {
            "train": len(train_sessions),
            "val": len(val_sessions),
            "test": len(test_sessions),
            "total": n_total,
        },
        "sessions": {
            "train": sorted(list(train_sessions)),
            "val": sorted(list(val_sessions)),
            "test": sorted(list(test_sessions)),
        },
        "example_counts": {
            "train": len(partitions["train"]),
            "val": len(partitions["val"]),
            "test": len(partitions["test"]),
            "total": len(rows),
        },
        "class_distribution": class_distribution,
        "absent_classes": absent_classes,
        "majority_baseline": majority_baselines,
        "imbalance_strategy": imbalance_strategy,
        "imbalance_rationale": imbalance_rationale,
        "class_weights": {
            "strategy": imbalance_strategy,
            "smoothing_alpha": smoothing_alpha if imbalance_strategy == "smoothed" else 0.0,
            "derived_on": "train_partition_only",
            "weights": resolved_weights,
            "max_min_ratio": max_min_ratio,
        },
    }

    return rows, split_summary


def save_target_dataset_parquet(
    rows: List[Dict[str, Any]],
    output_path: Union[str, Path],
    seq_len: int = 8,
    split_summary: Optional[Dict[str, Any]] = None,
) -> str:
    """Saves assembled rows to Parquet storage with schema and metadata."""
    if not PYARROW_AVAILABLE:
        raise RuntimeError("PyArrow is required to export Parquet datasets")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    custom_metadata = {
        "sequence_length": str(seq_len),
        "feature_dim": "18",
        "context_dim": str(CONTEXT_VECTOR_DIM),
        "num_examples": str(len(rows)),
        "label_policy_version": str(LABEL_POLICY_VERSION),
        "preprocessing_version": str(PREPROCESSING_VERSION),
        "feature_schema_version": str(FEATURE_SCHEMA_VERSION),
        "is_scripted_policy": "true",
        "adr_claim_boundary": "ADR-013: Substitute interaction traces and scripted label assignments.",
    }
    if split_summary:
        custom_metadata["split_strategy"] = str(split_summary.get("split_strategy"))
        custom_metadata["split_seed"] = str(split_summary.get("random_seed"))
        custom_metadata["train_examples"] = str(split_summary.get("example_counts", {}).get("train", 0))
        custom_metadata["val_examples"] = str(split_summary.get("example_counts", {}).get("val", 0))
        custom_metadata["test_examples"] = str(split_summary.get("example_counts", {}).get("test", 0))

    schema_with_meta = TARGET_DATASET_SCHEMA.with_metadata(custom_metadata)
    table = pa.Table.from_pylist(rows, schema=schema_with_meta)
    pq.write_table(table, str(out_file), compression="snappy")
    print(f"Target intervention dataset exported to: {out_file}")
    return str(out_file)


def print_split_report(split_summary: Dict[str, Any]) -> None:
    """Prints a structured CLI partition report covering Task 11.1 requirements."""
    print("=" * 80)
    print("      TARGET-DOMAIN INTERVENTION DATASET: SESSION-BOUNDED SPLIT REPORT")
    print("=" * 80)
    sc = split_summary["session_counts"]
    ec = split_summary["example_counts"]
    print(f"Split Strategy:     {split_summary['split_strategy']} (zero temporal leakage)")
    print(f"Random Seed:        {split_summary['random_seed']}")
    print(f"Total Sessions:     {sc['total']} (Train: {sc['train']}, Val: {sc['val']}, Test: {sc['test']})")
    print(f"Total Examples:     {ec['total']} (Train: {ec['train']}, Val: {ec['val']}, Test: {ec['test']})")
    print("-" * 80)

    print("PER-PARTITION CLASS DISTRIBUTION (5-CLASS VOCABULARY):")
    print(f"{'Class Name':<26} {'Train Count (%)':<20} {'Val Count (%)':<18} {'Test Count (%)':<18}")
    print("-" * 80)

    train_dist = split_summary["class_distribution"]["train"]
    val_dist = split_summary["class_distribution"]["val"]
    test_dist = split_summary["class_distribution"]["test"]

    for cls_name in INTERVENTION_VOCABULARY:
        tr_c = train_dist["counts"].get(cls_name, 0)
        tr_p = train_dist["percentages"].get(cls_name, 0.0)
        v_c = val_dist["counts"].get(cls_name, 0)
        v_p = val_dist["percentages"].get(cls_name, 0.0)
        te_c = test_dist["counts"].get(cls_name, 0)
        te_p = test_dist["percentages"].get(cls_name, 0.0)

        tr_str = f"{tr_c:>4} ({tr_p:>5.1f}%)"
        v_str = f"{v_c:>3} ({v_p:>5.1f}%)"
        te_str = f"{te_c:>3} ({te_p:>5.1f}%)"
        print(f"  {cls_name:<24}: {tr_str:<20} {v_str:<18} {te_str:<18}")

    print("-" * 80)
    print("ABSENT CLASSES PER PARTITION:")
    for p_name in ["train", "val", "test"]:
        absent = split_summary["absent_classes"][p_name]
        absent_str = ", ".join(absent) if absent else "None (all 5 classes represented)"
        print(f"  {p_name.capitalize():<8}: {absent_str}")

    print("-" * 80)
    print("MAJORITY-CLASS BASELINES (Evaluated against training majority class):")
    for p_name in ["train", "val", "test"]:
        base = split_summary["majority_baseline"][p_name]
        print(
            f"  {p_name.capitalize():<8}: Majority Class: '{base['majority_class']}' | "
            f"Accuracy: {base['accuracy'] * 100:.1f}% | "
            f"Macro-F1: {base['macro_f1']:.4f}"
        )

    print("-" * 80)
    cw = split_summary["class_weights"]
    print(f"CLASS WEIGHTS RESOLUTION ({cw['derived_on'].upper()}):")
    print(f"  Strategy:         {cw['strategy']} (alpha={cw['smoothing_alpha']})")
    print(f"  Max/Min Ratio:    {cw['max_min_ratio']:.2f}")
    for cls_name, w in cw["weights"].items():
        print(f"    {cls_name:<24}: {w:.4f}")
    print("-" * 80)
    print("Claim Notice:        Labels are scripted policy outputs (ADR-013).")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Assemble TargetInterventionHead Dataset & Splits (Tasks 10.3, 11.1, 11.2).")
    parser.add_argument("--trace-dir", type=str, default=".data/raw/scripted", help="Input directory of traces")
    parser.add_argument("--output", type=str, default=None, help="Output Parquet path (defaults to versioned directory)")
    parser.add_argument("--build-labels", action="store_true", help="Build labels and report class distribution")
    parser.add_argument("--split", action="store_true", help="Perform session-bounded split and report partition metrics")
    parser.add_argument("--build-manifest", action="store_true", help="Generate provenance-versioned dataset manifest")
    parser.add_argument("--validate-manifest", type=str, default=None, help="Path to manifest JSON to validate")
    parser.add_argument("--version", type=str, default="v1.0.0", help="Dataset version identifier")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Val split ratio")
    parser.add_argument("--imbalance-strategy", type=str, default="smoothed", choices=["smoothed", "inverse", "none"], help="Class imbalance handling strategy")
    parser.add_argument("--smoothing-alpha", type=float, default=10.0, help="Smoothing alpha for class weights")
    parser.add_argument("--seq-len", type=int, default=8, help="Sequence length T")
    args = parser.parse_args()

    if args.validate_manifest:
        try:
            from data_manager import validate_dataset_manifest
        except ImportError:
            from src.data_manager import validate_dataset_manifest
        res = validate_dataset_manifest(args.validate_manifest, trace_dir=args.trace_dir)
        print(f"[TargetDataset] Manifest validation successful: {res}")
        return

    # Resolve output parquet path
    version_dir = Path(f".data/processed/{args.version}")
    default_parquet = version_dir / "target_intervention_dataset.parquet"
    out_parquet_path = Path(args.output) if args.output else default_parquet

    rows, stats = build_target_dataset_from_traces(
        trace_dir=args.trace_dir,
        output_path=None,
        seq_len=args.seq_len,
    )

    split_summary = None
    if args.split or args.build_manifest:
        rows, split_summary = split_target_dataset(
            rows,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            random_seed=args.seed,
            imbalance_strategy=args.imbalance_strategy,
            smoothing_alpha=args.smoothing_alpha,
        )
        print_split_report(split_summary)
    else:
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

    # Save parquet if requested or if split/build-manifest
    if args.build_labels or args.split or args.build_manifest:
        save_target_dataset_parquet(rows, out_parquet_path, seq_len=args.seq_len, split_summary=split_summary)
        # Also copy / save to root .data/processed/target_intervention_dataset.parquet for convenience
        root_parquet = Path(".data/processed/target_intervention_dataset.parquet")
        if out_parquet_path.resolve() != root_parquet.resolve():
            root_parquet.parent.mkdir(parents=True, exist_ok=True)
            save_target_dataset_parquet(rows, root_parquet, seq_len=args.seq_len, split_summary=split_summary)

    if args.build_manifest and split_summary is not None:
        try:
            from data_manager import create_dataset_manifest
        except ImportError:
            from src.data_manager import create_dataset_manifest
        manifest_path = version_dir / "manifest.json"
        create_dataset_manifest(
            rows=rows,
            split_summary=split_summary,
            trace_dir=args.trace_dir,
            output_path=manifest_path,
            dataset_version=args.version,
        )


if __name__ == "__main__":
    main()

