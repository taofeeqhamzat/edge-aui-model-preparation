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
    from data_manager import find_project_root
    from semantic_targets import (
        DEFAULT_LOOKAHEAD_MS,
        NO_OUTCOME,
        RECORDED_ONLY_OUTCOMES,
        RECORDED_VOCABULARY_VERSION,
        TARGET_VOCABULARY,
        TARGET_VOCABULARY_VERSION,
        find_source_event,
        is_target_outcome,
        record_timestamp,
        resolve_target_label,
        select_target_record,
        target_outcome_id,
    )
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
    from src.semantic_targets import (
        DEFAULT_LOOKAHEAD_MS,
        NO_OUTCOME,
        RECORDED_ONLY_OUTCOMES,
        RECORDED_VOCABULARY_VERSION,
        TARGET_VOCABULARY,
        TARGET_VOCABULARY_VERSION,
        find_source_event,
        is_target_outcome,
        record_timestamp,
        resolve_target_label,
        select_target_record,
        target_outcome_id,
    )
    try:
        from src.data_manager import find_project_root
    except ImportError:
        find_project_root = lambda: Path(__file__).resolve().parent.parent


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
        # Semantic target fields (Task 40.2). Null on intervention-only rows.
        ("target_task_outcome", pa.string()),
        ("target_task_outcome_id", pa.int64()),
        ("target_vocabulary_version", pa.string()),
        ("recorded_vocabulary_version", pa.string()),
        ("task_definition_version", pa.string()),
        ("target_label_source", pa.string()),
        ("target_event_timestamp", pa.float64()),
        ("lead_time_ms", pa.float64()),
    ])


def _session_end_ms(trace: Dict[str, Any], micro_tensors: List[Dict[str, Any]]) -> float:
    """Latest observed timestamp in the session (events, outcome records or window ends)."""
    timestamps: List[float] = []
    for event in trace.get("behaviourEvents", []):
        try:
            timestamps.append(float(event.get("timestamp")))
        except (TypeError, ValueError):
            continue
    for record in trace.get("outcomes", []):
        if isinstance(record, dict):
            timestamp = record_timestamp(record)
            if timestamp is not None:
                timestamps.append(timestamp)
    for window in micro_tensors:
        try:
            timestamps.append(float(window.get("windowEnd")))
        except (TypeError, ValueError):
            continue
    return max(timestamps) if timestamps else 0.0


def _resolve_candidate(record: Dict[str, Any], behaviour_events: List[Dict[str, Any]],
                       has_semantic_contract: bool) -> Dict[str, Any]:
    """Resolve one outcome record to a candidate semantic act (or an implicit no-act)."""
    source_event = None
    if record.get("taskOutcome") is None and not has_semantic_contract:
        source_event = find_source_event(record, behaviour_events)
    name, source = resolve_target_label(record, has_semantic_contract, source_event)
    return {"name": name, "source": source}


def _semantic_candidates(
    outcomes: List[Dict[str, Any]],
    behaviour_events: List[Dict[str, Any]],
    has_semantic_contract: bool,
    horizon_start: float,
    horizon_end: float,
) -> List[Dict[str, Any]]:
    """Outcome records inside the horizon that represent a semantic act."""
    candidates: List[Dict[str, Any]] = []
    for record in outcomes:
        timestamp = record_timestamp(record)
        if timestamp is None or not (horizon_start <= timestamp <= horizon_end):
            continue
        resolved = _resolve_candidate(record, behaviour_events, has_semantic_contract)
        # An implicit "no semantic act" record is not a candidate: a later semantic act inside
        # the horizon is still the next target.
        if resolved["name"] == NO_OUTCOME and resolved["source"] == "trace_no_outcome":
            continue
        candidates.append({
            "timestamp": timestamp,
            "name": resolved["name"],
            "source": resolved["source"],
            "record": record,
        })
    return candidates


def _select_candidate(
    candidates: List[Dict[str, Any]],
    anchor_end_ms: float,
    lookahead_min_ms: float,
    lookahead_max_ms: float,
) -> Dict[str, Any]:
    synthetic = [
        {"sourceEventTimestamp": candidate["timestamp"], "taskOutcome": candidate["name"],
         "_candidate": candidate}
        for candidate in candidates
    ]
    selected = select_target_record(synthetic, anchor_end_ms, lookahead_min_ms, lookahead_max_ms)
    return selected["_candidate"]  # type: ignore[index]


def _resolve_semantic_target(
    outcomes: List[Dict[str, Any]],
    behaviour_events: List[Dict[str, Any]],
    session_end_ms: float,
    has_semantic_contract: bool,
    anchor_end_ms: float,
    lookahead_min_ms: float,
    lookahead_max_ms: float,
) -> Dict[str, Any]:
    """Resolve the future semantic target for one anchor window.

    Returns ``{"status": "labelled", ...}`` or an exclusion status. The horizon must close
    inside the session; an incomplete lookahead is excluded rather than labelled, because a
    missing future window is not evidence that nothing happened.
    """
    horizon_end = anchor_end_ms + lookahead_max_ms
    if horizon_end > session_end_ms:
        return {"status": "excluded_incomplete"}

    candidates = _semantic_candidates(
        outcomes, behaviour_events, has_semantic_contract,
        anchor_end_ms + lookahead_min_ms, horizon_end,
    )
    if not candidates:
        return {
            "status": "labelled",
            "label": NO_OUTCOME,
            "source": "lookahead_no_outcome",
            "timestamp": None,
            "lead_time_ms": None,
        }

    chosen = _select_candidate(candidates, anchor_end_ms, lookahead_min_ms, lookahead_max_ms)
    if chosen["name"] in RECORDED_ONLY_OUTCOMES:
        return {"status": "excluded_recorded_only", "label": chosen["name"]}

    return {
        "status": "labelled",
        "label": chosen["name"],
        "source": chosen["source"],
        "timestamp": chosen["timestamp"],
        "lead_time_ms": chosen["timestamp"] - anchor_end_ms,
    }


def _validate_build_options(
    target: str,
    lookahead_min_ms: Optional[float],
    lookahead_max_ms: Optional[float],
) -> Tuple[float, float]:
    if target not in ("intervention", "semantic"):
        raise ValueError(f"Unknown target '{target}'; expected 'intervention' or 'semantic'")
    minimum = DEFAULT_LOOKAHEAD_MS[0] if lookahead_min_ms is None else float(lookahead_min_ms)
    maximum = DEFAULT_LOOKAHEAD_MS[1] if lookahead_max_ms is None else float(lookahead_max_ms)
    if maximum <= minimum:
        raise ValueError(
            f"Invalid lookahead horizon: max ({maximum}) must exceed min ({minimum})"
        )
    return minimum, maximum


def _build_summary_stats(
    target: str,
    trace_count: int,
    candidate_windows: int,
    rows: List[Dict[str, Any]],
    exclusions: Dict[str, int],
    label_sources: Dict[str, int],
    lookahead_min: float,
    lookahead_max: float,
) -> Dict[str, Any]:
    return {
        "target": target,
        "total_traces": trace_count,
        "total_candidate_windows": candidate_windows,
        "total_retained_examples": len(rows),
        "excluded_missing_context": exclusions.get("excluded_missing_context", 0),
        "excluded_incomplete_lookahead": exclusions.get("excluded_incomplete", 0),
        "excluded_recorded_only": exclusions.get("excluded_recorded_only", 0),
        "label_sources": label_sources,
        "target_vocabulary_version": TARGET_VOCABULARY_VERSION,
        "recorded_vocabulary_version": RECORDED_VOCABULARY_VERSION,
        "lookahead_horizon_ms": [lookahead_min, lookahead_max],
        "distribution": compute_label_distribution(
            [row["target_intervention"] for row in rows]
        ),
    }


def _trace_files(trace_dir: Union[str, Path]) -> List[str]:
    files = sorted(glob.glob(str(Path(trace_dir) / "*.json")))
    files = [path for path in files if not Path(path).name.startswith("manifest")]
    if not files:
        raise FileNotFoundError(f"No trace JSON files found in {trace_dir}")
    return files


def _trace_context(trace_file: str) -> Dict[str, Any]:
    """Per-trace identifiers, records and session end, read once per file."""
    with open(trace_file, "r", encoding="utf-8") as fp:
        trace = json.load(fp)

    session = trace.get("session", {})
    task_info = trace.get("task", {})
    metadata = trace.get("metadata", {}) if isinstance(trace.get("metadata"), dict) else {}
    micro_tensors = trace.get("microTensors", [])
    if isinstance(micro_tensors, list):
        micro_tensors = sorted(micro_tensors, key=lambda window: window.get("windowId", 0))
    else:
        micro_tensors = []

    outcome_records = [
        record for record in trace.get("outcomes", []) if isinstance(record, dict)
    ]
    return {
        "session_id": str(session.get("sessionId", Path(trace_file).stem)),
        "experiment_id": str(session.get("experimentId", "unknown")),
        "condition_id": str(session.get("conditionId", "unknown")),
        "default_task_id": str(task_info.get("currentTaskId") or session.get("taskId") or "T1"),
        "task_definition_version": (
            metadata.get("taskDefinitionVersion") or task_info.get("definitionVersion")
        ),
        "recorded_vocabulary_version": metadata.get("semanticOutcomeVocabularyVersion"),
        "has_semantic_contract": bool(metadata.get("semanticOutcomeVocabularyVersion"))
        or str(trace.get("schemaVersion")) == "1.4.0",
        "outcomes": outcome_records,
        "outcomes_map": {
            record["windowId"]: record for record in outcome_records if "windowId" in record
        },
        "behaviour_events": trace.get("behaviourEvents", []),
        "micro_tensors": micro_tensors,
        "session_end_ms": _session_end_ms(trace, micro_tensors),
    }


def _sequence_matrix(window_slice: List[Dict[str, Any]]) -> List[List[float]]:
    matrix: List[List[float]] = []
    for window in window_slice:
        values = list(window.get("values", []))
        if len(values) < 18:
            values = values + [0.0] * (18 - len(values))
        matrix.append([float(value) for value in values[:18]])
    return matrix


def _sequence_event_ids(
    session_id: str,
    behaviour_events: List[Dict[str, Any]],
    start_ms: float,
    end_ms: float,
) -> List[str]:
    event_ids = [
        f"{session_id}:ev_{index}"
        for index, event in enumerate(behaviour_events)
        if start_ms <= float(event.get("timestamp", 0.0)) <= end_ms
    ]
    if not event_ids and behaviour_events:
        # Fallback to the nearest prior event when the span has no event of its own.
        event_ids = [f"{session_id}:ev_{len(behaviour_events) - 1}"]
    return event_ids


def _anchor_context(
    context: Dict[str, Any],
    anchor_window: Dict[str, Any],
    anchor_window_id: Any,
) -> Tuple[Dict[str, Any], str, str, Dict[str, Any]]:
    """Anchor outcome record, generic outcome name, anchor task id and UIContext snapshot."""
    anchor_end = float(anchor_window.get("windowEnd", 0))
    outcome_record = context["outcomes_map"].get(anchor_window_id, {})
    outcome_name = outcome_record.get("outcome", "NO_OUTCOME")
    task_id = outcome_record.get("taskId") or context["default_task_id"]
    step_id = outcome_record.get("taskStepId")
    route = outcome_record.get("route", "Analytics")
    if not step_id:
        for event in reversed(context["behaviour_events"]):
            if event.get("timestamp", 0) <= anchor_end and event.get("taskStepId"):
                step_id = event.get("taskStepId")
                route = event.get("route", route)
                break

    is_analytics = route == "Analytics"
    ui_context = {
        "route": route,
        "primaryActionAvailable": is_analytics,
        "helpAvailable": is_analytics,
        "expandable": is_analytics,
        "taskId": task_id,
        "taskStepId": step_id,
        "availableActions": ["click", "change", "hover"] if is_analytics else ["click"],
    }
    return outcome_record, outcome_name, task_id, ui_context


def _encode_context(ui_context: Dict[str, Any]) -> Optional[np.ndarray]:
    try:
        context_vec = encode_ui_context(ui_context)
        if not validate_context_vector(context_vec).valid:
            return None
        return context_vec
    except Exception:
        return None


_OUTCOME_TAXONOMY_MAP = {
    "NO_OUTCOME": 0,
    "CLICK": 1,
    "FORM_SUBMIT": 2,
    "BACKTRACK": 3,
    "RAPID_SCROLL": 4,
    "HOVER_DWELL": 5,
    "ABANDON": 6,
}


def _row_payload(
    context: Dict[str, Any],
    task_id: str,
    anchor_window_id: Any,
    seq_matrix: List[List[float]],
    context_vec: np.ndarray,
    seq_event_ids: List[str],
    label_info: Dict[str, Any],
    outcome_name: str,
    semantic: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """One dataset row. Branch-free: all label decisions were made by the caller."""
    semantic_label = semantic["label"] if semantic else None
    return {
        "session_id": context["session_id"],
        "experiment_id": context["experiment_id"],
        "condition_id": context["condition_id"],
        "task_id": task_id,
        "anchor_window_id": int(anchor_window_id),
        "sequence": seq_matrix,
        "context_vector": context_vec.tolist(),
        "target_outcome": outcome_name,
        "target_outcome_id": _OUTCOME_TAXONOMY_MAP.get(outcome_name, 0),
        "target_intervention": label_info["target_intervention"],
        "target_intervention_id": int(label_info["target_intervention_id"]),
        "label_policy_version": str(label_info["label_policy_version"]),
        "is_scripted_policy": True,
        "split": "unassigned",
        "source_event_ids": seq_event_ids,
        "preprocessing_version": PREPROCESSING_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "target_generation_version": LABEL_POLICY_VERSION,
        "target_task_outcome": semantic_label,
        "target_task_outcome_id": (
            target_outcome_id(semantic_label)
            if semantic_label and is_target_outcome(semantic_label)
            else None
        ),
        "target_vocabulary_version": TARGET_VOCABULARY_VERSION if semantic else None,
        "recorded_vocabulary_version": (
            context["recorded_vocabulary_version"] if semantic else None
        ),
        "task_definition_version": context["task_definition_version"],
        "target_label_source": semantic["source"] if semantic else None,
        "target_event_timestamp": semantic["timestamp"] if semantic else None,
        "lead_time_ms": semantic["lead_time_ms"] if semantic else None,
    }


def _build_example(
    context: Dict[str, Any],
    index: int,
    seq_len: int,
    target: str,
    lookahead_min: float,
    lookahead_max: float,
) -> Dict[str, Any]:
    """Build one example, or return an exclusion status."""
    window_slice = context["micro_tensors"][index: index + seq_len]
    anchor_window = window_slice[-1]
    anchor_window_id = anchor_window.get("windowId", 0)
    anchor_end = float(anchor_window.get("windowEnd", 0))

    semantic: Optional[Dict[str, Any]] = None
    if target == "semantic":
        semantic = _resolve_semantic_target(
            context["outcomes"],
            context["behaviour_events"],
            context["session_end_ms"],
            context["has_semantic_contract"],
            anchor_end,
            lookahead_min,
            lookahead_max,
        )
        if semantic["status"] != "labelled":
            return {"status": semantic["status"]}

    seq_matrix = _sequence_matrix(window_slice)
    seq_event_ids = _sequence_event_ids(
        context["session_id"], context["behaviour_events"],
        float(window_slice[0].get("windowStart", 0.0)) - 250.0, anchor_end,
    )
    _, outcome_name, task_id, ui_context = _anchor_context(
        context, anchor_window, anchor_window_id
    )

    context_vec = _encode_context(ui_context)
    if context_vec is None:
        return {"status": "excluded_missing_context"}

    ui_context["taskProgress"] = float(context_vec[4])
    label_info = assign_intervention_label(outcome_name, ui_context)
    row = _row_payload(
        context, task_id, anchor_window_id, seq_matrix, context_vec, seq_event_ids,
        label_info, outcome_name, semantic,
    )
    return {"status": "labelled", "row": row}


def build_target_dataset_from_traces(
    trace_dir: Union[str, Path] = ".data/raw/scripted",
    output_path: Optional[Union[str, Path]] = None,
    seq_len: int = 8,
    stride: int = 1,
    target: str = "intervention",
    lookahead_min_ms: Optional[float] = None,
    lookahead_max_ms: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build the target dataset from a directory of ExperimentTrace JSON files.

    ``target="intervention"`` keeps the prior-milestone behaviour: labels come from the anchor
    window's generic outcome and the scripted intervention policy. ``target="semantic"`` adds a
    causal future semantic label (Task 40.2) and excludes windows that cannot carry one.

    Returns ``(rows, summary_stats)``.
    """
    lookahead_min, lookahead_max = _validate_build_options(
        target, lookahead_min_ms, lookahead_max_ms
    )

    trace_files = _trace_files(trace_dir)
    all_rows: List[Dict[str, Any]] = []
    exclusions: Dict[str, int] = {}
    label_sources: Dict[str, int] = {}
    total_candidate_windows = 0

    for trace_file in trace_files:
        context = _trace_context(trace_file)
        if len(context["micro_tensors"]) < seq_len:
            continue
        for index in range(0, len(context["micro_tensors"]) - seq_len + 1, stride):
            total_candidate_windows += 1
            result = _build_example(
                context, index, seq_len, target, lookahead_min, lookahead_max
            )
            if result["status"] != "labelled":
                exclusions[result["status"]] = exclusions.get(result["status"], 0) + 1
                continue
            row = result["row"]
            if target == "semantic":
                label_sources[row["target_label_source"]] = (
                    label_sources.get(row["target_label_source"], 0) + 1
                )
            all_rows.append(row)

    summary_stats = _build_summary_stats(
        target, len(trace_files), total_candidate_windows, all_rows,
        exclusions, label_sources, lookahead_min, lookahead_max,
    )

    if output_path and PYARROW_AVAILABLE and all_rows:
        save_target_dataset_parquet(all_rows, output_path, seq_len=seq_len)

    return all_rows, summary_stats


def _session_partitions(
    rows: List[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    random_seed: int,
) -> Tuple[Set[str], Set[str], Set[str], List[str]]:
    """Deterministic session-bounded partition assignment (zero session/task leakage)."""
    sessions = sorted({row["session_id"] for row in rows})
    if len(sessions) < 3:
        raise ValueError(
            f"Insufficient distinct sessions for train/val/test split: found {len(sessions)}"
        )

    shuffled = sessions.copy()
    np.random.RandomState(random_seed).shuffle(shuffled)
    n_train = int(round(len(sessions) * train_ratio))
    n_val = int(round(len(sessions) * val_ratio))
    train = set(shuffled[:n_train])
    val = set(shuffled[n_train: n_train + n_val])
    test = set(shuffled[n_train + n_val:])

    assert train.isdisjoint(val), "Train and Val sessions must be disjoint"
    assert train.isdisjoint(test), "Train and Test sessions must be disjoint"
    assert val.isdisjoint(test), "Val and Test sessions must be disjoint"

    train_pairs = {(row["session_id"], row["task_id"]) for row in rows if row["session_id"] in train}
    val_pairs = {(row["session_id"], row["task_id"]) for row in rows if row["session_id"] in val}
    test_pairs = {(row["session_id"], row["task_id"]) for row in rows if row["session_id"] in test}
    assert train_pairs.isdisjoint(val_pairs), "Train and Val (session, task) pairs must be disjoint"
    assert train_pairs.isdisjoint(test_pairs), "Train and Test (session, task) pairs must be disjoint"
    assert val_pairs.isdisjoint(test_pairs), "Val and Test (session, task) pairs must be disjoint"
    return train, val, test, sessions


def _assign_splits(
    rows: List[Dict[str, Any]],
    train: Set[str],
    val: Set[str],
    test: Set[str],
) -> None:
    for row in rows:
        session_id = row["session_id"]
        if session_id in train:
            row["split"] = "train"
        elif session_id in val:
            row["split"] = "val"
        elif session_id in test:
            row["split"] = "test"
        else:
            raise ValueError(f"Unpartitioned session encountered: {session_id}")


def _partition_rows(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "train": [row for row in rows if row["split"] == "train"],
        "val": [row for row in rows if row["split"] == "val"],
        "test": [row for row in rows if row["split"] == "test"],
    }


def _partition_statistics(
    partitions: Dict[str, List[Dict[str, Any]]],
    rows: List[Dict[str, Any]],
    vocabulary: List[str],
    label_key: str,
) -> Tuple[Dict[str, Any], Dict[str, List[str]], Dict[str, Dict[str, int]], Dict[str, List[str]]]:
    class_distribution: Dict[str, Any] = {}
    absent_classes: Dict[str, List[str]] = {}
    task_coverage: Dict[str, Dict[str, int]] = {}

    for partition, partition_rows in partitions.items():
        distribution = compute_label_distribution(
            [row[label_key] for row in partition_rows], vocabulary=vocabulary
        )
        class_distribution[partition] = distribution
        absent_classes[partition] = [
            name for name, count in distribution["counts"].items() if count == 0
        ]
        coverage: Dict[str, int] = {}
        for row in partition_rows:
            task = str(row.get("task_id", "unknown"))
            coverage[task] = coverage.get(task, 0) + 1
        task_coverage[partition] = dict(sorted(coverage.items()))

    class_distribution["overall"] = compute_label_distribution(
        [row[label_key] for row in rows], vocabulary=vocabulary
    )
    all_tasks = sorted({str(row.get("task_id", "unknown")) for row in rows})
    tasks_absent = {
        partition: [task for task in all_tasks if task not in task_coverage[partition]]
        for partition in partitions
    }
    return class_distribution, absent_classes, task_coverage, tasks_absent


def _majority_baselines(
    partitions: Dict[str, List[Dict[str, Any]]],
    class_distribution: Dict[str, Any],
    vocabulary: List[str],
    majority_class: str,
    majority_id: int,
) -> Dict[str, Any]:
    baselines: Dict[str, Any] = {}
    for partition, partition_rows in partitions.items():
        total = len(partition_rows)
        if total == 0:
            baselines[partition] = {"accuracy": 0.0, "macro_f1": 0.0, "support": {}}
            continue

        counts = class_distribution[partition]["counts"]
        accuracy = float(counts.get(majority_class, 0) / total)
        f1_scores: List[float] = []
        per_class_f1: Dict[str, float] = {}
        for name in vocabulary:
            support = counts.get(name, 0)
            if name == majority_class and support > 0:
                precision = support / total
                f1 = (2 * precision / (precision + 1.0)) if precision else 0.0
            else:
                f1 = 0.0
            per_class_f1[name] = float(f1)
            if support > 0:
                f1_scores.append(f1)

        baselines[partition] = {
            "majority_class": majority_class,
            "majority_class_id": majority_id,
            "accuracy": accuracy,
            "macro_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
            "per_class_f1": per_class_f1,
            "support": counts,
        }
    return baselines


def _resolve_class_weights(
    partitions: Dict[str, List[Dict[str, Any]]],
    vocabulary: List[str],
    id_key: str,
    strategy: str,
    smoothing_alpha: float,
) -> Dict[str, Any]:
    train_targets = np.array([row[id_key] for row in partitions["train"]])
    num_classes = len(vocabulary)
    if strategy == "smoothed":
        weights = compute_class_weights(
            train_targets, num_classes=num_classes, smoothing_alpha=smoothing_alpha,
            allow_empty=True,
        )
    elif strategy == "inverse":
        weights = compute_class_weights(
            train_targets, num_classes=num_classes, smoothing_alpha=0.0, allow_empty=True
        )
    elif strategy == "none":
        weights = (
            torch.ones(num_classes, dtype=torch.float32)
            if TORCH_AVAILABLE else np.ones(num_classes, dtype=np.float32)
        )
    else:
        raise ValueError(
            f"Unknown imbalance_strategy: '{strategy}'. Expected 'smoothed', 'inverse', or 'none'."
        )

    values = weights.numpy() if hasattr(weights, "numpy") else np.asarray(weights)
    max_weight = float(np.max(values))
    positive = values[values > 0]
    min_weight = float(np.min(positive)) if positive.size else 0.0
    return {
        "strategy": strategy,
        "smoothing_alpha": smoothing_alpha if strategy == "smoothed" else 0.0,
        "derived_on": "train_partition_only",
        "weights": {vocabulary[i]: float(values[i]) for i in range(num_classes)},
        "max_min_ratio": (max_weight / min_weight) if min_weight > 0 else 0.0,
    }


def split_target_dataset(
    rows: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_seed: int = 42,
    imbalance_strategy: str = "smoothed",
    smoothing_alpha: float = 10.0,
    label_key: str = "target_intervention",
    id_key: str = "target_intervention_id",
    vocabulary: Optional[List[str]] = None,
    label_name: str = "intervention",
    build_stats: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Produce strictly session-bounded train / validation / test splits and resolve class weights
    exclusively on the training partition (Task 11.1 / Task 40.2).

    The label column is configurable so the same split contract serves the intervention dataset
    and the semantic task-outcome dataset. Session is the split unit (ADR-031); class weights,
    absent-class declarations and majority baselines are resolved from the declared partitions.
    """
    if not rows:
        raise ValueError("Cannot split an empty list of rows")

    vocab = list(vocabulary) if vocabulary is not None else list(INTERVENTION_VOCABULARY)
    if not vocab:
        raise ValueError("The split vocabulary must not be empty")

    train, val, test, sessions = _session_partitions(rows, train_ratio, val_ratio, random_seed)
    _assign_splits(rows, train, val, test)
    partitions = _partition_rows(rows)
    class_distribution, absent_classes, task_coverage, tasks_absent = _partition_statistics(
        partitions, rows, vocab, label_key
    )
    majority_class = class_distribution["train"]["majority_class"]
    majority_id = vocab.index(majority_class)
    baselines = _majority_baselines(
        partitions, class_distribution, vocab, majority_class, majority_id
    )
    class_weights = _resolve_class_weights(
        partitions, vocab, id_key, imbalance_strategy, smoothing_alpha
    )

    rationale = (
        f"Imbalance strategy: '{imbalance_strategy}' "
        f"(alpha={class_weights['smoothing_alpha']}). Resolved exclusively on the training "
        f"partition ({len(partitions['train'])} examples, {len(train)} sessions) to prevent "
        f"test-set label balance from leaking into training. Preserves full sample density "
        f"without synthetic duplication (no oversampling)."
    )
    summary: Dict[str, Any] = {
        "split_strategy": "session_bounded",
        "split_unit": "session",
        "label_key": label_key,
        "label_name": label_name,
        "label_vocabulary": vocab,
        "random_seed": random_seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "session_counts": {
            "train": len(train), "val": len(val), "test": len(test), "total": len(sessions),
        },
        "sessions": {
            "train": sorted(train), "val": sorted(val), "test": sorted(test),
        },
        "example_counts": {
            "train": len(partitions["train"]),
            "val": len(partitions["val"]),
            "test": len(partitions["test"]),
            "total": len(rows),
        },
        "class_distribution": class_distribution,
        "absent_classes": absent_classes,
        "task_coverage": task_coverage,
        "tasks_absent": tasks_absent,
        "majority_baseline": baselines,
        "imbalance_strategy": imbalance_strategy,
        "imbalance_rationale": rationale,
        "class_weights": class_weights,
    }
    if build_stats:
        summary["build_stats"] = {
            key: value for key, value in build_stats.items()
            if key in (
                "target", "total_traces", "total_candidate_windows",
                "excluded_missing_context", "excluded_incomplete_lookahead",
                "excluded_recorded_only", "label_sources",
                "target_vocabulary_version", "recorded_vocabulary_version",
                "lookahead_horizon_ms",
            )
        }
    return rows, summary


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
        custom_metadata["split_unit"] = str(split_summary.get("split_unit", "session"))
        custom_metadata["split_seed"] = str(split_summary.get("random_seed"))
        custom_metadata["train_examples"] = str(split_summary.get("example_counts", {}).get("train", 0))
        custom_metadata["val_examples"] = str(split_summary.get("example_counts", {}).get("val", 0))
        custom_metadata["test_examples"] = str(split_summary.get("example_counts", {}).get("test", 0))
    if rows and rows[0].get("target_vocabulary_version"):
        custom_metadata["target_vocabulary_version"] = str(rows[0]["target_vocabulary_version"])
        custom_metadata["recorded_vocabulary_version"] = str(rows[0].get("recorded_vocabulary_version"))

    schema_with_meta = TARGET_DATASET_SCHEMA.with_metadata(custom_metadata)
    table = pa.Table.from_pylist(rows, schema=schema_with_meta)
    pq.write_table(table, str(out_file), compression="snappy")
    print(f"Target intervention dataset exported to: {out_file}")
    return str(out_file)


def _print_class_table(summary: Dict[str, Any], vocabulary: List[str], label_name: str) -> None:
    print(f"PER-PARTITION CLASS DISTRIBUTION ({len(vocabulary)}-CLASS {label_name} VOCABULARY):")
    print(f"{'Class Name':<26} {'Train Count (%)':<20} {'Val Count (%)':<18} {'Test Count (%)':<18}")
    print("-" * 80)
    distributions = summary["class_distribution"]
    for name in vocabulary:
        cells = []
        for partition in ("train", "val", "test"):
            count = distributions[partition]["counts"].get(name, 0)
            percent = distributions[partition]["percentages"].get(name, 0.0)
            cells.append(f"{count:>4} ({percent:>5.1f}%)")
        print(f"  {name:<24}: {cells[0]:<20} {cells[1]:<18} {cells[2]:<18}")


def _print_coverage(summary: Dict[str, Any], vocabulary: List[str]) -> None:
    print("ABSENT CLASSES PER PARTITION:")
    for partition in ("train", "val", "test"):
        absent = summary["absent_classes"][partition]
        text = ", ".join(absent) if absent else f"None (all {len(vocabulary)} classes represented)"
        print(f"  {partition.capitalize():<8}: {text}")
    print("-" * 80)
    print("TASK-LEVEL COVERAGE PER PARTITION:")
    for partition in ("train", "val", "test"):
        coverage = summary.get("task_coverage", {}).get(partition, {})
        text = ", ".join(f"{task}={count}" for task, count in coverage.items()) or "none"
        print(f"  {partition.capitalize():<8}: {text}")


def _print_baselines(summary: Dict[str, Any]) -> None:
    print("MAJORITY-CLASS BASELINES (Evaluated against training majority class):")
    for partition in ("train", "val", "test"):
        base = summary["majority_baseline"][partition]
        print(
            f"  {partition.capitalize():<8}: Majority Class: '{base['majority_class']}' | "
            f"Accuracy: {base['accuracy'] * 100:.1f}% | Macro-F1: {base['macro_f1']:.4f}"
        )


def _print_class_weights(summary: Dict[str, Any]) -> None:
    weights = summary["class_weights"]
    print(f"CLASS WEIGHTS RESOLUTION ({weights['derived_on'].upper()}):")
    print(f"  Strategy:         {weights['strategy']} (alpha={weights['smoothing_alpha']})")
    print(f"  Max/Min Ratio:    {weights['max_min_ratio']:.2f}")
    for name, weight in weights["weights"].items():
        print(f"    {name:<24}: {weight:.4f}")


def print_split_report(split_summary: Dict[str, Any]) -> None:
    """Prints a structured CLI partition report (Task 11.1 / Task 40.3)."""
    vocabulary = split_summary.get("label_vocabulary") or list(INTERVENTION_VOCABULARY)
    label_name = str(split_summary.get("label_name") or "intervention").upper()
    session_counts = split_summary["session_counts"]
    example_counts = split_summary["example_counts"]

    print("=" * 80)
    print(f"      TARGET-DOMAIN {label_name} DATASET: SESSION-BOUNDED SPLIT REPORT")
    print("=" * 80)
    print(f"Split Strategy:     {split_summary['split_strategy']} (session unit, zero temporal leakage)")
    print(f"Random Seed:        {split_summary['random_seed']}")
    print(f"Total Sessions:     {session_counts['total']} (Train: {session_counts['train']}, "
          f"Val: {session_counts['val']}, Test: {session_counts['test']})")
    print(f"Total Examples:     {example_counts['total']} (Train: {example_counts['train']}, "
          f"Val: {example_counts['val']}, Test: {example_counts['test']})")
    print("-" * 80)
    _print_class_table(split_summary, vocabulary, label_name)
    print("-" * 80)
    _print_coverage(split_summary, vocabulary)
    print("-" * 80)
    _print_baselines(split_summary)
    print("-" * 80)
    _print_class_weights(split_summary)
    print("-" * 80)
    if label_name == "INTERVENTION":
        print("Claim Notice:        Labels are scripted policy outputs (ADR-013).")
    else:
        print("Claim Notice:        Labels are future semantic acts; scripted captures are engineering evidence.")
    print("=" * 80)


class TargetInterventionDataset(torch.utils.data.Dataset if TORCH_AVAILABLE else object):
    """
    PyTorch Dataset for target-domain intervention sequences (Phase D).
    Yields (sequence, context, target_intervention_id) tuples:
      - sequence: Tensor of shape (seq_len, 18), dtype float32
      - context: Tensor of shape (6,), dtype float32
      - target: Tensor of shape (), dtype long
    Also exposes row metadata (session_id, task_id, anchor_window_id, target_outcome, etc.).
    """
    def __init__(
        self,
        sequences: Union[np.ndarray, "torch.Tensor", List[Any]],
        contexts: Union[np.ndarray, "torch.Tensor", List[Any]],
        targets: Union[np.ndarray, "torch.Tensor", List[int]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ):
        if TORCH_AVAILABLE and torch is not None:
            if torch.is_tensor(sequences):
                self.sequences = sequences.float()
            else:
                self.sequences = torch.tensor(sequences, dtype=torch.float32)

            if torch.is_tensor(contexts):
                self.contexts = contexts.float()
            else:
                self.contexts = torch.tensor(contexts, dtype=torch.float32)

            if torch.is_tensor(targets):
                self.targets = targets.long()
            else:
                self.targets = torch.tensor(targets, dtype=torch.long)
        else:
            self.sequences = np.asarray(sequences, dtype=np.float32)
            self.contexts = np.asarray(contexts, dtype=np.float32)
            self.targets = np.asarray(targets, dtype=np.int64)

        self.metadata = metadata or []
        self.X = self.sequences
        self.Y = self.targets

        if len(self.sequences) != len(self.contexts) or len(self.sequences) != len(self.targets):
            raise ValueError(
                f"Dimension mismatch: sequences ({len(self.sequences)}), "
                f"contexts ({len(self.contexts)}), targets ({len(self.targets)})"
            )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        return self.sequences[idx], self.contexts[idx], self.targets[idx]

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        """Return the provenance metadata dict for a specific example index."""
        if 0 <= idx < len(self.metadata):
            return self.metadata[idx]
        return {}


def _empty_intervention_dataset() -> TargetInterventionDataset:
    if TORCH_AVAILABLE:
        return TargetInterventionDataset(
            torch.empty((0, 8, 18), dtype=torch.float32),
            torch.empty((0, 6), dtype=torch.float32),
            torch.empty(0, dtype=torch.long),
            metadata=[],
        )
    return TargetInterventionDataset(
        np.empty((0, 8, 18), dtype=np.float32),
        np.empty((0, 6), dtype=np.float32),
        np.empty(0, dtype=np.int64),
        metadata=[],
    )


def load_intervention_dataset(
    version: str = "v1.0.0",
    split: str = "train",
    data_dir: Optional[Union[str, Path]] = None,
    parquet_path: Optional[Union[str, Path]] = None,
) -> TargetInterventionDataset:
    """
    Loads target intervention sequences for a given split from versioned Parquet storage.

    Parameters:
      version: Dataset version identifier (e.g. 'v1.0.0').
      split: Split partition to load ('train', 'val', 'test', or 'all').
      data_dir: Optional project root or data directory override.
      parquet_path: Optional direct path to target Parquet file.

    Returns:
      TargetInterventionDataset initialized with sequences, contexts, and targets.
    """
    if parquet_path is None:
        root = Path(data_dir) if data_dir else Path(find_project_root())
        parquet_path = root / ".data" / "processed" / version / "target_intervention_dataset.parquet"
    else:
        parquet_path = Path(parquet_path)

    if not parquet_path.is_file():
        raise FileNotFoundError(f"Target intervention dataset not found at: {parquet_path}")

    if not PYARROW_AVAILABLE:
        raise RuntimeError("PyArrow is required to read target intervention Parquet files.")

    table = pq.read_table(str(parquet_path))
    df = table.to_pandas()

    if split != "all":
        df = df[df["split"] == split]

    if len(df) == 0:
        return _empty_intervention_dataset()

    seq_raw = df["sequence"].values
    ctx_raw = df["context_vector"].values
    sequences = np.stack([np.stack(s) for s in seq_raw]).astype(np.float32)
    contexts = np.stack(ctx_raw).astype(np.float32)
    targets = np.array(df["target_intervention_id"].values, dtype=np.int64)

    meta_cols = [
        c for c in df.columns if c not in ("sequence", "context_vector", "target_intervention_id")
    ]
    metadata = df[meta_cols].to_dict(orient="records")

    return TargetInterventionDataset(sequences, contexts, targets, metadata=metadata)


def get_intervention_class_weights(
    version: str = "v1.0.0",
    data_dir: Optional[Union[str, Path]] = None,
    manifest_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, Any]] = None,
) -> Any:
    """
    Loads pre-computed training-partition class weights for the 5-class intervention vocabulary.
    Derived exclusively on the training partition (Task 11.1).

    Returns:
      torch.Tensor of shape (5,) with float32 class weights.
    """
    if manifest_path is None:
        root = Path(data_dir) if data_dir else Path(find_project_root())
        manifest_path = root / ".data" / "processed" / version / "manifest.json"
    else:
        manifest_path = Path(manifest_path)

    weights_list = []
    if manifest_path.is_file():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        w_dict = manifest_data.get("class_weights", {}).get("weights", {})
        num_classes = len(INTERVENTION_VOCABULARY)
        weights_list = [w_dict.get(INTERVENTION_VOCABULARY[i], 1.0) for i in range(num_classes)]
    else:
        train_ds = load_intervention_dataset(version=version, split="train", data_dir=data_dir)
        t_weights = compute_class_weights(
            train_ds.targets,
            num_classes=len(INTERVENTION_VOCABULARY),
            smoothing_alpha=10.0,
            allow_empty=True
        )
        weights_list = [float(w) for w in t_weights]

    if TORCH_AVAILABLE and torch is not None:
        tensor_w = torch.tensor(weights_list, dtype=torch.float32)
        if device is not None:
            tensor_w = tensor_w.to(device)
        return tensor_w
    return np.asarray(weights_list, dtype=np.float32)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assemble TargetInterventionHead Dataset & Splits (Tasks 10.3, 11.1, 11.2)."
    )
    parser.add_argument("--trace-dir", type=str, default=".data/raw/scripted", help="Input directory of traces")
    parser.add_argument("--output", type=str, default=None, help="Output Parquet path (defaults to versioned directory)")
    parser.add_argument("--build-labels", action="store_true", help="Build labels and report class distribution")
    parser.add_argument("--split", action="store_true", help="Perform session-bounded split and report partition metrics")
    parser.add_argument("--build-manifest", action="store_true", help="Generate provenance-versioned dataset manifest")
    parser.add_argument("--validate-manifest", type=str, default=None, help="Path to manifest JSON to validate")
    parser.add_argument("--version", type=str, default="v1.0.0", help="Dataset version identifier")
    parser.add_argument(
        "--target", type=str, default="auto", choices=["auto", "intervention", "semantic"],
        help="Label target; 'auto' selects semantic when --version contains 'semantic'"
    )
    parser.add_argument("--lookahead-min-ms", type=float, default=None, help="Semantic lookahead minimum (ms)")
    parser.add_argument("--lookahead-max-ms", type=float, default=None, help="Semantic lookahead maximum (ms)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Val split ratio")
    parser.add_argument("--imbalance-strategy", type=str, default="smoothed",
                        choices=["smoothed", "inverse", "none"], help="Class imbalance handling strategy")
    parser.add_argument("--smoothing-alpha", type=float, default=10.0, help="Smoothing alpha for class weights")
    parser.add_argument("--seq-len", type=int, default=8, help="Sequence length T")
    return parser


def _resolve_target(args: argparse.Namespace) -> str:
    if args.target != "auto":
        return args.target
    return "semantic" if "semantic" in args.version else "intervention"


def _split_kwargs(args: argparse.Namespace, target: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = dict(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.seed,
        imbalance_strategy=args.imbalance_strategy,
        smoothing_alpha=args.smoothing_alpha,
        build_stats=stats,
    )
    if target == "semantic":
        kwargs.update(
            label_key="target_task_outcome",
            id_key="target_task_outcome_id",
            vocabulary=list(TARGET_VOCABULARY.values()),
            label_name="semantic",
        )
    return kwargs


def _print_dataset_summary(stats: Dict[str, Any], target: str) -> None:
    distribution = stats["distribution"]
    print("=" * 70)
    print("TARGET-DOMAIN INTERVENTION DATASET SUMMARY")
    print(f"Traces Ingested:           {stats['total_traces']}")
    print(f"Total Candidate Sequences: {stats['total_candidate_windows']}")
    print(f"Retained Examples:         {stats['total_retained_examples']}")
    print(f"Excluded Missing Context:  {stats['excluded_missing_context']}")
    print("-" * 70)
    print("CLASS DISTRIBUTION (5-CLASS INTERVENTION VOCABULARY):")
    for name in INTERVENTION_VOCABULARY:
        count = distribution["counts"][name]
        percent = distribution["percentages"][name]
        print(f"  {name:<25}: {count:>4} ({percent:>5.1f}%)")
    print("-" * 70)
    print(f"Majority Baseline Class:   {distribution['majority_class']} "
          f"({distribution['majority_percentage']:.1f}%)")
    print(f"Policy Version:            {LABEL_POLICY_VERSION}")
    if target == "semantic":
        print("-" * 70)
        print(f"Semantic Target Version:   {stats['target_vocabulary_version']}")
        print(f"Lookahead Horizon (ms):    {stats['lookahead_horizon_ms']}")
        print(f"Excluded Incomplete:       {stats['excluded_incomplete_lookahead']}")
        print(f"Excluded Recorded-Only:    {stats['excluded_recorded_only']}")
        print(f"Semantic Label Sources:    {stats['label_sources']}")
        print("Claim Notice:              Future semantic labels; engineering evidence only.")
    else:
        print("Claim Notice:              Labels are scripted policy outputs (ADR-013).")
    print("=" * 70)


def _validate_manifest_cli(path: str, trace_dir: str) -> None:
    try:
        from data_manager import validate_dataset_manifest
    except ImportError:
        from src.data_manager import validate_dataset_manifest
    result = validate_dataset_manifest(path, trace_dir=trace_dir)
    print(f"[TargetDataset] Manifest validation successful: {result}")


def _export_dataset(
    args: argparse.Namespace,
    rows: List[Dict[str, Any]],
    split_summary: Optional[Dict[str, Any]],
    version_dir: Path,
    out_parquet_path: Path,
) -> None:
    if not (args.build_labels or args.split or args.build_manifest):
        return
    save_target_dataset_parquet(
        rows, out_parquet_path, seq_len=args.seq_len, split_summary=split_summary
    )
    _write_convenience_copy(args, rows, split_summary, out_parquet_path)
    _write_manifest(args, rows, split_summary, version_dir)


def _write_convenience_copy(
    args: argparse.Namespace,
    rows: List[Dict[str, Any]],
    split_summary: Optional[Dict[str, Any]],
    out_parquet_path: Path,
) -> None:
    root_parquet = Path(".data/processed/target_intervention_dataset.parquet")
    if args.output is not None or out_parquet_path.resolve() == root_parquet.resolve():
        return
    save_target_dataset_parquet(
        rows, root_parquet, seq_len=args.seq_len, split_summary=split_summary
    )


def _write_manifest(
    args: argparse.Namespace,
    rows: List[Dict[str, Any]],
    split_summary: Optional[Dict[str, Any]],
    version_dir: Path,
) -> None:
    if not args.build_manifest or split_summary is None:
        return
    try:
        from data_manager import create_dataset_manifest
    except ImportError:
        from src.data_manager import create_dataset_manifest
    create_dataset_manifest(
        rows=rows,
        split_summary=split_summary,
        trace_dir=args.trace_dir,
        output_path=version_dir / "manifest.json",
        dataset_version=args.version,
    )


def main() -> None:
    args = _build_parser().parse_args()
    if args.validate_manifest:
        _validate_manifest_cli(args.validate_manifest, args.trace_dir)
        return

    target = _resolve_target(args)
    rows, stats = build_target_dataset_from_traces(
        trace_dir=args.trace_dir,
        output_path=None,
        seq_len=args.seq_len,
        target=target,
        lookahead_min_ms=args.lookahead_min_ms,
        lookahead_max_ms=args.lookahead_max_ms,
    )
    split_summary = None
    if args.split or args.build_manifest:
        rows, split_summary = split_target_dataset(rows, **_split_kwargs(args, target, stats))
        print_split_report(split_summary)
    else:
        _print_dataset_summary(stats, target)

    version_dir = Path(f".data/processed/{args.version}")
    out_parquet_path = (
        Path(args.output) if args.output
        else version_dir / "target_intervention_dataset.parquet"
    )
    _export_dataset(args, rows, split_summary, version_dir, out_parquet_path)


if __name__ == "__main__":
    main()

