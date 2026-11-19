"""
window_horizon.py
Task 40.3: the small, pre-declared window/horizon experiment for the semantic target.

Four configurations are evaluated; there is no open-ended search. The experiment measures
dataset/label feasibility (coverage, lead time, cadence, imbalance, incomplete windows) and a
declared majority-class baseline on the test partition. It does **not** train a head, so it
cannot and must not claim semantic-model performance or participant evidence.

Selection criteria are declared before the run:

1. training-partition coverage — the number of target classes with at least
   ``MIN_CLASS_SUPPORT`` examples (maximise);
2. training-partition balance — the largest single-class share (minimise);
3. earlier warning — the median lead time in ms (minimise);
4. the declared A, B, C, D order (final deterministic tie-break).

Splits are session-bounded (ADR-031); task-level coverage is reported per partition.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from data_manager import create_dataset_manifest
    from semantic_targets import TARGET_VOCABULARY, TARGET_VOCABULARY_VERSION
    from target_dataset import (
        build_target_dataset_from_traces,
        save_target_dataset_parquet,
        split_target_dataset,
    )
except ImportError:
    from src.data_manager import create_dataset_manifest
    from src.semantic_targets import TARGET_VOCABULARY, TARGET_VOCABULARY_VERSION
    from src.target_dataset import (
        build_target_dataset_from_traces,
        save_target_dataset_parquet,
        split_target_dataset,
    )

# Pre-declared experiment matrix. History windows are 500 ms each; stride is 250 ms.
EXPERIMENTS: Tuple[Dict[str, Any], ...] = (
    {"id": "A", "label": "baseline 8x500 ms, [500, 1500] ms", "history_windows": 8,
     "stride_ms": 250, "lookahead_ms": (500, 1500)},
    {"id": "B", "label": "8x500 ms, [1000, 2000] ms", "history_windows": 8,
     "stride_ms": 250, "lookahead_ms": (1000, 2000)},
    {"id": "C", "label": "4x500 ms, [500, 1500] ms", "history_windows": 4,
     "stride_ms": 250, "lookahead_ms": (500, 1500)},
    {"id": "D", "label": "12x500 ms, [500, 1500] ms", "history_windows": 12,
     "stride_ms": 250, "lookahead_ms": (500, 1500)},
)

# A class needs this many training examples to count towards coverage.
MIN_CLASS_SUPPORT = 5


def detect_provenance(trace_dir: Union[str, Path]) -> str:
    """Return the single declared provenance of a trace directory; refuse a mixed set."""
    files = sorted(
        path for path in Path(trace_dir).glob("*.json")
        if "manifest" not in path.name
    )
    if not files:
        raise ValueError(f"No trace JSON files found in {trace_dir}")
    values = set()
    for path in files:
        with open(path, "r", encoding="utf-8") as handle:
            trace = json.load(handle)
        value = (trace.get("session") or {}).get("provenance")
        if value not in ("scripted", "participant"):
            raise ValueError(
                f"Trace '{path.name}' declares invalid provenance '{value}'; participant and "
                f"scripted evidence cannot be pooled"
            )
        values.add(value)
    if len(values) > 1:
        raise ValueError(
            f"Trace directory mixes provenances {sorted(values)}; participant and scripted "
            f"evidence cannot be pooled"
        )
    return values.pop()


def _session_ids(trace_dir: Union[str, Path]) -> set:
    ids = set()
    for path in Path(trace_dir).glob("*.json"):
        if path.name.startswith("manifest"):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            trace = json.load(handle)
        session_id = (trace.get("session") or {}).get("sessionId")
        if session_id:
            ids.add(str(session_id))
    return ids


def _evidence_scope_text(provenance: str, session_count: int) -> str:
    if provenance == "participant":
        return (
            f"Participant captures ({session_count} sessions). These are participant-derived "
            f"engineering evidence, not a completed study: participant identifiers are not "
            f"approved (ADR-023), the splits are session-bounded (ADR-031), and the configuration "
            f"selection is provisional until more participant traces are collected. No semantic "
            f"head was trained, so no semantic-model performance is claimed."
        )
    return (
        "Scripted engineering captures only. This is a label-feasibility and majority-baseline "
        "report, not participant evidence. No semantic head was trained, so no semantic-model "
        "performance is claimed."
    )


def _percentiles(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"median": None, "p25": None, "p75": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
    }


def _time_between_actions(rows: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Median gap between consecutive semantic target acts within a session."""
    by_session: Dict[str, List[float]] = {}
    for row in rows:
        timestamp = row.get("target_event_timestamp")
        if timestamp is None:
            continue
        by_session.setdefault(str(row["session_id"]), []).append(float(timestamp))
    gaps: List[float] = []
    for timestamps in by_session.values():
        timestamps.sort()
        gaps.extend(b - a for a, b in zip(timestamps, timestamps[1:]) if b > a)
    return _percentiles(gaps)


def _majority_class_prf(summary: Dict[str, Any], vocabulary: List[str]) -> Dict[str, Any]:
    """Per-class precision/recall/F1 of the declared majority-class baseline on test."""
    test_counts = summary["class_distribution"]["test"]["counts"]
    total = summary["example_counts"]["test"]
    majority = summary["majority_baseline"]["test"]["majority_class"]
    per_class: Dict[str, Dict[str, float]] = {}
    for name in vocabulary:
        support = int(test_counts.get(name, 0))
        predicted_positive = total if name == majority else 0
        true_positive = support if name == majority else 0
        precision = (true_positive / predicted_positive) if predicted_positive else 0.0
        recall = (true_positive / support) if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }
    return {
        "majority_class": majority,
        "per_class": per_class,
        "macro_f1": float(summary["majority_baseline"]["test"]["macro_f1"]),
    }


def _empty_metrics(vocabulary: List[str]) -> Dict[str, Any]:
    return {
        "total_examples": 0,
        "per_outcome_frequency": {name: 0 for name in vocabulary},
        "time_between_actions_ms": {"median": None, "p25": None, "p75": None},
        "lead_time_ms": {"median": None, "p25": None, "p75": None},
        "lead_time_median_ms": None,
        "per_class_prf": {"majority_class": vocabulary[-1], "per_class": {}, "macro_f1": 0.0},
        "macro_f1": 0.0,
        "train_max_class_share": 1.0,
        "train_classes_with_min_support": 0,
        "false_early_predictions": 0,
        "excluded_incomplete_lookahead": 0,
        "excluded_recorded_only": 0,
        "task_coverage": {"train": {}, "val": {}, "test": {}},
        "tasks_absent": {"train": [], "val": [], "test": []},
    }


def summarize_config(
    rows: List[Dict[str, Any]],
    split_summary: Dict[str, Any],
    build_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute the pre-declared feasibility metrics for one configuration."""
    vocabulary = list(split_summary.get("label_vocabulary", TARGET_VOCABULARY.values()))
    train_counts = split_summary["class_distribution"]["train"]["counts"]
    train_total = split_summary["example_counts"]["train"]
    lead_times = [
        float(row["lead_time_ms"]) for row in rows if row.get("lead_time_ms") is not None
    ]
    lead = _percentiles(lead_times)
    majority = _majority_class_prf(split_summary, vocabulary)

    return {
        "total_examples": len(rows),
        "per_outcome_frequency": {name: int(train_counts.get(name, 0)) for name in vocabulary},
        "time_between_actions_ms": _time_between_actions(rows),
        "lead_time_ms": lead,
        "lead_time_median_ms": lead["median"],
        "per_class_prf": majority,
        "macro_f1": majority["macro_f1"],
        "train_max_class_share": (
            float(max(train_counts.values()) / train_total) if train_total else 1.0
        ),
        "train_classes_with_min_support": sum(
            1 for count in train_counts.values() if count >= MIN_CLASS_SUPPORT
        ),
        "false_early_predictions": sum(1 for value in lead_times if value <= 0.0),
        "excluded_incomplete_lookahead": int(build_stats.get("excluded_incomplete_lookahead", 0)),
        "excluded_recorded_only": int(build_stats.get("excluded_recorded_only", 0)),
        "task_coverage": split_summary.get("task_coverage", {}),
        "tasks_absent": split_summary.get("tasks_absent", {}),
    }


def run_config(
    trace_dir: Union[str, Path],
    config: Dict[str, Any],
    seed: int = 42,
) -> Dict[str, Any]:
    """Build, split and summarise one pre-declared configuration."""
    lookahead = tuple(config["lookahead_ms"])  # type: ignore[assignment]
    rows, build_stats = build_target_dataset_from_traces(
        trace_dir=trace_dir,
        target="semantic",
        seq_len=int(config["history_windows"]),
        lookahead_min_ms=float(lookahead[0]),
        lookahead_max_ms=float(lookahead[1]),
    )
    if not rows:
        return {"config": dict(config), "metrics": _empty_metrics(list(TARGET_VOCABULARY.values()))}

    _, split_summary = split_target_dataset(
        rows,
        random_seed=seed,
        vocabulary=list(TARGET_VOCABULARY.values()),
        label_key="target_task_outcome",
        id_key="target_task_outcome_id",
        label_name="semantic",
        build_stats=build_stats,
    )
    return {"config": dict(config), "metrics": summarize_config(rows, split_summary, build_stats)}


def _selection_key(result: Dict[str, Any]) -> Tuple[Any, ...]:
    metrics = result["metrics"]
    lead = metrics.get("lead_time_median_ms")
    return (
        -int(metrics.get("train_classes_with_min_support", 0)),
        float(metrics.get("train_max_class_share", 1.0)),
        float(lead) if lead is not None else float("inf"),
        int(result["order"]),
    )


SELECTION_CRITERIA = [
    f"maximise training classes with >= {MIN_CLASS_SUPPORT} examples",
    "minimise the largest single-class share in training",
    "minimise median lead time (earlier warning)",
    "declared A, B, C, D order",
]


def select_config(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply the pre-declared selection criteria; deterministic, no single aggregate."""
    if not results:
        raise ValueError("select_config requires at least one result")
    ranked = sorted(results, key=_selection_key)
    chosen = ranked[0]
    rationale = (
        f"Configuration {chosen['config']['id']} was selected by the pre-declared criteria: "
        f"{chosen['metrics']['train_classes_with_min_support']} target classes have at least "
        f"{MIN_CLASS_SUPPORT} training examples (highest), the largest training class holds "
        f"{chosen['metrics']['train_max_class_share'] * 100:.1f}% of examples, and the median "
        f"lead time is {_fmt(chosen['metrics']['lead_time_median_ms'], 1)} ms."
    )
    return {
        "config": chosen["config"],
        "order": chosen["order"],
        "criteria": SELECTION_CRITERIA,
        "ranking": [result["config"]["id"] for result in ranked],
        "rationale": rationale,
    }


def run_experiment(
    trace_dir: Union[str, Path],
    seed: int = 42,
    require_provenance: Optional[str] = None,
) -> Dict[str, Any]:
    """Run every pre-declared configuration and select one."""
    provenance = detect_provenance(trace_dir)
    if require_provenance and provenance != require_provenance:
        raise ValueError(
            f"Experiment requires '{require_provenance}' traces but '{trace_dir}' declares "
            f"'{provenance}'; refusing to label scripted evidence as participant"
        )
    results: List[Dict[str, Any]] = []
    for order, config in enumerate(EXPERIMENTS):
        result = run_config(trace_dir, config, seed=seed)
        result["order"] = order
        results.append(result)
    selection = select_config(results)
    if provenance == "participant":
        selection["provisional"] = True
        selection["rationale"] += (
            " This selection is provisional: it is participant-derived engineering evidence "
            "with a small session count and must be re-confirmed after further collection "
            "before the semantic head training run."
        )
    session_count = len(_session_ids(trace_dir))
    return {
        "experiments": results,
        "selection": selection,
        "provenance": provenance,
        "session_count": session_count,
        "evidence_scope": _evidence_scope_text(provenance, session_count),
    }


def _fmt(value: Optional[float], digits: int = 0) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def render_markdown(report: Dict[str, Any]) -> str:
    """Render the experiment report. Deterministic: no timestamps or machine paths."""
    lines: List[str] = []
    provenance = str(report.get("provenance", "scripted"))
    lines.append("# Window/Horizon Experiment — Semantic Target (Task 40.3)")
    lines.append("")
    lines.append(f"**Evidence provenance:** `{provenance}` "
                 f"({report.get('session_count', 'n/a')} sessions)")
    lines.append("")
    lines.append(f"> **Evidence scope:** {report['evidence_scope']}")
    lines.append("")
    lines.append(f"Target vocabulary: `{TARGET_VOCABULARY_VERSION}` ({len(TARGET_VOCABULARY)} IDs).")
    lines.append("Splits: session-bounded (ADR-031). Class weights: training partition only.")
    lines.append("")
    lines.append("## Pre-declared selection criteria")
    lines.append("")
    for index, criterion in enumerate(SELECTION_CRITERIA, start=1):
        lines.append(f"{index}. {criterion}")
    lines.append("")
    lines.append("## Configurations")
    lines.append("")
    lines.append(
        "| ID | History | Lookahead (ms) | Examples | Train coverage | Max class share | "
        "Lead time median (ms) | Majority macro-F1 | False early | Incomplete | Recorded-only |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for result in report["experiments"]:
        config = result["config"]
        metrics = result["metrics"]
        lines.append(
            f"| {config['id']} | {config['history_windows']}x500 ms | "
            f"{config['lookahead_ms'][0]}–{config['lookahead_ms'][1]} | "
            f"{metrics['total_examples']} | {metrics['train_classes_with_min_support']} | "
            f"{metrics['train_max_class_share'] * 100:.1f}% | "
            f"{_fmt(metrics['lead_time_median_ms'])} | {metrics['macro_f1']:.4f} | "
            f"{metrics['false_early_predictions']} | "
            f"{metrics['excluded_incomplete_lookahead']} | "
            f"{metrics['excluded_recorded_only']} |"
        )
    lines.append("")
    selection = report["selection"]
    lines.append(f"**Selected configuration: {selection['config']['id']}** "
                 f"({selection['config']['label']}).")
    lines.append("")
    lines.append(selection["rationale"])
    if selection.get("provisional"):
        lines.append("")
        lines.append(
            "> **Provisional:** this participant-derived selection must be re-confirmed after "
            "further participant collection before Phase 4B fixes the head configuration."
        )
    lines.append("")
    lines.append("## Per-outcome frequency (training partition, selected configuration)")
    lines.append("")
    selected = next(
        result for result in report["experiments"]
        if result["config"]["id"] == selection["config"]["id"]
    )
    lines.append("| Outcome | Count |")
    lines.append("| --- | --- |")
    for name, count in selected["metrics"]["per_outcome_frequency"].items():
        lines.append(f"| `{name}` | {count} |")
    lines.append("")
    lines.append("## Task-level coverage (selected configuration)")
    lines.append("")
    lines.append("| Partition | Tasks present (examples) | Tasks absent |")
    lines.append("| --- | --- | --- |")
    for partition in ("train", "val", "test"):
        coverage = selected["metrics"]["task_coverage"].get(partition, {})
        present = ", ".join(f"{task}={count}" for task, count in coverage.items()) or "none"
        absent = ", ".join(selected["metrics"]["tasks_absent"].get(partition, [])) or "none"
        lines.append(f"| {partition} | {present} | {absent} |")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.extend(_caveats(provenance))
    lines.append("")
    return "\n".join(lines)


def _caveats(provenance: str) -> List[str]:
    if provenance == "participant":
        return [
            "- These are participant captures, not scripted engineering runs. They were",
            "  retrofitted locally from an exported set that had been mislabelled; the",
            "  retrofit manifest records the authority and the original declarations.",
            "- Participant identifiers are not approved (ADR-023) and splits are",
            "  session-bounded (ADR-031), so no participant-held-out claim is made.",
            "- The example counts are small, so per-class support is sparse and the margin",
            "  between the top configurations may be within sampling noise. The selection",
            "  is provisional and must be re-confirmed after further collection.",
        ]
    return [
        "- The captures are scripted engineering runs; they are not participant evidence.",
        "- The example counts are small, so per-class support is sparse and the experiment",
        "  cannot select a production configuration on its own. The selected configuration",
        "  must be re-confirmed on participant data before the semantic head is trained.",
        "- This scripted report is pipeline evidence only; its configuration selection is",
        "  superseded by the participant re-run",
        "  (docs/experiments/window-horizon-participant/window-horizon-report.md).",
        "- Macro-F1 is the declared majority-class baseline, not a trained model's score.",
    ]


def write_report(report: Dict[str, Any], output_dir: Union[str, Path]) -> Dict[str, Path]:
    """Write the JSON result and the Markdown report."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "window-horizon-results.json"
    md_path = out_dir / "window-horizon-report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def write_selected_dataset(
    trace_dir: Union[str, Path],
    output_dir: Union[str, Path],
    version: str = "v2.0.0-semantic",
    seed: int = 42,
    require_provenance: Optional[str] = None,
) -> Dict[str, Path]:
    """Build the selected dataset version and its manifest, recording the configuration."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = run_experiment(trace_dir, seed=seed, require_provenance=require_provenance)
    selected = report["selection"]
    config = selected["config"]

    rows, build_stats = build_target_dataset_from_traces(
        trace_dir=trace_dir,
        target="semantic",
        seq_len=int(config["history_windows"]),
        lookahead_min_ms=float(config["lookahead_ms"][0]),
        lookahead_max_ms=float(config["lookahead_ms"][1]),
    )
    _, split_summary = split_target_dataset(
        rows,
        random_seed=seed,
        vocabulary=list(TARGET_VOCABULARY.values()),
        label_key="target_task_outcome",
        id_key="target_task_outcome_id",
        label_name="semantic",
        build_stats=build_stats,
    )

    version_dir = out_dir / version
    parquet_path = version_dir / "target_semantic_dataset.parquet"
    save_target_dataset_parquet(
        rows, parquet_path, seq_len=int(config["history_windows"]), split_summary=split_summary
    )
    manifest_path = version_dir / "manifest.json"
    create_dataset_manifest(
        rows=rows,
        split_summary=split_summary,
        trace_dir=trace_dir,
        output_path=manifest_path,
        dataset_version=version,
    )
    selection_path = out_dir / "window_horizon_selection.json"
    selection_path.write_text(
        json.dumps({
            "dataset_version": version,
            "provenance": report.get("provenance"),
            "provisional": bool(selected.get("provisional")),
            "selection": selected,
            "history_windows": config["history_windows"],
            "lookahead_ms": list(config["lookahead_ms"]),
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "parquet": parquet_path,
        "manifest": manifest_path,
        "selection": selection_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Window/horizon experiment (Task 40.3)")
    parser.add_argument("--trace-dir", type=str, default=".data/raw/scripted")
    parser.add_argument(
        "--report-dir", type=str, default="docs/experiments/window-horizon"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-dataset", action="store_true",
                        help="Write the selected dataset version and manifest")
    parser.add_argument(
        "--require-provenance", default=None, choices=["scripted", "participant"],
        help="Refuse the run unless the trace directory declares this provenance",
    )
    parser.add_argument("--dataset-dir", type=str, default=".data/processed")
    parser.add_argument("--version", type=str, default="v2.0.0-semantic")
    args = parser.parse_args()

    report = run_experiment(
        args.trace_dir, seed=args.seed, require_provenance=args.require_provenance
    )
    paths = write_report(report, args.report_dir)
    print(f"[WindowHorizon] Report: {paths['markdown']}")
    print(f"[WindowHorizon] Results: {paths['json']}")
    print(f"[WindowHorizon] Provenance: {report['provenance']} ({report['session_count']} sessions)")
    print(f"[WindowHorizon] Selected: {report['selection']['config']['id']} "
          f"({report['selection']['config']['label']})"
          f"{' [provisional]' if report['selection'].get('provisional') else ''}")

    if args.write_dataset:
        written = write_selected_dataset(
            trace_dir=args.trace_dir,
            output_dir=args.dataset_dir,
            version=args.version,
            seed=args.seed,
            require_provenance=args.require_provenance,
        )
        print(f"[WindowHorizon] Dataset: {written['parquet']}")
        print(f"[WindowHorizon] Manifest: {written['manifest']}")


if __name__ == "__main__":
    main()
