"""
trace_ingestion.py
Ingestion layer for exported browser ExperimentTrace (schemaVersion 1.4.0, plus
supported legacy 1.3.0 / 1.2.0 / 1.1.0 captures).
Converts exported runtime traces into the canonical telemetry dataset format (Parquet).

Implements Task 9.1 specifications:
- Strict, version-keyed validation of the ExperimentTrace contract against
  TRACE_KEY_ALLOWLIST / BEHAVIOUR_EVENT_KEY_ALLOWLIST. A trace declaring any other
  schemaVersion is rejected, never silently skipped.
- Explicit detection and rejection of unmapped trace and event fields.
- Non-positive / missing viewport dimension validation (fail visibly).
- Strict modality masking (no zero-imputation of missing modalities).
- Full provenance carrying: sessionId, experimentId, conditionId, taskId,
  windowId, sourceEventIds, preprocessingVersion, featureSchemaVersion, and
  provenance (scripted vs participant).
- Deterministic, byte-identical Parquet export across multiple executions.

Configuration caveat: this module never reads src/config.yaml. `load_config` is
imported by sibling modules but is not called on the trace-ingestion path, so the
effective values here are the hard-coded module defaults below (including the bare
`ts_ms // 250` window-correlation fallback). Do not assume config.yaml is honoured.
"""

import os
import sys
import glob
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

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
    from data import CANONICAL_EVENT_SCHEMA, find_project_root
except ImportError:
    try:
        from src.data import CANONICAL_EVENT_SCHEMA, find_project_root
    except ImportError:
        CANONICAL_EVENT_SCHEMA = None
        find_project_root = lambda: Path(os.getcwd()).resolve()

EXPERIMENT_TRACE_SCHEMA_VERSION = "1.4.0"
SUPPORTED_EXPERIMENT_TRACE_SCHEMA_VERSIONS: List[str] = ["1.4.0", "1.3.0", "1.2.0", "1.1.0"]

# Layer-1 *preprocessing* versions, deliberately NOT tied to the trace schema version.
# Adopting the 1.3.0 trace contract changed which records are accepted and added a
# provenance annotation; it did not change any windowing transform or feature value,
# so these version strings must not move with EXPERIMENT_TRACE_SCHEMA_VERSION.
PREPROCESSING_VERSION = "1.1.0"
FEATURE_SCHEMA_VERSION = "1.1.0"

# Top-level keys permitted by each supported ExperimentTrace version. This is the exact
# per-version contract: a top-level key absent from the relevant set is rejected rather
# than silently ignored, so a field added by the runtime cannot be dropped unnoticed.
_COMMON_TRACE_KEYS = {
    "schemaVersion",
    "exportedAt",
    "session",
    "task",
    "metadata",
    "effectiveConfig",
    "behaviourEvents",
    "microTensors",
    "macroInteractions",
    "outcomes",
    "predictions",
    "interventions",
    "taskEvents"
}

TRACE_KEY_ALLOWLIST: Dict[str, set] = {
    "1.1.0": set(_COMMON_TRACE_KEYS),
    "1.2.0": set(_COMMON_TRACE_KEYS),
    # 1.3.0 adds one policy verdict per evaluation, including rejections (F-07).
    "1.3.0": set(_COMMON_TRACE_KEYS) | {"policyDecisions"},
    # 1.4.0 is additive at the record level: semantic outcomes and the task-definition
    # version live inside `outcomes`, `taskEvents` and `metadata`. No new top-level key.
    "1.4.0": set(_COMMON_TRACE_KEYS) | {"policyDecisions"},
}

# Behaviour-event keys permitted by each supported ExperimentTrace version.
_COMMON_BEHAVIOUR_EVENT_KEYS = {
    "timestamp",
    "type",
    "x",
    "y",
    "scrollX",
    "scrollY",
    "scrollTopPx",
    "componentId",
    "componentRole",
    "route",
    "action",
    "taskId",
    "taskStepId",
    "targetTag",
    "viewport",
    "document"
}

BEHAVIOUR_EVENT_KEY_ALLOWLIST: Dict[str, set] = {
    "1.1.0": set(_COMMON_BEHAVIOUR_EVENT_KEYS),
    "1.2.0": set(_COMMON_BEHAVIOUR_EVENT_KEYS),
    # 1.3.0 records the observed post-action disclosure state (F-06).
    "1.3.0": set(_COMMON_BEHAVIOUR_EVENT_KEYS) | {"ariaExpanded"},
    # 1.4.0 adds no new behaviour-event field.
    "1.4.0": set(_COMMON_BEHAVIOUR_EVENT_KEYS) | {"ariaExpanded"},
}

# Outcome-record keys permitted by each supported trace version. The semantic outcome and its
# vocabulary version are additive in 1.4.0; legacy records keep the generic label only.
_COMMON_OUTCOME_RECORD_KEYS = {
    "timestamp",
    "outcome",
    "lookaheadComplete",
    "sessionId",
    "windowId",
    "experimentId",
    "conditionId",
    "sourceEventTimestamp",
    "sourceEventType",
    "componentId",
    "taskId",
    "taskStepId",
    "route",
    "derivation"
}

OUTCOME_RECORD_KEY_ALLOWLIST: Dict[str, set] = {
    "1.1.0": set(_COMMON_OUTCOME_RECORD_KEYS),
    "1.2.0": set(_COMMON_OUTCOME_RECORD_KEYS),
    "1.3.0": set(_COMMON_OUTCOME_RECORD_KEYS),
    "1.4.0": set(_COMMON_OUTCOME_RECORD_KEYS) | {
        "taskOutcome",
        "taskOutcomeVocabularyVersion",
        "taskOutcomeSource",
        "taskOutcomeRuleId",
    },
}

# Task-event keys permitted by each supported trace version. 1.4.0 adds the task-definition
# version and the failure/abandonment reason (timeout, wrong_action, ...).
_COMMON_TASK_EVENT_KEYS = {
    "timestamp",
    "type",
    "taskId",
    "status",
    "errors",
    "sessionId",
    "experimentId",
    "conditionId",
    "taskStepId",
    "durationMs"
}

TASK_EVENT_KEY_ALLOWLIST: Dict[str, set] = {
    "1.1.0": set(_COMMON_TASK_EVENT_KEYS),
    "1.2.0": set(_COMMON_TASK_EVENT_KEYS),
    "1.3.0": set(_COMMON_TASK_EVENT_KEYS),
    "1.4.0": set(_COMMON_TASK_EVENT_KEYS) | {"taskDefinitionVersion", "reason"},
}

# Failure/abandonment reasons accepted from a 1.4.0 task event (events.ts TaskEvent.reason).
TASK_EVENT_REASONS = {
    "reset",
    "replaced",
    "condition_change",
    "interrupted",
    "timeout",
    "wrong_action",
    "state_mismatch",
    "step_exhausted"
}


def _task_state_at(
    timeline: List[Tuple[float, str, str, Optional[str]]],
    task_id: str,
    timestamp: float
) -> Tuple[Optional[str], Optional[str]]:
    """Causal task-status snapshot: the last lifecycle event at/before ``timestamp``.

    ``reason`` is retained for a failure and cleared when the same task returns to
    ``In Progress``, so an abandoned trial cannot inherit a previous failure's reason.
    """
    status: Optional[str] = None
    reason: Optional[str] = None
    for event_ts, event_task, event_status, event_reason in timeline:
        if event_ts > timestamp:
            break
        if event_task and task_id and event_task != task_id:
            continue
        if event_status:
            status = event_status
        if event_reason:
            reason = event_reason
        elif status == "In Progress":
            reason = None
    return status, reason


def _load_trace(trace_data_or_path: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(trace_data_or_path, (str, Path)):
        trace_path = Path(trace_data_or_path).resolve()
        if not trace_path.is_file():
            raise FileNotFoundError(f"Trace file not found: {trace_path}")
        with open(trace_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    if isinstance(trace_data_or_path, dict):
        return trace_data_or_path
    raise TypeError(f"Expected path or dict, received: {type(trace_data_or_path)}")


def _validate_envelope(trace: Dict[str, Any]) -> Tuple[str, Dict[str, set]]:
    """Validate the declared schema version and every top-level key against it."""
    if not isinstance(trace, dict):
        raise ValueError("Trace payload must be a JSON object")

    schema_version = trace.get("schemaVersion")
    if schema_version not in TRACE_KEY_ALLOWLIST:
        raise ValueError(
            f"Invalid schemaVersion: '{schema_version}' is not supported; expected one of "
            f"{SUPPORTED_EXPERIMENT_TRACE_SCHEMA_VERSIONS}"
        )

    allowed = TRACE_KEY_ALLOWLIST[schema_version]
    for key in trace.keys():
        if key not in allowed:
            raise ValueError(f"Unmapped trace field: '{key}'")
    if not isinstance(trace.get("session"), dict):
        raise ValueError("Trace is missing required 'session' dictionary")
    allowed_keys = {
        "trace": allowed,
        "event": BEHAVIOUR_EVENT_KEY_ALLOWLIST[schema_version],
        "outcome": OUTCOME_RECORD_KEY_ALLOWLIST[schema_version],
        "task": TASK_EVENT_KEY_ALLOWLIST[schema_version],
    }
    return schema_version, allowed_keys


def _session_identity(session_obj: Dict[str, Any]) -> Dict[str, Any]:
    session_id = str(session_obj.get("sessionId", "")).strip()
    if not session_id:
        raise ValueError("Trace session is missing non-empty 'sessionId'")
    provenance = str(session_obj.get("provenance") or "scripted").strip() or "scripted"
    if provenance not in ("scripted", "participant"):
        raise ValueError(
            f"Invalid session.provenance: '{provenance}' (expected 'scripted' or 'participant')"
        )
    return {
        "session_id": session_id,
        "user_id": str(session_obj.get("userId", session_id)),
        "provenance": provenance,
        "experiment_id": str(session_obj.get("experimentId") or ""),
        "condition_id": str(session_obj.get("conditionId") or ""),
    }


def _trace_identity(trace: Dict[str, Any], session_obj: Dict[str, Any]) -> Dict[str, Any]:
    identity = _session_identity(session_obj)
    metadata = trace.get("metadata", {}) if isinstance(trace.get("metadata"), dict) else {}
    task_obj = trace.get("task", {}) if isinstance(trace.get("task"), dict) else {}
    identity.update({
        "metadata": metadata,
        "task": task_obj,
        "experiment_id": identity["experiment_id"] or str(metadata.get("experimentId") or ""),
        "condition_id": identity["condition_id"] or str(metadata.get("conditionId") or ""),
        "default_task_id": str(task_obj.get("currentTaskId") or ""),
        "semantic_vocabulary_version": metadata.get("semanticOutcomeVocabularyVersion"),
        "task_definition_version": (
            metadata.get("taskDefinitionVersion") or task_obj.get("definitionVersion")
        ),
    })
    return identity


def _window_spans(micro_windows: List[Dict[str, Any]]) -> List[Tuple[int, float, float]]:
    spans: List[Tuple[int, float, float]] = []
    for window in micro_windows:
        if isinstance(window, dict) and {"windowId", "windowStart", "windowEnd"} <= set(window):
            try:
                spans.append((
                    int(window["windowId"]),
                    float(window["windowStart"]),
                    float(window["windowEnd"]),
                ))
            except (ValueError, TypeError):
                continue
    spans.sort(key=lambda span: span[1])
    return spans


def _outcomes_by_window(records: Any, allowed_keys: set) -> Dict[int, Dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError("Trace 'outcomes' must be a list")
    by_window: Dict[int, Dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"outcomes[{index}] must be a dictionary")
        for key in record.keys():
            if key not in allowed_keys:
                raise ValueError(f"Unmapped outcome record field: '{key}'")
        window_id = record.get("windowId")
        if isinstance(window_id, (int, float)):
            by_window[int(window_id)] = record
    return by_window


def _validate_task_event(record: Any, index: int, allowed_keys: set) -> Optional[str]:
    if not isinstance(record, dict):
        raise ValueError(f"taskEvents[{index}] must be a dictionary")
    for key in record.keys():
        if key not in allowed_keys:
            raise ValueError(f"Unmapped task event field: '{key}'")
    if not isinstance(record.get("timestamp"), (int, float)):
        raise ValueError(f"taskEvents[{index}] is missing numeric 'timestamp'")
    reason = record.get("reason")
    if reason is None:
        return None
    if str(reason) not in TASK_EVENT_REASONS:
        raise ValueError(
            f"taskEvents[{index}] carries unknown failure reason '{reason}'; expected one of "
            f"{sorted(TASK_EVENT_REASONS)}"
        )
    return str(reason)


def _task_timeline(records: Any, allowed_keys: set) -> List[Tuple[float, str, str, Optional[str]]]:
    if not isinstance(records, list):
        raise ValueError("Trace 'taskEvents' must be a list")
    timeline: List[Tuple[float, str, str, Optional[str]]] = []
    for index, record in enumerate(records):
        reason = _validate_task_event(record, index, allowed_keys)
        timeline.append((
            float(record["timestamp"]),
            str(record.get("taskId") or ""),
            str(record.get("status") or ""),
            reason,
        ))
    timeline.sort(key=lambda entry: entry[0])
    return timeline


def _session_geometry(
    behaviour_events: List[Dict[str, Any]]
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """First valid viewport/document pair in the session."""
    for event in behaviour_events:
        if not isinstance(event, dict) or not isinstance(event.get("viewport"), dict):
            continue
        width = float(event["viewport"].get("width", 0))
        height = float(event["viewport"].get("height", 0))
        if width <= 0 or height <= 0:
            continue
        viewport = (width, height)
        document = None
        doc_obj = event.get("document")
        if isinstance(doc_obj, dict):
            doc_width = float(doc_obj.get("width", width))
            doc_height = float(doc_obj.get("height", height))
            if doc_width > 0 and doc_height > 0:
                document = (doc_width, doc_height)
        return viewport, document
    return None, None


def _validate_event(event: Any, index: int, allowed_keys: set) -> None:
    if not isinstance(event, dict):
        raise ValueError(f"behaviourEvents[{index}] must be a dictionary")
    for key in event.keys():
        if key not in allowed_keys:
            raise ValueError(f"Unmapped trace event field: '{key}'")
    if "timestamp" not in event or not isinstance(event["timestamp"], (int, float)):
        raise ValueError(f"behaviourEvents[{index}] is missing numeric 'timestamp'")
    if "type" not in event or not isinstance(event["type"], str):
        raise ValueError(f"behaviourEvents[{index}] is missing string 'type'")


def _viewport_for(
    event: Dict[str, Any], index: int, current: Optional[Tuple[float, float]]
) -> Tuple[float, float]:
    if "viewport" in event:
        viewport = event["viewport"]
        if not isinstance(viewport, dict):
            raise ValueError(f"behaviourEvents[{index}] 'viewport' must be a dictionary")
        width = float(viewport.get("width", 0))
        height = float(viewport.get("height", 0))
        if width <= 0.0 or height <= 0.0:
            raise ValueError(
                f"Non-positive or invalid viewport dimensions ({width}, {height}) "
                f"in behaviourEvents[{index}]"
            )
        return width, height
    if current is None:
        raise ValueError(f"behaviourEvents[{index}] is missing required 'viewport' dictionary")
    return current


def _document_for(
    event: Dict[str, Any],
    index: int,
    current: Optional[Tuple[float, float]],
    viewport: Tuple[float, float],
) -> Tuple[float, float]:
    if "document" in event:
        document = event["document"]
        if not isinstance(document, dict):
            raise ValueError(f"behaviourEvents[{index}] 'document' must be a dictionary")
        width = float(document.get("width", viewport[0]))
        height = float(document.get("height", viewport[1]))
        if width <= 0.0 or height <= 0.0:
            raise ValueError(
                f"Non-positive or invalid document dimensions ({width}, {height}) "
                f"in behaviourEvents[{index}]"
            )
        return width, height
    return current if current is not None else viewport


def _normalise_point(
    event: Dict[str, Any], viewport: Tuple[float, float]
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Returns (x_norm, y_norm, x_raw, y_raw)."""
    x_norm = y_norm = x_raw = y_raw = None
    if event.get("x") is not None:
        try:
            x_norm = float(np.clip(float(event["x"]), 0.0, 1.0))
            x_raw = float(x_norm * viewport[0])
        except (ValueError, TypeError):
            x_norm = x_raw = None
    if event.get("y") is not None:
        try:
            y_norm = float(np.clip(float(event["y"]), 0.0, 1.0))
            y_raw = float(y_norm * viewport[1])
        except (ValueError, TypeError):
            y_norm = y_raw = None
    return x_norm, y_norm, x_raw, y_raw


def _window_id_for(spans: List[Tuple[int, float, float]], timestamp: float) -> int:
    for window_id, start, end in spans:
        if start <= timestamp <= end:
            return window_id
    prior = [(window_id, end) for window_id, start, end in spans if end <= timestamp]
    return max(prior, key=lambda item: item[1])[0] if prior else -1


def _semantic_fields(
    outcome_record: Optional[Dict[str, Any]], trace_vocabulary_version: Optional[str]
) -> Dict[str, Any]:
    record = outcome_record or {}
    return {
        "task_outcome": record.get("taskOutcome"),
        "task_outcome_vocabulary_version": (
            record.get("taskOutcomeVocabularyVersion")
            or trace_vocabulary_version
        ),
        "task_outcome_source": record.get("taskOutcomeSource"),
        "task_outcome_rule_id": record.get("taskOutcomeRuleId"),
    }


def _canonical_row(
    dataset_id: str,
    identity: Dict[str, Any],
    event_values: Dict[str, Any],
) -> Dict[str, Any]:
    """One canonical row. Branch-free by design: validation happened before the call."""
    row = {
        "dataset_id": dataset_id,
        "session_id": identity["session_id"],
        "user_id": identity["user_id"],
        "trial_id": event_values["task_id"],
        "timestamp_ms": event_values["timestamp_ms"],
        "event_type": event_values["event_type"],
        "x": event_values["x"],
        "y": event_values["y"],
        "x_norm": event_values["x_norm"],
        "y_norm": event_values["y_norm"],
        "viewport_width": event_values["viewport_width"],
        "viewport_height": event_values["viewport_height"],
        "document_width": event_values["document_width"],
        "document_height": event_values["document_height"],
        "target_id": event_values["target_id"],
        "source_event_id": event_values["source_event_id"],
        "duration_ms": None,
        "angle_deg": None,
        "distance_px": None,
        "velocity_px_s": None,
        # Provenance columns (snake_case)
        "experiment_id": identity["experiment_id"],
        "condition_id": identity["condition_id"],
        "task_id": event_values["task_id"],
        "window_id": event_values["window_id"],
        "source_event_ids": event_values["source_event_id"],
        "provenance": identity["provenance"],
        "preprocessing_version": PREPROCESSING_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "task_definition_version": identity["task_definition_version"],
        "semantic_outcome_vocabulary_version": identity["semantic_vocabulary_version"],
        "task_status": event_values["task_status"],
        "task_failure_reason": event_values["task_failure_reason"],
        # CamelCase aliases for exact acceptance criteria verification
        "sessionId": identity["session_id"],
        "experimentId": identity["experiment_id"],
        "conditionId": identity["condition_id"],
        "taskId": event_values["task_id"],
        "windowId": event_values["window_id"],
        "sourceEventIds": event_values["source_event_id"],
        "preprocessingVersion": PREPROCESSING_VERSION,
        "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
    }
    row.update(event_values["semantic"])
    return row


def parse_experiment_trace(
    trace_data_or_path: Union[str, Path, Dict[str, Any]],
    dataset_id: str = "target_testbed"
) -> List[Dict[str, Any]]:
    """
    Parse a single SerializableExperimentTrace object or JSON file.
    Validates schema conformance, rejects unknown fields, validates viewport dimensions,
    and returns canonical event dictionaries with complete provenance metadata.
    """
    trace = _load_trace(trace_data_or_path)
    schema_version, allowed = _validate_envelope(trace)
    session_obj = trace["session"]
    identity = _trace_identity(trace, session_obj)

    micro_windows = trace.get("microTensors", [])
    if not isinstance(micro_windows, list):
        micro_windows = []
    window_spans = _window_spans(micro_windows)

    behaviour_events = trace.get("behaviourEvents", [])
    if not isinstance(behaviour_events, list):
        raise ValueError("Trace is missing required 'behaviourEvents' list")

    outcomes_by_window = _outcomes_by_window(trace.get("outcomes", []), allowed["outcome"])
    task_timeline = _task_timeline(trace.get("taskEvents", []), allowed["task"])

    session_vp, session_doc = _session_geometry(behaviour_events)
    current_vp, current_doc = session_vp, session_doc or session_vp

    rows: List[Dict[str, Any]] = []
    for index, event in enumerate(behaviour_events):
        _validate_event(event, index, allowed["event"])
        timestamp = float(event["timestamp"])
        timestamp_ms = int(round(timestamp))
        event_type = str(event["type"])

        current_vp = _viewport_for(event, index, current_vp)
        current_doc = _document_for(event, index, current_doc, current_vp)
        x_norm, y_norm, x_raw, y_raw = _normalise_point(event, current_vp)

        task_id = str(event.get("taskId") or identity["default_task_id"])
        window_id = _window_id_for(window_spans, timestamp)
        task_status, task_failure_reason = _task_state_at(task_timeline, task_id, timestamp)
        semantic = _semantic_fields(
            outcomes_by_window.get(window_id), identity["semantic_vocabulary_version"]
        )

        rows.append(_canonical_row(dataset_id, identity, {
            "timestamp_ms": timestamp_ms,
            "event_type": event_type,
            "x": x_raw,
            "y": y_raw,
            "x_norm": x_norm,
            "y_norm": y_norm,
            "viewport_width": current_vp[0],
            "viewport_height": current_vp[1],
            "document_width": current_doc[0],
            "document_height": current_doc[1],
            "target_id": str(event.get("componentId") or event.get("targetTag") or ""),
            "task_id": task_id,
            "source_event_id": f"{identity['session_id']}:ev_{index}",
            "window_id": window_id,
            "task_status": task_status,
            "task_failure_reason": task_failure_reason,
            "semantic": semantic,
        }))

    rows.sort(key=lambda row: (row["timestamp_ms"], row["source_event_id"]))
    return rows


class UnsupportedTraceSchemaError(ValueError):
    """
    Raised when one or more trace files declare a schemaVersion outside
    SUPPORTED_EXPERIMENT_TRACE_SCHEMA_VERSIONS.

    Carries a machine-readable ``skipped`` report: a list of
    ``{"path": str, "schema_version": Any, "reason": str}`` entries, one per file.
    """

    def __init__(self, skipped: List[Dict[str, Any]]):
        self.skipped = skipped
        details = "; ".join(
            f"{Path(entry['path']).name} (schemaVersion={entry['schema_version']!r})"
            for entry in skipped
        )
        super().__init__(
            f"{len(skipped)} trace file(s) declare an unsupported schemaVersion and were "
            f"not ingested: {details}. Supported versions: "
            f"{SUPPORTED_EXPERIMENT_TRACE_SCHEMA_VERSIONS}. Re-run with strict=False only if "
            f"excluding these files is intended."
        )


def _report_skipped_traces(skipped: List[Dict[str, Any]]) -> None:
    """Print a loud, per-file report of traces excluded for an unsupported schemaVersion."""
    print(
        f"[Trace Ingestion] WARNING: {len(skipped)} trace file(s) declared a schemaVersion "
        f"outside {SUPPORTED_EXPERIMENT_TRACE_SCHEMA_VERSIONS} and were NOT ingested:",
        file=sys.stderr,
    )
    for entry in skipped:
        print(
            f"[Trace Ingestion]   - {entry['path']} "
            f"(schemaVersion={entry['schema_version']!r}): {entry['reason']}",
            file=sys.stderr,
        )


def _require_supported_rows(
    rows: List[Dict[str, Any]],
    skipped: List[Dict[str, Any]],
    strict: bool,
    directory: Path,
) -> None:
    """Report skipped versions, fail loudly in strict mode, and refuse an empty result."""
    if skipped:
        _report_skipped_traces(skipped)
        if strict:
            raise UnsupportedTraceSchemaError(skipped)
    if not rows:
        raise ValueError(
            f"No traces with a supported schemaVersion "
            f"{SUPPORTED_EXPERIMENT_TRACE_SCHEMA_VERSIONS} found in: {directory}"
        )


def _canonical_output_path(output_path: Optional[Union[str, Path]]) -> Path:
    if output_path is not None:
        return Path(output_path).resolve()
    return find_project_root() / ".data" / "interim" / "traces" / "canonical_traces.parquet"


def _collect_trace_rows(
    json_files: List[str], dataset_id: str
) -> Tuple[List[Dict[str, Any]], int, List[Dict[str, Any]]]:
    """Parse every trace file; collect supported rows and unsupported-version reports."""
    rows: List[Dict[str, Any]] = []
    ingested = 0
    skipped: List[Dict[str, Any]] = []
    for path in json_files:
        with open(path, "r", encoding="utf-8") as handle:
            trace = json.load(handle)
        try:
            rows.extend(parse_experiment_trace(trace, dataset_id=dataset_id))
        except ValueError as error:
            declared = trace.get("schemaVersion") if isinstance(trace, dict) else None
            if declared not in TRACE_KEY_ALLOWLIST:
                skipped.append({
                    "path": str(path),
                    "schema_version": declared,
                    "reason": str(error),
                })
                continue
            raise
        ingested += 1
    return rows, ingested, skipped


def _write_canonical_parquet(rows: List[Dict[str, Any]], out_file: Path) -> None:
    try:
        from data import CANONICAL_EVENT_SCHEMA
    except ImportError:
        from src.data import CANONICAL_EVENT_SCHEMA
    if CANONICAL_EVENT_SCHEMA is None:
        raise RuntimeError("CANONICAL_EVENT_SCHEMA is not defined.")
    table = pa.Table.from_pylist(rows, schema=CANONICAL_EVENT_SCHEMA)
    pq.write_table(table, str(out_file), compression="snappy")


def ingest_trace_directory(
    trace_dir: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    dataset_id: str = "target_testbed",
    force: bool = False,
    strict: bool = True
) -> str:
    """
    Ingests all JSON experiment traces from trace_dir into a single canonical Parquet file.
    Produces deterministic, byte-identical output across consecutive runs over identical inputs.

    Any file declaring a schemaVersion outside SUPPORTED_EXPERIMENT_TRACE_SCHEMA_VERSIONS is
    never silently dropped: by default (``strict=True``) the run raises
    UnsupportedTraceSchemaError after reporting every offending file. Pass ``strict=False`` to
    ingest the supported subset explicitly, in which case the skipped files are still reported
    on stderr and listed on the exception should no supported trace remain.
    """
    if not PYARROW_AVAILABLE:
        raise RuntimeError("PyArrow is required for canonical Parquet export.")

    in_dir = Path(trace_dir).resolve()
    if not in_dir.is_dir():
        raise FileNotFoundError(f"Trace directory not found: {in_dir}")

    out_file = _canonical_output_path(output_path)
    if not force and out_file.is_file() and out_file.stat().st_size > 0:
        print(f"[Trace Ingestion] Using cached canonical trace Parquet: {out_file}")
        return str(out_file)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    json_files = sorted(glob.glob(str(in_dir / "*.json")))
    if not json_files:
        raise FileNotFoundError(f"No JSON trace files found in: {in_dir}")

    all_rows, ingested_count, skipped = _collect_trace_rows(json_files, dataset_id)
    _require_supported_rows(all_rows, skipped, strict, in_dir)

    # Global deterministic ordering across all sessions: (session_id, timestamp_ms, source_event_id)
    all_rows.sort(key=lambda row: (row["session_id"], row["timestamp_ms"], row["source_event_id"]))
    _write_canonical_parquet(all_rows, out_file)

    print(
        f"[Trace Ingestion] Successfully ingested {ingested_count} trace file(s) "
        f"({len(skipped)} skipped for unsupported schemaVersion), "
        f"wrote {len(all_rows)} canonical events to {out_file} "
        f"({out_file.stat().st_size / 1024 / 1024:.2f} MB)"
    )
    return str(out_file)
