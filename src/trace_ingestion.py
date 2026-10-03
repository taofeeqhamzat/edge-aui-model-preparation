"""
trace_ingestion.py
Ingestion layer for exported browser ExperimentTrace (schemaVersion 1.3.0, plus
supported legacy 1.2.0 / 1.1.0 captures).
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
from typing import List, Dict, Any, Optional, Union

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

EXPERIMENT_TRACE_SCHEMA_VERSION = "1.3.0"
SUPPORTED_EXPERIMENT_TRACE_SCHEMA_VERSIONS: List[str] = ["1.3.0", "1.2.0", "1.1.0"]

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
}


def parse_experiment_trace(
    trace_data_or_path: Union[str, Path, Dict[str, Any]],
    dataset_id: str = "target_testbed"
) -> List[Dict[str, Any]]:
    """
    Parse a single SerializableExperimentTrace object or JSON file.
    Validates schema conformance, rejects unknown fields, validates viewport dimensions,
    and returns canonical event dictionaries with complete provenance metadata.
    """
    if isinstance(trace_data_or_path, (str, Path)):
        trace_path = Path(trace_data_or_path).resolve()
        if not trace_path.is_file():
            raise FileNotFoundError(f"Trace file not found: {trace_path}")
        with open(trace_path, "r", encoding="utf-8") as f:
            trace = json.load(f)
    elif isinstance(trace_data_or_path, dict):
        trace = trace_data_or_path
    else:
        raise TypeError(f"Expected path or dict, received: {type(trace_data_or_path)}")

    # 1. Validate the declared schema version, then the top-level keys permitted by it.
    if not isinstance(trace, dict):
        raise ValueError("Trace payload must be a JSON object")

    schema_version = trace.get("schemaVersion")
    if schema_version not in TRACE_KEY_ALLOWLIST:
        raise ValueError(
            f"Invalid schemaVersion: '{schema_version}' is not supported; expected one of "
            f"{SUPPORTED_EXPERIMENT_TRACE_SCHEMA_VERSIONS}"
        )

    allowed_trace_keys = TRACE_KEY_ALLOWLIST[schema_version]
    allowed_event_keys = BEHAVIOUR_EVENT_KEY_ALLOWLIST[schema_version]
    for key in trace.keys():
        if key not in allowed_trace_keys:
            raise ValueError(f"Unmapped trace field: '{key}'")

    session_obj = trace.get("session")
    if not isinstance(session_obj, dict):
        raise ValueError("Trace is missing required 'session' dictionary")

    session_id = str(session_obj.get("sessionId", "")).strip()
    if not session_id:
        raise ValueError("Trace session is missing non-empty 'sessionId'")

    user_id = str(session_obj.get("userId", session_id))

    # Provenance records scripted/synthetic vs participant-derived origin (ADR-018).
    # Legacy 1.1.0/1.2.0 traces predate the field and are scripted substitute captures,
    # so they default to "scripted" and can never be pooled with participant data.
    provenance = str(session_obj.get("provenance") or "scripted").strip() or "scripted"
    if provenance not in ("scripted", "participant"):
        raise ValueError(
            f"Invalid session.provenance: '{provenance}' (expected 'scripted' or 'participant')"
        )

    metadata_obj = trace.get("metadata", {}) if isinstance(trace.get("metadata"), dict) else {}
    task_obj = trace.get("task", {}) if isinstance(trace.get("task"), dict) else {}

    experiment_id = str(
        session_obj.get("experimentId") or metadata_obj.get("experimentId") or ""
    )
    condition_id = str(
        session_obj.get("conditionId") or metadata_obj.get("conditionId") or ""
    )
    default_task_id = str(
        task_obj.get("currentTaskId") or ""
    )

    micro_windows = trace.get("microTensors", [])
    if not isinstance(micro_windows, list):
        micro_windows = []

    # Map windows for fast windowId lookup: list of (windowId, windowStart, windowEnd)
    window_spans = []
    for mw in micro_windows:
        if isinstance(mw, dict) and "windowId" in mw and "windowStart" in mw and "windowEnd" in mw:
            try:
                window_spans.append((int(mw["windowId"]), float(mw["windowStart"]), float(mw["windowEnd"])))
            except (ValueError, TypeError):
                continue
    window_spans.sort(key=lambda x: x[1])

    behaviour_events = trace.get("behaviourEvents", [])
    if not isinstance(behaviour_events, list):
        raise ValueError("Trace is missing required 'behaviourEvents' list")

    # Find first valid viewport in session
    session_vp: Optional[Tuple[float, float]] = None
    session_doc: Optional[Tuple[float, float]] = None
    for ev in behaviour_events:
        if isinstance(ev, dict) and isinstance(ev.get("viewport"), dict):
            vw = float(ev["viewport"].get("width", 0))
            vh = float(ev["viewport"].get("height", 0))
            if vw > 0 and vh > 0:
                session_vp = (vw, vh)
                doc_obj = ev.get("document")
                if isinstance(doc_obj, dict):
                    dw = float(doc_obj.get("width", vw))
                    dh = float(doc_obj.get("height", vh))
                    if dw > 0 and dh > 0:
                        session_doc = (dw, dh)
                break

    canonical_events: List[Dict[str, Any]] = []
    curr_vp = session_vp
    curr_doc = session_doc or session_vp

    for idx, ev in enumerate(behaviour_events):
        if not isinstance(ev, dict):
            raise ValueError(f"behaviourEvents[{idx}] must be a dictionary")

        # Strict check for unmapped event fields against the version's allow-list
        for ev_key in ev.keys():
            if ev_key not in allowed_event_keys:
                raise ValueError(f"Unmapped trace event field: '{ev_key}'")

        if "timestamp" not in ev or not isinstance(ev["timestamp"], (int, float)):
            raise ValueError(f"behaviourEvents[{idx}] is missing numeric 'timestamp'")
        if "type" not in ev or not isinstance(ev["type"], str):
            raise ValueError(f"behaviourEvents[{idx}] is missing string 'type'")

        ts_raw = float(ev["timestamp"])
        ts_ms = int(round(ts_raw))
        ev_type = str(ev["type"])

        # Viewport geometry validation
        if "viewport" in ev:
            if not isinstance(ev["viewport"], dict):
                raise ValueError(f"behaviourEvents[{idx}] 'viewport' must be a dictionary")
            vp = ev["viewport"]
            vp_w = float(vp.get("width", 0))
            vp_h = float(vp.get("height", 0))
            if vp_w <= 0.0 or vp_h <= 0.0:
                raise ValueError(
                    f"Non-positive or invalid viewport dimensions ({vp_w}, {vp_h}) "
                    f"in behaviourEvents[{idx}]"
                )
            curr_vp = (vp_w, vp_h)
        elif curr_vp is not None:
            vp_w, vp_h = curr_vp
        else:
            raise ValueError(f"behaviourEvents[{idx}] is missing required 'viewport' dictionary")

        # Document geometry validation
        if "document" in ev:
            if not isinstance(ev["document"], dict):
                raise ValueError(f"behaviourEvents[{idx}] 'document' must be a dictionary")
            doc = ev["document"]
            dw = float(doc.get("width", vp_w))
            dh = float(doc.get("height", vp_h))
            if dw <= 0.0 or dh <= 0.0:
                raise ValueError(
                    f"Non-positive or invalid document dimensions ({dw}, {dh}) "
                    f"in behaviourEvents[{idx}]"
                )
            curr_doc = (dw, dh)
            doc_w, doc_h = curr_doc
        elif curr_doc is not None:
            doc_w, doc_h = curr_doc
        else:
            doc_w, doc_h = vp_w, vp_h

        # Coordinate normalization
        x_val = ev.get("x")
        y_val = ev.get("y")
        x_norm: Optional[float] = None
        y_norm: Optional[float] = None
        x_raw: Optional[float] = None
        y_raw: Optional[float] = None

        if x_val is not None:
            try:
                x_float = float(x_val)
                x_norm = float(np.clip(x_float, 0.0, 1.0))
                x_raw = float(x_norm * vp_w)
            except (ValueError, TypeError):
                x_norm = None
                x_raw = None

        if y_val is not None:
            try:
                y_float = float(y_val)
                y_norm = float(np.clip(y_float, 0.0, 1.0))
                y_raw = float(y_norm * vp_h)
            except (ValueError, TypeError):
                y_norm = None
                y_raw = None

        target_id = str(ev.get("componentId") or ev.get("targetTag") or "")
        task_id = str(ev.get("taskId") or default_task_id)
        source_ev_id = f"{session_id}:ev_{idx}"

        # Correlate windowId
        #
        # The trace's window bounds are half-open `[start, end)`, so an event exactly on a
        # boundary belongs to the closing window; the comparison here is inclusive to keep such
        # an event attributed rather than dropped.
        matched_window_id: int = -1
        for wid, wstart, wend in window_spans:
            if wstart <= ts_raw <= wend:
                matched_window_id = wid
                break
        if matched_window_id == -1:
            # Fall back to the latest window that ends at or before this event.
            #
            # The previous fallback was `int(ts_ms // 250)`. That arithmetic assumed a short
            # monotonic timestamp; against epoch milliseconds (schema 1.3.0) it produces a window
            # id in the billions, which overflows the int32 `window_id` column and aborts the
            # entire ingestion run. A mis-attributed id is also worse than none, because it
            # silently corrupts the downstream `window_id` join instead of failing.
            candidates = [(wid, wend) for wid, _wstart, wend in window_spans if wend <= ts_raw]
            if candidates:
                matched_window_id = max(candidates, key=lambda item: item[1])[0]

        row: Dict[str, Any] = {
            "dataset_id": dataset_id,
            "session_id": session_id,
            "user_id": user_id,
            "trial_id": task_id,
            "timestamp_ms": ts_ms,
            "event_type": ev_type,
            "x": x_raw,
            "y": y_raw,
            "x_norm": x_norm,
            "y_norm": y_norm,
            "viewport_width": vp_w,
            "viewport_height": vp_h,
            "document_width": doc_w,
            "document_height": doc_h,
            "target_id": target_id,
            "source_event_id": source_ev_id,
            "duration_ms": None,
            "angle_deg": None,
            "distance_px": None,
            "velocity_px_s": None,
            # Provenance columns (snake_case)
            "experiment_id": experiment_id,
            "condition_id": condition_id,
            "task_id": task_id,
            "window_id": matched_window_id,
            "source_event_ids": source_ev_id,
            "provenance": provenance,
            "preprocessing_version": PREPROCESSING_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            # CamelCase aliases for exact acceptance criteria verification
            "sessionId": session_id,
            "experimentId": experiment_id,
            "conditionId": condition_id,
            "taskId": task_id,
            "windowId": matched_window_id,
            "sourceEventIds": source_ev_id,
            "preprocessingVersion": PREPROCESSING_VERSION,
            "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
        }
        canonical_events.append(row)

    # Sort deterministically by timestamp, then by source_event_id
    canonical_events.sort(key=lambda r: (r["timestamp_ms"], r["source_event_id"]))
    return canonical_events


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

    if output_path is None:
        root = find_project_root()
        out_file = root / ".data" / "interim" / "traces" / "canonical_traces.parquet"
    else:
        out_file = Path(output_path).resolve()

    if not force and out_file.is_file() and out_file.stat().st_size > 0:
        print(f"[Trace Ingestion] Using cached canonical trace Parquet: {out_file}")
        return str(out_file)

    out_file.parent.mkdir(parents=True, exist_ok=True)

    json_files = sorted(glob.glob(str(in_dir / "*.json")))
    if not json_files:
        raise FileNotFoundError(f"No JSON trace files found in: {in_dir}")

    all_rows: List[Dict[str, Any]] = []
    ingested_count = 0
    skipped: List[Dict[str, Any]] = []
    for fpath in json_files:
        with open(fpath, "r", encoding="utf-8") as f:
            trace = json.load(f)
        try:
            rows = parse_experiment_trace(trace, dataset_id=dataset_id)
        except ValueError as e:
            declared_version = trace.get("schemaVersion") if isinstance(trace, dict) else None
            if declared_version not in TRACE_KEY_ALLOWLIST:
                skipped.append({
                    "path": str(fpath),
                    "schema_version": declared_version,
                    "reason": str(e),
                })
                continue
            raise
        all_rows.extend(rows)
        ingested_count += 1

    if skipped:
        _report_skipped_traces(skipped)
        if strict:
            # Fail loudly instead of reporting success while entire (newer) traces vanish.
            raise UnsupportedTraceSchemaError(skipped)

    if not all_rows:
        raise ValueError(
            f"No traces with a supported schemaVersion "
            f"{SUPPORTED_EXPERIMENT_TRACE_SCHEMA_VERSIONS} found in: {in_dir}"
        )

    # Global deterministic ordering across all sessions: (session_id, timestamp_ms, source_event_id)
    all_rows.sort(key=lambda r: (r["session_id"], r["timestamp_ms"], r["source_event_id"]))

    try:
        from data import CANONICAL_EVENT_SCHEMA
    except ImportError:
        from src.data import CANONICAL_EVENT_SCHEMA
    if CANONICAL_EVENT_SCHEMA is None:
        raise RuntimeError("CANONICAL_EVENT_SCHEMA is not defined.")

    table = pa.Table.from_pylist(all_rows, schema=CANONICAL_EVENT_SCHEMA)
    pq.write_table(table, str(out_file), compression="snappy")

    print(
        f"[Trace Ingestion] Successfully ingested {ingested_count} trace file(s) "
        f"({len(skipped)} skipped for unsupported schemaVersion), "
        f"wrote {len(all_rows)} canonical events to {out_file} "
        f"({out_file.stat().st_size / 1024 / 1024:.2f} MB)"
    )
    return str(out_file)
