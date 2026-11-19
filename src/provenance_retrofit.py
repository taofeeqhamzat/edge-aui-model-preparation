"""
provenance_retrofit.py
Local, audited retrofit of a mislabelled provenance declaration.

The Supabase rows and the exported trace bodies are the *record*; this tool never edits them.
It works on an exported copy and rewrites only the two declared provenance fields
(`session.provenance`, `metadata.provenance`) under an explicit researcher authority, leaving
`participantId` null because no participant-identifier scheme is approved (ADR-023).

Every run writes `retrofit-manifest.json` beside the corrected traces: per session, the
original declarations, the corrected declaration, the authority, and the source/output file
hashes. Re-running is idempotent (already-corrected sessions are skipped), and the tool refuses
to write into the source directory, to invent an identifier, or to accept an unknown
provenance value.

Provenance is declared, not inferred: this tool applies a declaration the researcher made; it
never guesses from behaviour, timing or device.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

RETROFIT_SCHEMA_VERSION = "1.0.0"
ALLOWED_PROVENANCE = ("scripted", "participant")
MANIFEST_NAME = "retrofit-manifest.json"
TRACE_GLOB = "experiment-trace-*.json"


class ProvenanceRetrofitError(ValueError):
    """Raised when a retrofit cannot be applied safely or auditable."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_trace(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            trace = json.load(handle)
    except Exception as exc:
        raise ProvenanceRetrofitError(f"Failed to parse trace '{path.name}': {exc}")
    if not isinstance(trace, dict):
        raise ProvenanceRetrofitError(f"Trace '{path.name}' is not a JSON object")
    session = trace.get("session")
    if not isinstance(session, dict) or not session.get("sessionId"):
        raise ProvenanceRetrofitError(f"Trace '{path.name}' has no session.sessionId")
    return trace


def _declared_provenance(trace: Dict[str, Any], path: Path) -> tuple:
    session = trace["session"]
    metadata = trace.get("metadata")
    if not isinstance(metadata, dict):
        raise ProvenanceRetrofitError(f"Trace '{path.name}' has no metadata object")
    session_provenance = session.get("provenance")
    metadata_provenance = metadata.get("provenance")
    for label, value in (
        ("session.provenance", session_provenance),
        ("metadata.provenance", metadata_provenance),
    ):
        if value not in ALLOWED_PROVENANCE:
            raise ProvenanceRetrofitError(
                f"Trace '{path.name}' declares {label}='{value}', which is not one of "
                f"{list(ALLOWED_PROVENANCE)}; refusing to launder an unknown declaration"
            )
    return session_provenance, metadata_provenance


def _prepare_output(source: Path, output: Path, force: bool) -> None:
    if output == source:
        raise ProvenanceRetrofitError("Output directory must differ from the source directory")
    if source in output.parents:
        raise ProvenanceRetrofitError("Output directory must not be inside the source directory")
    if not output.exists():
        output.mkdir(parents=True)
        return
    existing = sorted(output.glob("*.json"))
    if existing and not force:
        raise ProvenanceRetrofitError(
            f"Output directory already contains {len(existing)} JSON file(s); pass force=True "
            f"to replace a previous retrofit"
        )
    for path in existing:
        if path.name == MANIFEST_NAME or path.match(TRACE_GLOB):
            path.unlink()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _retrofit_one(path: Path, output: Path, target: str) -> Dict[str, Any]:
    trace = _load_trace(path)
    original_session, original_metadata = _declared_provenance(trace, path)
    already_corrected = original_session == target and original_metadata == target
    entry: Dict[str, Any] = {
        "session_id": str(trace["session"]["sessionId"]),
        "source_file": path.name,
        "task_id": str((trace.get("task") or {}).get("currentTaskId") or ""),
        "schema_version": str(trace.get("schemaVersion") or ""),
        "original_session_provenance": original_session,
        "original_metadata_provenance": original_metadata,
        "corrected_provenance": target,
        "source_sha256": _sha256(path),
    }
    output_path = output / path.name
    if already_corrected:
        output_path.write_bytes(path.read_bytes())
        entry["status"] = "already_corrected"
    else:
        _correct_session(trace, target)
        _write_json(output_path, trace)
        entry["status"] = "corrected"
        entry["participant_id"] = trace["session"].get("participantId")
    entry["output_file"] = output_path.name
    entry["output_sha256"] = _sha256(output_path)
    return entry


def _correct_session(
    trace: Dict[str, Any], target: str
) -> tuple:
    original_session, original_metadata = (
        trace["session"].get("provenance"),
        (trace.get("metadata") or {}).get("provenance"),
    )
    trace["session"]["provenance"] = target
    trace["metadata"]["provenance"] = target
    return original_session, original_metadata


def retrofit_provenance(
    source_dir: Union[str, Path],
    output_dir: Union[str, Path],
    provenance: str = "participant",
    authority: str = "",
    force: bool = False,
) -> Dict[str, Any]:
    """Write an audited, corrected copy of every exported trace under ``output_dir``."""
    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    if not source.is_dir():
        raise ProvenanceRetrofitError(f"Source directory not found: {source}")
    if provenance not in ALLOWED_PROVENANCE:
        raise ProvenanceRetrofitError(
            f"Target provenance '{provenance}' is not one of {list(ALLOWED_PROVENANCE)}"
        )
    if not isinstance(authority, str) or not authority.strip():
        raise ProvenanceRetrofitError(
            "An explicit, non-empty authority is required for a provenance retrofit"
        )

    _prepare_output(source, output, force)
    trace_paths = sorted(source.glob(TRACE_GLOB))
    if not trace_paths:
        raise ProvenanceRetrofitError(f"No '{TRACE_GLOB}' files found in {source}")

    sessions = [_retrofit_one(path, output, provenance) for path in trace_paths]
    corrected = sum(1 for entry in sessions if entry["status"] == "corrected")
    skipped = len(sessions) - corrected

    manifest = {
        "retrofit_schema_version": RETROFIT_SCHEMA_VERSION,
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "target_provenance": provenance,
        "authority": authority.strip(),
        "rule": (
            "session.provenance and metadata.provenance rewritten on the exported copy only; "
            "participantId untouched (null); source export unchanged; session-bounded splits "
            "remain the grouping unit (ADR-031)"
        ),
        "source": {"directory": str(source), "file_count": len(trace_paths)},
        "counts": {
            "total": len(sessions),
            "corrected": corrected,
            "skipped": skipped,
        },
        "sessions": sorted(sessions, key=lambda entry: entry["session_id"]),
    }
    _write_json(output / MANIFEST_NAME, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply an audited, local provenance retrofit to exported traces."
    )
    parser.add_argument("--source", required=True, help="Exported trace directory (read only)")
    parser.add_argument("--output", required=True, help="Corrected trace directory")
    parser.add_argument("--provenance", default="participant", choices=list(ALLOWED_PROVENANCE))
    parser.add_argument("--authority", required=True, help="Who authorized the correction, and why")
    parser.add_argument("--force", action="store_true", help="Replace a previous output")
    args = parser.parse_args()

    try:
        manifest = retrofit_provenance(
            args.source, args.output,
            provenance=args.provenance, authority=args.authority, force=args.force,
        )
    except ProvenanceRetrofitError as exc:
        print(f"[Retrofit] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)

    counts = manifest["counts"]
    print(
        f"[Retrofit] {counts['corrected']} corrected, {counts['skipped']} already correct "
        f"-> {args.output}"
    )
    print(f"[Retrofit] Manifest: {Path(args.output) / MANIFEST_NAME}")


if __name__ == "__main__":
    main()
