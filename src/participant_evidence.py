"""
participant_evidence.py
Stage participant traces as analysis input, gated by the framework trace verifier.

The framework verifier (`edge-aui-framework/scripts/verify-trace.mjs`) is the authoritative
trace check. A participant capture that fails it is still a record, but it is not analysis
input: the staging step copies only verified traces and writes `verification-manifest.json`
with every pass/fail and its detail.

Guards:

- the source must be participant-labelled in both `session.provenance` and
  `metadata.provenance`; scripted traces are refused rather than pooled;
- a trace without a verification result fails closed (excluded);
- the output directory may not be the source directory.
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

EVIDENCE_SCHEMA_VERSION = "1.0.0"
MANIFEST_NAME = "verification-manifest.json"
TRACE_GLOB = "experiment-trace-*.json"
_ALLOWED_PROVENANCE = ("scripted", "participant")


class ParticipantEvidenceError(ValueError):
    """Raised when participant evidence cannot be staged safely."""


def _load_trace(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            trace = json.load(handle)
    except Exception as exc:
        raise ParticipantEvidenceError(f"Failed to parse trace '{path.name}': {exc}")
    if not isinstance(trace, dict):
        raise ParticipantEvidenceError(f"Trace '{path.name}' is not a JSON object")
    return trace


def _session_id(trace: Dict[str, Any], path: Path) -> str:
    session = trace.get("session")
    if not isinstance(session, dict) or not session.get("sessionId"):
        raise ParticipantEvidenceError(f"Trace '{path.name}' has no session.sessionId")
    return str(session["sessionId"])


def _require_participant(trace: Dict[str, Any], path: Path) -> None:
    session_provenance = (trace.get("session") or {}).get("provenance")
    metadata_provenance = (trace.get("metadata") or {}).get("provenance")
    if session_provenance != "participant" or metadata_provenance != "participant":
        raise ParticipantEvidenceError(
            f"Trace '{path.name}' declares provenance "
            f"session={session_provenance!r}, metadata={metadata_provenance!r}; participant "
            f"evidence requires both to be 'participant'"
        )


def _detail_from_output(output: str) -> str:
    lines = output.splitlines()
    for index, line in enumerate(lines):
        if "FAIL" not in line:
            continue
        detail = line.strip()
        for follow in lines[index + 1: index + 4]:
            if follow.strip():
                return f"{detail} — {follow.strip()}"
        return detail
    for line in lines:
        if line.strip():
            return line.strip()
    return "verifier produced no detail"


def verify_traces(
    trace_paths: Iterable[Path],
    verifier_script: Union[str, Path],
    framework_root: Union[str, Path],
) -> Dict[str, Dict[str, Any]]:
    """Run the framework verifier once per trace; return ``session_id -> result``."""
    script = Path(verifier_script).resolve()
    if not script.is_file():
        raise ParticipantEvidenceError(f"Trace verifier not found: {script}")
    root = Path(framework_root).resolve()
    results: Dict[str, Dict[str, Any]] = {}
    for path in trace_paths:
        trace = _load_trace(path)
        session_id = _session_id(trace, path)
        process = subprocess.run(
            ["node", str(script), str(path), "--verbose"],
            cwd=str(root), capture_output=True, text=True,
        )
        output = (process.stdout or "") + (process.stderr or "")
        results[session_id] = {
            "verified": process.returncode == 0,
            "detail": "verified" if process.returncode == 0 else _detail_from_output(output),
        }
    return results


def _classify_sessions(
    trace_paths: Iterable[Path],
    verification: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Validate the source and decide which sessions are analysis input."""
    sessions: List[Dict[str, Any]] = []
    for path in trace_paths:
        trace = _load_trace(path)
        _require_participant(trace, path)
        session_id = _session_id(trace, path)
        result = verification.get(session_id)
        if result is None:
            status, detail = "excluded", "no verification result"
        elif result.get("verified"):
            status, detail = "verified", str(result.get("detail") or "verified")
        else:
            status, detail = "excluded", str(result.get("detail") or "verification failed")
        sessions.append({
            "session_id": session_id,
            "status": status,
            "detail": detail,
            "task_id": str((trace.get("task") or {}).get("currentTaskId") or ""),
            "source_file": path.name,
        })
    return sessions


def _copy_verified(
    source: Path, output: Path, sessions: List[Dict[str, Any]]
) -> None:
    for entry in sessions:
        target = output / entry["source_file"]
        if entry["status"] != "verified":
            if target.is_file():
                target.unlink()
            continue
        shutil.copyfile(source / entry["source_file"], target)


def stage_verified(
    source_dir: Union[str, Path],
    output_dir: Union[str, Path],
    verification: Dict[str, Dict[str, Any]],
    force: bool = False,
) -> Dict[str, Any]:
    """Copy verified participant traces to ``output_dir`` and write the audit manifest."""
    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    if not source.is_dir():
        raise ParticipantEvidenceError(f"Source directory not found: {source}")
    if output == source:
        raise ParticipantEvidenceError("Output directory must differ from the source directory")

    trace_paths = sorted(source.glob(TRACE_GLOB))
    if not trace_paths:
        raise ParticipantEvidenceError(f"No '{TRACE_GLOB}' files found in {source}")

    sessions = _classify_sessions(trace_paths, verification)
    output.mkdir(parents=True, exist_ok=True)
    if not force:
        stale = output / MANIFEST_NAME
        if stale.is_file():
            stale.unlink()
    _copy_verified(source, output, sessions)

    verified = sum(1 for entry in sessions if entry["status"] == "verified")
    manifest = {
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "provenance": "participant",
        "source": {"directory": str(source), "file_count": len(trace_paths)},
        "counts": {
            "total": len(sessions),
            "verified": verified,
            "excluded": len(sessions) - verified,
        },
        "sessions": sorted(sessions, key=lambda entry: entry["session_id"]),
    }
    (output / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def prepare_participant_evidence(
    source_dir: Union[str, Path],
    output_dir: Union[str, Path],
    framework_root: Union[str, Path],
    verifier_script: Optional[Union[str, Path]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Verify every trace in ``source_dir``, then stage the verified subset."""
    root = Path(framework_root).resolve()
    script = Path(verifier_script) if verifier_script else root / "scripts" / "verify-trace.mjs"
    traces = sorted(Path(source_dir).resolve().glob(TRACE_GLOB))
    verification = verify_traces(traces, script, root)
    return stage_verified(source_dir, output_dir, verification, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify participant traces and stage the verified subset for analysis."
    )
    parser.add_argument("--source", required=True, help="Corrected participant trace directory")
    parser.add_argument("--output", required=True, help="Directory for verified traces")
    parser.add_argument(
        "--framework-root", default="../edge-aui-framework",
        help="Path to the framework repository containing scripts/verify-trace.mjs",
    )
    parser.add_argument("--verifier", default=None, help="Explicit verifier script path")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        manifest = prepare_participant_evidence(
            args.source, args.output, args.framework_root,
            verifier_script=args.verifier, force=args.force,
        )
    except ParticipantEvidenceError as exc:
        print(f"[ParticipantEvidence] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)

    counts = manifest["counts"]
    print(
        f"[ParticipantEvidence] {counts['verified']} verified, {counts['excluded']} excluded "
        f"of {counts['total']} -> {args.output}"
    )


if __name__ == "__main__":
    main()
