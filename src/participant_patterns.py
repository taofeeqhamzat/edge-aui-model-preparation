"""
participant_patterns.py
Derive Fast Gate pattern candidates from participant macro sequences (Task 50.2 input).

The previous pattern basis was the scripted canonical paths, which made "aligned to TD1–TD4"
self-fulfilling. This tool reads participant traces only, collapses the duplicate macro symbols
one participant act emits, drops symbols that are unreachable or misleading (assessment
limitation #6), and reports contiguous n-grams with their session support and task coverage.

Candidates are evidence, not configuration: 50.1 correctness is still open, so a candidate is
validated as a declared pattern in 50.2 rather than copied into the Fast Gate map.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

EVIDENCE_SCHEMA_VERSION = "1.0.0"
MANIFEST_NAME = "participant-pattern-candidates.json"
TRACE_GLOB = "experiment-trace-*.json"
DEFAULT_MIN_SUPPORT = 3
MAX_PATTERN_LENGTH = 4

# Symbols the macro layer cannot currently produce, or produces misleadingly (plan §2.8).
# Mining them would turn a defect into a pattern.
EXCLUDED_SYMBOLS = frozenset({
    "TABLE_PAGE_NEXT",
    "TABLE_PAGE_PREV",
    "TABLE_SORT",
    "TABLE_ROW_SELECT",
    "IDLE_DWELL",
    "RAPID_SCROLL",
    "BACKTRACK",
    "SELECT_SEGMENT",
})

# Symbols whose macro derivation is still known to be noisy; candidates containing them are
# reported with a flag rather than silently dropped.
RISKY_SYMBOLS = {
    "SELECT_REGION": "may include tooltip-region misclassification until the macro rule is fixed",
}


class ParticipantPatternError(ValueError):
    """Raised when the participant pattern analysis cannot run on the supplied evidence."""


def collapse_symbols(symbols: Iterable[str]) -> List[str]:
    """One participant act can emit several events; keep one symbol per run."""
    collapsed: List[str] = []
    for symbol in symbols:
        if not symbol:
            continue
        if collapsed and collapsed[-1] == symbol:
            continue
        collapsed.append(str(symbol))
    return collapsed


def _require_participant(trace: Dict[str, Any], path: Path) -> None:
    session_provenance = (trace.get("session") or {}).get("provenance")
    metadata_provenance = (trace.get("metadata") or {}).get("provenance")
    if session_provenance != "participant" or metadata_provenance != "participant":
        raise ParticipantPatternError(
            f"Trace '{path.name}' is not declared participant in both places "
            f"(session={session_provenance!r}, metadata={metadata_provenance!r}); "
            f"scripted canonical paths must not drive the 50.2 pattern set"
        )


def _symbols_of(trace: Dict[str, Any]) -> List[str]:
    macros = sorted(
        (record for record in trace.get("macroInteractions", [])
         if isinstance(record, dict) and record.get("symbol")),
        key=lambda record: float(record.get("timestamp", 0.0)),
    )
    raw_symbols = [str(record["symbol"]) for record in macros]
    return [
        symbol for symbol in collapse_symbols(raw_symbols)
        if symbol not in EXCLUDED_SYMBOLS
    ]


def load_sessions(trace_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load participant macro sequences; refuse anything not declared participant."""
    directory = Path(trace_dir)
    if not directory.is_dir():
        raise ParticipantPatternError(f"Trace directory not found: {directory}")
    paths = sorted(directory.glob(TRACE_GLOB))
    if not paths:
        raise ParticipantPatternError(f"No '{TRACE_GLOB}' files found in {directory}")

    sessions: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            trace = json.load(handle)
        _require_participant(trace, path)
        sessions.append({
            "session_id": str((trace.get("session") or {}).get("sessionId")),
            "task_id": str((trace.get("task") or {}).get("currentTaskId") or "unknown"),
            "symbols": _symbols_of(trace),
            "raw_symbol_count": len(trace.get("macroInteractions", [])),
        })
    return sessions


def derive_candidates(
    sessions: Sequence[Dict[str, Any]],
    min_support: int = DEFAULT_MIN_SUPPORT,
    max_length: int = MAX_PATTERN_LENGTH,
) -> List[Dict[str, Any]]:
    """Contiguous n-grams with support counted across sessions, not event frequency."""
    support: Dict[Tuple[str, ...], int] = defaultdict(int)
    occurrences: Dict[Tuple[str, ...], int] = defaultdict(int)
    pattern_tasks: Dict[Tuple[str, ...], Set[str]] = defaultdict(set)

    for session in sessions:
        symbols = session["symbols"]
        seen_in_session: Set[Tuple[str, ...]] = set()
        for length in range(2, max_length + 1):
            for start in range(0, len(symbols) - length + 1):
                pattern = tuple(symbols[start:start + length])
                occurrences[pattern] += 1
                pattern_tasks[pattern].add(session["task_id"])
                if pattern not in seen_in_session:
                    support[pattern] += 1
                    seen_in_session.add(pattern)

    candidates = []
    for pattern, session_support in support.items():
        if session_support < min_support:
            continue
        risks = [
            RISKY_SYMBOLS[symbol] for symbol in pattern if symbol in RISKY_SYMBOLS
        ]
        candidates.append({
            "pattern": list(pattern),
            "length": len(pattern),
            "support_sessions": session_support,
            "occurrences": occurrences[pattern],
            "tasks": sorted(pattern_tasks[pattern]),
            "risk_flags": sorted(set(risks)),
        })
    candidates.sort(key=lambda item: (-item["support_sessions"], -item["length"], item["pattern"]))
    return candidates


def run_analysis(
    trace_dir: Union[str, Path],
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> Dict[str, Any]:
    sessions = load_sessions(trace_dir)
    candidates = derive_candidates(sessions, min_support=min_support)
    task_counts = Counter(session["task_id"] for session in sessions)
    symbol_counts = Counter(
        symbol for session in sessions for symbol in session["symbols"]
    )
    return {
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "provenance": "participant",
        "session_count": len(sessions),
        "task_counts": dict(sorted(task_counts.items())),
        "sessions_without_macros": sum(1 for session in sessions if not session["symbols"]),
        "min_support": min_support,
        "max_pattern_length": MAX_PATTERN_LENGTH,
        "excluded_symbols": sorted(EXCLUDED_SYMBOLS),
        "symbol_counts": dict(symbol_counts.most_common()),
        "candidates": candidates,
        "evidence_scope": (
            "Participant-derived macro sequences. Candidate patterns are evidence for Task "
            "50.2, not the gate configuration; 50.1 PrefixSpan correctness remains open and "
            "the selection is provisional until more participant traces are collected."
        ),
    }


def _fmt_pattern(pattern: Sequence[str]) -> str:
    return " > ".join(f"`{symbol}`" for symbol in pattern)


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Participant-derived Fast Gate pattern candidates (Task 50.2)")
    lines.append("")
    lines.append(f"**Evidence provenance:** `{report['provenance']}` "
                 f"({report['session_count']} sessions)")
    lines.append("")
    lines.append(f"> **Evidence scope:** {report['evidence_scope']}")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("- Reads participant macro sequences only; scripted traces are refused.")
    lines.append("- Collapses consecutive duplicate symbols, because one participant act emits")
    lines.append("  several macro events (for example three `OPEN_REPORT` records in 20 ms).")
    lines.append("- Counts contiguous 2–4 symbol n-grams; support is the number of sessions that")
    lines.append("  contain the pattern, not the number of occurrences.")
    lines.append(f"- Minimum support: {report['min_support']} sessions.")
    lines.append("- Excludes symbols that are unreachable or misleading "
                 f"({', '.join(f'`{s}`' for s in report['excluded_symbols'])}); a pattern that")
    lines.append("  cannot be produced cannot be a Fast Gate pattern.")
    lines.append("")
    lines.append("## Participants and tasks")
    lines.append("")
    lines.append(f"- Sessions: {report['session_count']}")
    lines.append(f"- Sessions with no usable macro symbols: {report['sessions_without_macros']}")
    lines.append(f"- Task coverage: {report['task_counts']}")
    lines.append("")
    lines.append("## Candidates")
    lines.append("")
    lines.append("| Pattern | Length | Support (sessions) | Occurrences | Tasks | Risk |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for candidate in report["candidates"][:60]:
        risk = "; ".join(candidate["risk_flags"]) or "none"
        lines.append(
            f"| {_fmt_pattern(candidate['pattern'])} | {candidate['length']} | "
            f"{candidate['support_sessions']} | {candidate['occurrences']} | "
            f"{', '.join(candidate['tasks'])} | {risk} |"
        )
    if not report["candidates"]:
        lines.append("| _none met the support threshold_ | — | — | — | — | — |")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- Candidate patterns are evidence, not the declared Fast Gate configuration.")
    lines.append("  50.1 PrefixSpan correctness and boundedness remain open; 50.2 must validate")
    lines.append("  each candidate (support, length, session isolation, no-match behaviour)")
    lines.append("  before it enters the pattern map.")
    lines.append("- `SELECT_REGION` is still produced by the macro `tooltip-region` substring rule")
    lines.append("  (assessment limitation #6), so candidates containing it may include tooltip")
    lines.append("  misclassifications; they are flagged, not silently accepted.")
    lines.append("- The participant set is small and has no approved participant identifier")
    lines.append("  (ADR-023). No participant-held-out generalisation is claimed.")
    lines.append("- The selection is provisional until further participant traces are collected.")
    lines.append("")
    return "\n".join(lines)


def write_report(report: Dict[str, Any], output_dir: Union[str, Path]) -> Dict[str, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / MANIFEST_NAME
    markdown_path = directory / "participant-pattern-candidates.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive Fast Gate pattern candidates from participant macro sequences."
    )
    parser.add_argument("--trace-dir", default=".data/raw/participant-verified")
    parser.add_argument("--report-dir", default="docs/experiments/participant-patterns")
    parser.add_argument("--min-support", type=int, default=DEFAULT_MIN_SUPPORT)
    args = parser.parse_args()

    try:
        report = run_analysis(args.trace_dir, min_support=args.min_support)
    except ParticipantPatternError as exc:
        print(f"[ParticipantPatterns] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)

    paths = write_report(report, args.report_dir)
    print(f"[ParticipantPatterns] {len(report['candidates'])} candidate(s) from "
          f"{report['session_count']} participant sessions")
    print(f"[ParticipantPatterns] Report: {paths['markdown']}")
    print(f"[ParticipantPatterns] Results: {paths['json']}")


if __name__ == "__main__":
    main()
