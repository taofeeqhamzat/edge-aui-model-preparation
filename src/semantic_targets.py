"""
semantic_targets.py
Semantic task-outcome targets for the Slow Gate's task-outcome head (Task 40.2).

The runtime records 15 semantic outcome IDs; the model's target vocabulary is the approved 14
(ADR-030). `inspect_help` is recorded evidence, not a target, and its windows are excluded from
training labels rather than relabelled.

Two label sources are supported, and the provenance is recorded per example:

- ``trace_semantic_outcome`` — the trace already carries ``taskOutcome`` for the window. This is
  the normal path for schema 1.4.0 captures and is preferred, because it is the runtime's own
  derivation.
- ``re_derived`` — a legacy capture predates the semantic field. The rules below mirror
  ``edge-aui-framework/src/testbed/config/semanticOutcomes.ts`` exactly (exact identifiers before
  substring rules, committed acts only, no hover/blur guessing), so the same event yields the same
  ID in both languages.

Future-outcome selection is causal: only records whose source event falls inside
``(anchor_end + min, anchor_end + max]`` are candidates, so no label can see the anchor window.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from config import load_config
except ImportError:  # pragma: no cover - import style used when run as src.*
    from src.config import load_config

_CONFIG = load_config().semantic_targets

TARGET_VOCABULARY: Dict[int, str] = dict(_CONFIG.target_vocabulary)
TARGET_VOCABULARY_VERSION: str = _CONFIG.target_vocabulary_version
RECORDED_VOCABULARY_ID: str = _CONFIG.recorded_vocabulary_id
RECORDED_VOCABULARY_VERSION: str = _CONFIG.recorded_vocabulary_version
RECORDED_ONLY_OUTCOMES = frozenset(_CONFIG.recorded_only_outcomes)
DEFAULT_LOOKAHEAD_MS: Tuple[int, int] = tuple(_CONFIG.lookahead_horizon_ms)  # type: ignore[assignment]

TARGET_NAME_TO_ID: Dict[str, int] = {name: idx for idx, name in TARGET_VOCABULARY.items()}

NO_OUTCOME_ID: int = TARGET_NAME_TO_ID["NO_OUTCOME"]
NO_OUTCOME: str = TARGET_VOCABULARY[NO_OUTCOME_ID]

# Deterministic tie-break when two semantic acts share the earliest timestamp. Terminal,
# committed actions win over state changes, which win over navigation. Declared here so a tie
# can never depend on dictionary iteration order.
SEMANTIC_TIE_BREAK_ORDER: List[str] = [
    "export_report",
    "reset_filter",
    "apply_filter",
    "select_report",
    "select_category",
    "select_date_range",
    "select_region",
    "open_filter_section",
    "close_filter_section",
    "navigate_reports",
    "navigate_analytics",
    "navigate_overview",
    "backtrack",
    NO_OUTCOME,
]
_TIE_BREAK_RANK: Dict[str, int] = {
    name: len(SEMANTIC_TIE_BREAK_ORDER) - idx
    for idx, name in enumerate(SEMANTIC_TIE_BREAK_ORDER)
}

# Event type -> canonical action kind, mirroring the framework's canonical action layer.
_KIND_BY_EVENT_TYPE = {
    "click": "click",
    "change": "change",
    "input": "input",
    "submit": "submit",
    "focus": "focus",
    "blur": "blur",
    "mouseover": "hover",
    "scroll": "scroll",
    "wheel": "scroll",
    "navigation": "navigate",
    "popstate": "navigate",
    "hashchange": "navigate",
    "pagehide": "lifecycle",
    "beforeunload": "lifecycle",
    "unload": "lifecycle",
}

_NAVIGATION_DESTINATIONS = {
    "nav-Overview": "navigate_overview",
    "nav-Analytics": "navigate_analytics",
    "nav-Reports": "navigate_reports",
}


def is_target_outcome(name: str) -> bool:
    """True when the ID is one of the 14 approved target classes."""
    return name in TARGET_NAME_TO_ID


def target_outcome_id(name: str) -> int:
    """Integer class ID for a target outcome; raises for a non-target ID."""
    if name not in TARGET_NAME_TO_ID:
        raise ValueError(
            f"'{name}' is not in the {TARGET_VOCABULARY_VERSION} target vocabulary; "
            f"recorded-only IDs {sorted(RECORDED_ONLY_OUTCOMES)} are never labels"
        )
    return TARGET_NAME_TO_ID[name]


def _action_kind(event: Dict[str, Any]) -> Optional[str]:
    return _KIND_BY_EVENT_TYPE.get(str(event.get("type") or "").lower())


def _is_committed_click(kind: Optional[str]) -> bool:
    return kind in ("click", "submit")


def _is_submit_control(kind: Optional[str], role: str) -> bool:
    if kind == "submit":
        return True
    return kind == "click" and "submit" in role


def _match_primary_control(kind: Optional[str], component_id: str, role: str) -> Optional[str]:
    """Export, reset and apply. Exact IDs first; apply is decided by the submit annotation."""
    if not _is_committed_click(kind):
        return None
    if component_id == "btn-export":
        return "export_report"
    if component_id == "btn-reset-filters":
        return "reset_filter"
    if _is_submit_control(kind, role):
        return "apply_filter"
    return None


def _match_catalogue(kind: Optional[str], component_id: str) -> Optional[str]:
    if _is_committed_click(kind) and component_id.startswith("open-report-"):
        return "select_report"
    return None


def _match_contextual_help(kind: Optional[str], component_id: str, role: str) -> Optional[str]:
    if not _is_committed_click(kind):
        return None
    if component_id == "tooltip-region" or role == "tooltip":
        return "inspect_help"
    return None


def _match_disclosure(kind: Optional[str], role: str, aria_expanded: Any) -> Optional[str]:
    if not _is_committed_click(kind) or role != "accordion":
        return None
    if aria_expanded is True:
        return "open_filter_section"
    if aria_expanded is False:
        return "close_filter_section"
    return None


def _match_filter_field(kind: Optional[str], component_id: str) -> Optional[str]:
    if kind != "change":
        return None
    if component_id == "filter-Region-select":
        return "select_region"
    if component_id == "filter-date-input":
        return "select_date_range"
    if component_id.startswith("filter-category-"):
        return "select_category"
    return None


def _match_navigation(kind: Optional[str], role: str, component_id: str) -> Optional[str]:
    if not _is_committed_click(kind) or role != "navigation":
        return None
    return _NAVIGATION_DESTINATIONS.get(component_id)


# Ordered rule table. Exact identity rules precede prefix/role rules, mirroring the TypeScript
# declaration order so a new control cannot be captured by an accidental substring.
_RULES = (
    lambda kind, cid, role, aria: _match_primary_control(kind, cid, role),
    lambda kind, cid, role, aria: _match_catalogue(kind, cid),
    lambda kind, cid, role, aria: _match_contextual_help(kind, cid, role),
    lambda kind, cid, role, aria: _match_disclosure(kind, role, aria),
    lambda kind, cid, role, aria: _match_filter_field(kind, cid),
    lambda kind, cid, role, aria: _match_navigation(kind, role, cid),
)


def derive_semantic_outcome(event: Dict[str, Any]) -> Optional[str]:
    """Derive one semantic outcome ID from a raw behaviour event.

    Hovers, blur/focus, uncommitted inputs and system-initiated acts never produce an outcome.
    Returns ``None`` when no rule matches.
    """
    if not isinstance(event, dict) or event.get("systemInitiated") is True:
        return None
    kind = _action_kind(event)
    if kind is None:
        return None

    component_id = str(event.get("componentId") or "")
    role = str(event.get("componentRole") or "").lower()
    aria_expanded = event.get("ariaExpanded")
    for rule in _RULES:
        outcome = rule(kind, component_id, role, aria_expanded)
        if outcome is not None:
            return outcome
    return None


class RouteTracker:
    """Route-level backtracking, mirroring the framework's tracker.

    Only a deliberate navigation to a previously visited, different page is a reversal. Clicks,
    blurs, disclosure toggles and filter edits never are.
    """

    def __init__(self, starting_route: Optional[str] = None):
        self.reset(starting_route)

    def reset(self, starting_route: Optional[str] = None) -> None:
        self.current = starting_route
        self.visited: List[str] = [starting_route] if starting_route else []

    def observe(self, event: Dict[str, Any]) -> bool:
        if not isinstance(event, dict) or event.get("systemInitiated") is True:
            return False
        if _action_kind(event) != "navigate":
            return False
        destination = event.get("route")
        if not destination or destination == self.current:
            return False
        backtrack = destination in self.visited
        if not backtrack:
            self.visited.append(str(destination))
        self.current = str(destination)
        return backtrack


def derive_semantic_outcomes(
    events: Iterable[Dict[str, Any]],
    starting_route: Optional[str] = None,
) -> List[Optional[str]]:
    """Derive outcomes for a whole event stream, in order, so backtracking stays coherent."""
    tracker = RouteTracker(starting_route)
    outcomes: List[Optional[str]] = []
    for event in events:
        if tracker.observe(event):
            outcomes.append("backtrack")
        else:
            outcomes.append(derive_semantic_outcome(event))
    return outcomes


def record_timestamp(record: Dict[str, Any]) -> Optional[float]:
    raw = record.get("sourceEventTimestamp")
    if raw is None:
        raw = record.get("timestamp")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def select_target_record(
    records: Iterable[Dict[str, Any]],
    anchor_end_ms: float,
    lookahead_min_ms: float = DEFAULT_LOOKAHEAD_MS[0],
    lookahead_max_ms: float = DEFAULT_LOOKAHEAD_MS[1],
) -> Optional[Dict[str, Any]]:
    """Pick the earliest future outcome record inside the lookahead horizon.

    Records are supplied by the caller already resolved to semantic IDs. A record exactly on a
    horizon boundary is included (the horizon is closed), and ties are broken by
    ``SEMANTIC_TIE_BREAK_ORDER``.
    """
    start = float(anchor_end_ms) + float(lookahead_min_ms)
    end = float(anchor_end_ms) + float(lookahead_max_ms)

    candidates: List[Tuple[float, int, int, Dict[str, Any]]] = []
    for index, record in enumerate(records):
        timestamp = record_timestamp(record)
        if timestamp is None or not (start <= timestamp <= end):
            continue
        name = record.get("taskOutcome") or NO_OUTCOME
        candidates.append((timestamp, _TIE_BREAK_RANK.get(str(name), 0), -index, record))

    if not candidates:
        return None

    earliest = min(candidate[0] for candidate in candidates)
    tied = [candidate for candidate in candidates if abs(candidate[0] - earliest) <= 1.0]
    tied.sort(key=lambda candidate: (candidate[1], candidate[2]), reverse=True)
    return tied[0][3]


def find_source_event(
    record: Dict[str, Any],
    events: Iterable[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Find the raw event that produced an outcome record (used for legacy re-derivation)."""
    timestamp = record_timestamp(record)
    if timestamp is None:
        return None
    component_id = record.get("componentId")

    for event in events:
        try:
            event_ts = float(event.get("timestamp"))
        except (TypeError, ValueError):
            continue
        if abs(event_ts - timestamp) > 1.0:
            continue
        if component_id and event.get("componentId") != component_id:
            continue
        return event
    return None


def resolve_target_label(
    record: Dict[str, Any],
    has_semantic_contract: bool,
    source_event: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """Resolve a record to ``(semantic outcome, label source)``.

    A recorded ``taskOutcome`` always wins. When the trace declares the semantic contract but
    the record carries no ID, no rule matched and the label is ``NO_OUTCOME``. Only a trace that
    predates the contract (``has_semantic_contract=False``) is re-derived from its source event.
    """
    recorded = record.get("taskOutcome")
    if recorded is not None:
        name = str(recorded)
        if name not in TARGET_NAME_TO_ID and name not in RECORDED_ONLY_OUTCOMES:
            raise ValueError(
                f"Unknown recorded semantic outcome '{name}': it is in neither the "
                f"{TARGET_VOCABULARY_VERSION} target vocabulary nor the recorded-only set"
            )
        return name, "trace_semantic_outcome"

    if has_semantic_contract:
        return NO_OUTCOME, "trace_no_outcome"

    if source_event is not None:
        derived = derive_semantic_outcome(source_event)
        if derived is not None:
            return derived, "re_derived"

    return NO_OUTCOME, "re_derived_no_match"
