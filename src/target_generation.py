"""
target_generation.py
Self-Supervised Lookahead Outcome Extraction & Target UI Intervention Taxonomy.
Implements outcome-driven self-supervision (Wu et al., 2024) mapping downstream
observable interaction events to discrete target labels.
"""

from typing import List, Dict, Tuple, Any, Optional

# Concrete, observable downstream outcome taxonomy
OUTCOME_TAXONOMY = {
    0: "IDLE_ABANDON",
    1: "CLICK",
    2: "FORM_SUBMIT",
    3: "BACKTRACK",
    4: "RAPID_SCROLL",
    5: "HOVER_DWELL"
}

OUTCOME_NAME_TO_ID = {v: k for k, v in OUTCOME_TAXONOMY.items()}

# Downstream target intervention vocabulary (for transfer learning head)
TARGET_INTERVENTIONS = {
    0: "simplify_options",
    1: "highlight_primary_action",
    2: "offer_assistance",
    3: "expand_tooltip",
    4: "no_op"
}

INTERVENTION_NAME_TO_ID = {v: k for k, v in TARGET_INTERVENTIONS.items()}


def extract_lookahead_outcome(
    future_events: List[Dict[str, Any]],
    idle_threshold_ms: float = 3000.0,
    rapid_scroll_min_events: int = 4,
    hover_dwell_min_events: int = 2
) -> Tuple[int, str]:
    """
    Extract the downstream UI outcome label from forward-looking temporal events.
    Hierarchy:
    1. FORM_SUBMIT / CLICK: User commits action on target component.
    2. BACKTRACK: User navigates back, switches tab, or blurs window.
    3. RAPID_SCROLL: User exhibits fast scanning behavior.
    4. HOVER_DWELL: User lingers over an interactive element without clicking.
    5. IDLE_ABANDON: Inactivity or empty event stream.
    """
    if not future_events:
        return OUTCOME_NAME_TO_ID["IDLE_ABANDON"], "IDLE_ABANDON"

    # Collect event types and target identifiers
    event_types = set()
    xpaths: List[str] = []
    scroll_count = 0
    hover_count = 0

    for ev in future_events:
        etype = str(ev.get("event_type") or ev.get("event") or "").lower()
        event_types.add(etype)

        xpath = str(ev.get("target_id") or ev.get("xpath") or "")
        if xpath:
            xpaths.append(xpath.lower())

        if etype in ("scroll", "wheel"):
            scroll_count += 1
        elif etype == "mouseover":
            hover_count += 1

    # 1. Action Commit (CLICK / FORM_SUBMIT)
    if "click" in event_types or "mousedown" in event_types:
        for xp in xpaths:
            if any(term in xp for term in ["submit", "btn", "button", "input", "form"]):
                return OUTCOME_NAME_TO_ID["FORM_SUBMIT"], "FORM_SUBMIT"
        return OUTCOME_NAME_TO_ID["CLICK"], "CLICK"

    # 2. Navigation Reversal / Backtracking
    if any(ev in event_types for ev in ["beforeunload", "blur", "pagehide", "unload"]):
        return OUTCOME_NAME_TO_ID["BACKTRACK"], "BACKTRACK"

    # 3. Rapid Scanning / Information Search
    if scroll_count >= rapid_scroll_min_events:
        return OUTCOME_NAME_TO_ID["RAPID_SCROLL"], "RAPID_SCROLL"

    # 4. Cognitive Evaluation / Hover Linger
    if hover_count >= hover_dwell_min_events:
        return OUTCOME_NAME_TO_ID["HOVER_DWELL"], "HOVER_DWELL"

    # Default to IDLE_ABANDON if only ambient movement or idle
    return OUTCOME_NAME_TO_ID["IDLE_ABANDON"], "IDLE_ABANDON"
