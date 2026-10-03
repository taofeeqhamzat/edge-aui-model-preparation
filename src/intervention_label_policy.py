"""
Scripted Intervention-Target Label Policy (Phase B, Task 10.3).

Assigns a deterministic target intervention label to each sequence example based on
the observable outcome (from target_generation) and active UIContext snapshot.

ADR-013 Claim Boundary & Guardrails:
- The label is a scripted policy output, NOT ground truth or human observation.
- The outcome vocabulary is strictly behavioral (no affect, confusion, or cognitive state terms).
- The intervention vocabulary is:
  ['simplify_options', 'highlight_primary_action', 'offer_assistance', 'expand_tooltip', 'no_op'].
- Changing policy configuration produces measurable label shifts for ablation testing.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import yaml
from pathlib import Path

LABEL_POLICY_VERSION: str = "1.0.0"

INTERVENTION_VOCABULARY: Tuple[str, ...] = (
    "simplify_options",
    "highlight_primary_action",
    "offer_assistance",
    "expand_tooltip",
    "no_op",
)

INTERVENTION_TO_ID: Dict[str, int] = {
    name: idx for idx, name in enumerate(INTERVENTION_VOCABULARY)
}

DEFAULT_OUTCOME_TO_INTERVENTION: Dict[str, str] = {
    "HOVER_DWELL": "expand_tooltip",
    "BACKTRACK": "offer_assistance",
    "RAPID_SCROLL": "simplify_options",
    "ABANDON": "offer_assistance",
    "CLICK": "no_op",
    "FORM_SUBMIT": "no_op",
    "NO_OUTCOME": "no_op",
}


def load_policy_config_from_yaml(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Loads intervention policy mapping from config.yaml if present."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    target_gen = data.get("target_generation", {})
    return target_gen.get("outcome_intervention_policy", {})


def derive_intervention_label(
    outcome: str,
    context: Optional[Dict[str, Any]] = None,
    custom_policy: Optional[Dict[str, Any]] = None,
) -> Tuple[str, int]:
    """
    Deterministically maps (outcome, UIContext) -> (intervention_label, intervention_id).

    Policy:
    1. If custom_policy defines an override for outcome, evaluate it.
    2. Context-aware conditioning:
       - If outcome == 'HOVER_DWELL' and context has primaryActionAvailable=True and taskProgress > 0.5:
         Recommend 'highlight_primary_action' to guide completion of current task step.
         The comparison is strictly greater-than: taskProgress == 0.5 falls through to
         'expand_tooltip'. This boundary is observable for exactly one case (T2 step 2 at
         taskProgress == 0.5) and should be treated as a methodology question rather than
         changed silently, because it defines training labels.
       - If outcome == 'HOVER_DWELL' without primary action focus:
         Recommend 'expand_tooltip'.
       - If outcome == 'BACKTRACK' or 'ABANDON':
         Recommend 'offer_assistance'.
       - If outcome == 'RAPID_SCROLL':
         Recommend 'simplify_options'.
       - Otherwise ('CLICK', 'FORM_SUBMIT', 'NO_OUTCOME'):
         Recommend 'no_op'.
    """
    if custom_policy and outcome in custom_policy:
        target = custom_policy[outcome]
        if target in INTERVENTION_TO_ID:
            return target, INTERVENTION_TO_ID[target]

    if outcome == "HOVER_DWELL":
        if context and isinstance(context, dict):
            primary_avail = bool(context.get("primaryActionAvailable"))
            task_progress = float(context.get("taskProgress", 0.0))
            # Strict '>' by design (matches the docstring): taskProgress == 0.5 routes to
            # 'expand_tooltip', not 'highlight_primary_action'. This defines training labels;
            # flag the boundary to the supervisor before ever changing it.
            if primary_avail and task_progress > 0.5:
                return "highlight_primary_action", INTERVENTION_TO_ID["highlight_primary_action"]
        return "expand_tooltip", INTERVENTION_TO_ID["expand_tooltip"]

    target = DEFAULT_OUTCOME_TO_INTERVENTION.get(outcome, "no_op")
    return target, INTERVENTION_TO_ID[target]


def assign_intervention_label(
    outcome: str,
    context: Optional[Dict[str, Any]] = None,
    custom_policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assigns a versioned intervention label dictionary to a sequence example.

    Returns:
        {
            'target_intervention': str,
            'target_intervention_id': int,
            'label_policy_version': str,
            'is_scripted_policy': True
        }
    """
    label_name, label_id = derive_intervention_label(outcome, context, custom_policy)
    return {
        "target_intervention": label_name,
        "target_intervention_id": label_id,
        "label_policy_version": LABEL_POLICY_VERSION,
        "is_scripted_policy": True,
    }


def compute_label_distribution(labels: List[str]) -> Dict[str, Any]:
    """Computes class counts and percentages across the 5-class intervention vocabulary."""
    total = len(labels)
    counts = {name: 0 for name in INTERVENTION_VOCABULARY}
    for lbl in labels:
        if lbl in counts:
            counts[lbl] += 1
        else:
            counts["no_op"] += 1

    percentages = {
        name: (float(count) / total * 100.0) if total > 0 else 0.0
        for name, count in counts.items()
    }

    majority_class = max(counts, key=counts.get) if total > 0 else "no_op"
    majority_percentage = percentages.get(majority_class, 0.0)

    return {
        "total_examples": total,
        "counts": counts,
        "percentages": percentages,
        "majority_class": majority_class,
        "majority_percentage": majority_percentage,
    }
