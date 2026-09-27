"""
UI Context Ingestion and R⁶ Context Encoding (Phase B, Task 10.2).

Binds the offline model-preparation pipeline to the edge-aui-framework runtime encoding
(edge-aui-framework/src/types/contextVector.ts). Produces the mandatory 6-dimensional UI context
vector expected by TargetInterventionHead.

Encoding Specification:
| Index | Field              | Encoding                                                        |
|-------|--------------------|-----------------------------------------------------------------|
| 0     | route              | Stable index into ROUTE_VOCABULARY, normalized by route count     |
| 1     | primaryAction      | 1.0 when a primary action is available, else 0.0                 |
| 2     | helpAvailable      | 1.0 when contextual help/tooltip is available, else 0.0          |
| 3     | expandable         | 1.0 when an expandable/accordion surface is present, else 0.0    |
| 4     | taskProgress       | current step index / total steps (0.0 when no task is active)     |
| 5     | actionAvailability | available action count normalized by ACTION_VOCABULARY size       |

All components are strictly bounded in [0.0, 1.0].
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

CONTEXT_VECTOR_DIM: int = 6

ROUTE_VOCABULARY: Tuple[str, ...] = (
    "Overview",
    "Analytics",
    "Reports",
    "Customers",
    "Settings",
)

ACTION_VOCABULARY: Tuple[str, ...] = (
    "click",
    "change",
    "toggle",
    "hover",
    "select",
    "input",
    "focus",
)

TASK_STEP_COUNTS: Dict[str, int] = {
    "T1": 4,
    "T2": 2,
    "T3": 4,
}


@dataclass
class ContextValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)


def validate_context_vector(vector: np.ndarray) -> ContextValidationResult:
    """Validates that an encoded context vector satisfies the TargetInterventionHead contract."""
    errors: List[str] = []

    if not isinstance(vector, np.ndarray):
        errors.append(f"Context vector must be a numpy ndarray, got {type(vector)}")
        return ContextValidationResult(valid=False, errors=errors)

    if vector.shape != (CONTEXT_VECTOR_DIM,):
        errors.append(
            f"Context vector must have shape ({CONTEXT_VECTOR_DIM},), got {vector.shape}"
        )

    for i, val in enumerate(vector):
        if not np.isfinite(val):
            errors.append(f"Context vector element {i} is not finite: {val}")
        elif val < 0.0 or val > 1.0:
            errors.append(f"Context vector element {i} out of [0, 1] bounds: {val}")

    return ContextValidationResult(valid=len(errors) == 0, errors=errors)


def encode_ui_context(context: Optional[Dict[str, Any]]) -> np.ndarray:
    """
    Encodes a UIContext dictionary into the 6-dimensional context vector expected
    by TargetInterventionHead.

    Deterministic and pure. Matches edge-aui-framework/src/types/contextVector.ts.
    Raises ValueError if context is None or not a dict.
    """
    if context is None or not isinstance(context, dict):
        raise ValueError(
            "UIContext dictionary is required. Missing context cannot be imputed with zeros "
            "(plan 1 Task 10.2 / ADR-001)."
        )

    vector = np.zeros(CONTEXT_VECTOR_DIM, dtype=np.float32)

    # 0. Route index, normalized by vocabulary size
    route = context.get("route")
    if route in ROUTE_VOCABULARY:
        route_index = ROUTE_VOCABULARY.index(route)
        vector[0] = float(route_index) / len(ROUTE_VOCABULARY)
    else:
        vector[0] = 0.0

    # 1-3. Capability flags
    vector[1] = 1.0 if bool(context.get("primaryActionAvailable")) else 0.0
    vector[2] = 1.0 if bool(context.get("helpAvailable")) else 0.0
    vector[3] = 1.0 if bool(context.get("expandable")) else 0.0

    # 4. Task progress derived from current step index
    task_id = context.get("taskId")
    task_step_id = context.get("taskStepId")

    if task_id and task_step_id and isinstance(task_step_id, str):
        step_count = TASK_STEP_COUNTS.get(str(task_id))
        if step_count and step_count > 0:
            tokens = task_step_id.split("-")
            last_token = tokens[-1] if tokens else ""
            try:
                step_number = int(last_token)
                if step_number > 0:
                    progress = float(step_number - 1) / step_count
                    vector[4] = min(1.0, max(0.0, progress))
            except (ValueError, TypeError):
                vector[4] = 0.0

    # 5. Breadth of currently available actions
    raw_actions = context.get("availableActions")
    if raw_actions and isinstance(raw_actions, (list, tuple, set)):
        unique_actions = set(raw_actions)
        action_count = len(unique_actions)
        vector[5] = min(1.0, float(action_count) / len(ACTION_VOCABULARY))
    else:
        vector[5] = 0.0

    return vector


@dataclass
class ContextBatchExtractionStats:
    total_examples: int = 0
    encoded_examples: int = 0
    excluded_missing_context: int = 0
    excluded_invalid_vector: int = 0


def encode_context_batch(
    contexts: Sequence[Optional[Dict[str, Any]]],
) -> Tuple[np.ndarray, List[int], ContextBatchExtractionStats]:
    """
    Encodes a sequence of UIContext snapshots for a batch of sequence examples.

    Examples with missing or malformed context are EXCLUDED and counted in the returned
    statistics, rather than imputed with zeros.

    Returns:
        (encoded_matrix, retained_indices, stats)
    """
    stats = ContextBatchExtractionStats(total_examples=len(contexts))
    encoded_vectors: List[np.ndarray] = []
    retained_indices: List[int] = []

    for idx, ctx in enumerate(contexts):
        if ctx is None or not isinstance(ctx, dict):
            stats.excluded_missing_context += 1
            continue

        try:
            vec = encode_ui_context(ctx)
            val_res = validate_context_vector(vec)
            if not val_res.valid:
                stats.excluded_invalid_vector += 1
                continue

            encoded_vectors.append(vec)
            retained_indices.append(idx)
            stats.encoded_examples += 1
        except Exception:
            stats.excluded_missing_context += 1

    if encoded_vectors:
        result_matrix = np.stack(encoded_vectors, axis=0).astype(np.float32)
    else:
        result_matrix = np.empty((0, CONTEXT_VECTOR_DIM), dtype=np.float32)

    return result_matrix, retained_indices, stats
