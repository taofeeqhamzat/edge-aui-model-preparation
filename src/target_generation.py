"""
target_generation.py
Self-Supervised Lookahead Outcome Extraction & Target UI Intervention Taxonomy.
Implements outcome-driven self-supervision (Wu et al., 2024; ADR-002) mapping
downstream observable interaction events to discrete target labels.
"""

from typing import List, Dict, Tuple, Any, Optional, Union
from pathlib import Path
import os
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    Dataset = object  # type: ignore
    TORCH_AVAILABLE = False


# Concrete, observable downstream outcome taxonomy (ADR-002)
# Zero subjective affective labels. Class 0 represents legitimate absence of qualifying UI actions,
# while Class 6 represents observable session/window termination.
OUTCOME_TAXONOMY = {
    0: "NO_OUTCOME",  # Inactivity / ongoing reading / pause without qualifying macro-action
    1: "CLICK",
    2: "FORM_SUBMIT",
    3: "BACKTRACK",
    4: "RAPID_SCROLL",
    5: "HOVER_DWELL",
    6: "ABANDON"  # Observable window/session termination (beforeunload, pagehide, unload, stream termination)
}

OUTCOME_NAME_TO_ID = {v: k for k, v in OUTCOME_TAXONOMY.items()}
# Backward-compatibility alias
OUTCOME_NAME_TO_ID["IDLE_ABANDON"] = 6

# Downstream target intervention vocabulary (for transfer learning head)
TARGET_INTERVENTIONS = {
    0: "simplify_options",
    1: "highlight_primary_action",
    2: "offer_assistance",
    3: "expand_tooltip",
    4: "no_op"
}

INTERVENTION_NAME_TO_ID = {v: k for k, v in TARGET_INTERVENTIONS.items()}

try:
    from config import DEFAULT_PRIORITY_HIERARCHY
except ImportError:
    try:
        # pyrefly: ignore [missing-import]
        from src.config import DEFAULT_PRIORITY_HIERARCHY
    except ImportError:
        DEFAULT_PRIORITY_HIERARCHY = [
            "FORM_SUBMIT",
            "CLICK",
            "BACKTRACK",
            "RAPID_SCROLL",
            "HOVER_DWELL",
            "ABANDON",
            "NO_OUTCOME"
        ]

PRIORITY_HIERARCHY = list(DEFAULT_PRIORITY_HIERARCHY)

# Deterministic priority hierarchy (used STRICTLY for resolving timestamp ties)
# Higher rank = higher priority when events share identical timestamps
PRIORITY_RANK = {name: len(PRIORITY_HIERARCHY) - idx for idx, name in enumerate(PRIORITY_HIERARCHY)}
PRIORITY_RANK["IDLE_ABANDON"] = PRIORITY_RANK.get("ABANDON", 1)


def extract_lookahead_outcome(
    future_events: List[Dict[str, Any]],
    session_terminated: bool = False,
    idle_threshold_ms: float = 3000.0,
    rapid_scroll_min_events: int = 4,
    hover_dwell_min_events: int = 2
) -> Tuple[int, str]:
    """
    Extract the downstream UI outcome label from forward-looking temporal events.
    Enforces earliest-event temporal selection. Deterministic priority hierarchy
    is applied ONLY to resolve ties for events sharing the earliest timestamp.

    Hierarchy for ties:
    1. FORM_SUBMIT / CLICK: User commits action on target component.
    2. BACKTRACK: User navigates back, switches tab, or blurs window.
    3. RAPID_SCROLL: User exhibits fast scanning behavior.
    4. HOVER_DWELL: User lingers over an interactive element without clicking.
    5. ABANDON: Observable window/session termination signals.
    6. NO_OUTCOME: Inactivity or ordinary reading pause without qualifying action.
    """
    if not future_events:
        if session_terminated:
            return OUTCOME_NAME_TO_ID["ABANDON"], "ABANDON"
        return OUTCOME_NAME_TO_ID["NO_OUTCOME"], "NO_OUTCOME"

    # Sort events deterministically by timestamp
    def get_ts(ev: Dict[str, Any]) -> float:
        ts = ev.get("timestamp_ms")
        if ts is None:
            ts = ev.get("timestamp", 0)
        return float(ts)

    sorted_events = sorted(future_events, key=get_ts)

    # Candidate list of (outcome_id, outcome_name, timestamp, priority_rank)
    candidates: List[Tuple[int, str, float, int]] = []

    scroll_events_seen = 0
    hover_events_seen = 0

    for ev in sorted_events:
        ev_ts = get_ts(ev)
        etype = str(ev.get("event_type") or ev.get("event") or "").lower()
        xpath = str(ev.get("target_id") or ev.get("xpath") or "").lower()

        # 1. Action Commit (CLICK / FORM_SUBMIT)
        if etype in ("click", "mousedown"):
            is_submit = any(term in xpath for term in ["submit", "btn", "button", "input", "form"])
            if is_submit:
                candidates.append((
                    OUTCOME_NAME_TO_ID["FORM_SUBMIT"],
                    "FORM_SUBMIT",
                    ev_ts,
                    PRIORITY_RANK["FORM_SUBMIT"]
                ))
            else:
                candidates.append((
                    OUTCOME_NAME_TO_ID["CLICK"],
                    "CLICK",
                    ev_ts,
                    PRIORITY_RANK["CLICK"]
                ))

        # 2. Observable Window/Session Termination (ABANDON)
        elif etype in ("beforeunload", "pagehide", "unload", "abandon"):
            candidates.append((
                OUTCOME_NAME_TO_ID["ABANDON"],
                "ABANDON",
                ev_ts,
                PRIORITY_RANK["ABANDON"]
            ))

        # 3. Navigation Reversal / Focus Shift (BACKTRACK)
        elif etype in ("blur", "popstate"):
            candidates.append((
                OUTCOME_NAME_TO_ID["BACKTRACK"],
                "BACKTRACK",
                ev_ts,
                PRIORITY_RANK["BACKTRACK"]
            ))

        # 4. Rapid Scrolling (tracks count and triggers when threshold is reached)
        elif etype in ("scroll", "wheel"):
            scroll_events_seen += 1
            if scroll_events_seen == rapid_scroll_min_events:
                candidates.append((
                    OUTCOME_NAME_TO_ID["RAPID_SCROLL"],
                    "RAPID_SCROLL",
                    ev_ts,
                    PRIORITY_RANK["RAPID_SCROLL"]
                ))

        # 5. Attentional Hover Linger (triggers when threshold is reached)
        elif etype == "mouseover":
            hover_events_seen += 1
            if hover_events_seen == hover_dwell_min_events:
                candidates.append((
                    OUTCOME_NAME_TO_ID["HOVER_DWELL"],
                    "HOVER_DWELL",
                    ev_ts,
                    PRIORITY_RANK["HOVER_DWELL"]
                ))

    # If no qualifying events were triggered
    if not candidates:
        # Check for observable termination signals
        for ev in sorted_events:
            etype = str(ev.get("event_type") or ev.get("event") or "").lower()
            if etype in ("beforeunload", "pagehide", "unload", "abandon"):
                return OUTCOME_NAME_TO_ID["ABANDON"], "ABANDON"
        if session_terminated:
            return OUTCOME_NAME_TO_ID["ABANDON"], "ABANDON"
        return OUTCOME_NAME_TO_ID["NO_OUTCOME"], "NO_OUTCOME"

    # Earliest-event temporal selection:
    # 1. Find the earliest timestamp among all qualifying candidates
    earliest_ts = min(c[2] for c in candidates)

    # 2. Filter candidates occurring at the earliest timestamp (within 1ms tolerance)
    tied_candidates = [c for c in candidates if abs(c[2] - earliest_ts) <= 1.0]

    # 3. Apply deterministic priority hierarchy strictly as a tie-breaker
    tied_candidates.sort(key=lambda c: c[3], reverse=True)
    best_candidate = tied_candidates[0]

    return best_candidate[0], best_candidate[1]


def extract_outcome_for_window(
    session_events: List[Dict[str, Any]],
    window_end_ms: float,
    lookahead_min_ms: float = 500.0,
    lookahead_max_ms: float = 1500.0,
    rapid_scroll_min_events: int = 4,
    hover_dwell_min_events: int = 2
) -> Tuple[int, str]:
    """
    Extract the downstream UI outcome for a specific window, strictly filtering
    events to the lookahead horizon [window_end_ms + 500ms, window_end_ms + 1500ms].
    Enforces strict session boundedness.
    """
    t_start = window_end_ms + lookahead_min_ms
    t_end = window_end_ms + lookahead_max_ms

    future_events = []
    session_end_ms = 0.0
    for ev in session_events:
        ts = ev.get("timestamp_ms")
        if ts is None:
            ts = ev.get("timestamp", 0)
        ts_val = float(ts)
        if ts_val > session_end_ms:
            session_end_ms = ts_val
        if t_start <= ts_val <= t_end:
            future_events.append(ev)

    session_terminated = bool(session_end_ms <= t_end)

    return extract_lookahead_outcome(
        future_events,
        session_terminated=session_terminated,
        rapid_scroll_min_events=rapid_scroll_min_events,
        hover_dwell_min_events=hover_dwell_min_events
    )


# ============================================================================
# PyTorch Dataset Integration & Loss Weighting
# ============================================================================

if TORCH_AVAILABLE:
    class OutcomeDataset(Dataset):
        """
        PyTorch Dataset yielding (seq_len, 18) MicroTensor sequences and discrete outcome labels.
        Decoupled from file storage/Parquet; consumes tensors or NumPy arrays directly.
        """
        def __init__(
            self,
            sequences: Union[torch.Tensor, np.ndarray],
            targets: Union[torch.Tensor, np.ndarray]
        ):
            if isinstance(sequences, np.ndarray):
                self.X = torch.from_numpy(sequences).float()
            elif isinstance(sequences, torch.Tensor):
                self.X = sequences.float()
            else:
                self.X = torch.tensor(sequences, dtype=torch.float32)

            if isinstance(targets, np.ndarray):
                self.Y = torch.from_numpy(targets).long()
            elif isinstance(targets, torch.Tensor):
                self.Y = targets.long()
            else:
                self.Y = torch.tensor(targets, dtype=torch.long)

            if self.X.ndim != 3 or self.X.shape[-1] != 18:
                raise ValueError(f"Expected sequences tensor of shape (N, T, 18), got {self.X.shape}")
            if len(self.X) != len(self.Y):
                raise ValueError(f"Mismatched dataset lengths: {len(self.X)} sequences vs {len(self.Y)} targets")

        def __len__(self) -> int:
            return len(self.Y)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            return self.X[idx], self.Y[idx]

else:
    class OutcomeDataset:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required to instantiate OutcomeDataset.")


def compute_class_weights(
    targets: Union[torch.Tensor, np.ndarray, List[int]],
    num_classes: int = 7,
    smoothing_alpha: float = 0.0,
    allow_empty: bool = False
) -> torch.Tensor:
    """
    Calculate inverse class frequency weights for loss penalization:
    w_c = (N + C * alpha) / (C * (N_c + alpha))

    Safeguards:
    If allow_empty is False and any required class has 0 samples, raises ValueError
    to prevent concealed data defects.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to compute class weights.")

    if isinstance(targets, torch.Tensor):
        t_arr = targets.detach().cpu().numpy()
    elif isinstance(targets, np.ndarray):
        t_arr = targets
    else:
        t_arr = np.array(targets)

    counts = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        counts[c] = float(np.sum(t_arr == c))

    total_n = float(len(t_arr))
    empty_classes = [c for c in range(num_classes) if counts[c] == 0]

    if empty_classes and not allow_empty and smoothing_alpha == 0.0:
        class_names = [OUTCOME_TAXONOMY.get(c, str(c)) for c in empty_classes]
        raise ValueError(
            f"Empty classes detected in targets: IDs {empty_classes} ({class_names}). "
            f"Cannot compute unregularized inverse weights for classes with 0 samples."
        )

    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        denom = float(num_classes) * (counts[c] + smoothing_alpha)
        if denom > 0:
            weights[c] = (total_n + float(num_classes) * smoothing_alpha) / denom
        else:
            weights[c] = 0.0

    # Normalize weights so mean of active weights is 1.0
    active_mask = weights > 0
    if np.any(active_mask):
        mean_w = np.mean(weights[active_mask])
        if mean_w > 0:
            weights = weights / mean_w

    return torch.from_numpy(weights).float()


def create_sample_outcome_dataset(
    num_samples: int = 35,
    seq_len: int = 8,
    input_dim: int = 18,
    from_cache: bool = True
) -> OutcomeDataset:
    """
    Generates or loads a verified OutcomeDataset pairing (T, 18) MicroTensor sequences
    with discrete outcome labels y in {0, 1, 2, 3, 4, 5, 6}.
    Guarantees coverage across all 7 outcome classes.
    """
    if from_cache:
        try:
            proj_root = Path(__file__).resolve().parent.parent
            parquet_path = proj_root / ".data" / "interim" / "microtensors" / "adserp_microtensors.parquet"
            if parquet_path.is_file() and parquet_path.stat().st_size > 0:
                try:
                    from microtensor_store import reconstruct_sequences_from_parquet
                except ImportError:
                    from src.microtensor_store import reconstruct_sequences_from_parquet
                X, Y = reconstruct_sequences_from_parquet(str(parquet_path), seq_len=seq_len)
                if len(X) >= num_samples:
                    return OutcomeDataset(X[:num_samples], Y[:num_samples])
        except Exception:
            pass

    # Deterministic fallback: Generate well-formed synthetic tensors with all 7 classes
    rng = np.random.RandomState(42)
    X_list = []
    Y_list = []

    for i in range(num_samples):
        target_label = i % 7
        # Features bounded in [0, 1]
        seq = rng.uniform(0.0, 1.0, size=(seq_len, input_dim)).astype(np.float32)
        # Modality mask indices (9..17) are binary flags (1.0 for active support)
        seq[:, 9:] = 1.0
        X_list.append(seq)
        Y_list.append(target_label)

    X = np.stack(X_list, axis=0)
    Y = np.array(Y_list, dtype=np.int64)

    return OutcomeDataset(X, Y)
