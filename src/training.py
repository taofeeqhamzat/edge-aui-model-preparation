"""
training.py
Foundational PyTorch training loop for the GRU model (Edge-AUI Framework).
"""

import os
import sys
import argparse
from typing import Optional, Dict, Any, List, Union, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from preprocessing import MicroInteractionSequenceDataset, FEATURE_NAMES, MICROTENSOR_DIM
except ImportError:
    # pyrefly: ignore [missing-import]
    from src.preprocessing import MicroInteractionSequenceDataset, FEATURE_NAMES, MICROTENSOR_DIM

try:
    from data_manager import find_project_root, is_colab, is_kaggle
except ImportError:
    try:
        # pyrefly: ignore [missing-import]
        from src.data_manager import find_project_root, is_colab, is_kaggle
    except ImportError:
        find_project_root = lambda: os.getcwd()
        is_colab = lambda: False
try:
    from target_generation import (
        compute_class_weights,
        compute_class_weight_diagnostics,
        OUTCOME_TAXONOMY
    )
except ImportError:
    try:
        # pyrefly: ignore [missing-import]
        from src.target_generation import (
            compute_class_weights,
            compute_class_weight_diagnostics,
            OUTCOME_TAXONOMY
        )
    except ImportError:
        compute_class_weights = None  # type: ignore
        compute_class_weight_diagnostics = None  # type: ignore
        OUTCOME_TAXONOMY = {}

# Default Configuration
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".data", "raw"))
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3
HIDDEN_DIM = 64
NUM_LAYERS = 2
NUM_CLASSES = 7  # NO_OUTCOME, CLICK, FORM_SUBMIT, BACKTRACK, RAPID_SCROLL, HOVER_DWELL, ABANDON


def get_device(device_override: Optional[str] = None) -> torch.device:
    """
    Select optimal execution provider (CUDA > MPS > CPU).
    In hosted environments like Colab/Kaggle, CUDA is preferred.
    """
    if device_override:
        return torch.device(device_override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FoundationOutcomeHead(nn.Module):
    """
    Modular classification head mapping latent behavioral representation h_T in R^64
    to the 7 foundational outcome classes (ADR-002, ADR-003).
    """
    def __init__(self, hidden_dim: int = 64, num_classes: int = 7):
        super(FoundationOutcomeHead, self).__init__()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, h_T: torch.Tensor) -> torch.Tensor:
        return self.fc(h_T)


class TargetInterventionHead(nn.Module):
    """
    Modular projection head mapping latent behavioral representation h_T in R^64
    and mandatory target UI context vector C in R^C_dim to the 5 target UI intervention actions (ADR-003).
    Conditioned as [h_T, C] -> 5 actions via MLP. Context vector C is strictly mandatory.
    Intervention classes represent system adaptation decisions rather than user actions.
    Interface updated in Task 4.2; training deferred to Phase 5.
    """
    def __init__(self, hidden_dim: int = 64, context_dim: int = 6, num_classes: int = 5):
        super(TargetInterventionHead, self).__init__()
        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive, got {context_dim}. UI context vector is mandatory.")
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_classes = num_classes
        in_features = hidden_dim + context_dim
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, h_T: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Forward pass requiring both latent state h_T and UI context tensor."""
        if context is None:
            raise ValueError("UI context tensor is mandatory for TargetInterventionHead.")
        features = torch.cat([h_T, context], dim=-1)
        return self.fc(features)


class EdgeAUIGRU(nn.Module):
    """
    Decoupled lightweight Gated Recurrent Unit (GRU) for edge inference (ADR-003).
    The recurrent backbone encodes kinematics and temporal dynamics into an
    interface-agnostic latent representation h_T in R^64, solving target Out-of-Vocabulary (OOV).
    Modular classification heads project h_T onto task-specific vocabularies.
    """
    def __init__(
        self,
        input_dim: int = MICROTENSOR_DIM,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 7,
        head: Optional[nn.Module] = None
    ):
        super(EdgeAUIGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Recurrent GRU backbone (theta_base)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

        # Modular classification head (theta_head)
        if head is not None:
            self.head = head
        else:
            self.head = FoundationOutcomeHead(hidden_dim, num_classes)

    @property
    def fc(self) -> nn.Module:
        """Backward-compatibility accessor for existing self.fc references."""
        if hasattr(self.head, "fc"):
            return self.head.fc
        return self.head

    def attach_head(self, head: nn.Module) -> None:
        """
        Replace the modular classification head without reconstructing the GRU backbone.
        """
        self.head = head

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        return_latent: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through recurrent backbone and active modular head.
        Extracts terminal recurrent representation h_T = out[:, -1, :].
        (For a unidirectional GRU, the final timestep output is mathematically
        equivalent to the final layer hidden state h_n[-1]).
        """
        out, _ = self.gru(x)
        # Latent behavioral representation h_T in R^(batch_size, hidden_dim)
        h_T = out[:, -1, :]
        if isinstance(self.head, TargetInterventionHead):
            if context is None:
                raise ValueError("UI context tensor is mandatory when using TargetInterventionHead.")
            logits = self.head(h_T, context=context)
        elif context is not None:
            logits = self.head(h_T, context=context)
        else:
            logits = self.head(h_T)

        if return_latent:
            return logits, h_T
        return logits

    def backbone_parameters(self):
        """Return parameters of the recurrent GRU backbone (theta_base)."""
        return self.gru.parameters()

    def head_parameters(self):
        """Return parameters of the modular classification head (theta_head)."""
        return self.head.parameters()

    def freeze_backbone(self) -> None:
        """Freeze all recurrent backbone parameters (Strict Freezing)."""
        for p in self.gru.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all recurrent backbone parameters (Full Fine-Tuning)."""
        for p in self.gru.parameters():
            p.requires_grad = True

    def unfreeze_terminal_layer(self) -> None:
        """
        Freeze lower GRU layers and unfreeze exclusively the terminal GRU layer (Partial Fine-Tuning).
        """
        terminal_layer_idx = self.num_layers - 1
        for name, p in self.gru.named_parameters():
            if f"_l{terminal_layer_idx}" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False


def load_foundation_dataset(
    data_dir: Optional[str] = None,
    max_sequences: Optional[int] = None,
    split: str = "train",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_seed: int = 42,
    enforce_temporal_continuity: bool = False
) -> MicroInteractionSequenceDataset:
    """
    Load model-ready (N, 8, 18) sequence tensors from flattened Parquet storage,
    enforcing leak-free user/session splitting BEFORE sequence construction.
    """
    from pathlib import Path
    root = Path(find_project_root())
    interim_micro = root / ".data" / "interim" / "microtensors" / "adserp_microtensors.parquet"

    # Auto-extract if interim microtensors do not yet exist
    if not interim_micro.is_file():
        canon_path = root / ".data" / "canonical" / "adserp" / "data.parquet"
        if not canon_path.is_file():
            try:
                from data import convert_adserp_to_canonical
            except ImportError:
                # pyrefly: ignore [missing-import]
                from src.data import convert_adserp_to_canonical
            convert_adserp_to_canonical()
        try:
            from microtensor_store import extract_microtensors_from_canonical
        except ImportError:
            # pyrefly: ignore [missing-import]
            from src.microtensor_store import extract_microtensors_from_canonical
        extract_microtensors_from_canonical(str(canon_path), str(interim_micro))

    try:
        from microtensor_store import split_users_leak_free, reconstruct_sequences_from_parquet, SplitResult
    except ImportError:
        # pyrefly: ignore [missing-import]
        from src.microtensor_store import split_users_leak_free, reconstruct_sequences_from_parquet, SplitResult

    split_res = split_users_leak_free(
        str(interim_micro),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_seed=random_seed
    )

    X, Y = reconstruct_sequences_from_parquet(
        str(interim_micro),
        seq_len=8,
        split_result=split_res,
        split_partition=split,
        enforce_temporal_continuity=enforce_temporal_continuity
    )

    if max_sequences and len(X) > max_sequences:
        X = X[:max_sequences]
        Y = Y[:max_sequences]

    return MicroInteractionSequenceDataset(X, Y)


def evaluate_majority_baseline(
    train_dataset: MicroInteractionSequenceDataset,
    eval_dataset: MicroInteractionSequenceDataset,
    num_classes: int = NUM_CLASSES
) -> Dict[str, Any]:
    """
    Evaluate a majority-class baseline on the evaluation partition.
    Yields baseline accuracy, Macro-F1, Weighted-F1, and per-class metrics.
    Accuracy alone is deeply misleading on imbalanced distributions; Macro-F1 reveals
    the failure of a trivial majority predictor on minority classes.
    """
    import numpy as np

    if hasattr(train_dataset, "Y"):
        train_y = train_dataset.Y.numpy() if hasattr(train_dataset.Y, "numpy") else np.asarray(train_dataset.Y)
    else:
        train_y = train_dataset.numpy() if hasattr(train_dataset, "numpy") else np.asarray(train_dataset)

    if hasattr(eval_dataset, "Y"):
        eval_y = eval_dataset.Y.numpy() if hasattr(eval_dataset.Y, "numpy") else np.asarray(eval_dataset.Y)
    else:
        eval_y = eval_dataset.numpy() if hasattr(eval_dataset, "numpy") else np.asarray(eval_dataset)

    # Determine majority class from training data only
    train_counts = np.bincount(train_y, minlength=num_classes)
    majority_class = int(np.argmax(train_counts))
    majority_name = OUTCOME_TAXONOMY.get(majority_class, str(majority_class))

    eval_n = len(eval_y)
    if eval_n == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}

    eval_counts = np.bincount(eval_y, minlength=num_classes)
    correct = int(eval_counts[majority_class])
    acc = float(correct / eval_n)

    per_class_f1 = {}
    f1_list = []
    weighted_f1_sum = 0.0

    for c in range(num_classes):
        c_name = OUTCOME_TAXONOMY.get(c, str(c))
        support = int(eval_counts[c])
        if c == majority_class:
            tp = support
            fp = eval_n - support
            fn = 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = 1.0 if support > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        else:
            f1 = 0.0

        per_class_f1[c_name] = float(f1)
        if support > 0:
            f1_list.append(f1)
            weighted_f1_sum += f1 * support

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    weighted_f1 = float(weighted_f1_sum / eval_n) if eval_n > 0 else 0.0

    return {
        "majority_class_id": majority_class,
        "majority_class_name": majority_name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": per_class_f1,
        "support": {OUTCOME_TAXONOMY.get(c, str(c)): int(eval_counts[c]) for c in range(num_classes)}
    }


def compute_ranking_diagnostics(
    logits: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    ks: Tuple[int, ...] = (1, 3)
) -> Dict[str, Any]:
    """
    Compute secondary ranking diagnostics: Hit Rate@K (Top-K accuracy) and Mean Reciprocal Rank (MRR).

    Methodological Note (ADR-003, Task 4.2):
    Ranking diagnostics (HR@K, MRR) evaluate relative score ordering among downstream candidate
    outcomes. In accordance with the project's interaction-outcome framing, these serve as auxiliary
    ranking diagnostics rather than primary classification criteria. Macro-F1 and per-class metrics
    remain the primary criteria.

    Parameters:
        logits: Model output logits or predicted probabilities of shape (N, num_classes).
        targets: Ground-truth class labels of shape (N,).
        ks: Tuple of K thresholds for Hit Rate@K (default: (1, 3)).

    Returns:
        Dict containing hr_at_k for each k in ks, mrr, and descriptive diagnostic note.
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    logits = np.asarray(logits)
    targets = np.asarray(targets)
    n_samples = len(targets)

    if n_samples == 0:
        return {
            "hr@1": 0.0,
            "hr@3": 0.0,
            "mrr": 0.0,
            "n_samples": 0,
            "diagnostic_note": "Secondary ranking diagnostics evaluated over 0 samples."
        }

    # Sort logits descending: shape (N, num_classes)
    sorted_indices = np.argsort(-logits, axis=1)

    hr_results = {}
    for k in ks:
        # Check if true target is within top-k sorted predictions
        hits = np.any(sorted_indices[:, :k] == targets[:, None], axis=1)
        hr_results[f"hr@{k}"] = float(np.mean(hits))

    # Mean Reciprocal Rank (MRR)
    # rank is 1-indexed position of targets[i] in sorted_indices[i]
    ranks = np.where(sorted_indices == targets[:, None])[1] + 1
    mrr = float(np.mean(1.0 / ranks))

    return {
        **hr_results,
        "mrr": mrr,
        "n_samples": int(n_samples),
        "diagnostic_note": "Secondary ranking diagnostics (HR@K, MRR) evaluate relative outcome ordering rather than primary classification criteria."
    }


def evaluate_foundation_model(
    model: nn.Module,
    eval_dataset: Union[MicroInteractionSequenceDataset, DataLoader],
    criterion: Optional[nn.Module] = None,
    device: Optional[Union[str, torch.device]] = None,
    batch_size: int = BATCH_SIZE,
    num_classes: int = NUM_CLASSES,
    train_dataset: Optional[MicroInteractionSequenceDataset] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of EdgeAUIGRU with FoundationOutcomeHead on interaction sequences.

    Primary Metrics:
      - Macro-F1: Unweighted mean of per-class F1 scores across classes with support.
      - Weighted-F1: Support-weighted mean of per-class F1 scores.
      - Per-Class Metrics: Precision, Recall, F1, and Support per outcome class.
      - Confusion Matrix: Empirical (num_classes x num_classes) transition matrix.
      - Majority-Class Baseline Comparison: Direct delta comparison against majority predictor.

    Secondary Diagnostics:
      - Hit Rate@1, Hit Rate@3, and Mean Reciprocal Rank (MRR), explicitly documented
        as secondary ranking diagnostics rather than primary classification criteria.
    """
    target_device = get_device(device) if device is None or isinstance(device, str) else device
    model.eval()
    model.to(target_device)

    if isinstance(eval_dataset, DataLoader):
        dataloader = eval_dataset
        total_eval_samples = len(eval_dataset.dataset) if hasattr(eval_dataset, "dataset") else 0
    else:
        dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        total_eval_samples = len(eval_dataset)

    if total_eval_samples == 0:
        if verbose:
            print("[Evaluation] Warning: Evaluation dataset is empty.")
        return {
            "eval_loss": 0.0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "per_class": {},
            "confusion_matrix": [],
            "ranking_diagnostics": compute_ranking_diagnostics(np.empty((0, num_classes)), np.empty(0)),
            "majority_baseline": None,
            "n_samples": 0
        }

    total_loss = 0.0
    total_samples = 0
    all_logits_list = []
    all_targets_list = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(target_device)
            batch_y = batch_y.to(target_device)

            outputs = model(batch_x)
            if criterion is not None:
                loss = criterion(outputs, batch_y)
                total_loss += loss.item() * batch_x.size(0)

            total_samples += batch_y.size(0)
            all_logits_list.append(outputs.detach().cpu().numpy())
            all_targets_list.append(batch_y.detach().cpu().numpy())

    logits_arr = np.concatenate(all_logits_list, axis=0)
    targets_arr = np.concatenate(all_targets_list, axis=0)
    preds_arr = np.argmax(logits_arr, axis=1)

    avg_loss = total_loss / max(total_samples, 1) if criterion is not None else 0.0
    accuracy = float(np.mean(preds_arr == targets_arr)) * 100.0

    # Empirical Confusion Matrix: shape (num_classes, num_classes)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(targets_arr, preds_arr):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    # Per-Class Precision, Recall, F1, and Support
    per_class_metrics = {}
    f1_list = []
    weighted_f1_sum = 0.0

    for c in range(num_classes):
        c_name = OUTCOME_TAXONOMY.get(c, str(c))
        tp = int(cm[c, c])
        fp = int(np.sum(cm[:, c]) - tp)
        fn = int(np.sum(cm[c, :]) - tp)
        support = int(np.sum(cm[c, :]))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class_metrics[c_name] = {
            "class_id": c,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support
        }
        if support > 0:
            f1_list.append(f1)
            weighted_f1_sum += f1 * support

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    weighted_f1 = float(weighted_f1_sum / total_samples) if total_samples > 0 else 0.0

    # Secondary Ranking Diagnostics
    ranking_diagnostics = compute_ranking_diagnostics(logits_arr, targets_arr, ks=(1, 3))

    # Majority Baseline Comparison
    baseline_metrics = None
    if train_dataset is not None:
        if isinstance(eval_dataset, MicroInteractionSequenceDataset):
            baseline_metrics = evaluate_majority_baseline(train_dataset, eval_dataset, num_classes=num_classes)
        else:
            baseline_metrics = evaluate_majority_baseline(train_dataset, targets_arr, num_classes=num_classes)

    if verbose:
        print("\n" + "=" * 80)
        print("           FOUNDATION GRU OUTCOME CLASSIFICATION EVALUATION REPORT")
        print("=" * 80)
        print(f"{'Class Name':<18} {'Class ID':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 80)
        for c_name, m in per_class_metrics.items():
            print(f"{c_name:<18} {m['class_id']:<10} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {m['support']:<10}")
        print("-" * 80)
        print(f"Overall Accuracy:  {accuracy:.2f}% ({int(np.sum(preds_arr == targets_arr))}/{total_samples})")
        print(f"Macro-F1:          {macro_f1:.4f}  (Primary unweighted metric across classes with support)")
        print(f"Weighted-F1:       {weighted_f1:.4f}")
        if criterion is not None:
            print(f"Evaluation Loss:   {avg_loss:.4f}")

        print("\n[Secondary Ranking Diagnostics - Auxiliary Diagnostics, Not Primary Criteria]")
        print(f" - Hit Rate @ 1 (HR@1):           {ranking_diagnostics['hr@1'] * 100:.2f}%")
        print(f" - Hit Rate @ 3 (HR@3):           {ranking_diagnostics['hr@3'] * 100:.2f}%")
        print(f" - Mean Reciprocal Rank (MRR):     {ranking_diagnostics['mrr']:.4f}")

        if baseline_metrics is not None:
            print("\n[Majority-Class Baseline Comparison]")
            print(f" - Majority Class:                {baseline_metrics['majority_class_name']} (ID {baseline_metrics['majority_class_id']})")
            print(f" - Baseline Accuracy:             {baseline_metrics['accuracy'] * 100:.2f}%")
            print(f" - Baseline Macro-F1:             {baseline_metrics['macro_f1']:.4f}")
            delta_macro = macro_f1 - baseline_metrics['macro_f1']
            print(f" - Model Macro-F1 vs Baseline:    {macro_f1:.4f} vs {baseline_metrics['macro_f1']:.4f} (Delta: {'+' if delta_macro >= 0 else ''}{delta_macro:.4f})")
        print("=" * 80 + "\n")

    return {
        "eval_loss": float(avg_loss),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "ranking_diagnostics": ranking_diagnostics,
        "majority_baseline": baseline_metrics,
        "n_samples": int(total_samples)
    }


def train_foundation_model(
    data_dir: Optional[str] = None,
    hf_repo_id: str = "T40/edge-aui-framework-data",
    hf_token: Optional[str] = None,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    hidden_dim: int = HIDDEN_DIM,
    num_layers: int = NUM_LAYERS,
    max_sequences: Optional[int] = None,
    max_files_per_dataset: Optional[int] = None,
    device: Optional[str] = None,
    output_dir: Optional[str] = None,
    class_weights: Optional[Union[torch.Tensor, str]] = "smoothed",
    evaluate_val: bool = True,
    val_dataset: Optional[MicroInteractionSequenceDataset] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Foundational training loop for EdgeAUIGRU with FoundationOutcomeHead.
    Supports session-bounded leak-free training/validation partitions and empty-class safe loss weighting.

    Methodological Invariant:
    If class_weights is 'inverse' or 'smoothed', weights are calculated strictly from the
    training partition dataset.Y (never from global or validation targets).
    """
    target_data_dir = data_dir if data_dir is not None else DATA_DIR
    target_device = get_device(device)

    if verbose:
        print(f"[Training] Target execution device: {target_device}")
        print(f"[Training] Loading interaction sequences from Parquet storage (train split)...")

    dataset = load_foundation_dataset(
        data_dir=target_data_dir,
        max_sequences=max_sequences,
        split="train"
    )

    if len(dataset) == 0:
        print(f"[Training] Warning: No valid sequences found in {target_data_dir}. Ensure data is populated.")
        return {"model": None, "history": {"loss": [], "accuracy": []}}

    # Load validation split if requested
    if val_dataset is None and evaluate_val:
        if verbose:
            print(f"[Training] Loading interaction sequences from Parquet storage (val split)...")
        max_val = max(1, int(max_sequences * 0.2)) if max_sequences else None
        try:
            val_dataset = load_foundation_dataset(
                data_dir=target_data_dir,
                max_sequences=max_val,
                split="val"
            )
        except Exception as e:
            if verbose:
                print(f"[Training] Notice: Could not load validation dataset ({e}). Proceeding without validation split.")
            val_dataset = None

    # Resolve class weights strictly on training partition targets
    weights_tensor: Optional[torch.Tensor] = None
    weight_diagnostics: Optional[Dict[str, Any]] = None

    if isinstance(class_weights, str):
        if compute_class_weights is not None:
            alpha = 100.0 if class_weights == "smoothed" else 0.0
            weights_tensor = compute_class_weights(
                dataset.Y,
                num_classes=NUM_CLASSES,
                smoothing_alpha=alpha,
                allow_empty=True
            )
            if compute_class_weight_diagnostics is not None:
                weight_diagnostics = compute_class_weight_diagnostics(
                    dataset.Y,
                    num_classes=NUM_CLASSES,
                    smoothing_alpha=alpha,
                    allow_empty=True
                )
        else:
            raise RuntimeError("compute_class_weights not available.")
    elif isinstance(class_weights, torch.Tensor):
        weights_tensor = class_weights
        if compute_class_weight_diagnostics is not None:
            weight_diagnostics = compute_class_weight_diagnostics(
                dataset.Y,
                num_classes=NUM_CLASSES,
                allow_empty=True
            )

    if verbose and weight_diagnostics is not None:
        print(f"[Training] Training Partition Class Weight Diagnostics:")
        print(f" - Max weight: {weight_diagnostics['max_weight']:.4f}")
        print(f" - Min nonzero weight: {weight_diagnostics['min_nonzero_weight']:.4f}")
        print(f" - Max/Min nonzero ratio: {weight_diagnostics['max_min_nonzero_ratio']:.2f}")
        if weight_diagnostics['max_min_nonzero_ratio'] > 1000.0:
            print(f" [ALERT] Severe class imbalance: max/min weight ratio is {weight_diagnostics['max_min_nonzero_ratio']:.1f}x. Monitor training stability.")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = EdgeAUIGRU(
        input_dim=MICROTENSOR_DIM,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=NUM_CLASSES
    ).to(target_device)

    if weights_tensor is not None:
        criterion = nn.CrossEntropyLoss(weight=weights_tensor.to(target_device))
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history: Dict[str, List[float]] = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": []
    }

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset and len(val_dataset) > 0 else None

    if verbose:
        print(f"[Training] Starting Foundational Training: {epochs} epochs over {len(dataset)} sequences on {target_device}...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(target_device)
            batch_y = batch_y.to(target_device)

            optimizer.zero_grad()

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        epoch_loss = total_loss / max(total, 1)
        epoch_acc = 100.0 * correct / max(total, 1)
        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_acc)

        # Validation step
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for v_x, v_y in val_loader:
                    v_x = v_x.to(target_device)
                    v_y = v_y.to(target_device)
                    v_out = model(v_x)
                    v_l = criterion(v_out, v_y)
                    val_loss += v_l.item() * v_x.size(0)
                    _, v_pred = torch.max(v_out.data, 1)
                    val_total += v_y.size(0)
                    val_correct += (v_pred == v_y).sum().item()
                    val_preds.append(v_pred.cpu().numpy())
                    val_targets.append(v_y.cpu().numpy())

            epoch_val_loss = val_loss / max(val_total, 1)
            epoch_val_acc = 100.0 * val_correct / max(val_total, 1)

            # Compute quick val macro-f1
            vp = np.concatenate(val_preds, axis=0) if val_preds else np.empty(0)
            vt = np.concatenate(val_targets, axis=0) if val_targets else np.empty(0)
            if len(vt) > 0:
                epoch_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
                for t_i, p_i in zip(vt, vp):
                    if 0 <= t_i < NUM_CLASSES and 0 <= p_i < NUM_CLASSES:
                        epoch_cm[t_i, p_i] += 1
                f1s = []
                for c in range(NUM_CLASSES):
                    s_c = np.sum(epoch_cm[c, :])
                    if s_c > 0:
                        tp_c = epoch_cm[c, c]
                        fp_c = np.sum(epoch_cm[:, c]) - tp_c
                        fn_c = s_c - tp_c
                        p_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
                        r_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
                        f1_c = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) > 0 else 0.0
                        f1s.append(f1_c)
                epoch_val_macro_f1 = float(np.mean(f1s)) if f1s else 0.0
            else:
                epoch_val_macro_f1 = 0.0

            history["val_loss"].append(epoch_val_loss)
            history["val_accuracy"].append(epoch_val_acc)
            history["val_macro_f1"].append(epoch_val_macro_f1)

            if verbose:
                print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%, Val Macro-F1: {epoch_val_macro_f1:.4f}")
        else:
            if verbose:
                print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")

    # Final comprehensive evaluation on validation split
    val_eval_results = None
    if val_dataset is not None and len(val_dataset) > 0:
        if verbose:
            print("[Training] Running comprehensive final validation evaluation...")
        val_eval_results = evaluate_foundation_model(
            model=model,
            eval_dataset=val_dataset,
            criterion=criterion,
            device=target_device,
            num_classes=NUM_CLASSES,
            train_dataset=dataset,
            verbose=verbose
        )

    # Resolve model output path
    if output_dir is None:
        proj_root = find_project_root()
        models_path = os.path.join(proj_root, "models")
    else:
        models_path = os.path.abspath(output_dir)

    os.makedirs(models_path, exist_ok=True)
    model_path = os.path.join(models_path, "foundational_gru.pth")
    torch.save(model.state_dict(), model_path)

    # Sanity check saved checkpoint
    assert os.path.isfile(model_path), f"Failed to save model to {model_path}"
    saved_state = torch.load(model_path, map_location="cpu", weights_only=True)
    assert "gru.weight_ih_l0" in saved_state, "Corrupted state dict: missing GRU weights"

    if verbose:
        print(f"[Training] Foundational model successfully saved to: {model_path} ({os.path.getsize(model_path) / 1024:.1f} KB)")

    return {
        "model": model,
        "model_path": model_path,
        "history": history,
        "dataset_size": len(dataset),
        "val_dataset_size": len(val_dataset) if val_dataset else 0,
        "device": str(target_device),
        "class_weights": weights_tensor,
        "weight_diagnostics": weight_diagnostics,
        "val_evaluation": val_eval_results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Foundational GRU Training for Edge-AUI Framework")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to raw dataset directory")
    parser.add_argument("--hf-repo", type=str, default="T40/edge-aui-framework-data", help="Hugging Face repo ID")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--max-sequences", type=int, default=None, help="Cap max sequences for fast iteration")
    parser.add_argument("--device", type=str, default=None, help="Execution device (cuda/mps/cpu)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument("--class-weights", type=str, default="smoothed", choices=["smoothed", "inverse", "none"], help="Class weighting scheme")
    args = parser.parse_args()

    cw = None if args.class_weights == "none" else args.class_weights
    train_foundation_model(
        data_dir=args.data_dir,
        hf_repo_id=args.hf_repo,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_sequences=args.max_sequences,
        device=args.device,
        output_dir=args.output_dir,
        class_weights=cw
    )
