"""
training.py
Foundational PyTorch training loop for the GRU model (Edge-AUI Framework).
"""

import os
import sys
import argparse
from typing import Optional, Dict, Any, List, Union, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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
    to the 5 target UI intervention actions (ADR-003).
    Intervention classes represent system adaptation decisions rather than user actions.
    Interface established in Task 4.1; training deferred to Task 4.2.
    """
    def __init__(self, hidden_dim: int = 64, num_classes: int = 5):
        super(TargetInterventionHead, self).__init__()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, h_T: torch.Tensor) -> torch.Tensor:
        return self.fc(h_T)


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
    class_weights: Optional[Union[torch.Tensor, str]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Foundational training loop for EdgeAUIGRU.
    Supports local datasets as well as automatic hosted synchronization from Hugging Face Hub.

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
    
    history: Dict[str, List[float]] = {"loss": [], "accuracy": []}

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

        if verbose:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        
    # Resolve model output path
    if output_dir is None:
        proj_root = find_project_root()
        models_path = os.path.join(proj_root, "models")
    else:
        models_path = os.path.abspath(output_dir)

    os.makedirs(models_path, exist_ok=True)
    model_path = os.path.join(models_path, "foundational_gru.pth")
    torch.save(model.state_dict(), model_path)

    if verbose:
        print(f"[Training] Foundational model successfully saved to: {model_path}")

    return {
        "model": model,
        "model_path": model_path,
        "history": history,
        "dataset_size": len(dataset),
        "device": str(target_device),
        "class_weights": weights_tensor,
        "weight_diagnostics": weight_diagnostics
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
    args = parser.parse_args()

    train_foundation_model(
        data_dir=args.data_dir,
        hf_repo_id=args.hf_repo,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_sequences=args.max_sequences,
        device=args.device,
        output_dir=args.output_dir
    )
