"""
semantic_training.py
Task 40.4: train and evaluate the semantic task-outcome head.

The head has the approved 14-class target vocabulary (ADR-030) and the same contract as the
legacy intervention head — ``(sequence[T, 18], context[6]) -> logits[14]`` — so the foundation
GRU checkpoint and the previous five-class artifacts are preserved unchanged.

Evidence rules enforced here:

- the target is ``target_task_outcome_id``; the generic seven-class outcome and the five-class
  intervention label are never used as the semantic target;
- class weights come from the dataset manifest, which resolves them on the training partition
  only;
- the declared split unit is ``session`` (ADR-031); no participant-held-out claim is made;
- epoch selection uses the validation partition, and the test partition is touched once, after
  the checkpoint is frozen;
- absent classes are reported, never given a fabricated F1.

CLI: ``python3 -m src.semantic_training --experiment e1`` (or ``--evaluate-suite`` for the
freeze / partial / full ablation).
"""

import argparse
import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from training import (
        EdgeAUIGRU,
        TargetSemanticHead,
        get_device,
        BATCH_SIZE,
        EPOCHS,
        HIDDEN_DIM,
        NUM_LAYERS,
        SEMANTIC_NUM_CLASSES,
        compute_ranking_diagnostics,
    )
    from target_dataset import (
        TargetInterventionDataset,
        load_semantic_dataset,
        get_semantic_class_weights,
    )
    from semantic_targets import (
        TARGET_VOCABULARY,
        TARGET_VOCABULARY_VERSION,
        RECORDED_VOCABULARY_VERSION,
    )
    from config import load_config
    from data_manager import find_project_root
except ImportError:
    from src.training import (
        EdgeAUIGRU,
        TargetSemanticHead,
        get_device,
        BATCH_SIZE,
        EPOCHS,
        HIDDEN_DIM,
        NUM_LAYERS,
        SEMANTIC_NUM_CLASSES,
        compute_ranking_diagnostics,
    )
    from src.target_dataset import (
        TargetInterventionDataset,
        load_semantic_dataset,
        get_semantic_class_weights,
    )
    from src.semantic_targets import (
        TARGET_VOCABULARY,
        TARGET_VOCABULARY_VERSION,
        RECORDED_VOCABULARY_VERSION,
    )
    from src.config import load_config
    from src.data_manager import find_project_root

VOCABULARY: List[str] = [TARGET_VOCABULARY[i] for i in sorted(TARGET_VOCABULARY)]
SPLIT_UNIT = "session"
REPORT_SCHEMA_VERSION = "1.0.0"
EXPERIMENTS = ("e1", "e2", "e3")


class SemanticTrainingError(ValueError):
    """Raised when the semantic training/evaluation contract cannot be honoured."""


# ---------------------------------------------------------------------------
# Model assembly
# ---------------------------------------------------------------------------

def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return get_device(device)


def build_semantic_model(
    foundation_checkpoint: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[EdgeAUIGRU, torch.device]:
    """Build the GRU + 14-class semantic head and load the foundation backbone."""
    target_device = _resolve_device(device)
    head = TargetSemanticHead(hidden_dim=HIDDEN_DIM, context_dim=6,
                              num_classes=SEMANTIC_NUM_CLASSES)
    model = EdgeAUIGRU(
        input_dim=18,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=SEMANTIC_NUM_CLASSES,
        head=head,
    ).to(target_device)

    if foundation_checkpoint is not None:
        path = Path(foundation_checkpoint)
        if not path.is_absolute():
            path = Path(find_project_root()) / path
        if not path.is_file():
            raise FileNotFoundError(f"Foundation checkpoint not found: {path}")
        base_state = torch.load(str(path), map_location="cpu", weights_only=True)
        gru_state = {k: v for k, v in base_state.items() if k.startswith("gru.")}
        if not gru_state:
            raise SemanticTrainingError(
                f"Foundation checkpoint '{path}' carries no 'gru.*' backbone weights"
            )
        model.load_state_dict(gru_state, strict=False)
    return model, target_device


def _configure_optimizer(
    model: EdgeAUIGRU, experiment: str, cfg: Any
) -> Tuple[optim.Optimizer, Dict[str, Any]]:
    """Freeze / partial / full ablation, identical semantics to the intervention head."""
    if hasattr(cfg, "ablation"):
        head_lr = float(getattr(cfg.ablation, "strict_freezing_head_lr", 1e-3))
        gru_lr = float(getattr(cfg.ablation, "partial_gru_lr", 1e-4))
        partial_head_lr = float(getattr(cfg.ablation, "partial_head_lr", 1e-3))
        full_lr = float(getattr(cfg.ablation, "full_lr", 5e-4))
    else:
        head_lr = 1e-3
        gru_lr = 1e-4
        partial_head_lr = 1e-3
        full_lr = 5e-4

    if experiment == "e1":
        model.freeze_backbone()
        return (
            optim.Adam(model.head_parameters(), lr=head_lr),
            {"scheme": "strict_freezing", "backbone_frozen": True, "head_lr": head_lr},
        )
    if experiment == "e2":
        model.unfreeze_terminal_layer()
        terminal = [p for name, p in model.gru.named_parameters()
                    if f"_l{NUM_LAYERS - 1}" in name]
        return (
            optim.Adam([
                {"params": terminal, "lr": gru_lr},
                {"params": list(model.head_parameters()), "lr": partial_head_lr},
            ]),
            {
                "scheme": "partial_finetuning",
                "backbone_terminal_layer": f"_l{NUM_LAYERS - 1}",
                "gru_lr": gru_lr,
                "head_lr": partial_head_lr,
                "terminal_params_count": sum(p.numel() for p in terminal),
                "head_params_count": sum(p.numel() for p in model.head_parameters()),
            },
        )
    if experiment == "e3":
        model.unfreeze_backbone()
        return (
            optim.Adam(model.parameters(), lr=full_lr),
            {"scheme": "full_finetuning", "all_layers_trainable": True, "lr": full_lr},
        )
    raise SemanticTrainingError(
        f"Unknown experiment '{experiment}'; expected one of {EXPERIMENTS}"
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _percentiles(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"count": 0, "median_ms": None, "p25_ms": None, "p75_ms": None}
    return {
        "count": int(arr.size),
        "median_ms": float(np.median(arr)),
        "p25_ms": float(np.percentile(arr, 25)),
        "p75_ms": float(np.percentile(arr, 75)),
    }


def _per_class_metrics(
    targets: np.ndarray, predictions: np.ndarray, vocabulary: List[str]
) -> Tuple[Dict[str, Dict[str, Any]], np.ndarray]:
    num_classes = len(vocabulary)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(targets, predictions):
        if 0 <= int(true) < num_classes and 0 <= int(pred) < num_classes:
            confusion[int(true), int(pred)] += 1

    per_class: Dict[str, Dict[str, Any]] = {}
    for class_id, name in enumerate(vocabulary):
        true_positive = int(confusion[class_id, class_id])
        false_positive = int(confusion[:, class_id].sum() - true_positive)
        false_negative = int(confusion[class_id, :].sum() - true_positive)
        support = int(confusion[class_id, :].sum())
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[name] = {
            "class_id": class_id,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
            "absent": support == 0,
        }
    return per_class, confusion


def _aggregate_f1(
    per_class: Dict[str, Dict[str, Any]], targets: np.ndarray
) -> Tuple[float, float, float]:
    supported = [m["f1"] for m in per_class.values() if m["support"] > 0]
    all_f1 = [m["f1"] for m in per_class.values()]
    macro_supported = float(np.mean(supported)) if supported else 0.0
    macro_all = float(np.mean(all_f1)) if all_f1 else 0.0
    total = int(targets.size)
    weighted = (
        float(sum(m["f1"] * m["support"] for m in per_class.values()) / total)
        if total else 0.0
    )
    return macro_supported, macro_all, weighted


def _ece_and_reliability(
    probabilities: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Expected calibration error and reliability table for the predicted class."""
    if probabilities.size == 0:
        return {"ece": None, "brier": None, "bins": []}
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == targets).astype(float)

    bins = []
    ece = 0.0
    for index in range(n_bins):
        low = index / n_bins
        high = (index + 1) / n_bins
        if index == n_bins - 1:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)
        count = int(mask.sum())
        if count == 0:
            continue
        accuracy = float(correct[mask].mean())
        confidence = float(confidences[mask].mean())
        ece += (count / len(confidences)) * abs(accuracy - confidence)
        bins.append({
            "lower": low,
            "upper": high,
            "count": count,
            "mean_confidence": confidence,
            "accuracy": accuracy,
            "gap": accuracy - confidence,
        })

    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(targets)), targets] = 1.0
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    return {"ece": float(ece), "brier": brier, "bins": bins, "n_bins": n_bins}


def _lead_time_report(
    metadata: List[Dict[str, Any]], targets: np.ndarray, vocabulary: List[str]
) -> Dict[str, Any]:
    per_class: Dict[str, Any] = {}
    overall: List[float] = []
    for class_id, name in enumerate(vocabulary):
        values = []
        for index, true in enumerate(targets):
            if int(true) != class_id or index >= len(metadata):
                continue
            value = metadata[index].get("lead_time_ms")
            if value is not None and np.isfinite(float(value)):
                values.append(float(value))
                overall.append(float(value))
        per_class[name] = _percentiles(values)
    return {"overall": _percentiles(overall), "per_class": per_class}


def _task_level_report(
    metadata: List[Dict[str, Any]],
    targets: np.ndarray,
    predictions: np.ndarray,
    vocabulary: List[str],
) -> Dict[str, Any]:
    tasks: Dict[str, List[int]] = {}
    for index in range(len(targets)):
        task = str((metadata[index] if index < len(metadata) else {}).get("task_id") or "unknown")
        tasks.setdefault(task, []).append(index)

    report: Dict[str, Any] = {}
    for task, indices in sorted(tasks.items()):
        task_targets = targets[indices]
        task_predictions = predictions[indices]
        per_class, _ = _per_class_metrics(task_targets, task_predictions, vocabulary)
        macro_supported, macro_all, weighted = _aggregate_f1(per_class, task_targets)
        report[task] = {
            "n_samples": int(len(indices)),
            "accuracy": float(np.mean(task_targets == task_predictions)) if len(indices) else 0.0,
            "macro_f1": macro_supported,
            "macro_f1_all": macro_all,
            "weighted_f1": weighted,
            "absent_classes": [name for name, m in per_class.items() if m["support"] == 0],
        }
    return report


def _majority_baseline(
    train_targets: np.ndarray,
    eval_targets: np.ndarray,
    vocabulary: List[str],
) -> Dict[str, Any]:
    if train_targets.size == 0 or eval_targets.size == 0:
        return {}
    counts = np.bincount(train_targets, minlength=len(vocabulary))
    majority_id = int(np.argmax(counts))
    predictions = np.full_like(eval_targets, majority_id)
    per_class, _ = _per_class_metrics(eval_targets, predictions, vocabulary)
    macro_supported, macro_all, weighted = _aggregate_f1(per_class, eval_targets)
    return {
        "majority_class": vocabulary[majority_id],
        "majority_class_id": majority_id,
        "train_support": int(counts[majority_id]),
        "accuracy": float(np.mean(eval_targets == predictions)),
        "macro_f1": macro_supported,
        "macro_f1_all": macro_all,
        "weighted_f1": weighted,
        "per_class": per_class,
    }


def _class_distribution(targets: np.ndarray, vocabulary: List[str]) -> Dict[str, Any]:
    counts = np.bincount(targets, minlength=len(vocabulary)) if targets.size else np.zeros(len(vocabulary))
    total = int(targets.size)
    return {
        "total": total,
        "counts": {name: int(counts[index]) for index, name in enumerate(vocabulary)},
        "percentages": {
            name: (100.0 * int(counts[index]) / total if total else 0.0)
            for index, name in enumerate(vocabulary)
        },
        "majority_class": vocabulary[int(np.argmax(counts))] if total else None,
        "majority_percentage": (100.0 * float(counts.max()) / total if total else None),
    }


def collect_predictions(
    model: nn.Module,
    dataset: Union[Any, DataLoader],
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (logits, targets, contexts) for a dataset without updating the model."""
    dataloader = dataset if isinstance(dataset, DataLoader) else DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    model.eval()
    logits_parts, target_parts, context_parts = [], [], []
    with torch.no_grad():
        for batch_x, batch_ctx, batch_y in dataloader:
            outputs = model(batch_x.to(device), context=batch_ctx.to(device))
            logits_parts.append(outputs.detach().cpu().numpy())
            target_parts.append(batch_y.detach().cpu().numpy())
            context_parts.append(batch_ctx.detach().cpu().numpy())
    if not logits_parts:
        return (
            np.empty((0, len(VOCABULARY)), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty((0, 6), dtype=np.float32),
        )
    return (
        np.concatenate(logits_parts, axis=0),
        np.concatenate(target_parts, axis=0).astype(np.int64),
        np.concatenate(context_parts, axis=0),
    )


def evaluate_semantic_dataset(
    model: nn.Module,
    dataset: Union[Any, DataLoader],
    train_targets: np.ndarray,
    split: str = "test",
    device: Optional[Union[str, torch.device]] = None,
    batch_size: int = BATCH_SIZE,
    criterion: Optional[nn.Module] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Full semantic evaluation for one split, including absent-class reporting."""
    target_device = _resolve_device(device)
    metadata = list(getattr(dataset, "metadata", []) or [])
    logits, targets, _ = collect_predictions(model, dataset, target_device, batch_size=batch_size)

    if targets.size == 0:
        return {
            "split": split,
            "split_unit": SPLIT_UNIT,
            "n_samples": 0,
            "accuracy": None,
            "macro_f1": None,
            "macro_f1_all": None,
            "weighted_f1": None,
            "per_class": {},
            "confusion_matrix": [],
            "absent_classes": VOCABULARY,
            "majority_baseline": {},
            "task_level": {},
            "lead_time": {"overall": _percentiles([]), "per_class": {}},
            "calibration": {"ece": None, "brier": None, "bins": []},
            "class_distribution": _class_distribution(targets, VOCABULARY),
            "loss": None,
        }

    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    per_class, confusion = _per_class_metrics(targets, predictions, VOCABULARY)
    macro_supported, macro_all, weighted = _aggregate_f1(per_class, targets)

    # Evaluation loss is unweighted so it is comparable across arms; the training loss uses
    # the training-only class weights.
    loss = float(
        nn.functional.cross_entropy(
            torch.from_numpy(logits), torch.from_numpy(targets)
        ).item()
    )

    result = {
        "split": split,
        "split_unit": SPLIT_UNIT,
        "n_samples": int(targets.size),
        "accuracy": float(np.mean(predictions == targets)),
        "macro_f1": macro_supported,
        "macro_f1_all": macro_all,
        "weighted_f1": weighted,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "absent_classes": [name for name, m in per_class.items() if m["support"] == 0],
        "majority_baseline": _majority_baseline(train_targets, targets, VOCABULARY),
        "task_level": _task_level_report(metadata, targets, predictions, VOCABULARY),
        "lead_time": _lead_time_report(metadata, targets, VOCABULARY),
        "calibration": _ece_and_reliability(probabilities, targets),
        "class_distribution": _class_distribution(targets, VOCABULARY),
        "ranking": compute_ranking_diagnostics(logits, targets, ks=(1, 3)),
        "loss": loss,
        "vocabulary": VOCABULARY,
        "target_vocabulary_version": TARGET_VOCABULARY_VERSION,
        "recorded_vocabulary_version": RECORDED_VOCABULARY_VERSION,
    }

    if verbose:
        print(f"\nSemantic outcome head — {split} split ({targets.size} examples)")
        print(f"{'Outcome':<24}{'P':>9}{'R':>9}{'F1':>9}{'Support':>10}")
        for name, metrics in per_class.items():
            if metrics["support"] == 0:
                print(f"{name:<24}{'—':>9}{'—':>9}{'—':>9}{0:>10}  (absent)")
            else:
                print(
                    f"{name:<24}{metrics['precision']:>9.3f}{metrics['recall']:>9.3f}"
                    f"{metrics['f1']:>9.3f}{metrics['support']:>10}"
                )
        print(
            f"accuracy={result['accuracy']:.3f} macro-F1={result['macro_f1']:.3f} "
            f"weighted-F1={result['weighted_f1']:.3f} "
            f"majority-accuracy={result['majority_baseline'].get('accuracy', float('nan')):.3f} "
            f"majority-macro-F1={result['majority_baseline'].get('macro_f1', float('nan')):.3f}"
        )
    return result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    try:
        import subprocess
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(root), capture_output=True, text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _manifest_for(version: str, root: Path) -> Dict[str, Any]:
    path = root / ".data" / "processed" / version / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _training_config(
    experiment: str,
    dataset_version: str,
    manifest: Dict[str, Any],
    lr_info: Dict[str, Any],
    seed: int,
    epochs: int,
    batch_size: int,
    class_weights: List[float],
    checkpoint_path: Path,
    best_epoch: int,
    history: Dict[str, List[float]],
    device: torch.device,
    sequence_length: int,
) -> Dict[str, Any]:
    target_semantics = manifest.get("target_semantics", {})
    return {
        "training_schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "head": "TargetSemanticHead",
        "experiment": experiment,
        "head_class_count": SEMANTIC_NUM_CLASSES,
        "target_vocabulary": VOCABULARY,
        "target_vocabulary_version": TARGET_VOCABULARY_VERSION,
        "recorded_vocabulary_version": RECORDED_VOCABULARY_VERSION,
        "label_key": "target_task_outcome_id",
        "dataset_version": dataset_version,
        "dataset_content_hash": manifest.get("content_hash"),
        "split_unit": manifest.get("split_summary", {}).get("split_unit", SPLIT_UNIT),
        "history_windows": sequence_length,
        "lookahead_horizon_ms": target_semantics.get("lookahead_horizon_ms"),
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate_scheme": lr_info,
        "class_weights": dict(zip(VOCABULARY, class_weights)),
        "class_weights_source": manifest.get("class_weights", {}).get(
            "derived_on", "train_partition_only"
        ),
        "best_epoch": best_epoch,
        "selection_rule": "highest validation macro-F1 (test evaluated once after selection)",
        "history": history,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "device": str(device),
        "environment": {
            "python": str(os.sys.version.split()[0]),
            "pytorch": torch.__version__,
        },
    }


def train_semantic_model(
    experiment: str = "e1",
    dataset_version: str = "v2.2.0-participant",
    foundation_checkpoint: Optional[str] = "models/foundational_gru.pth",
    seed: int = 42,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    output_dir: Optional[str] = None,
    dataset_dir: Optional[str] = None,
    evaluate_test: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train one semantic-head ablation arm and evaluate the declared held-out split."""
    experiment = experiment.lower()
    if experiment not in EXPERIMENTS:
        raise SemanticTrainingError(
            f"Unknown experiment '{experiment}'; expected one of {EXPERIMENTS}"
        )

    root = Path(find_project_root())
    cfg = load_config()
    epochs = int(epochs if epochs is not None else getattr(getattr(cfg, "training", None), "epochs", EPOCHS))
    batch_size = int(
        batch_size if batch_size is not None
        else getattr(getattr(cfg, "training", None), "batch_size", BATCH_SIZE)
    )
    version_dir = (
        Path(dataset_dir)
        if dataset_dir is not None
        else root / ".data" / "processed" / dataset_version
    )
    parquet_path = version_dir / "target_semantic_dataset.parquet"
    manifest_path = version_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dataset = load_semantic_dataset(parquet_path=parquet_path, split="train")
    if len(train_dataset) == 0:
        raise SemanticTrainingError(f"Train partition of '{dataset_version}' is empty")
    val_dataset = load_semantic_dataset(parquet_path=parquet_path, split="val")
    test_dataset = load_semantic_dataset(parquet_path=parquet_path, split="test")

    model, target_device = build_semantic_model(foundation_checkpoint, device)
    optimizer, lr_info = _configure_optimizer(model, experiment, cfg)
    class_weights = get_semantic_class_weights(manifest_path=manifest_path).tolist()
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=target_device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    history: Dict[str, List[float]] = {
        "train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": [], "val_macro_f1": [],
    }
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_macro_f1 = -1.0
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch_x, batch_ctx, batch_y in train_loader:
            batch_x = batch_x.to(target_device)
            batch_ctx = batch_ctx.to(target_device)
            batch_y = batch_y.to(target_device)
            optimizer.zero_grad()
            logits = model(batch_x, context=batch_ctx)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * batch_x.size(0)
            correct += int((logits.argmax(dim=1) == batch_y).sum().item())
            total += int(batch_y.size(0))

        history["train_loss"].append(total_loss / max(total, 1))
        history["train_accuracy"].append(correct / max(total, 1))

        val_metrics = evaluate_semantic_dataset(
            model, val_dataset, train_dataset.targets, split="val",
            device=target_device, batch_size=batch_size, criterion=criterion, verbose=False,
        )
        val_macro = val_metrics.get("macro_f1") or 0.0
        history["val_loss"].append(float(val_metrics.get("loss") or 0.0))
        history["val_accuracy"].append(float(val_metrics.get("accuracy") or 0.0))
        history["val_macro_f1"].append(float(val_macro))

        if val_macro > best_macro_f1:
            best_macro_f1 = float(val_macro)
            best_epoch = epoch + 1
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

        if verbose:
            print(
                f"epoch {epoch + 1}/{epochs} train_loss={history['train_loss'][-1]:.4f} "
                f"train_acc={history['train_accuracy'][-1]:.3f} val_macro_f1={val_macro:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    models_dir = Path(output_dir) if output_dir else root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / f"semantic_head_{experiment}.pth"
    torch.save(model.state_dict(), str(checkpoint_path))

    training_config = _training_config(
        experiment, dataset_version, manifest, lr_info, seed, epochs, batch_size,
        class_weights, checkpoint_path, best_epoch, history, target_device,
        int(train_dataset.sequences.shape[1]),
    )
    training_config_path = models_dir / f"semantic_head_{experiment}_training.json"
    training_config_path.write_text(json.dumps(training_config, indent=2) + "\n", encoding="utf-8")

    val_evaluation = evaluate_semantic_dataset(
        model, val_dataset, train_dataset.targets, split="val",
        device=target_device, batch_size=batch_size, criterion=criterion, verbose=verbose,
    )
    test_evaluation = None
    if evaluate_test:
        test_evaluation = evaluate_semantic_dataset(
            model, test_dataset, train_dataset.targets, split="test",
            device=target_device, batch_size=batch_size, criterion=criterion, verbose=verbose,
        )

    return {
        "experiment": experiment,
        "dataset_version": dataset_version,
        "seed": seed,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "model": model,
        "checkpoint_path": str(checkpoint_path),
        "training_config_path": str(training_config_path),
        "training_config": training_config,
        "history": history,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "val_evaluation": val_evaluation,
        "test_evaluation": test_evaluation,
        "device": str(target_device),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _result_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    test = result.get("test_evaluation") or {}
    val = result.get("val_evaluation") or {}
    return {
        "experiment": result["experiment"],
        "dataset_version": result["dataset_version"],
        "seed": result["seed"],
        "epochs": result["epochs"],
        "best_epoch": result["best_epoch"],
        "device": result["device"],
        "train_samples": result["train_samples"],
        "val_samples": result["val_samples"],
        "test_samples": result["test_samples"],
        "checkpoint": result["checkpoint_path"],
        "val": {
            "accuracy": val.get("accuracy"),
            "macro_f1": val.get("macro_f1"),
            "weighted_f1": val.get("weighted_f1"),
        },
        "test": {
            "accuracy": test.get("accuracy"),
            "macro_f1": test.get("macro_f1"),
            "macro_f1_all": test.get("macro_f1_all"),
            "weighted_f1": test.get("weighted_f1"),
            "majority_accuracy": (test.get("majority_baseline") or {}).get("accuracy"),
            "majority_macro_f1": (test.get("majority_baseline") or {}).get("macro_f1"),
            "ece": (test.get("calibration") or {}).get("ece"),
            "brier": (test.get("calibration") or {}).get("brier"),
            "absent_classes": test.get("absent_classes"),
            "n_samples": test.get("n_samples"),
            "ranking": test.get("ranking"),
            "per_class": test.get("per_class"),
            "confusion_matrix": test.get("confusion_matrix"),
            "majority_baseline": test.get("majority_baseline"),
            "task_level": test.get("task_level"),
            "lead_time": test.get("lead_time"),
            "calibration": test.get("calibration"),
            "class_distribution": test.get("class_distribution"),
        },
        "training_config": {
            key: result["training_config"].get(key)
            for key in (
                "head_class_count", "target_vocabulary_version", "dataset_version",
                "dataset_content_hash", "split_unit", "history_windows",
                "lookahead_horizon_ms", "seed", "epochs", "class_weights",
                "class_weights_source", "best_epoch", "learning_rate_scheme",
            )
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Semantic task-outcome head training and evaluation (Task 40.4)"
    )
    parser.add_argument("--experiment", default="e1", choices=list(EXPERIMENTS))
    parser.add_argument("--dataset-version", default="v2.2.0-participant")
    parser.add_argument("--foundation-checkpoint", default="models/foundational_gru.pth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset-dir", default=None,
                        help="Versioned dataset directory (overrides --dataset-version lookup)")
    parser.add_argument("--evaluate-suite", action="store_true",
                        help="Train and evaluate E1, E2 and E3")
    parser.add_argument("--report", default=None,
                        help="Optional JSON path for the evaluation report")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    experiments = list(EXPERIMENTS) if args.evaluate_suite else [args.experiment]
    results = []
    for experiment in experiments:
        result = train_semantic_model(
            experiment=experiment,
            dataset_version=args.dataset_version,
            foundation_checkpoint=args.foundation_checkpoint,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            output_dir=args.output_dir,
            dataset_dir=args.dataset_dir,
        )
        results.append(result)

    summary = [_result_summary(result) for result in results]
    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": args.dataset_version,
        "split_unit": SPLIT_UNIT,
        "seed": args.seed,
        "target_vocabulary_version": TARGET_VOCABULARY_VERSION,
        "recorded_vocabulary_version": RECORDED_VOCABULARY_VERSION,
        "ablation": summary,
    }
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[SemanticTraining] Report written to {path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
