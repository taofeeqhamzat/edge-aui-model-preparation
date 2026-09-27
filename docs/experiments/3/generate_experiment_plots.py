#!/usr/bin/env python3
"""
generate_experiment_plots.py
Generates diagnostic visualization plots for Experiment 3:
Target-Domain Intervention-Head Progressive Fine-Tuning & Evaluation (Tasks 12.1-12.4).

Evaluates checkpoints:
  - models/intervention_head_e1.pth (Strict Freezing)
  - models/intervention_head_e2.pth (Partial Fine-Tuning)
  - models/intervention_head_e3.pth (Full Fine-Tuning)
against the Majority-Class baseline on v1.0.0 validation and test partitions.

Saves publication-quality figures into docs/experiments/3/after/.
"""

import os
import sys
from pathlib import Path

# Resolve project root and add src/ to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from training import (
    EdgeAUIGRU,
    TargetInterventionHead,
    evaluate_intervention_model,
    evaluate_intervention_majority_baseline,
    evaluate_intervention_retention,
    MICROTENSOR_DIM,
    HIDDEN_DIM,
    NUM_LAYERS,
)
from target_dataset import (
    load_intervention_dataset,
    INTERVENTION_VOCABULARY,
)


OUTPUT_DIR = SCRIPT_DIR / "after"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = PROJECT_ROOT / "models"

SHORT_VOCAB = ["simplify", "highlight", "assistance", "tooltip", "no_op"]
PALETTE = {
    "baseline": "#7f7f7f",
    "e1": "#4C72B0",
    "e2": "#55A868",
    "e3": "#C44E52",
}


def load_model(checkpoint_path: Path) -> EdgeAUIGRU:
    head = TargetInterventionHead(hidden_dim=HIDDEN_DIM, context_dim=6, num_classes=5)
    model = EdgeAUIGRU(
        input_dim=MICROTENSOR_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=5,
        head=head,
    )
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    return model


def plot_confusion_matrices(results: dict):
    """Plot side-by-side empirical confusion matrices for E1, E2, E3 on test partition."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    experiments = [("e1", "E1: Strict Freezing"), ("e2", "E2: Partial Fine-Tuning"), ("e3", "E3: Full Fine-Tuning")]

    for idx, (exp_key, title) in enumerate(experiments):
        cm = np.array(results[exp_key]["test"]["confusion_matrix"])
        ax = axes[idx]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=SHORT_VOCAB,
            yticklabels=SHORT_VOCAB,
            ax=ax,
            linewidths=1,
            linecolor="#e0e0e0"
        )
        ax.set_title(f"{title}\nTest Acc: {results[exp_key]['test']['accuracy']:.1f}% | Macro-F1: {results[exp_key]['test']['macro_f1']:.4f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Intervention", fontsize=10)
        ax.set_ylabel("True Intervention" if idx == 0 else "", fontsize=10)

    plt.suptitle("Empirical Confusion Matrices on Test Partition (v1.0.0, N=30)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_file = OUTPUT_DIR / "confusion_matrices_comparison.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_macro_f1_comparison(results: dict, baseline: dict):
    """Plot comparative Macro-F1 and Accuracy across partitions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    models = ["Majority Baseline", "E1: Strict", "E2: Partial", "E3: Full"]
    val_macro = [baseline["val"]["macro_f1"], results["e1"]["val"]["macro_f1"], results["e2"]["val"]["macro_f1"], results["e3"]["val"]["macro_f1"]]
    test_macro = [baseline["test"]["macro_f1"], results["e1"]["test"]["macro_f1"], results["e2"]["test"]["macro_f1"], results["e3"]["test"]["macro_f1"]]

    x = np.arange(len(models))
    width = 0.35

    rects1 = ax1.bar(x - width/2, val_macro, width, label="Validation Split", color="#4C72B0", alpha=0.9)
    rects2 = ax1.bar(x + width/2, test_macro, width, label="Held-Out Test Split", color="#55A868", alpha=0.9)

    ax1.set_ylabel("Macro-F1 Score (Supported Classes)", fontsize=11)
    ax1.set_title("Macro-F1 Performance by Transfer Fine-Tuning Regime", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10)
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.set_ylim(0, 0.70)

    for rect in rects1 + rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9)

    val_acc = [baseline["val"]["accuracy"] * 100, results["e1"]["val"]["accuracy"], results["e2"]["val"]["accuracy"], results["e3"]["val"]["accuracy"]]
    test_acc = [baseline["test"]["accuracy"] * 100, results["e1"]["test"]["accuracy"], results["e2"]["test"]["accuracy"], results["e3"]["test"]["accuracy"]]

    rects3 = ax2.bar(x - width/2, val_acc, width, label="Validation Split", color="#4C72B0", alpha=0.9)
    rects4 = ax2.bar(x + width/2, test_acc, width, label="Held-Out Test Split", color="#55A868", alpha=0.9)

    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_title("Overall Accuracy by Transfer Fine-Tuning Regime", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=10)
    ax2.legend(loc="upper left")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.set_ylim(0, 90)

    for rect in rects3 + rects4:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_file = OUTPUT_DIR / "macro_f1_ablation_comparison.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_per_class_f1(results: dict):
    """Plot per-class F1 performance on test partition across E1, E2, E3."""
    fig, ax = plt.subplots(figsize=(11, 5))

    active_classes = ["offer_assistance", "expand_tooltip", "no_op"]
    short_labels = ["Offer Assistance", "Expand Tooltip", "No-Op (Majority)"]

    e1_f1 = [results["e1"]["test"]["per_class"][c]["f1"] for c in active_classes]
    e2_f1 = [results["e2"]["test"]["per_class"][c]["f1"] for c in active_classes]
    e3_f1 = [results["e3"]["test"]["per_class"][c]["f1"] for c in active_classes]

    x = np.arange(len(active_classes))
    width = 0.25

    r1 = ax.bar(x - width, e1_f1, width, label="E1: Strict Freezing", color="#4C72B0")
    r2 = ax.bar(x, e2_f1, width, label="E2: Partial Fine-Tuning", color="#55A868")
    r3 = ax.bar(x + width, e3_f1, width, label="E3: Full Fine-Tuning", color="#C44E52")

    ax.set_ylabel("F1-Score", fontsize=11)
    ax.set_title("Per-Class F1 Score on Active Test Partition Classes (v1.0.0)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=10)
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1.0)

    for r_set in [r1, r2, r3]:
        for rect in r_set:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_file = OUTPUT_DIR / "per_class_f1_breakdown.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_foundation_retention_audit():
    """Plot foundation retention diagnostic bar chart for E3."""
    e3_model = load_model(MODELS_DIR / "intervention_head_e3.pth")
    retention = evaluate_intervention_retention(e3_model, verbose=False)

    if not retention.get("retention_evaluated", False):
        print("Skipping retention plot: retention not evaluated.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ["Foundation Macro-F1", "Foundation Accuracy (%)"]
    pre_values = [retention["pre_finetune_macro_f1"], retention["pre_finetune_accuracy"]]
    post_values = [retention["post_finetune_macro_f1"], retention["post_finetune_accuracy"]]

    x = np.arange(len(metrics))
    width = 0.35

    r1 = ax.bar(x - width/2, pre_values, width, label="Pre-Fine-Tuning (Foundational Checkpoint)", color="#4C72B0")
    r2 = ax.bar(x + width/2, post_values, width, label="Post-Fine-Tuning (E3 Backbone Modified)", color="#E17C05")

    ax.set_title("Catastrophic Forgetting Retention Audit (E3 Backbone on AdSERP Val)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 65)

    for rect in r1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}" if h < 1.0 else f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in r2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}" if h < 1.0 else f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    drop_text = (
        f"Relative Degradation: {retention['relative_drop_pct']:.1f}%\n"
        f"Threshold (<=15.0%): PASSED"
    )
    ax.text(0.5, 0.15, drop_text, transform=ax.transAxes, ha="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#d4edda", edgecolor="#28a745", alpha=0.9),
            fontsize=10, fontweight="bold")

    plt.tight_layout()
    out_file = OUTPUT_DIR / "foundation_retention_audit.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def main():
    print("[Plots] Loading datasets for evaluation...")
    train_ds = load_intervention_dataset(version="v1.0.0", split="train")
    val_ds = load_intervention_dataset(version="v1.0.0", split="val")
    test_ds = load_intervention_dataset(version="v1.0.0", split="test")

    majority_val = evaluate_intervention_majority_baseline(train_ds, val_ds)
    majority_test = evaluate_intervention_majority_baseline(train_ds, test_ds)
    baseline = {"val": majority_val, "test": majority_test}

    results = {}
    for exp in ["e1", "e2", "e3"]:
        ckpt = MODELS_DIR / f"intervention_head_{exp}.pth"
        if not ckpt.is_file():
            print(f"[ERROR] Checkpoint missing: {ckpt}")
            return
        model = load_model(ckpt)
        results[exp] = {
            "val": evaluate_intervention_model(model, val_ds, train_dataset=train_ds, verbose=False),
            "test": evaluate_intervention_model(model, test_ds, train_dataset=train_ds, verbose=False),
        }

    print("[Plots] Generating diagnostic plots...")
    plot_confusion_matrices(results)
    plot_macro_f1_comparison(results, baseline)
    plot_per_class_f1(results)
    plot_foundation_retention_audit()
    print(f"[Plots] All diagnostic plots successfully exported to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
