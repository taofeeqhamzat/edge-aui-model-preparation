#!/usr/bin/env python3
"""
generate_audit_plots.py
Generates diagnostic visualisation plots for Experiment 2:
Methodological Audit of Tasks 3.1, 3.2, and 4.1.

Reads the live adserp_microtensors.parquet, runs leak-free splitting,
computes class distributions, class weights, and majority-baseline metrics.
Saves publication-quality figures into docs/experiments/2/after/.

Usage:
    cd model-preparation
    .venv/bin/python docs/experiments/2/generate_audit_plots.py
"""

import sys
import os
from pathlib import Path

# Resolve project root and add src/ to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless PNG generation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ------------------------------------------------------------------
# Project imports
# ------------------------------------------------------------------
from microtensor_store import split_users_leak_free, SplitResult
from target_generation import (
    OUTCOME_TAXONOMY,
    compute_class_weights,
    compute_class_weight_diagnostics,
)

try:
    import pyarrow.parquet as pq
    import pandas as pd
except ImportError:
    print("[ERROR] pyarrow and pandas are required. Install via: pip install pyarrow pandas")
    sys.exit(1)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
PARQUET_PATH = PROJECT_ROOT / ".data" / "interim" / "microtensors" / "adserp_microtensors.parquet"
OUTPUT_DIR = SCRIPT_DIR / "after"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 7
CLASS_NAMES = [OUTCOME_TAXONOMY[i] for i in range(NUM_CLASSES)]
SHORT_NAMES = ["NO_OUT", "CLICK", "FORM", "BACK", "SCROLL", "HOVER", "ABANDON"]

# Colour palette: muted professional tones
PARTITION_COLOURS = {
    "Corpus": "#4C72B0",
    "Train":  "#55A868",
    "Val":    "#C44E52",
    "Test":   "#8172B2",
}

SEVERITY_COLOURS = {
    "BLOCKER": "#d32f2f",
    "HIGH":    "#f57c00",
    "MEDIUM":  "#fbc02d",
    "LOW":     "#66bb6a",
}


def load_parquet_df() -> pd.DataFrame:
    """Load the canonical MicroTensor Parquet into a DataFrame."""
    if not PARQUET_PATH.is_file():
        print(f"[ERROR] Parquet file not found: {PARQUET_PATH}")
        sys.exit(1)
    table = pq.read_table(str(PARQUET_PATH), columns=["user_id", "session_id", "target_label"])
    return table.to_pandas()


def compute_partition_dfs(df: pd.DataFrame, split: SplitResult):
    """Partition the DataFrame by the SplitResult."""
    col = split.split_by
    train_df = df[df[col].isin(split.train_ids)]
    val_df   = df[df[col].isin(split.val_ids)]
    test_df  = df[df[col].isin(split.test_ids)]
    return train_df, val_df, test_df


def count_distribution(df: pd.DataFrame, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Count target label occurrences for each class."""
    counts = np.zeros(num_classes, dtype=np.int64)
    labels = df["target_label"].values
    for c in range(num_classes):
        counts[c] = int(np.sum(labels == c))
    return counts


# ==================================================================
# Plot 1: Target Distribution by Partition
# ==================================================================
def plot_target_distribution(corpus_df, train_df, val_df, test_df):
    """Grouped bar chart of 7-class window distribution across partitions."""
    corpus_counts = count_distribution(corpus_df)
    train_counts  = count_distribution(train_df)
    val_counts    = count_distribution(val_df)
    test_counts   = count_distribution(test_df)

    # Convert to percentages
    corpus_pct = corpus_counts / corpus_counts.sum() * 100
    train_pct  = train_counts  / train_counts.sum()  * 100
    val_pct    = val_counts    / val_counts.sum()     * 100
    test_pct   = test_counts   / test_counts.sum()    * 100

    x = np.arange(NUM_CLASSES)
    width = 0.20

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [3, 1]})

    # Left panel: major classes (0, 1, 4, 5)
    major_idx = [0, 1, 4, 5]
    x_major = np.arange(len(major_idx))
    for i, (label, pct, colour) in enumerate([
        ("Corpus", corpus_pct, PARTITION_COLOURS["Corpus"]),
        ("Train",  train_pct,  PARTITION_COLOURS["Train"]),
        ("Val",    val_pct,    PARTITION_COLOURS["Val"]),
        ("Test",   test_pct,   PARTITION_COLOURS["Test"]),
    ]):
        bars = ax1.bar(x_major + i * width, pct[major_idx], width, label=label, color=colour, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, pct[major_idx]):
            if val > 2:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f"{val:.1f}%", ha="center", va="bottom", fontsize=7)

    ax1.set_xticks(x_major + 1.5 * width)
    ax1.set_xticklabels([SHORT_NAMES[i] for i in major_idx], fontsize=10)
    ax1.set_ylabel("Percentage of Windows (%)", fontsize=10)
    ax1.set_title("Major Classes", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, max(corpus_pct[major_idx]) * 1.15)
    ax1.grid(axis="y", alpha=0.3)

    # Right panel: rare classes (2, 3, 6)
    rare_idx = [2, 3, 6]
    x_rare = np.arange(len(rare_idx))
    for i, (label, pct, colour) in enumerate([
        ("Corpus", corpus_pct, PARTITION_COLOURS["Corpus"]),
        ("Train",  train_pct,  PARTITION_COLOURS["Train"]),
        ("Val",    val_pct,    PARTITION_COLOURS["Val"]),
        ("Test",   test_pct,   PARTITION_COLOURS["Test"]),
    ]):
        bars = ax2.bar(x_rare + i * width, pct[rare_idx], width, label=label, color=colour, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, pct[rare_idx]):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                         f"{val:.3f}%", ha="center", va="bottom", fontsize=6.5, rotation=45)

    ax2.set_xticks(x_rare + 1.5 * width)
    ax2.set_xticklabels([SHORT_NAMES[i] for i in rare_idx], fontsize=10)
    ax2.set_title("Rare Classes (note y-axis scale)", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, max(corpus_pct[rare_idx].max(), 0.06) * 1.8)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Target Outcome Distribution by Partition (AdSERP, 112,865 windows)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = OUTPUT_DIR / "target_distribution_by_partition.png"
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVED] {out_path}")


# ==================================================================
# Plot 2: Class Weight Comparison (exact vs smoothed)
# ==================================================================
def plot_class_weight_comparison(train_df):
    """Side-by-side bar chart comparing exact (α=0) vs smoothed (α=100) weights."""
    targets = train_df["target_label"].values

    diag_exact    = compute_class_weight_diagnostics(targets, smoothing_alpha=0.0,   allow_empty=True)
    diag_smoothed = compute_class_weight_diagnostics(targets, smoothing_alpha=100.0, allow_empty=True)

    weights_exact    = diag_exact["weights"].numpy()
    weights_smoothed = diag_smoothed["weights"].numpy()

    # Replace 0.0 with a tiny value for log display
    we_plot = np.where(weights_exact > 0, weights_exact, 1e-4)
    ws_plot = np.where(weights_smoothed > 0, weights_smoothed, 1e-4)

    x = np.arange(NUM_CLASSES)
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, we_plot, width, label="Exact (α=0)", color="#e53935", edgecolor="white", alpha=0.85)
    bars2 = ax.bar(x + width / 2, ws_plot, width, label="Smoothed (α=100)", color="#1e88e5", edgecolor="white", alpha=0.85)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_NAMES, fontsize=10)
    ax.set_ylabel("Class Weight (log scale)", fontsize=11)
    ax.set_title("Training-Partition Class Weights: Exact vs Smoothed", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3, which="both")

    # Annotate extreme values
    for bar, val in zip(bars1, weights_exact):
        if val > 10:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#b71c1c")
    for bar, val in zip(bars2, weights_smoothed):
        if val > 10:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#0d47a1")

    # Weight ratio annotation
    ratio_exact = diag_exact["max_min_nonzero_ratio"]
    ratio_smooth = diag_smoothed["max_min_nonzero_ratio"]
    ax.text(0.98, 0.95,
            f"Exact ratio: {ratio_exact:,.0f}:1\nSmoothed ratio: {ratio_smooth:,.0f}:1",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray", alpha=0.9))

    fig.tight_layout()
    out_path = OUTPUT_DIR / "class_weight_comparison.png"
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVED] {out_path}")


# ==================================================================
# Plot 3: Majority Baseline Performance
# ==================================================================
def plot_majority_baseline(val_df, test_df):
    """Horizontal bar chart of majority baseline metrics."""
    from sklearn.metrics import accuracy_score, f1_score

    def compute_baseline_metrics(df: pd.DataFrame) -> dict:
        y_true = df["target_label"].values
        majority_class = 0  # NO_OUTCOME
        y_pred = np.full_like(y_true, majority_class)
        return {
            "Accuracy":    accuracy_score(y_true, y_pred) * 100,
            "Macro-F1":    f1_score(y_true, y_pred, average="macro", zero_division=0),
            "Weighted-F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

    val_metrics  = compute_baseline_metrics(val_df)
    test_metrics = compute_baseline_metrics(test_df)

    metric_names = ["Accuracy (%)", "Macro-F1", "Weighted-F1"]
    val_values   = [val_metrics["Accuracy"], val_metrics["Macro-F1"], val_metrics["Weighted-F1"]]
    test_values  = [test_metrics["Accuracy"], test_metrics["Macro-F1"], test_metrics["Weighted-F1"]]

    y = np.arange(len(metric_names))
    height = 0.30

    fig, ax = plt.subplots(figsize=(10, 4))
    bars_val  = ax.barh(y - height / 2, val_values, height, label=f"Validation (n={len(val_df):,})",
                        color=PARTITION_COLOURS["Val"], edgecolor="white", alpha=0.85)
    bars_test = ax.barh(y + height / 2, test_values, height, label=f"Test (n={len(test_df):,})",
                        color=PARTITION_COLOURS["Test"], edgecolor="white", alpha=0.85)

    for bars in [bars_val, bars_test]:
        for bar in bars:
            w = bar.get_width()
            fmt = f"{w:.1f}%" if w > 1 else f"{w:.4f}"
            ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2,
                    fmt, ha="left", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(metric_names, fontsize=11)
    ax.set_xlabel("Score", fontsize=11)
    ax.set_title("Majority-Class Baseline Performance (Predicts NO_OUTCOME for all instances)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0, max(max(val_values), max(test_values)) * 1.25)
    ax.grid(axis="x", alpha=0.3)

    # Callout
    ax.text(0.98, 0.05,
            "Macro-F1 ≈ 0.08 is the hurdle\nfor GRU Task 4.2 evaluation",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff9c4", edgecolor="gray", alpha=0.9))

    fig.tight_layout()
    out_path = OUTPUT_DIR / "majority_baseline_performance.png"
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVED] {out_path}")


# ==================================================================
# Plot 4: Issue Resolution Summary
# ==================================================================
def plot_issue_resolution_summary():
    """Severity-coded table summarising the 10 issues and resolution status."""
    issues = [
        ("ISSUE-01", "BLOCKER", "Sequence cross-split leakage",                    "RESOLVED", "test_split_before_sequence_construction_zero_leakage"),
        ("ISSUE-02", "BLOCKER", "Corpus-wide class weight contamination",           "RESOLVED", "test_train_only_weighting_invariant"),
        ("ISSUE-03", "HIGH",    "Distorted weights via mean normalization",          "RESOLVED", "test_compute_class_weights_exact_formula"),
        ("ISSUE-04", "HIGH",    "ABANDON overclaiming without lifecycle events",     "RESOLVED", "test_observable_abandonment_vs_stream_exhaustion"),
        ("ISSUE-05", "HIGH",    "Priority overriding earliest temporal outcome",     "RESOLVED", "test_earliest_event_wins_before_priority"),
        ("ISSUE-06", "MEDIUM",  "Brittle user threshold and placeholder handling",   "RESOLVED", "test_split_result_contract_and_reproducibility"),
        ("ISSUE-07", "MEDIUM",  "Window validity conflated with stride continuity",  "RESOLVED", "test_temporal_continuity_enforcement"),
        ("ISSUE-08", "MEDIUM",  "Incomplete distribution reporting",                 "RESOLVED", "02_target_distribution.ipynb §3"),
        ("ISSUE-09", "MEDIUM",  "Absent majority-class baseline",                   "RESOLVED", "evaluate_majority_baseline"),
        ("ISSUE-10", "LOW",     "Unverified gradient isolation on backbone",         "RESOLVED", "test_behavioral_gradient_flow"),
    ]

    fig, ax = plt.subplots(figsize=(16, 5.5))
    ax.axis("off")

    col_labels = ["Issue ID", "Severity", "Description", "Status", "Verification"]
    col_widths = [0.08, 0.08, 0.36, 0.08, 0.40]

    table_data = [[i[0], i[1], i[2], i[3], i[4]] for i in issues]

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        colWidths=col_widths,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#37474f")
        cell.set_text_props(color="white", fontweight="bold", fontsize=9)
        cell.set_edgecolor("white")

    # Style rows by severity
    for i, issue in enumerate(issues):
        severity = issue[1]
        bg_color = {
            "BLOCKER": "#ffcdd2",
            "HIGH":    "#ffe0b2",
            "MEDIUM":  "#fff9c4",
            "LOW":     "#c8e6c9",
        }.get(severity, "#ffffff")

        for j in range(len(col_labels)):
            cell = table[i + 1, j]
            cell.set_facecolor(bg_color)
            cell.set_edgecolor("#e0e0e0")
            # Bold the severity cell
            if j == 1:
                cell.set_text_props(fontweight="bold", color=SEVERITY_COLOURS.get(severity, "black"))
            # Green "RESOLVED" text
            if j == 3:
                cell.set_text_props(fontweight="bold", color="#2e7d32")

    fig.suptitle("Issue Resolution Summary — Pipeline Audit Tasks 3.1, 3.2, 4.1",
                 fontsize=13, fontweight="bold", y=0.97)

    # Legend patches
    legend_patches = [
        mpatches.Patch(color="#ffcdd2", label="BLOCKER"),
        mpatches.Patch(color="#ffe0b2", label="HIGH"),
        mpatches.Patch(color="#fff9c4", label="MEDIUM"),
        mpatches.Patch(color="#c8e6c9", label="LOW"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, title="Severity", title_fontsize=9)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "issue_resolution_summary.png"
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVED] {out_path}")


# ==================================================================
# Plot 5: Traceability Matrix
# ==================================================================
def plot_traceability_matrix():
    """Formatted table rendering of the Data-Fact-vs-Modelling-Decision matrix."""
    rows = [
        ("112,865 windows, 47 users",
         "Canonical Parquet",
         "Dataset scale for capacity planning",
         "Yes",
         "Descriptive corpus fact"),
        ("500 ms window, 250 ms stride overlap",
         "Window specification",
         "Partition before sequence building",
         "Yes",
         "Enforced: filter by partition before slicing"),
        ("47 distinct user_id strings",
         "Canonical metadata",
         "User-level split mandatory",
         "Yes",
         "Enforced: split_users_leak_free by user_id"),
        ("Class imbalance 9,854:1 in train",
         "Training partition",
         "Weight comparison needed",
         "Yes",
         "Enforced: weights from training dataset.Y"),
        ("FORM_SUBMIT 1 user/part, BACKTRACK 0 in test",
         "Partition distribution",
         "Cannot generalise rare classes",
         "Yes",
         "Documented: expected failure modes"),
        ("Zero DOM lifecycle events",
         "Raw AdSERP JSON",
         "ABANDON is recording proxy",
         "Yes",
         "Enforced: metadata + documentation"),
        ("13.27% step gaps > 250 ms",
         "Interim timestamps",
         "Consecutive active windows only",
         "Yes",
         "Enforced: temporal continuity param"),
        ("Baseline ~39% acc, 0.08 Macro-F1",
         "Val/Test partitions",
         "Accuracy not primary metric",
         "Yes",
         "Enforced: Macro-F1 reporting"),
        ("No intervention ground truth",
         "Corpus semantic audit",
         "TargetInterventionHead is scaffold",
         "Yes",
         "Enforced: zero synthetic labels"),
    ]

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.axis("off")

    col_labels = ["Observation", "Source", "Modelling Implication", "Valid?", "Status"]
    col_widths = [0.24, 0.13, 0.22, 0.05, 0.36]

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        colWidths=col_widths,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.55)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#1565c0")
        cell.set_text_props(color="white", fontweight="bold", fontsize=9)
        cell.set_edgecolor("white")

    # Alternating row colours
    for i in range(len(rows)):
        bg = "#f5f5f5" if i % 2 == 0 else "#ffffff"
        for j in range(len(col_labels)):
            cell = table[i + 1, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor("#e0e0e0")
            # Green checkmark column
            if j == 3:
                cell.set_text_props(fontweight="bold", color="#2e7d32", ha="center")

    fig.suptitle("Data Fact vs Modelling Decision — Traceability Matrix",
                 fontsize=13, fontweight="bold", y=0.97)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "traceability_matrix.png"
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVED] {out_path}")


# ==================================================================
# Main
# ==================================================================
def main():
    print("=" * 60)
    print("Experiment 2: Pipeline Audit Diagnostic Plot Generation")
    print("=" * 60)

    # Load data
    print("\n[1/6] Loading parquet data...")
    df = load_parquet_df()
    print(f"  Total windows: {len(df):,}")
    print(f"  Unique users:  {df['user_id'].nunique()}")
    print(f"  Unique sessions: {df['session_id'].nunique()}")

    # Split
    print("\n[2/6] Computing leak-free partition split...")
    split = split_users_leak_free(str(PARQUET_PATH))
    print(f"  Split by: {split.split_by}")
    print(f"  Train IDs: {len(split.train_ids)}, Val IDs: {len(split.val_ids)}, Test IDs: {len(split.test_ids)}")

    train_df, val_df, test_df = compute_partition_dfs(df, split)
    print(f"  Train windows: {len(train_df):,}")
    print(f"  Val windows:   {len(val_df):,}")
    print(f"  Test windows:  {len(test_df):,}")

    # Generate plots
    print("\n[3/6] Generating target distribution plot...")
    plot_target_distribution(df, train_df, val_df, test_df)

    print("\n[4/6] Generating class weight comparison plot...")
    plot_class_weight_comparison(train_df)

    print("\n[5/6] Generating majority baseline performance plot...")
    plot_majority_baseline(val_df, test_df)

    print("\n[6/6] Generating issue summary and traceability plots...")
    plot_issue_resolution_summary()
    plot_traceability_matrix()

    print("\n" + "=" * 60)
    print(f"All 5 diagnostic plots saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
