#!/usr/bin/env python3
"""
generate_intervention_experiments_nb.py
Programmatically generates the interactive reproducibility notebook for Phase D:
`notebooks/03_intervention_head_experiments.ipynb`.

Mirrors Plan 0's notebook generator pattern (generate_training_nb.py, generate_target_distribution_nb.py).
Equipped with:
- Google Colab badge pointing to active git branch.
- Automated hosted environment detection (Colab / Kaggle / Local).
- Hardware accelerator selection (CUDA / MPS / CPU).
- Interactive execution of transfer-learning regimes E1, E2, E3.
- Comprehensive multi-metric evaluation table and failure case analysis.
- Publication-quality inline diagnostic visualisations.
"""

import json
import os
import subprocess
from pathlib import Path


def get_current_branch(default: str = "explore/pipeline/revision/1") -> str:
    env_branch = os.environ.get("GITHUB_REF_NAME") or os.environ.get("GIT_BRANCH")
    if env_branch:
        return env_branch
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        b = res.stdout.strip()
        if b and b != "HEAD":
            return b
    except Exception:
        pass
    return default


def create_intervention_notebook(branch: str = "explore/pipeline/revision/1") -> dict:
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Experiment 3: Target-Domain Intervention-Head Progressive Fine-Tuning\n",
                    "\n",
                    f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/taofeeqhamzat/edge-aui-model-preparation/blob/{branch}/notebooks/03_intervention_head_experiments.ipynb)\n",
                    "\n",
                    "This interactive notebook reproduces **Phase D (Tasks 12.1–12.4)** of the Edge-AUI Framework model-preparation pipeline:\n",
                    "1. Ingests the provenance-versioned, session-bounded target dataset (`v1.0.0`, 289 examples across 18 sessions).\n",
                    "2. Evaluates the **Majority-Class Baseline** as the primary lower-bound reference on imbalanced distributions.\n",
                    "3. Executes **Experiment E1 (Strict Freezing)**: Frozen GRU backbone + trained `TargetInterventionHead`.\n",
                    "4. Executes **Experiment E2 (Partial Fine-Tuning)**: Frozen lower GRU layer + unfrozen terminal layer `_l1` + trained head with differential learning rates.\n",
                    "5. Executes **Experiment E3 (Full Fine-Tuning)**: Full GRU backbone + trained head, paired with a foundation outcome retention audit to monitor catastrophic forgetting.\n",
                    "6. Generates a multi-metric comparative report (Macro-F1, Accuracy, HR@K, MRR, per-class metrics, confusion matrices, and failure cases).\n",
                    "7. Adheres strictly to **ADR-013 Claim Boundaries**: labels reflect the scripted policy teacher, proving pipeline convergence and edge readiness without claiming human participant generalization."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Environment Setup & Hardware Accelerator Detection\n",
                    "Configures dependencies, sets up repository paths, and initializes execution hardware (CUDA / Apple Silicon MPS / CPU)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Autoreload modules for development\n",
                    "try:\n",
                    "    ip = get_ipython()\n",
                    "    if ip is not None:\n",
                    "        ip.run_line_magic('load_ext', 'autoreload')\n",
                    "        ip.run_line_magic('autoreload', '2')\n",
                    "except Exception:\n",
                    "    pass\n",
                    "\n",
                    "import os\n",
                    "import sys\n",
                    "import json\n",
                    "from pathlib import Path\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "import torch\n",
                    "import torch.nn as nn\n",
                    "\n",
                    "# Environment detection\n",
                    "IN_COLAB = 'google.colab' in sys.modules or 'COLAB_GPU' in os.environ\n",
                    "IN_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ\n",
                    "\n",
                    "if IN_COLAB:\n",
                    "    print('[Setup] Detected Google Colaboratory environment.')\n",
                    f"    if not os.path.exists('src'):\n",
                    f"        !git clone -b {branch} https://github.com/taofeeqhamzat/edge-aui-model-preparation.git\n",
                    f"        %cd edge-aui-model-preparation\n",
                    "    !pip install -q pyarrow pandas matplotlib seaborn torch scikit-learn pyyaml\n",
                    "elif IN_KAGGLE:\n",
                    "    print('[Setup] Detected Kaggle Kernel environment.')\n",
                    "    !pip install -q pyarrow pandas matplotlib seaborn torch scikit-learn pyyaml\n",
                    "else:\n",
                    "    print('[Setup] Detected Local / Self-Hosted Environment.')\n",
                    "\n",
                    "# Add src to sys.path across possible execution directories\n",
                    "for path_candidate in ['src', '../src', './model-preparation/src']:\n",
                    "    abs_p = os.path.abspath(path_candidate)\n",
                    "    if os.path.isdir(abs_p) and abs_p not in sys.path:\n",
                    "        sys.path.insert(0, abs_p)\n",
                    "        print(f'[System] Added to sys.path: {abs_p}')\n",
                    "        break\n",
                    "\n",
                    "# Hardware accelerator detection\n",
                    "from training import get_device\n",
                    "device = get_device()\n",
                    "print(f'[Hardware] Optimal Execution Provider: {device}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Ingestion of Versioned Dataset & Manifest Inspection\n",
                    "Loads the frozen `v1.0.0` session-bounded dataset (`target_intervention_dataset.parquet`) and inspects class distributions and train-derived class weights."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from target_dataset import load_intervention_dataset, get_intervention_class_weights, INTERVENTION_VOCABULARY\n",
                    "\n",
                    "DATASET_VERSION = 'v1.0.0'\n",
                    "train_ds = load_intervention_dataset(version=DATASET_VERSION, split='train')\n",
                    "val_ds = load_intervention_dataset(version=DATASET_VERSION, split='val')\n",
                    "test_ds = load_intervention_dataset(version=DATASET_VERSION, split='test')\n",
                    "class_weights = get_intervention_class_weights(version=DATASET_VERSION, device=device)\n",
                    "\n",
                    "print('=' * 70)\n",
                    "print(f'DATASET v1.0.0 PARTITION SUMMARY:')\n",
                    "print(f' - Train Partition:  {len(train_ds)} sequences (13 sessions)')\n",
                    "print(f' - Val Partition:    {len(val_ds)} sequences (3 sessions)')\n",
                    "print(f' - Test Partition:   {len(test_ds)} sequences (2 sessions)')\n",
                    "print(f' - Total Examples:   {len(train_ds) + len(val_ds) + len(test_ds)}')\n",
                    "print('-' * 70)\n",
                    "print('RESOLVED TRAINING-PARTITION CLASS WEIGHTS:')\n",
                    "for i, w in enumerate(class_weights.cpu().numpy()):\n",
                    "    print(f'  [{i}] {INTERVENTION_VOCABULARY[i]:<25}: {w:.4f}')\n",
                    "print('=' * 70)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Majority-Class Baseline Evaluation\n",
                    "Computes the majority-class lower bound evaluated using the training majority class (`no_op`).\n",
                    "On imbalanced distributions, accuracy alone can be misleading; Macro-F1 reveals the true performance floor."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from training import evaluate_intervention_majority_baseline\n",
                    "\n",
                    "maj_val = evaluate_intervention_majority_baseline(train_ds, val_ds)\n",
                    "maj_test = evaluate_intervention_majority_baseline(train_ds, test_ds)\n",
                    "\n",
                    "print(f\"Majority Class (Train-Derived): '{maj_val['majority_class_name']}' (ID {maj_val['majority_class_id']})\")\n",
                    "print(f\"Validation Partition: Accuracy = {maj_val['accuracy']*100:.1f}%, Macro-F1 = {maj_val['macro_f1']:.4f}\")\n",
                    "print(f\"Test Partition:       Accuracy = {maj_test['accuracy']*100:.1f}%, Macro-F1 = {maj_test['macro_f1']:.4f}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Experiment E1: Strict Freezing (Frozen Backbone + Trained Head)\n",
                    "Loads the pre-trained recurrent GRU backbone from `models/foundational_gru.pth`.\n",
                    "Backbone parameters are strictly frozen ($\\\\frac{\\\\partial \\\\mathcal{L}}{\\\\partial \\\\theta_{\\\\text{base}}} = 0$), and only the `TargetInterventionHead` is trained with learning rate $0.001$."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from training import train_intervention_model\n",
                    "\n",
                    "print('[Training] Launching Experiment E1 (Strict Freezing)...')\n",
                    "res_e1 = train_intervention_model(\n",
                    "    experiment='e1',\n",
                    "    dataset_version=DATASET_VERSION,\n",
                    "    seed=42,\n",
                    "    epochs=5,\n",
                    "    batch_size=64,\n",
                    "    device=str(device),\n",
                    "    verbose=True,\n",
                    ")\n",
                    "\n",
                    "print(f\"E1 Final Val Accuracy: {res_e1['val_evaluation']['accuracy']:.1f}%\")\n",
                    "print(f\"E1 Final Val Macro-F1: {res_e1['val_evaluation']['macro_f1']:.4f}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Experiment E2: Partial Fine-Tuning (Terminal Layer + Head)\n",
                    "Freezes lower GRU layer `_l0` and unfreezes the terminal GRU layer `_l1`.\n",
                    "Trains jointly with the head using disjoint parameter groups with differential learning rates:\n",
                    "- Terminal GRU layer `_l1`: $\\\\text{lr} = 0.0001$\n",
                    "- `TargetInterventionHead`: $\\\\text{lr} = 0.001$"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('[Training] Launching Experiment E2 (Partial Fine-Tuning)...')\n",
                    "res_e2 = train_intervention_model(\n",
                    "    experiment='e2',\n",
                    "    dataset_version=DATASET_VERSION,\n",
                    "    seed=42,\n",
                    "    epochs=5,\n",
                    "    batch_size=64,\n",
                    "    device=str(device),\n",
                    "    verbose=True,\n",
                    ")\n",
                    "\n",
                    "print(f\"E2 Final Val Accuracy: {res_e2['val_evaluation']['accuracy']:.1f}%\")\n",
                    "print(f\"E2 Final Val Macro-F1: {res_e2['val_evaluation']['macro_f1']:.4f}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Experiment E3: Full Fine-Tuning & Retention Audit\n",
                    "Unfreezes all layers of the `EdgeAUIGRU` backbone and trains end-to-end with a conservative uniform learning rate ($\\\\text{lr} = 0.0005$).\n",
                    "Audits **catastrophic forgetting** by evaluating whether the backbone retains its outcome prediction capability on foundational AdSERP sequences."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('[Training] Launching Experiment E3 (Full Fine-Tuning)...')\n",
                    "res_e3 = train_intervention_model(\n",
                    "    experiment='e3',\n",
                    "    dataset_version=DATASET_VERSION,\n",
                    "    seed=42,\n",
                    "    epochs=5,\n",
                    "    batch_size=64,\n",
                    "    device=str(device),\n",
                    "    verbose=True,\n",
                    ")\n",
                    "\n",
                    "print(f\"E3 Final Val Accuracy: {res_e3['val_evaluation']['accuracy']:.1f}%\")\n",
                    "print(f\"E3 Final Val Macro-F1: {res_e3['val_evaluation']['macro_f1']:.4f}\")\n",
                    "if res_e3.get('retention_diagnostics'):\n",
                    "    rd = res_e3['retention_diagnostics']\n",
                    "    print(f\"Foundation Retention Drop: {rd['relative_drop_pct']:.1f}% (Threshold <= 15.0%: {'PASSED' if rd['retention_passed'] else 'FLAGGED'})\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Comparative Multi-Metric Evaluation on Held-Out Test Data\n",
                    "Evaluates the converged checkpoints on the held-out test partition ($N=30$) and compares all candidates directly against the majority-class baseline."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from training import evaluate_intervention_suite\n",
                    "\n",
                    "suite_results = evaluate_intervention_suite(dataset_version=DATASET_VERSION, device=str(device))\n",
                    "\n",
                    "# Build Pandas summary table\n",
                    "table_data = []\n",
                    "base_val = suite_results['majority_baseline']['val']\n",
                    "base_test = suite_results['majority_baseline']['test']\n",
                    "table_data.append({\n",
                    "    'Regime': 'Majority Baseline',\n",
                    "    'Val Macro-F1': f\"{base_val['macro_f1']:.4f}\",\n",
                    "    'Val Acc (%)': f\"{base_val['accuracy']*100:.1f}%\",\n",
                    "    'Test Macro-F1': f\"{base_test['macro_f1']:.4f}\",\n",
                    "    'Test Acc (%)': f\"{base_test['accuracy']*100:.1f}%\",\n",
                    "    'Payload (KB)': '-'\n",
                    "})\n",
                    "\n",
                    "for exp_key in ['e1', 'e2', 'e3']:\n",
                    "    exp_data = suite_results['experiments'][exp_key]\n",
                    "    table_data.append({\n",
                    "        'Regime': f'Exp {exp_key.upper()}',\n",
                    "        'Val Macro-F1': f\"{exp_data['val']['macro_f1']:.4f}\",\n",
                    "        'Val Acc (%)': f\"{exp_data['val']['accuracy']:.1f}%\",\n",
                    "        'Test Macro-F1': f\"{exp_data['test']['macro_f1']:.4f}\",\n",
                    "        'Test Acc (%)': f\"{exp_data['test']['accuracy']:.1f}%\",\n",
                    "        'Payload (KB)': f\"{exp_data['size_kb']:.1f}\"\n",
                    "    })\n",
                    "\n",
                    "df_summary = pd.DataFrame(table_data)\n",
                    "display(df_summary)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 8. Interactive Diagnostic Visualizations\n",
                    "Generates side-by-side empirical confusion matrices, Macro-F1 ablation charts, and per-class performance comparisons."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "short_vocab = ['simplify', 'highlight', 'assistance', 'tooltip', 'no_op']\n",
                    "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
                    "\n",
                    "for idx, exp_key in enumerate(['e1', 'e2', 'e3']):\n",
                    "    cm = np.array(suite_results['experiments'][exp_key]['test']['confusion_matrix'])\n",
                    "    ax = axes[idx]\n",
                    "    sns.heatmap(\n",
                    "        cm,\n",
                    "        annot=True,\n",
                    "        fmt='d',\n",
                    "        cmap='Blues',\n",
                    "        cbar=False,\n",
                    "        xticklabels=short_vocab,\n",
                    "        yticklabels=short_vocab,\n",
                    "        ax=ax,\n",
                    "        linewidths=1\n",
                    "    )\n",
                    "    acc = suite_results['experiments'][exp_key]['test']['accuracy']\n",
                    "    f1 = suite_results['experiments'][exp_key]['test']['macro_f1']\n",
                    "    ax.set_title(f\"Exp {exp_key.upper()} (Test Acc: {acc:.1f}% | Macro-F1: {f1:.4f})\", fontweight='bold')\n",
                    "    ax.set_xlabel('Predicted Intervention')\n",
                    "    ax.set_ylabel('True Intervention' if idx == 0 else '')\n",
                    "\n",
                    "plt.suptitle('Test Partition Empirical Confusion Matrices (v1.0.0, N=30)', fontsize=14, fontweight='bold', y=1.03)\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 9. Failure Case Analysis & ADR-013 Methodological Boundary\n",
                    "Examines representative misclassified sequences to understand model error boundaries, and summarizes methodological conclusions."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "e3_test = suite_results['experiments']['e3']['test']\n",
                    "print(f\"Experiment E3 Total Misclassifications on Test: {len(e3_test['failure_cases'])} / {e3_test['n_samples']}\")\n",
                    "print('-' * 70)\n",
                    "print('DOMINANT CONFUSION PAIRS:')\n",
                    "for cp in e3_test['dominant_confusion_pairs'][:5]:\n",
                    "    print(f\"  - True: '{cp['true_class']:<18}' -> Predicted: '{cp['pred_class']:<18}' ({cp['count']} occurrences)\")\n",
                    "\n",
                    "if e3_test['failure_cases']:\n",
                    "    sample = e3_test['failure_cases'][0]\n",
                    "    print('-' * 70)\n",
                    "    print('REPRESENTATIVE MISCLASSIFIED SAMPLE:')\n",
                    "    print(f\"  - Session ID:       {sample['session_id']}\")\n",
                    "    print(f\"  - Task ID:          {sample['task_id']}\")\n",
                    "    print(f\"  - Anchor Window ID: {sample['anchor_window_id']}\")\n",
                    "    print(f\"  - True Target:      {sample['true_class']}\")\n",
                    "    print(f\"  - Predicted:        {sample['pred_class']} (Confidence: {sample['confidence']*100:.1f}%)\")\n",
                    "    print(f\"  - UIContext R^6:    {[round(x, 4) for x in sample['context_vector']]}\")\n",
                    "print('=' * 70)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### ADR-013 Methodological Conclusion\n",
                    "\n",
                    "> **Claim Notice:** Results from this experiment demonstrate that:\n",
                    "> 1. The target-domain dataset preparation, sequence vectorization, and model training path executes end-to-end.\n",
                    "> 2. `TargetInterventionHead` conditioned on $R^6$ UIContext converges cleanly across progressive fine-tuning regimes.\n",
                    "> 3. The INT8-quantizable GRU artifact satisfies the edge storage budget ($184.3\\\\text{ KB} \\\\le 200\\\\text{ KB}$).\n",
                    ">\n",
                    "> In accordance with **ADR-013**, results **must NOT be reported as evidence of outperforming the deterministic policy** (as the head learned from that policy as its teacher) nor as generalizable human participant usability findings."
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 5,
                "pygments_lexer": "ipython3",
                "version": "3.11.15"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def main():
    root = Path(__file__).resolve().parent
    notebooks_dir = root / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)
    out_path = notebooks_dir / "03_intervention_head_experiments.ipynb"

    branch = get_current_branch()
    print(f"[Generator] Target branch for Colab badge: '{branch}'")

    nb = create_intervention_notebook(branch=branch)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"[Generator] Successfully generated notebook: {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
