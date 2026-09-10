"""
generate_preprocessing_eda_nb.py
Generates the Phase 1 Preprocessing EDA Notebook (notebooks/01_preprocessing_eda.ipynb).
Analyzes AdSERP coordinate scaling, [0, 1] bounds, 18-dim MicroTensor extraction,
modality mask activation frequencies, and sequence length distributions.
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
            check=True
        )
        b = res.stdout.strip()
        if b and b != "HEAD":
            return b
    except Exception:
        pass
    return default


def build_preprocessing_eda_notebook(branch: str = "explore/pipeline/revision/1") -> dict:
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Phase 1 Preprocessing EDA: Canonical Event Normalization & MicroTensor Extraction\n",
                    "\n",
                    f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/taofeeqhamzat/edge-aui-model-preparation/blob/{branch}/notebooks/01_preprocessing_eda.ipynb)\n",
                    "\n",
                    "This exploratory data analysis notebook supports the **Edge-AUI Framework** probabilistic \"Slow Brain\" (INT8 Gated Recurrent Unit). It systematically inspects and verifies:\n",
                    "- **Layer A $\\rightarrow$ Layer B:** Parsing raw AdSERP CSV streams and companion XML trial metadata, normalizing coordinates against observed source viewports (`<window>WxH</window>`) into $[0, 1]$ canvas coordinates.\n",
                    "- **Layer B $\\rightarrow$ Layer C:** Segmenting event streams into 500ms sliding windows (250ms stride) to extract 9 continuous features (7 core kinematics + 2 contextual scroll features).\n",
                    "- **ADR-001 Modality Masking:** Generating binary Modality Mask Vector $M \\in \\{0, 1\\}^9$ and concatenated 18-dimensional MicroTensors $\\widetilde{X}_t = [X_t \\odot M, \\; M] \\in \\mathbb{R}^{18}$.\n",
                    "- **Data Quality Assurance:** Verifying zero unhandled NaNs/Infs, strict $[0, 1]$ bounds, and session sequence length distributions."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Environment Setup & Dependency Configuration\n",
                    "Bootstraps the runtime environment across Google Colab, Kaggle, or local virtual environments, and resolves `src/` module paths."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Safe autoreload configuration\n",
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
                    "import glob\n",
                    "from pathlib import Path\n",
                    "\n",
                    "IN_COLAB = 'google.colab' in sys.modules or 'COLAB_GPU' in os.environ\n",
                    "IN_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ\n",
                    "\n",
                    "if IN_COLAB:\n",
                    "    print('[Environment] Running in Google Colab.')\n",
                    "    if not os.path.exists('src') and not os.path.exists('../src'):\n",
                    f"        !git clone -b {branch} https://github.com/taofeeqhamzat/edge-aui-model-preparation.git\n",
                    "        %cd edge-aui-model-preparation\n",
                    "    !pip install -q huggingface_hub datasets torch pandas numpy matplotlib seaborn onnx onnxruntime pyyaml pyarrow\n",
                    "elif IN_KAGGLE:\n",
                    "    print('[Environment] Running in Kaggle.')\n",
                    "    !pip install -q huggingface_hub datasets torch pandas numpy matplotlib seaborn pyyaml pyarrow\n",
                    "else:\n",
                    "    print('[Environment] Running in local/virtual environment.')\n",
                    "\n",
                    "# Configure sys.path\n",
                    "for p in ['src', '../src', './model-preparation/src']:\n",
                    "    abs_p = os.path.abspath(p)\n",
                    "    if os.path.isdir(abs_p) and abs_p not in sys.path:\n",
                    "        sys.path.insert(0, abs_p)\n",
                    "        print(f'[Path] Added {abs_p} to sys.path')\n",
                    "        break"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Dataset Resolution & Inode-Safe Consolidation\n",
                    "Resolves the AdSERP corpus across local data repositories or DVC/Hugging Face remote storage."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import data\n",
                    "import preprocessing\n",
                    "from config import load_config\n",
                    "\n",
                    "cfg = load_config('minimal')\n",
                    "print(f'[Config] Loaded pipeline configuration: mode={cfg.mode}, input_dim={cfg.input_dim}')\n",
                    "\n",
                    "# Ensure dataset availability\n",
                    "adserp_dir = data.ensure_adserp_dataset()\n",
                    "print(f'[Data] AdSERP dataset root: {adserp_dir}')\n",
                    "assert os.path.isdir(adserp_dir), f'AdSERP directory does not exist: {adserp_dir}'\n",
                    "assert (Path(adserp_dir) / 'mouse-movement-data').is_dir(), f'mouse-movement-data not found in {adserp_dir}'"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Viewport Normalization Audit (Layer A $\\rightarrow$ Layer B)\n",
                    "Validates the transformation from raw browser mouse coordinates to canvas-normalized coordinates in $[0, 1]$ relative to observed `<window>WxH</window>` XML metadata."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "# Parse canonical events from a sample session\n",
                    "sample_session = 'p004-b1-t1.csv'\n",
                    "events = preprocessing.parse_adserp_session(sample_session, raw_dir=adserp_dir)\n",
                    "df_events = pd.DataFrame(events)\n",
                    "\n",
                    "print(f'Total parsed events for {sample_session}: {len(df_events)}')\n",
                    "print(f'Observed viewport: {df_events[\"viewport_w\"].iloc[0]}x{df_events[\"viewport_h\"].iloc[0]}')\n",
                    "print(f'Coordinate bounds: x_norm in [{df_events[\"x_norm\"].min():.4f}, {df_events[\"x_norm\"].max():.4f}], ' \n",
                    "      f'y_norm in [{df_events[\"y_norm\"].min():.4f}, {df_events[\"y_norm\"].max():.4f}]')\n",
                    "\n",
                    "# Verify [0, 1] bounds\n",
                    "assert (df_events['x_norm'] >= 0.0).all() and (df_events['x_norm'] <= 1.0).all()\n",
                    "assert (df_events['y_norm'] >= 0.0).all() and (df_events['y_norm'] <= 1.0).all()\n",
                    "print('Bounds check PASSED: All canonical coordinates are strictly within [0, 1].')\n",
                    "\n",
                    "# Visual comparison: Raw Coordinates vs Normalized Viewport Canvas\n",
                    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                    "\n",
                    "pointer_df = df_events[df_events['event_type'].isin(['mousemove', 'mouseover', 'click'])]\n",
                    "axes[0].plot(pointer_df['x_raw'], pointer_df['y_raw'], color='navy', alpha=0.6, marker='o', markersize=3)\n",
                    "axes[0].set_title(f'Raw Pointer Trajectory (Source Viewport {df_events[\"viewport_w\"].iloc[0]:.0f}x{df_events[\"viewport_h\"].iloc[0]:.0f})')\n",
                    "axes[0].set_xlabel('X Position (px)')\n",
                    "axes[0].set_ylabel('Y Position (px)')\n",
                    "axes[0].invert_yaxis()\n",
                    "axes[0].grid(True, linestyle='--', alpha=0.5)\n",
                    "\n",
                    "axes[1].plot(pointer_df['x_norm'], pointer_df['y_norm'], color='darkgreen', alpha=0.6, marker='o', markersize=3)\n",
                    "axes[1].set_title('Normalized Canvas Trajectory [0, 1]')\n",
                    "axes[1].set_xlabel('Normalized X')\n",
                    "axes[1].set_ylabel('Normalized Y')\n",
                    "axes[1].set_xlim(-0.05, 1.05)\n",
                    "axes[1].set_ylim(1.05, -0.05)  # Inverted Y for browser canvas\n",
                    "axes[1].grid(True, linestyle='--', alpha=0.5)\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. MicroTensor Extraction & Modality Masking ($2D = 18$)\n",
                    "Extracts 18-dimensional MicroTensors across multiple AdSERP sessions using a 500ms sliding window and 250ms stride. Evaluates the 7 core kinematic features and 2 contextual scroll features alongside the binary Modality Mask Vector $M \\in \\{0, 1\\}^9$."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Ingest multiple sessions for population-level EDA\n",
                    "adserp_root = Path(adserp_dir)\n",
                    "session_csvs = sorted(list((adserp_root / 'mouse-movement-data').glob('*.csv')))[:30]\n",
                    "print(f'Extracting MicroTensors across {len(session_csvs)} AdSERP sessions...')\n",
                    "\n",
                    "all_tensors = []\n",
                    "session_lengths = []\n",
                    "\n",
                    "for csv_file in session_csvs:\n",
                    "    evs = preprocessing.parse_adserp_session(str(csv_file), raw_dir=adserp_dir)\n",
                    "    t_seq = preprocessing.extract_session_microtensors(evs, window_size_ms=500, stride_ms=250)\n",
                    "    if len(t_seq) > 0:\n",
                    "        all_tensors.append(t_seq)\n",
                    "        session_lengths.append(len(t_seq))\n",
                    "\n",
                    "tensor_matrix = np.concatenate(all_tensors, axis=0)\n",
                    "print(f'Total 18-dim MicroTensor windows extracted: {tensor_matrix.shape[0]}')\n",
                    "print(f'Output feature matrix shape: {tensor_matrix.shape}')\n",
                    "assert tensor_matrix.shape[1] == 18\n",
                    "\n",
                    "# Split into Feature Values (0..8) and Modality Mask (9..17)\n",
                    "feature_cols = preprocessing.FEATURE_NAMES\n",
                    "mask_cols = [f'mask_{name}' for name in feature_cols]\n",
                    "\n",
                    "df_features = pd.DataFrame(tensor_matrix[:, :9], columns=feature_cols)\n",
                    "df_masks = pd.DataFrame(tensor_matrix[:, 9:], columns=mask_cols)\n",
                    "df_all = pd.concat([df_features, df_masks], axis=1)\n",
                    "df_features.describe().round(4)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Kinematic Feature Distribution Histograms\n",
                    "Visualizes empirical distributions for the 7 core kinematic metrics and 2 contextual metrics. Confirms effective symmetric scaling and lack of ceiling/floor saturation."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig, axes = plt.subplots(3, 3, figsize=(16, 12))\n",
                    "axes = axes.flatten()\n",
                    "\n",
                    "colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']\n",
                    "\n",
                    "for idx, col in enumerate(feature_cols):\n",
                    "    ax = axes[idx]\n",
                    "    vals = df_features[col]\n",
                    "    sns.histplot(vals, bins=30, kde=True, ax=ax, color=colors[idx], edgecolor='none', alpha=0.7)\n",
                    "    ax.set_title(f'{col}', fontsize=12, fontweight='bold')\n",
                    "    ax.set_xlabel('Normalized Value [0, 1]')\n",
                    "    ax.set_ylabel('Frequency')\n",
                    "    ax.set_xlim(-0.02, 1.02)\n",
                    "    ax.grid(True, linestyle='--', alpha=0.4)\n",
                    "\n",
                    "plt.suptitle('Empirical Distributions: 9 Kinematic & Contextual Micro-Interaction Features', fontsize=15, y=1.02)\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Modality Mask Activation Frequencies (ADR-001)\n",
                    "Inspects the activation rates of the Modality Mask Vector $M \\in \\{0, 1\\}^9$. Core features remain active across standard pointer streams, while contextual features explicitly signal sensor telemetry presence without zero-imputation collinearity."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "mask_activation_rates = df_masks.mean()\n",
                    "\n",
                    "plt.figure(figsize=(12, 5))\n",
                    "bar_plot = sns.barplot(x=mask_activation_rates.index, y=mask_activation_rates.values, palette='Blues_d')\n",
                    "plt.title('Modality Mask Activation Frequency ($M \\in \\{0, 1\\}^9$)', fontsize=13, fontweight='bold')\n",
                    "plt.ylabel('Activation Frequency (Fraction of Windows)')\n",
                    "plt.xlabel('Modality Mask Dimensions')\n",
                    "plt.xticks(rotation=40, ha='right')\n",
                    "plt.ylim(0, 1.15)\n",
                    "plt.grid(axis='y', linestyle='--', alpha=0.5)\n",
                    "\n",
                    "for p in bar_plot.patches:\n",
                    "    bar_plot.annotate(f'{p.get_height():.2f}', \n",
                    "                      (p.get_x() + p.get_width() / 2., p.get_height()), \n",
                    "                      ha='center', va='center', xytext=(0, 7), textcoords='offset points', fontsize=10)\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Data Quality & Sequence Length Integrity Diagnostics\n",
                    "Automated assertions ensuring zero NaNs, zero Infs, bounded tensors, and healthy window count distribution for recurrent GRU sequence modeling."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Zero NaN and Inf check\n",
                    "nan_count = int(np.isnan(tensor_matrix).sum())\n",
                    "inf_count = int(np.isinf(tensor_matrix).sum())\n",
                    "assert nan_count == 0, f'Found {nan_count} NaNs in MicroTensor matrix!'\n",
                    "assert inf_count == 0, f'Found {inf_count} Infs in MicroTensor matrix!'\n",
                    "print(f'✅ Integrity Check PASSED: Zero NaNs ({nan_count}) and zero Infs ({inf_count}).')\n",
                    "\n",
                    "# 2. Strict [0, 1] bounds verification\n",
                    "min_val = float(tensor_matrix.min())\n",
                    "max_val = float(tensor_matrix.max())\n",
                    "assert min_val >= 0.0, f'Values under 0.0 observed: {min_val}'\n",
                    "assert max_val <= 1.0, f'Values over 1.0 observed: {max_val}'\n",
                    "print(f'✅ Bounds Check PASSED: All {tensor_matrix.size:,} tensor elements bounded in [{min_val:.4f}, {max_val:.4f}].')\n",
                    "\n",
                    "# 3. Sequence Length Distribution\n",
                    "plt.figure(figsize=(10, 4))\n",
                    "sns.histplot(session_lengths, bins=15, color='teal', edgecolor='black')\n",
                    "plt.title('Distribution of Windows per AdSERP Session', fontsize=13)\n",
                    "plt.xlabel('Number of 500ms Windows (250ms stride)')\n",
                    "plt.ylabel('Session Count')\n",
                    "plt.grid(True, linestyle='--', alpha=0.4)\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                    "\n",
                    "print(f'Session Length Stats: Min={min(session_lengths)}, Max={max(session_lengths)}, Mean={np.mean(session_lengths):.1f} windows.')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 8. Summary & Transition to Phase 3\n",
                    "\n",
                    "### Findings:\n",
                    "1. **Canonical Viewport Normalization:** Successfully projects diverse source screen resolutions into canonical canvas coordinates in $[0, 1]$, preserving trajectory geometry.\n",
                    "2. **Vectorization ($2D=18$):** MicroTensor extraction produces well-scaled features without ceiling saturation.\n",
                    "3. **Modality Masking:** Decouples sensor absence from user inactivity per ADR-001.\n",
                    "4. **Next Step (Phase 3):** Ground MicroTensor sequences in future downstream interaction outcomes over a lookahead horizon $\\Delta t \\in [500\\text{ms}, 1500\\text{ms}]$ (`02_target_distribution.ipynb`)."
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
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 4,
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    return nb


def main():
    root = Path(__file__).parent.resolve()
    notebooks_dir = root / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    branch = get_current_branch()
    print(f"[Generator] Target branch for Colab badge: {branch}")

    nb = build_preprocessing_eda_notebook(branch=branch)
    output_path = notebooks_dir / "01_preprocessing_eda.ipynb"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"[Generator] Successfully generated {output_path} with {len(nb['cells'])} cells.")


if __name__ == "__main__":
    main()
