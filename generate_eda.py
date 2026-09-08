import json
import os
import subprocess


def get_current_branch(default: str = "main") -> str:
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


def create_eda_notebook(branch: str = "main") -> dict:
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Exploratory Data Analysis: Edge-AUI Behavioral Datasets\n",
                    "\n",
                    f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/taofeeqhamzat/edge-aui-model-preparation/blob/{branch}/notebooks/EDA.ipynb)\n",
                    "\n",
                    "This notebook analyzes the continuous kinematic interaction datasets supporting the **Edge-AUI Framework** (Continuous Kinematics 2020 and High-Volume Trajectories 20226).\n",
                    "It extracts 9-dimensional MicroTensors across continuous datasets, evaluates unit harmonization and symmetric scaling, and visualizes feature distributions and correlation structures without discrete category distortion."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Hosted Environment Setup & Dependency Installation\n",
                    "Detects Google Colab or Kaggle runtimes, installs dependencies, sets up module paths, and synchronizes hosted datasets from Hugging Face Hub (`T40/edge-aui-framework-data`)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Safe autoreload (Python 3.13 removed legacy 'imp' module used by older IPython)\n",
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
                    "\n",
                    "# Environment detection\n",
                    "IN_COLAB = 'google.colab' in sys.modules or 'COLAB_GPU' in os.environ\n",
                    "IN_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ\n",
                    "\n",
                    "if IN_COLAB:\n",
                    "    print(\"[Environment] Running in Google Colab. Setting up dependencies...\")\n",
                    "    # Clone repo if executing standalone in Colab, or pull latest updates\n",
                    "    if not os.path.exists(\"src\") and not os.path.exists(\"../src\"):\n",
                    f"        !git clone -b {branch} https://github.com/taofeeqhamzat/edge-aui-model-preparation.git\n",
                    "        %cd edge-aui-model-preparation\n",
                    "    else:\n",
                    "        # Sync latest commits if already cloned\n",
                    "        try:\n",
                    f"            !git checkout {branch}\n",
                    f"            !git pull origin {branch}\n",
                    "        except Exception:\n",
                    "            pass\n",
                    "    !pip install -q huggingface_hub datasets torch pandas numpy matplotlib seaborn onnx onnxruntime onnxscript scikit-learn tqdm\n",
                "elif IN_KAGGLE:\n",
                "    print(\"[Environment] Running in Kaggle. Setting up dependencies...\")\n",
                "    !pip install -q huggingface_hub datasets torch pandas numpy matplotlib seaborn onnx onnxruntime scikit-learn tqdm\n",
                "else:\n",
                "    print(\"[Environment] Running in local/self-hosted environment.\")\n",
                "\n",
                "# Add src to sys.path across all possible working directory locations\n",
                "for path_candidate in [\"src\", \"../src\", \"./model-preparation/src\"]:\n",
                "    abs_p = os.path.abspath(path_candidate)\n",
                "    if os.path.isdir(abs_p) and abs_p not in sys.path:\n",
                "        sys.path.insert(0, abs_p)\n",
                "        print(f\"[System] Added to sys.path: {abs_p}\")\n",
                "        break"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Hosted Data Synchronization\n",
                "Connects to the Hugging Face Hub dataset repository (`T40/edge-aui-framework-data`) using `data_manager.ensure_dataset()`. If the data is already present locally, it is used with zero network transfer.\n",
                "> **Note on HTTP 429 Rate Limits:** Google Colab shares public IP addresses across many users. To avoid rate limits, set `HF_TOKEN` in Colab Secrets (🔑). `ensure_dataset()` will automatically throttle requests and fall back to single-stream Git clone if rate-limited."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from data_manager import ensure_dataset, get_hf_token, is_colab\n",
                "from preprocessing import (\n",
                "    process_ck_session,\n",
                "    process_hvt_session,\n",
                "    FEATURE_NAMES\n",
                ")\n",
                "\n",
                "# Check Hugging Face authentication token\n",
                "token = get_hf_token()\n",
                "if token is None and is_colab():\n",
                "    print(\"ℹ️ TIP: To ensure maximum rate limits, consider setting HF_TOKEN in Colab Secrets (🔑).\")\n",
                "\n",
                "# Automatically verify local cache or sync from Hugging Face Hub (with Git clone fallback)\n",
                "DATA_ROOT = ensure_dataset(repo_id=\"T40/edge-aui-framework-data\", token=token)\n",
                "print(f\"[DataManager] Data root successfully resolved: {DATA_ROOT}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Preprocessing & Continuous Kinematic Ingestion (CK & HVT)\n",
                "The continuous MicroTensor pipeline exclusively processes continuous kinematic datasets (`continuous-kinematics-2020` and `high-volume-trajectories-20226`).\n",
                "Discrete UI logs (`structural-hmi-sequences-2023` and `client-side-action-paths-2021`) are reserved for deterministic Fast Gate (PrefixSpan) benchmarking."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import importlib\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "import preprocessing\n",
                "import data_manager\n",
                "importlib.reload(preprocessing)\n",
                "importlib.reload(data_manager)\n",
                "\n",
                "try:\n",
                "    from data_manager import find_dataset_dir\n",
                "except ImportError:\n",
                "    def find_dataset_dir(root, dataset_name):\n",
                "        from pathlib import Path\n",
                "        p = Path(root).resolve()\n",
                "        candidates = [p / dataset_name, p / \"raw\" / dataset_name, p / \".data\" / \"raw\" / dataset_name, p.parent / dataset_name, p.parent / \"raw\" / dataset_name]\n",
                "        for c in candidates:\n",
                "            if c.is_dir() and any(c.iterdir()):\n",
                "                return str(c)\n",
                "        matches = glob.glob(os.path.join(str(p), \"**\", dataset_name), recursive=True)\n",
                "        for m in matches:\n",
                "            if os.path.isdir(m) and any(os.scandir(m)):\n",
                "                return m\n",
                "        return str(p / dataset_name)\n",
                "\n",
                "from preprocessing import (\n",
                "    process_ck_session,\n",
                "    process_hvt_session,\n",
                "    FEATURE_NAMES\n",
                ")\n",
                "\n",
                "# 1. Continuous Kinematics 2020\n",
                "ck_dir = find_dataset_dir(DATA_ROOT, \"continuous-kinematics-2020\")\n",
                "ck_logs = glob.glob(os.path.join(ck_dir, \"**\", \"*.csv\"), recursive=True)[:5]\n",
                "ck_windows = []\n",
                "for log in ck_logs:\n",
                "    ck_windows.extend(process_ck_session(log))\n",
                "df_ck = pd.DataFrame([w[\"features\"] for w in ck_windows], columns=FEATURE_NAMES)\n",
                "df_ck[\"Dataset\"] = \"Continuous Kinematics\"\n",
                "\n",
                "# 2. High-Volume Trajectories 20226 (sampled for efficient memory footprint)\n",
                "hvt_dir = find_dataset_dir(DATA_ROOT, \"high-volume-trajectories-20226\")\n",
                "hvt_candidates = [f for f in glob.glob(os.path.join(hvt_dir, \"**\", \"*.csv\"), recursive=True) if \"dataset.csv\" in f]\n",
                "hvt_logs = hvt_candidates[:1] if hvt_candidates else glob.glob(os.path.join(hvt_dir, \"**\", \"*.csv\"), recursive=True)[:1]\n",
                "hvt_windows = []\n",
                "for log in hvt_logs:\n",
                "    hvt_windows.extend(process_hvt_session(log, max_rows=10000, max_windows=200))\n",
                "df_hvt = pd.DataFrame([w[\"features\"] for w in hvt_windows], columns=FEATURE_NAMES)\n",
                "df_hvt[\"Dataset\"] = \"High-Volume Trajectories\"\n",
                "\n",
                "print(f\"Sample counts: CK={len(df_ck)}, HVT={len(df_hvt)}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Feature Distribution Visualizations (Velocity & Hesitation)\n",
                "Compares continuous kinematic features across `df_ck` and `df_hvt`. Unit harmonization (converting HVT $\\mu s \\to ms$ and $\\text{px/s} \\to \\text{px/ms}$) and symmetric velocity scaling ($/ 10.0$) prevent saturation and preserve true behavioral variance."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df_all = pd.concat([df_ck, df_hvt], ignore_index=True)\n",
                "\n",
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                "\n",
                "sns.kdeplot(data=df_all, x=\"meanVelocity\", hue=\"Dataset\", fill=True, ax=axes[0], common_norm=False)\n",
                "axes[0].set_title(\"Distribution of Mean Normalized Velocity\")\n",
                "axes[0].set_xlabel(\"Normalized Velocity [0, 1]\")\n",
                "\n",
                "sns.kdeplot(data=df_all, x=\"hesitationCount\", hue=\"Dataset\", fill=True, ax=axes[1], common_norm=False)\n",
                "axes[1].set_title(\"Distribution of Angular Hesitation Count\")\n",
                "axes[1].set_xlabel(\"Normalized Hesitation Count [0, 1]\")\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Cross-Dataset Feature Correlation Matrix (Harmonized Kinematics)\n",
                "Validates feature correlations across continuous kinematic sources. By excluding discrete structural logs from the continuous pipeline, artificial collinearity caused by zero-padded placeholders is completely resolved."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "corr_matrix = df_all.drop(columns=[\"Dataset\"]).corr()\n",
                "plt.figure(figsize=(10, 8))\n",
                "sns.heatmap(corr_matrix, annot=True, cmap=\"coolwarm\", fmt=\".2f\", cbar=True)\n",
                "plt.title(\"Cross-Dataset Feature Correlation Matrix (Harmonized Kinematics)\")\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Summary & Edge Implications\n",
                "- **Dynamic Range:** Normalized features span cleanly between $[0.0, 1.0]$ without clipping saturation at 1.0, preserving natural variance.\n",
                "- **Symmetric Scaling:** Both `meanVelocity` and `maxVelocity` use symmetric normalization denominators ($/ 10.0$), resolving mechanical artifact skew.\n",
                "- **Collinearity Elimination:** Segregating continuous motor datasets from discrete logs eliminates degenerate states and artificial collinearity.\n",
                "- **Ready for Training:** Run `Training.ipynb` to execute foundational GRU training, dynamic ONNX export, and INT8 edge quantization."
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
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}


# Default module-level export targeting current branch
current_branch = get_current_branch()
notebook = create_eda_notebook(current_branch)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate EDA notebook for Edge-AUI framework.")
    parser.add_argument("--branch", default=None, help="Target git branch (defaults to current branch).")
    args = parser.parse_args()

    target_branch = args.branch if args.branch else get_current_branch()
    print(f"[Generate EDA] Generating notebook targeting branch: '{target_branch}'")
    nb = create_eda_notebook(target_branch)

    out_dir = "notebooks"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "EDA.ipynb")
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=4)
    print(f"Successfully generated {out_path} (branch: '{target_branch}')")
