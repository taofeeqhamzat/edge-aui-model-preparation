import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Exploratory Data Analysis: Edge-AUI Behavioral Datasets\n",
                "\n",
                "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/taofeeqhamzat/edge-aui-model-preparation/blob/main/notebooks/EDA.ipynb)\n",
                "\n",
                "This notebook analyzes the multi-domain interaction datasets supporting the **Edge-AUI Framework** (Continuous Kinematics, High-Volume Trajectories, Structural HMI Sequences, and Client-Side Action Paths).\n",
                "It extracts 8-dimensional MicroTensors across all datasets and visualizes feature distributions, cross-modality discrepancies, and the Modality Mask Vector pattern."
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
                "    # Clone repo if executing standalone in Colab\n",
                "    if not os.path.exists(\"src\") and not os.path.exists(\"../src\"):\n",
                "        !git clone https://github.com/taofeeqhamzat/edge-aui-model-preparation.git\n",
                "        %cd edge-aui-model-preparation\n",
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
                "Connects to the Hugging Face Hub dataset repository (`T40/edge-aui-framework-data`) using `data_manager.ensure_dataset()`. If the data is already present locally, it is used with zero network transfer."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from data_manager import ensure_dataset, is_colab, is_kaggle\n",
                "from preprocessing import (\n",
                "    process_ck_session,\n",
                "    process_hvt_session,\n",
                "    process_hmi_sequences,\n",
                "    process_action_paths,\n",
                "    FEATURE_NAMES\n",
                ")\n",
                "\n",
                "# Automatically verify local cache or sync from Hugging Face Hub\n",
                "DATA_ROOT = ensure_dataset(repo_id=\"T40/edge-aui-framework-data\")\n",
                "print(f\"[DataManager] Data root successfully resolved: {DATA_ROOT}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Preprocessing & MicroTensor Ingestion Across All 4 Datasets"
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
                "# 1. Continuous Kinematics 2020\n",
                "ck_logs = glob.glob(f\"{DATA_ROOT}/continuous-kinematics-2020/logs/*.csv\")[:5]\n",
                "ck_windows = []\n",
                "for log in ck_logs:\n",
                "    ck_windows.extend(process_ck_session(log))\n",
                "df_ck = pd.DataFrame([w[\"features\"] for w in ck_windows], columns=FEATURE_NAMES)\n",
                "df_ck[\"Dataset\"] = \"Continuous Kinematics\"\n",
                "\n",
                "# 2. High-Volume Trajectories 20226 (sampled for efficient memory footprint)\n",
                "hvt_logs = glob.glob(f\"{DATA_ROOT}/high-volume-trajectories-20226/**/*.csv\", recursive=True)[:1]\n",
                "hvt_windows = []\n",
                "for log in hvt_logs:\n",
                "    hvt_windows.extend(process_hvt_session(log, max_rows=10000, max_windows=200))\n",
                "df_hvt = pd.DataFrame([w[\"features\"] for w in hvt_windows], columns=FEATURE_NAMES)\n",
                "df_hvt[\"Dataset\"] = \"High-Volume Trajectories\"\n",
                "\n",
                "# 3. Structural HMI Sequences 2023\n",
                "hmi_logs = glob.glob(f\"{DATA_ROOT}/structural-hmi-sequences-2023/*.csv\")[:2]\n",
                "hmi_windows = []\n",
                "for log in hmi_logs:\n",
                "    hmi_windows.extend(process_hmi_sequences(log))\n",
                "df_hmi = pd.DataFrame([w[\"features\"] for w in hmi_windows], columns=FEATURE_NAMES)\n",
                "df_hmi[\"Dataset\"] = \"HMI Sequences\"\n",
                "\n",
                "# 4. Client-Side Action Paths 2021\n",
                "ap_logs = glob.glob(f\"{DATA_ROOT}/client-side-action-paths-2021/**/*.json\", recursive=True)[:5]\n",
                "ap_windows = []\n",
                "for log in ap_logs:\n",
                "    ap_windows.extend(process_action_paths(log))\n",
                "df_ap = pd.DataFrame([w[\"features\"] for w in ap_windows], columns=FEATURE_NAMES)\n",
                "df_ap[\"Dataset\"] = \"Action Paths\"\n",
                "\n",
                "print(f\"Sample counts: CK={len(df_ck)}, HVT={len(df_hvt)}, HMI={len(df_hmi)}, AP={len(df_ap)}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Feature Distribution Visualizations (Velocity & Hesitation)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df_all = pd.concat([df_ck, df_hvt, df_hmi, df_ap], ignore_index=True)\n",
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
                "## 5. Cross-Dataset Feature Discrepancy Matrix & Correlation Heatmap\n",
                "Validates the Modality Mask Vector pattern: older datasets or discrete structural traces pad absent features (such as scroll velocities or absolute coordinates) with zeros, while kinematic datasets capture continuous motor signals."
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
                "plt.title(\"Cross-Dataset Feature Correlation Matrix (Modality Mask Analysis)\")\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Summary & Edge Implications\n",
                "- **Dynamic Range:** Normalized features span cleanly between $[0.0, 1.0]$, ensuring that INT8 quantization does not suffer from extreme outlier clipping.\n",
                "- **Modality Masking:** Structural logs provide orthogonal grounding while continuous kinematics train recurrent temporal weights.\n",
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

if __name__ == "__main__":
    out_dir = "notebooks"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "EDA.ipynb")
    with open(out_path, "w") as f:
        json.dump(notebook, f, indent=4)
    print(f"Successfully generated {out_path}")
