import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Edge-AUI Foundational Model Training & Edge Quantization\n",
                "\n",
                "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/taofeeqhamzat/edge-aui-model-preparation/blob/main/notebooks/Training.ipynb)\n",
                "\n",
                "This notebook executes the end-to-end pipeline for the **Edge-AUI Framework** in hosted environments (Google Colab / Kaggle):\n",
                "1. Synchronizes interaction datasets directly from the Hugging Face Hub (`T40/edge-aui-framework-data`).\n",
                "2. Extracts 8-dimensional MicroTensors from continuous behavioral trajectories.\n",
                "3. Trains a lightweight PyTorch Gated Recurrent Unit (GRU) model with hardware acceleration (CUDA / MPS / CPU).\n",
                "4. Exports the PyTorch model to dynamic ONNX and applies INT8 Post-Training Quantization (PTQ).\n",
                "5. Empirically validates edge constraints: **Memory Footprint < 20MB** and **Inference Latency < 50ms**."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Hosted Environment Setup & Accelerator Detection\n",
                "Automatically configures required dependencies, sets up repository paths, and inspects available GPU accelerators."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%load_ext autoreload\n",
                "%autoreload 2\n",
                "\n",
                "import os\n",
                "import sys\n",
                "import torch\n",
                "\n",
                "# Environment detection\n",
                "IN_COLAB = 'google.colab' in sys.modules or 'COLAB_GPU' in os.environ\n",
                "IN_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ\n",
                "\n",
                "if IN_COLAB:\n",
                "    print(\"[Setup] Detected Google Colaboratory. Setting up environment...\")\n",
                "    if not os.path.exists(\"src\") and not os.path.exists(\"../src\"):\n",
                "        !git clone https://github.com/taofeeqhamzat/edge-aui-model-preparation.git\n",
                "        %cd edge-aui-model-preparation\n",
                "    !pip install -q huggingface_hub datasets torch pandas numpy matplotlib seaborn onnx onnxruntime onnxscript scikit-learn tqdm\n",
                "elif IN_KAGGLE:\n",
                "    print(\"[Setup] Detected Kaggle Kernel. Setting up environment...\")\n",
                "    !pip install -q huggingface_hub datasets torch pandas numpy matplotlib seaborn onnx onnxruntime onnxscript scikit-learn tqdm\n",
                "else:\n",
                "    print(\"[Setup] Detected Local / Self-Hosted Environment.\")\n",
                "\n",
                "# Add src to sys.path across all possible working directory locations\n",
                "for path_candidate in [\"src\", \"../src\", \"./model-preparation/src\"]:\n",
                "    abs_p = os.path.abspath(path_candidate)\n",
                "    if os.path.isdir(abs_p) and abs_p not in sys.path:\n",
                "        sys.path.insert(0, abs_p)\n",
                "        print(f\"[System] Added to sys.path: {abs_p}\")\n",
                "        break\n",
                "\n",
                "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"mps\" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else \"cpu\")\n",
                "print(f\"[Hardware] Active Execution Device: {device}\")\n",
                "if device.type == \"cuda\":\n",
                "    print(f\"[Hardware] GPU Name: {torch.cuda.get_device_name(0)}\")\n",
                "    print(f\"[Hardware] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Hosted Data Synchronization (Hugging Face Hub)\n",
                "Verifies local dataset cache or synchronizes from `T40/edge-aui-framework-data`.\n",
                "> **Note on HTTP 429 Rate Limits:** Google Colab shares public IP addresses across many users. To avoid rate limits, set `HF_TOKEN` in Colab Secrets (🔑). `ensure_dataset()` will automatically throttle concurrent requests and fall back to single-stream Git clone if rate-limited."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from data_manager import ensure_dataset, get_hf_token, is_colab\n",
                "\n",
                "# Check Hugging Face authentication token\n",
                "token = get_hf_token()\n",
                "if token is None and is_colab():\n",
                "    print(\"ℹ️ TIP: To ensure maximum rate limits, consider setting HF_TOKEN in Colab Secrets (🔑).\")\n",
                "\n",
                "# Synchronize datasets (handles rate limits, throttles requests, and auto-falls back to Git stream sync)\n",
                "DATA_ROOT = ensure_dataset(repo_id=\"T40/edge-aui-framework-data\", token=token)\n",
                "print(f\"[Data] Datasets verified at: {DATA_ROOT}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Dataset Loading & Feature Inspection\n",
                "Extracts 500ms sliding windows and constructs temporal sequences (`seq_len=8`)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import importlib\n",
                "import preprocessing\n",
                "importlib.reload(preprocessing)\n",
                "from preprocessing import MicroInteractionSequenceDataset, FEATURE_NAMES, LABEL_MAP\n",
                "\n",
                "# Initialize sequence dataset\n",
                "dataset = MicroInteractionSequenceDataset(data_root=DATA_ROOT, max_files_per_dataset=10)\n",
                "print(f\"[Dataset] Total behavioral sequences extracted: {len(dataset)}\")\n",
                "\n",
                "if len(dataset) > 0:\n",
                "    sample_x, sample_y = dataset[0]\n",
                "    print(f\"[Dataset] Input Tensor Shape (seq_len, features): {sample_x.shape}\")\n",
                "    print(f\"[Dataset] Target Outcome Label: {sample_y.item()}\")\n",
                "    print(\"[Dataset] MicroTensor Features:\", FEATURE_NAMES)\n",
                "    print(\"[Dataset] Outcome Classes:\", LABEL_MAP)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Train Foundational GRU Model\n",
                "Executes the recurrent training loop, learning cross-domain motor dynamics and structural interaction patterns."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from training import train_foundation_model\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "# Train model\n",
                "results = train_foundation_model(\n",
                "    data_dir=DATA_ROOT,\n",
                "    epochs=5,\n",
                "    batch_size=64,\n",
                "    lr=1e-3,\n",
                "    max_files_per_dataset=10,\n",
                "    device=str(device),\n",
                "    verbose=True\n",
                ")\n",
                "\n",
                "history = results[\"history\"]\n",
                "\n",
                "# Plot Training Metrics\n",
                "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))\n",
                "ax1.plot(range(1, len(history[\"loss\"]) + 1), history[\"loss\"], marker='o', color='crimson')\n",
                "ax1.set_title(\"Training Loss Progression\")\n",
                "ax1.set_xlabel(\"Epoch\")\n",
                "ax1.set_ylabel(\"CrossEntropy Loss\")\n",
                "ax1.grid(True, linestyle='--', alpha=0.6)\n",
                "\n",
                "ax2.plot(range(1, len(history[\"accuracy\"]) + 1), history[\"accuracy\"], marker='s', color='navy')\n",
                "ax2.set_title(\"Accuracy Progression\")\n",
                "ax2.set_xlabel(\"Epoch\")\n",
                "ax2.set_ylabel(\"Accuracy (%)\")\n",
                "ax2.grid(True, linestyle='--', alpha=0.6)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Dynamic ONNX Export & INT8 Quantization\n",
                "Compiles the PyTorch graph to ONNX format and applies dynamic Post-Training Quantization (QUInt8) to compress the model weights."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from export import export_and_quantize\n",
                "\n",
                "metrics = export_and_quantize(\n",
                "    model_path=results[\"model_path\"],\n",
                "    output_dir=\"models\"\n",
                ")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Edge Constraint Verification\n",
                "Empirically asserts strict edge-native deployment constraints:\n",
                "- **Memory Footprint:** $\\le 20\\text{ MB}$\n",
                "- **Inference Latency:** $\\le 50\\text{ ms}$ (zero UI jank threshold)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"=\" * 50)\n",
                "print(\"       EDGE ARCHITECTURE BENCHMARK REPORT\")\n",
                "print(\"=\" * 50)\n",
                "print(f\"Quantized Model Path:    {metrics['quantized_onnx_path']}\")\n",
                "print(f\"Storage Footprint:       {metrics['file_size_mb']:.4f} MB  (Threshold: < 20.0 MB) -> {'PASS' if metrics['file_size_mb'] < 20 else 'FAIL'}\")\n",
                "print(f\"Average Latency (CPU):   {metrics['avg_latency_ms']:.4f} ms  (Threshold: < 50.0 ms) -> {'PASS' if metrics['avg_latency_ms'] < 50 else 'FAIL'}\")\n",
                "print(f\"P95 Latency (CPU):       {metrics['p95_latency_ms']:.4f} ms\")\n",
                "print(\"=\" * 50)\n",
                "\n",
                "assert metrics['file_size_mb'] < 20.0, \"Memory constraint exceeded!\"\n",
                "assert metrics['avg_latency_ms'] < 50.0, \"Latency constraint exceeded!\"\n",
                "print(\"All Edge-AUI production deployment constraints passed!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Artifact Download (Colab / Local Export)\n",
                "Provides one-click download of the quantized ONNX artifact (`model_int8.onnx`) for direct integration with the browser runtime."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if IN_COLAB:\n",
                "    from google.colab import files\n",
                "    print(\"Triggering browser download of quantized ONNX model...\")\n",
                "    files.download(metrics['quantized_onnx_path'])\n",
                "else:\n",
                "    print(f\"Model artifact is ready locally at: {os.path.abspath(metrics['quantized_onnx_path'])}\")"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
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
    out_path = os.path.join(out_dir, "Training.ipynb")
    with open(out_path, "w") as f:
        json.dump(notebook, f, indent=4)
    print(f"Successfully generated {out_path}")
