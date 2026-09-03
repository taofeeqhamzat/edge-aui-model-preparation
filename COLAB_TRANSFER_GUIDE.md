# Google Colab Transfer & Hosted Training Guide

This guide walks you through transferring and executing the entire **Edge-AUI Model Preparation** environment in **Google Colaboratory**.

---

## 1. Why Transfer to Google Colab?

- **Hardware Acceleration:** Leverage free NVIDIA T4 (or A100/V100 on Colab Pro) GPUs to accelerate GRU training and feature vectorization.
- **Hosted Data Direct Pipeline:** Stream and synchronize interaction datasets directly from Hugging Face Hub (`T40/edge-aui-framework-data`) over cloud datacenter network links.
- **Resource Offloading:** Prevent local machine memory pressure and CPU bottlenecks from processing gigabyte-scale raw mouse trajectories.

---

## 2. Opening the Notebooks in Google Colab

### Method A: One-Click GitHub Integration (Recommended)
You can launch the notebooks directly in Google Colab using their GitHub URLs:

- **Foundational Training & Edge Quantization:**  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/taofeeqhamzat/edge-aui-model-preparation/blob/main/notebooks/Training.ipynb)  
  URL: `https://colab.research.google.com/github/taofeeqhamzat/edge-aui-model-preparation/blob/main/notebooks/Training.ipynb`

- **Exploratory Data Analysis (EDA):**  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/taofeeqhamzat/edge-aui-model-preparation/blob/main/notebooks/EDA.ipynb)  
  URL: `https://colab.research.google.com/github/taofeeqhamzat/edge-aui-model-preparation/blob/main/notebooks/EDA.ipynb`

### Method B: Manual File Upload
1. Navigate to [Google Colab](https://colab.research.google.com/).
2. Select the **Upload** tab.
3. Choose either `notebooks/Training.ipynb` or `notebooks/EDA.ipynb` from your local workspace.

---

## 3. Configuring Hardware Acceleration in Colab

To ensure training executes on a GPU rather than CPU:
1. In the Colab top navigation menu, click **Runtime** → **Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU** (available on free tier) or **A100 GPU** (Colab Pro).
3. Click **Save**.

The first cell in `Training.ipynb` will automatically verify your GPU:
```python
# Hardware accelerator check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Hardware] Active Execution Device: {device}")
# Output: [Hardware] Active Execution Device: cuda
# Output: [Hardware] GPU Name: Tesla T4
```

---

## 4. Hugging Face Authentication (Colab Secrets)

If your dataset repository (`T40/edge-aui-framework-data`) is private or gated:
1. In Colab's left sidebar, click the **Secrets** icon (the key 🔑).
2. Click **+ Add new secret**.
3. Set the Name to: `HF_TOKEN`
4. Paste your Hugging Face User Access Token (read permission) into Value.
5. Toggle **Notebook access** to **ON**.

The pipeline's `data_manager.py` will automatically detect and authenticate using Colab Secrets via `google.colab.userdata.get('HF_TOKEN')`. If the repository is public, no token is required.

---

## 5. End-to-End Execution Flow

Once open in Colab, click **Runtime** → **Run all** (or `Ctrl+F9` / `Cmd+F9`). The notebook executes the following stages sequentially:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Hosted Setup & Repo Clone                                │
│    git clone https://github.com/taofeeqhamzat/edge-aui-model-preparation.git
│    pip install requirements (torch, huggingface_hub, onnx)  │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ 2. Hosted Data Sync (Hugging Face Hub)                      │
│    ensure_dataset(repo_id="T40/edge-aui-framework-data")    │
│    Caches raw datasets to /content/.data/raw                │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ 3. MicroTensor Feature Vectorization                        │
│    500ms sliding windows, 8D feature extraction             │
│    Modality Mask Vector for missing coordinates/scroll      │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ 4. Foundational GRU Training (PyTorch on CUDA)              │
│    2-layer GRU, Hidden Dim: 32, Output Classes: 6           │
│    Interactive loss & accuracy progression curves           │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ 5. Dynamic ONNX Export & INT8 Quantization                  │
│    model.onnx -> model_int8.onnx (QUInt8)                   │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ 6. Edge Constraint Assertions                               │
│    Assert Storage Footprint < 20 MB (Actual: ~0.045 MB)     │
│    Assert Inference Latency < 50 ms (Actual: ~0.032 ms)     │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ 7. Browser Download & Artifact Export                       │
│    files.download('models/model_int8.onnx')                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Transferring Model Artifacts to the Front-End Framework

After training and quantization complete in Colab:
1. The last cell in `Training.ipynb` automatically triggers a browser download for `model_int8.onnx`.
2. Move the downloaded `model_int8.onnx` file into your local project:
   ```bash
   # In your local repository:
   cp ~/Downloads/model_int8.onnx edge-aui-framework/public/models/
   cp ~/Downloads/model_int8.onnx model-preparation/models/
   ```
3. The front-end ONNX Runtime Web Worker (`edge-aui-framework/src/worker.ts`) will now consume the newly trained INT8 model directly for zero-latency in-browser inference!
