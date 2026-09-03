**System Context and Core Objective**
You are an expert Data Engineer and Machine Learning Developer. Your objective is to build a modular Python codebase (using scripts and `.ipynb` notebooks) to train a lightweight Gated Recurrent Unit (GRU). This model supports an edge-native adaptive user interface (AUI) framework that predicts user intent from high-frequency micro-interactions.

Your implementation must reflect a multi-tiered transfer learning strategy:

- **Foundational Pre-training:** The model must first learn cross-domain human motor behaviors (kinematic features) using large-scale public datasets.

- **Target UI Fine-tuning:** The model's classification head must then be fine-tuned on localized logs from a target testbed, such as a Security Operations Center (SOC) dashboard, to map physical movements to specific, localized user interface (UI) outcomes. This step resolves the Out-of-Vocabulary (OOV) target action problem caused by domain mismatch.
