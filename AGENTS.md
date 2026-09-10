# AGENTS.md

## System Context and Core Objective

You are an expert Data Engineer and Machine Learning Developer. Your objective is to build a modular Python codebase (using scripts and Jupyter notebooks) to train a lightweight Gated Recurrent Unit (GRU). This neural architecture serves as the probabilistic "Slow Gate" supporting an edge-native adaptive user interface (AUI) framework for the primary research goal: **"IMPLEMENTING A LIGHTWEIGHT AI-ASSISTED FRAMEWORK FOR BEHAVIOURAL PATTERN EXTRACTION AND ADAPTIVE UI RECOMMENDATIONS ON THE WEB"**.

Rather than attempting subjective affective computing or classifying unobservable emotional states, the GRU operates as a **latent behavioural-state encoder trained through self-supervised prediction of subsequent UI outcomes**. The system does not attempt to diagnose internal user psychology; it extracts statistical regularities from continuous micro-interaction streams to forecast observable subsequent interactions and evaluate whether an interface intervention is beneficial.

```

Raw Interaction Stream
│
▼
Micro-Interaction Aggregation (500 ms window)
│
▼
MicroTensor Sequence (T timesteps)
│
▼
Latent Behavioural Representation (GRU Hidden State)
│
▼
UI-Outcome Prediction (Target-Specific Head)
│
▼
Adaptive UI Recommendation

```

---

## Conceptual Guidelines & Terminology Guardrails

- **Interaction-Outcome Framing:** Never state that "the system detects frustration," "the model infers user confusion," or "the architecture identifies negative emotional affect". Always state: **"The system learns behavioural patterns that predict observable interaction outcomes"**.
- **Auxiliary Interpretation:** Learned latent patterns that statistically correlate with behaviours traditionally associated with cognitive friction or hesitation (e.g., elevated cursor dwell time, angular deviation, rapid scroll reversals) may be evaluated post-hoc during error and ablation analysis, but must not serve as supervised classification labels.
- **Latent Intent vs. Outcome Prediction:** Avoid ungrounded references to "latent intent detection". Describe the model explicitly as a sequence-to-outcome predictor mapping continuous motor dynamics to downstream UI resolutions (e.g., `CLICK`, `FORM_SUBMIT`, `BACKTRACK`, `IDLE`).
- **Output-Space Alignment (No Recurrent OOV):** The foundational GRU recurrent layers do not suffer from an Out-of-Vocabulary (OOV) problem regarding target interface actions. The recurrent backbone maps MicroTensor sequences to an interface-agnostic latent behavioural representation. The target-specific classification head contains the discrete output vocabulary aligned with the deployment environment's intervention space (e.g., expanding detail panels, adjusting layout density, providing contextual guidance).

---

## Prioritized Implementation Roadmap

### Priority 1: Dataset Sourcing and Validation

- **Dataset Selection Criterion:** Reject datasets whose sole utility is subjective emotion classification or biometric authentication without observable UI context. Prioritize longitudinal, timestamped interaction and session datasets containing both high-resolution continuous telemetry (pointer coordinates, timestamps, scroll offsets) and consequential downstream interface actions.
- **Candidate Tiering:**
  - _Target Fine-Tuning & Web Semantics:_ AdSERP (contains millisecond timestamps, coordinates, and DOM XPath targets), Admission Test Cursor Trajectory Dataset.
  - _Foundational Motor Dynamics:_ CaptchaSolve30k, VideoCUA, Attentive Cursor Dataset (Leiva & Arapakis, 2020).
  - _Structural & Sequential Validation:_ Structured HMI Sequences (Carrera-Rivera et al., 2023), Client-Side Action Paths (Ou et al., 2021).
- **Integrity Gate:** Require raw datasets to provide millisecond-level timestamps and trajectory coordinates; never fabricate missing continuous telemetry with arbitrary zeroes.

### Priority 2: Raw-Event to MicroTensor Preprocessing

- **Temporal Sliding Window:** Segment high-frequency pointer and scroll streams into discrete 500 ms windows with a 250 ms stride to mirror the browser runtime buffer.
- **MicroTensor Specification:** Compute eight core continuous kinematic metrics per window:
  1. Instantaneous and mean Euclidean velocity ($v_k = \frac{\Delta d}{\Delta t}$)
  2. Maximum Euclidean velocity
  3. Mean acceleration ($a_k = \frac{\Delta v}{\Delta t}$)
  4. Hesitation count (directional turning angles $\theta > 45^\circ$)
  5. Total trajectory length ($\sum \Delta d$)
  6. Dwell time on interactive DOM targets ($d_t$)
  7. Normalized scroll depth percentage
  8. Scroll velocity ($v_{\text{scroll}}$)
- **Virtual Viewport Canonicalization:** Normalize raw coordinate offsets against a standard $1920 \times 1080$ pixel reference frame to preserve spatial invariance across heterogeneous screen resolutions.
- **Modality Masking:** When ingesting corpora that omit viewport or scrolling telemetry, append a binary Modality Mask Vector ($M \in \{0, 1\}^D$) rather than imputing artificial zero values.

### Priority 3: Training-Example & Self-Supervised Target Generation

- **Supervisory Horizon:** Implement an outcome-driven self-supervised extraction pipeline (Wu et al., 2024).
- **Target Pairing:** For each sequence of $T$ consecutive 500 ms MicroTensors, extract naturally occurring macro-interactions within an adjacent lookahead horizon ($\Delta t \in [500\text{ ms}, 1500\text{ ms}]$).
- **Outcome Vocabulary:** Target labels must represent concrete, observable events (e.g., `CLICK`, `DWELL`, `BACKTRACK`, `FORM_SUBMIT`, `ABANDON`).

### Priority 4: Foundation GRU Pre-Training

- **Recurrent Architecture:** Construct a multi-layer Gated Recurrent Unit (GRU4Rec baseline; Hidasi et al., 2016; Quadrana et al., 2018) with hidden dimension $d_h = 64$ and sequence length $T$.
- **Representation Learning:** Train the recurrent encoder on cross-domain kinematic streams to model baseline human motor control, trajectory entropy, and temporal decay without tying representations to a specific interface geometry.
- **Optimization:** Utilize multi-task self-supervised loss functions, monitoring latent cluster separation and representation stability.

### Priority 5: Target-Domain Fine-Tuning

- **Output-Space Alignment:** Initialize a target-specific classification head that projects the 64-dimensional latent embedding $h_T$ onto the discrete intervention space $|\mathcal{C}|$ of the target web application (e.g., `Expand_Detail_Panel`, `Pin_Filter_Bar`, `No_Op`).
- **Transfer Learning Workflow:** Programmatically freeze the pre-trained recurrent GRU weights ($\frac{\partial \mathcal{L}}{\partial \theta_{\text{base}}} = 0$) and train exclusively the target classification head using localized testbed logs.

### Priority 6: Fast Gate / Slow Gate Integration

- **Deterministic Fast Gate:** Route incoming macro-interaction tokens through a WebAssembly-compiled PrefixSpan pattern miner for immediate, exact sequence matches ($O(1)$ lookup).
- **Probabilistic Slow Gate:** Invoke the GRU worker only when the Fast Gate returns a cache miss or encounters ambiguous, non-indexed micro-interaction sequences.
- **Concurrency:** Ensure asynchronous message passing between the main thread event observer and Web Workers to eliminate main-thread layout thrashing and maintain 60 FPS execution.

### Priority 7: Model Evaluation & Ablation Testing

- **Ablation Protocol:** Execute controlled empirical ablations comparing:
  1. Head-only fine-tuning (frozen GRU backbone).
  2. Partial fine-tuning (unfreezing the terminal GRU layer + head).
  3. Full end-to-end fine-tuning across all layers.
- **Predictive Metrics:** Quantify task performance using Hit Rate at $K$ ($\text{HR}@K \ge 80\%$ for $K=3$) and Mean Reciprocal Rank ($\text{MRR} \ge 0.70$).
- **Ablation of Feature Modalities:** Assess model degradation when masking velocity versus turning angles versus scroll dynamics.

### Priority 8: INT8 Quantization & Browser Resource Testing

- **Graph Compilation:** Export converged PyTorch models to the ONNX graph format (`torch.onnx.export`).
- **Post-Training Quantization (PTQ):** Apply INT8 quantization to model weights and activations via ONNX Runtime Web.
- **Hardware & Runtime Validation:**
  - Resident memory footprint $\le 20\text{ MB}$.
  - On-device inference latency $\le 50\text{ ms}$ via WebGPU / WASM execution providers.
  - Total serialized model artifact payload $\le 200\text{ KB}$ (total client bundle $< 500\text{ KB}$).

### Priority 9: Data & Model Failure Debugging

- Isolate cross-dataset distribution shifts and resolve Pearson correlation artifacts ($r < 0.80$) caused by improper feature normalization.
- Debug vanishing gradients or activation saturation resulting from unmasked missing modalities or extreme outliers.

### Priority 10: Methodology & Empirical Results Documentation

- Document implemented data ingestion schemas, mathematical transformations, ablation results, and latency profiles for Chapters 4 and 5 of the thesis following Design Science Research guidelines (Hevner et al., 2004; Peffers et al., 2007).
