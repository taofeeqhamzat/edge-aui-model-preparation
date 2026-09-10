# Implementation Plan: Modular Behavioral GRU Pipeline & Progressive Fine-Tuning

Architect and execute a modular, configuration-driven Python pipeline to train a highly compressed Gated Recurrent Unit (GRU) acting as the probabilistic "Slow Gate" for **IMPLEMENTING A LIGHTWEIGHT AI-ASSISTED FRAMEWORK FOR BEHAVIOURAL PATTERN EXTRACTION AND ADAPTIVE UI RECOMMENDATIONS ON THE WEB**.

This refined plan strictly incorporates:

1. **User Review Comments:**
   - **Continuous Kinematics Removed:** All references to Continuous Kinematics have been eliminated. Sourcing is dedicated strictly to `AdSERP` for MVP (`mode: minimal`), and `CaptchaSolve30k` $\rightarrow$ `VideoCUA` $\rightarrow$ `AdSERP` for extended mode (`mode: extended`).
   - **Branch Confirmed:** Active branch `explore/pipeline/revision/1` is confirmed and checked out.
   - **DVC Remote Data Repository Updated:** Remote data storage tracking via DVC (`hfremote` S3 gateway / `dvc pull` / `dvc push`), updated `dvc.yaml` pipeline stages, and partitioned data directory management.
2. **Task Breakdown (`planning-and-task-breakdown`):** Vertically sliced tasks with explicit acceptance criteria, verification commands, file targets, and checkpoints.
3. **Architecture Decision Records (`documentation-and-adrs`):** Five formal ADRs capturing the rationale, constraints, and trade-offs of critical technical decisions.

---

## Architecture Decisions & ADR Index

The following Architecture Decision Records govern the implementation:

- **[ADR-001: Modality Masking for Missing Feature Collinearity Resolution](#adr-001-modality-masking-for-missing-feature-collinearity-resolution)**
- **[ADR-002: Outcome-Driven Self-Supervised Lookahead Horizon](#adr-002-outcome-driven-self-supervised-lookahead-horizon)**
- **[ADR-003: Latent Encoder Decoupling & Target-Domain Output-Space Alignment](#adr-003-latent-encoder-decoupling--target-domain-output-space-alignment)**
- **[ADR-004: Remote Data Repository Integration via DVC & Parquet Inode Protection](#adr-004-remote-data-repository-integration-via-dvc--parquet-inode-protection)**
- **[ADR-005: Progressive Fine-Tuning Ablation Protocol with Non-Failing Metrics](#adr-005-progressive-fine-tuning-ablation-protocol-with-non-failing-metrics)**

---

## Prioritized Implementation Roadmap

```
Remote Data Repo (DVC / HF S3) ──► .data/raw (AdSERP / CaptchaSolve30k / VideoCUA)
                                              │
                                              ▼
                             Layer A: Inode-Safe Parquet Consolidation
                                              │
                                              ▼
                             Layer B: Canonical Event Schema ([x, y] in [0, 1], ms timestamps)
                                              │
                                              ▼
                             Layer C: MicroTensor Extraction (500ms window, 250ms stride)
                                      + Modality Masking [X_t ⊙ M, M] in R^18
                                              │
                                              ├──► Notebook: 01_preprocessing_eda.ipynb
                                              ▼
                             Phase 2: Derived Future-Outcome Labeler (500–1500ms Horizon)
                                              │
                                              ├──► Notebook: 02_target_distribution.ipynb
                                              ▼
                             Phase 3: Foundation GRU Pre-Training (d_h=64 latent encoder)
                                              │
                                              ▼
                             Phase 3: Target-Domain Fine-Tuning & Progressive Ablations
                                      (Strict Freezing vs. Partial vs. Full Fine-Tuning)
                                              │
                                              ▼
                             Phase 4: INT8 Dynamic PTQ Export & Edge Profiling (<20MB, <50ms)
                                              │
                                              ▼
                             Production Artifacts (models/model_int8.onnx -> Edge Web Worker)
```

---

## Detailed Task Breakdown

### Phase 1: Environment, DVC Remote Data & Configuration

#### Task 1.1: [task/1.1.md](./task/1.1.md)

#### Task 1.2: [task/1.2.md](./task/1.2.md)

#### Task 1.3: [task/1.3.md](./task/1.3.md)

### Checkpoint 1: Foundation & Data Sync

- [ ] DVC configuration and pipeline stages verified.
- [ ] Central YAML configuration loads both `minimal` and `extended` modes.
- [ ] AdSERP data ingestion succeeds with inode-safe Parquet caching.

---

### Phase 2: Tiered Preprocessing & MicroTensor Generation

#### Task 2.1: [task/2.1.md](./task/2.1.md)

#### Task 2.2: [task/2.2.md](./task/2.2.md)

#### Task 2.3: [task/2.3.md](./task/2.3.md)

### Checkpoint 2: Preprocessing & Vectorization

- [ ] Canonical event schema processes raw AdSERP logs with viewport normalization.
- [ ] 18-dimensional MicroTensors ($2D=18$) generated with valid modality masks.
- [ ] `01_preprocessing_eda.ipynb` notebook generated and verified.

---

### Phase 3: Target Generation & Outcome Distribution Analysis

#### Task 3.1: [task/3.1.md](./task/3.1.md)

#### Task 3.2: [task/3.2.md](./task/3.2.md)

### Checkpoint 3: Target Generation & Distribution Diagnostics

- [ ] Self-supervised outcome lookahead horizon extracts ground-truth macro-actions.
- [ ] Label imbalance quantified and class-weighted loss computed.
- [ ] `02_target_distribution.ipynb` generated and verified.

---

### Phase 4: Foundation Pre-Training & Target Fine-Tuning Ablations

#### Task 4.1: [task/4.1.md](./task/4.1.md)

#### Task 4.2: [task/4.2.md](./task/4.2.md)

### Checkpoint 4: Training & Progressive Fine-Tuning

- [ ] Latent encoder architecture forwards $(B, T, 18) \rightarrow (B, 64) \rightarrow (B, C)$.
- [ ] Foundation model trains successfully on AdSERP.
- [ ] Progressive unfreezing ablation compares Strict, Partial, and Full fine-tuning with HR@K and MRR metrics.

---

### Phase 5: INT8 Quantization, Edge Profiling & Pipeline Integration

#### Task 5.1: [task/5.1.md](./task/5.1.md)

#### Task 5.2: [task/5.2.md](./task/5.2.md)

#### Task 5.3: [task/5.3.md](./task/5.3.md)

### Checkpoint 5: Complete Pipeline Verification

- [ ] End-to-end MVP execution (`train_pipeline.py --mode minimal`) completes cleanly.
- [ ] ONNX and INT8 quantized models generated in `models/` (<200KB payload).
- [ ] All three notebooks (`01_preprocessing_eda.ipynb`, `02_target_distribution.ipynb`, `Training.ipynb`) generated and validated.

---

## Architecture Decision Records (ADRs)

### ADR-001: Modality Masking for Missing Feature Collinearity Resolution

#### Status

Accepted

#### Date

2026-09-10

#### Context

Different behavioral datasets provide differing telemetry subsets (e.g., `AdSERP` has coordinates and DOM XPaths but intermittent vertical scroll telemetry; other datasets omit DOM targets). Zero-imputation causes severe artificial collinearity because $0$ carries semantic meaning in kinematic streams (e.g., $v=0$ denotes a stationary cursor; zero scroll velocity denotes no scrolling).

#### Decision

Append a binary Modality Mask Vector $M \in \{0, 1\}^D$ to the normalized feature vector:
$$\widetilde{X}_t = \big[X_t \odot M, \; M\big] \in \mathbb{R}^{2D}$$
For $D=9$ features (7 core kinematics + 2 optional contextual), input dimensionality is $2D = 18$. The GRU input layer accepts $18$ features, allowing it to learn independent weights for observed vs. unobserved modalities.

#### Alternatives Considered

- _Zero Imputation:_ Rejected due to induced distribution shift and false correlation artifacts.
- _Dropping Contextual Features:_ Rejected as it discards valuable scroll and DOM signals when present.

#### Consequences

- Disentangles sensor absence from user inactivity.
- Small increase in input layer parameters ($18 \times 64$ vs $9 \times 64$), representing $< 1\text{ KB}$ in memory footprint.

---

### ADR-002: Outcome-Driven Self-Supervised Lookahead Horizon

#### Status

Accepted

#### Date

2026-09-10

#### Context

Affective computing approaches attempt to infer subjective internal states (e.g. "frustration", "confusion") which are unobservable, subjective, and prone to severe annotator bias. The framework's goal is adaptive UI recommendation.

#### Decision

Ground training targets entirely in verifiable downstream interaction outcomes occurring in a lookahead horizon $\Delta t \in [500\text{ms}, 1500\text{ms}]$:
$$\text{Taxonomy} \in \{\text{IDLE\_ABANDON}, \text{CLICK}, \text{FORM\_SUBMIT}, \text{BACKTRACK}, \text{RAPID\_SCROLL}, \text{HOVER\_DWELL}\}$$

#### Alternatives Considered

- _Subjective Emotion Annotation:_ Rejected due to unreliability and lack of actionable UI semantics.
- _Fixed Next-Token Event Prediction:_ Rejected because individual mouse movements at 16ms intervals do not represent meaningful macro UI actions.

#### Consequences

- Produces objective, verifiable supervisory targets directly from raw web interaction logs.
- Directly informs downstream UI interventions (e.g., predicting `BACKTRACK` triggers assistance).

---

### ADR-003: Latent Encoder Decoupling & Target-Domain Output-Space Alignment

#### Status

Accepted

#### Date

2026-09-10

#### Context

Web applications have distinct interaction geometries and adaptive intervention spaces. Tying the recurrent model directly to target-specific actions creates an Out-of-Vocabulary (OOV) problem when transferring across web environments.

#### Decision

Decouple the architecture into:

1. **Recurrent Latent Encoder (GRU Backbone):** Maps micro-interaction sequences $\widetilde{X}_{1:T}$ to interface-agnostic latent behavioural representation $h_T \in \mathbb{R}^{64}$.
2. **Modular Classification Head:** Linear projection from $h_T$ to the deployment application's discrete intervention vocabulary:
   `['simplify_options', 'highlight_primary_action', 'offer_assistance', 'expand_tooltip', 'no_op']`.

#### Alternatives Considered

- _End-to-End Retraining for Every UI:_ Rejected due to heavy data and compute requirements on target environments.
- _Shared Global Action Vocabulary:_ Rejected because target interfaces do not share identical intervention spaces.

#### Consequences

- Solves the recurrent OOV problem.
- Enables efficient transfer learning via freezing/progressive unfreezing on localized testbed logs.

---

### ADR-004: Remote Data Repository Integration via DVC & Parquet Inode Protection

#### Status

Accepted

#### Date

2026-09-10

#### Context

Interaction datasets contain over 8,300 individual CSV and XML files. Downloading and reading thousands of tiny files causes inode exhaustion on shared systems, triggering HTTP 429 rate limits and extreme disk I/O latency.

#### Decision

1. Configure DVC with `hfremote` pointing to Hugging Face's S3 gateway (`s3://edge-aui-framework-data/dvc-store`) for version-controlled remote data storage.
2. Ingest granular CSVs into compressed columnar Parquet containers stored in `.data/interim/` and `.data/processed/`.

#### Alternatives Considered

- _Direct Uncompressed Ingestion of Thousands of CSVs:_ Rejected due to inode exhaustion and file system bottlenecks.
- _Pure Git LFS:_ Rejected due to rigid tracking and bandwidth limits on large datasets.

#### Consequences

- Accelerates data loading by $>10\times$ via Parquet vectorized reads.
- Guarantees reproducibility and version control via DVC remote storage.

---

### ADR-005: Progressive Fine-Tuning Ablation Protocol with Non-Failing Metrics

#### Status

Accepted

#### Date

2026-09-10

#### Context

Determining the optimal transfer learning strategy requires empirical comparison of computational cost versus prediction accuracy. In automated CI/CD pipelines, hardcoded metric thresholds or latency assertions can cause false build failures.

#### Decision

Implement a 3-tier ablation suite:

1. Strict Freezing ($\frac{\partial \mathcal{L}}{\partial \theta_{\text{base}}} = 0$, train head only).
2. Partial Fine-Tuning (unfreeze terminal layer with GRU LR `1e-4`, Head LR `1e-3`).
3. Full Fine-Tuning (unfreeze all layers).
   Evaluate using $\text{HR}@1$, $\text{HR}@3$, and $\text{MRR}$. Report metrics as advisory structured output without hard assertion failures.

#### Alternatives Considered

- _Single Fixed Fine-Tuning Strategy:_ Rejected because ablation is required to justify edge efficiency trade-offs.
- _Hard Threshold Assertions (e.g. assert HR@3 >= 0.80):_ Rejected to allow training and export across variable host hardware.

#### Consequences

- Provides rigorous empirical evidence for thesis methodology while maintaining continuous build reliability.

---

## Verification Plan

### Automated Verification

1. **DVC & Data Management:**
   - Verify `dvc.yaml` syntax and dependency DAG.
   - Verify AdSERP ingestion into `.data/interim/*.parquet`.
2. **Preprocessing & Modality Masking:**
   - Assert canonical coordinate normalization $[0, 1]$ relative to observed viewport metadata.
   - Assert MicroTensor output dimensionality is $18$ ($9$ features + $9$ mask flags).
3. **Target Generation:**
   - Verify lookahead horizon $\Delta t \in [500\text{ms}, 1500\text{ms}]$ produces discrete outcome labels $0–5$.
4. **GRU Latent Forward Pass:**
   - Pass batch `(B=4, T=8, D=18)` to `EdgeAUIGRU(18, 64)`.
   - Verify latent representation shape `(4, 64)` and output logits shape `(4, 6)`.
5. **Ablation Suite:**
   - Execute Strict Freezing, Partial Fine-Tuning, and Full Fine-Tuning on testbed split.
   - Confirm computation of $\text{HR}@1$, $\text{HR}@3$, and $\text{MRR}$.
6. **INT8 Quantization & Profiling:**
   - Export ONNX model and quantize with `QUInt8`.
   - Measure artifact size (verifying $< 20\text{ MB}$, expected $< 200\text{ KB}$) and p50/p95 latency.
7. **End-to-End Pipeline Execution:**
   - Run `python train_pipeline.py --mode minimal --epochs 2` on branch `explore/pipeline/revision/1`.
8. **Notebooks Generation:**
   - Run `generate_preprocessing_eda_nb.py`, `generate_target_distribution_nb.py`, `generate_training_nb.py`.
   - Confirm valid `.ipynb` JSON and Colab badge links to `explore/pipeline/revision/1`.

### Manual Verification

- Verify `git status` on branch `explore/pipeline/revision/1` reflects clean tracked commits.
