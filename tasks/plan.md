# Implementation Plan: Phase D — Target-Domain Intervention-Head Training Experiments & Evaluation

Continuation of [Implementation Plan 1](../docs/plan/1/model-preparation-intervention-head-plan.md), Phase D (Tasks 12.1–12.4).
Paralleling Plan 0's methodology by integrating interactive notebook generation for experiment documentation and reproducibility.

---

## 1. Executive Summary & Objectives

Phase D transitions the `model-preparation` pipeline from dataset curation (Phase A–C) to empirical transfer-learning experimentation. 
The objective is to train, evaluate, and document the `TargetInterventionHead` using the pre-trained `EdgeAUIGRU` backbone on the frozen `v1.0.0` session-bounded intervention dataset.

The core experimental inquiry evaluates three progressive fine-tuning regimes:
1. **Experiment E1 (Task 12.1):** Strict Freezing (`freeze_backbone()` + trained `TargetInterventionHead`).
2. **Experiment E2 (Task 12.2):** Partial Fine-Tuning (Frozen lower GRU layer + unfrozen terminal layer `_l1` + trained `TargetInterventionHead` with differential learning rates).
3. **Experiment E3 (Task 12.3):** Full Fine-Tuning (All layers trainable with small learning rate + catastrophic forgetting retention check).
4. **Multi-Metric Evaluation & Experiment Record (Task 12.4):** Comprehensive evaluation across validation and test partitions, confusion matrices, failure case analysis, baseline comparison, and generation of publication-quality diagnostic plots in `docs/experiments/3/after/` and report in `docs/experiments/3/README.md`.
5. **Interactive Reproducibility (Plan 0 Parallel):** Programmatic notebook generation via `generate_intervention_experiments_nb.py` producing `notebooks/03_intervention_head_experiments.ipynb` equipped with Colab badges, automated accelerator detection, interactive execution, and inline plots.

---

## 2. Architectural Decisions & Methodological Guardrails

1. **ADR-003 & ADR-006 Compliance:** The `TargetInterventionHead` interface takes `[h_T, context_vector]` where `context_vector in R^6` is mandatory. Zero context or missing context is prohibited.
2. **ADR-013 Claim Boundary:** The dataset was generated using scripted testbed traces with labels assigned via the deterministic policy (`LABEL_POLICY_VERSION = "1.0.0"`). A learned head trained on these labels **cannot be reported as outperforming that policy** (its teacher).
3. **Evidence Discipline (Brief §18):** No fabricated measurements. Every metric must be explicitly tagged as `Measured`, `Verified by automated test`, `Inferred`, `Not measured`, `Deferred`, or `Blocked`.
4. **No Single-Metric Model Selection (Brief §10):** Models must not be selected on accuracy alone. Evaluation mandates Macro-F1 (unweighted across supported classes), Weighted-F1, per-class Precision/Recall/F1, Confusion Matrix, Majority Baseline comparison, and secondary ranking diagnostics (HR@1, HR@3, MRR).
5. **Test Partition Isolation:** Hyperparameters and model training decisions use training and validation splits only. The test partition is evaluated exclusively during final benchmark reporting.
6. **Bit-Identical Parameter Freeze Verification:** Freezing must be verified through automated tests comparing model state dictionaries before and after an optimizer step.
7. **Disjoint Parameter Groups:** E2 differential learning rates must be verified via test to guarantee parameter groups are mutually disjoint and cover all trainable parameters.

---

## 3. Dependency Graph & Pipeline Flow

```text
.data/processed/v1.0.0/target_intervention_dataset.parquet + manifest.json
                           │
                           ▼
Task D.1: Target Intervention Dataset Loader & Loss Weighting Integration
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
Task D.2: E1 (Strict)  Task D.3: E2 (Partial) Task D.4: E3 (Full)
  models/e1.pth          models/e2.pth          models/e3.pth
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                           ▼
Task D.5: Multi-Metric Evaluation Suite, Experiment 3 Record & Diagnostic Plots
         (docs/experiments/3/README.md + docs/experiments/3/after/*.png)
                           │
                           ▼
Task D.6: Interactive Reproducibility Notebook Generator & Verification
         (generate_intervention_experiments_nb.py -> notebooks/03_intervention_head_experiments.ipynb)
                           │
                           ▼
Task D.7: Full Suite Regression Verification & Phase D Checkpoint Sign-off
```

---

## 4. Detailed Task Breakdown

### Task D.1: Target Intervention PyTorch Dataset & Data Loader Integration
- **Description:** Implement `TargetInterventionDataset(Dataset)` and `load_intervention_dataset()` in `src/target_dataset.py` / `src/training.py` to ingest the session-bounded Parquet dataset (`.data/processed/{version}/target_intervention_dataset.parquet`). Deliver `(sequence, context_vector, target_intervention_id)` batches of shapes `(B, 8, 18)`, `(B, 6)`, `(B,)`. Supply training-partition class weights directly to `CrossEntropyLoss`.
- **Acceptance criteria:**
  - `TargetInterventionDataset` correctly indexes and yields `(x, context, y)` with matching tensor types.
  - `load_intervention_dataset` accepts `split in ['train', 'val', 'test']` and `dataset_version`.
  - Loss weights match the pre-computed training-partition weights (`split_summary["class_weights"]`).
- **Verification:**
  - Automated tests in `tests/test_intervention_head_training.py` asserting tensor dimensions, partition filtering, and batch collating.
- **Estimated Scope:** S (2 files: `src/target_dataset.py`, `tests/test_intervention_head_training.py`).

### Task D.2: Experiment E1 — Strict Freezing (Frozen Backbone + Trained Head) (Task 12.1)
- **Description:** Implement Experiment E1 in `src/training.py`: load foundation weights from `models/foundational_gru.pth`, freeze backbone using `model.freeze_backbone()`, attach `TargetInterventionHead(hidden_dim=64, context_dim=6, num_classes=5)`. Train only head parameters with `head_lr: 0.001` (`ablation.strict_freezing`).
- **Acceptance criteria:**
  - E1 runs with a fixed seed (`seed=42`) and dataset `v1.0.0`.
  - Automated test asserts backbone weights are bit-identical before and after an optimizer step, while head weights change.
  - Model checkpoint saved to `models/intervention_head_e1.pth`.
  - Training and validation metrics logged per epoch.
- **Verification:**
  - `pytest tests/test_intervention_head_training.py -k test_e1_freeze_contract`
  - `.venv/bin/python -m src.training --experiment e1 --dataset-version v1.0.0 --seed 42`
- **Estimated Scope:** S (2 files: `src/training.py`, `tests/test_intervention_head_training.py`).

### Task D.3: Experiment E2 — Partial Fine-Tuning (Terminal Layer + Head) (Task 12.2)
- **Description:** Implement Experiment E2 in `src/training.py`: start from `models/foundational_gru.pth`, unfreeze terminal GRU layer `_l1` (`unfreeze_terminal_layer()`), configure disjoint optimizer parameter groups with differential learning rates (`gru_lr: 0.0001`, `head_lr: 0.001` from `ablation.partial_finetuning`).
- **Acceptance criteria:**
  - E2 runs from a fixed seed starting from foundation weights.
  - Automated test asserts parameter groups are disjoint, lower layer `_l0` weights are bit-identical after an optimizer step, and terminal layer `_l1` + head weights change.
  - Model checkpoint saved to `models/intervention_head_e2.pth`.
- **Verification:**
  - `pytest tests/test_intervention_head_training.py -k test_e2_partial_finetune_contract`
  - `.venv/bin/python -m src.training --experiment e2 --dataset-version v1.0.0 --seed 42`
- **Estimated Scope:** S (2 files: `src/training.py`, `tests/test_training_contract.py`).

### Task D.4: Experiment E3 — Full Fine-Tuning & Retention Diagnostics (Task 12.3)
- **Description:** Implement Experiment E3 in `src/training.py`: start from `models/foundational_gru.pth`, unfreeze all layers (`unfreeze_backbone()`), train with uniform `lr: 0.0005` (`ablation.full_finetuning`). Implement catastrophic forgetting / foundation-head retention check: compare foundation outcome head predictions before and after fine-tuning.
- **Acceptance criteria:**
  - E3 runs from a fixed seed starting from foundation weights.
  - Automated test asserts all parameters are trainable.
  - Overfitting monitor logs train vs validation divergence.
  - Foundation outcome retention diagnostic recorded.
  - Model checkpoint saved to `models/intervention_head_e3.pth`.
- **Verification:**
  - `pytest tests/test_intervention_head_training.py -k test_e3_full_finetune_contract`
  - `.venv/bin/python -m src.training --experiment e3 --dataset-version v1.0.0 --seed 42`
- **Estimated Scope:** S (2 files: `src/training.py`, `tests/test_intervention_head_training.py`).

### Task D.5: Multi-Metric Evaluation Suite, Experiment 3 Record & Diagnostic Plots (Task 12.4)
- **Description:** Implement `evaluate_intervention_suite()` in `src/training.py`. Evaluate E1, E2, and E3 on held-out test partition against the majority baseline. Generate publication-quality diagnostic plots via `docs/experiments/3/generate_experiment_plots.py` into `docs/experiments/3/after/`. Document complete empirical findings in `docs/experiments/3/README.md` following `docs/experiments/skeleton.md` with explicit evidence discipline tags and concrete failure cases.
- **Acceptance criteria:**
  - Directly comparable evaluation table for E1, E2, E3, and Majority Baseline.
  - All metrics reported: Accuracy, Macro-F1, Weighted-F1, Per-class Precision/Recall/F1, Confusion Matrix, HR@1, HR@3, MRR.
  - Every numerical entry carries an evidence label (`Measured`, `Verified by automated test`, etc.).
  - Concrete failure case sequences documented with context.
  - Diagnostic plots generated: confusion matrices, loss curves, per-class F1 comparisons, ablation delta bars.
- **Verification:**
  - `.venv/bin/python -m src.training --evaluate-suite --dataset-version v1.0.0`
  - `.venv/bin/python docs/experiments/3/generate_experiment_plots.py`
  - Confirm `docs/experiments/3/README.md` and `docs/experiments/3/after/` plots exist and match.
- **Estimated Scope:** M (3 files: `src/training.py`, `docs/experiments/3/README.md`, `docs/experiments/3/generate_experiment_plots.py`).

### Task D.6: Interactive Reproducibility Notebook Generator & Verification (Plan 0 Parallel)
- **Description:** Build generator script `generate_intervention_experiments_nb.py` and compile `notebooks/03_intervention_head_experiments.ipynb`. Modelled after `generate_training_nb.py`, providing hosted accelerator detection (CUDA/MPS/CPU), Colab badge pointing to `explore/pipeline/revision/1`, dataset loading, live execution of E1/E2/E3, evaluation reporting, and inline visualization plots.
- **Acceptance criteria:**
  - `generate_intervention_experiments_nb.py` runs and outputs valid `notebooks/03_intervention_head_experiments.ipynb`.
  - Notebook contains valid JSON schema, cell metadata, Colab badge, and complete narrative.
  - Notebook is self-contained and reproducible.
- **Verification:**
  - `.venv/bin/python generate_intervention_experiments_nb.py`
  - Verify notebook JSON validity and execution smoke test.
- **Estimated Scope:** S (2 files: `generate_intervention_experiments_nb.py`, `notebooks/03_intervention_head_experiments.ipynb`).

### Task D.7: Regression Testing & Phase D Checkpoint Sign-Off
- **Description:** Run full repository test suite (`pytest tests`), verify all 97+ tests pass without regression, verify task documentation files (`docs/plan/1/tasks/12.1.md` through `12.4.md`), and update Checkpoint D status in `docs/plan/1/model-preparation-intervention-head-plan.md`.
- **Acceptance criteria:**
  - `pytest tests` passes 100% green.
  - Checkpoint D criteria verified and documented.
  - Review summary prepared for human researcher before Phase E.
- **Verification:**
  - `.venv/bin/pytest tests`
- **Estimated Scope:** XS (documentation and test execution).

---

## 5. Checkpoints & Gateways

### Checkpoint D: Training Experiments Complete
- [ ] E1, E2, E3 each run from a fixed seed (`seed=42`) and the frozen dataset version `v1.0.0`.
- [ ] No model selected on a single metric; Macro-F1 is primary unweighted metric.
- [ ] Class distribution, class weighting, and failure cases recorded.
- [ ] Majority baseline reported beside every experiment.
- [ ] Bit-identical parameter freezing verified by unit tests.
- [ ] Interactive notebook `03_intervention_head_experiments.ipynb` generated and verified.
- [ ] Review with human researcher before proceeding to Phase E (ONNX export).
