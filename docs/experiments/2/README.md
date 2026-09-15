# Experiment 2: Methodological Audit of Target Generation, Leak-Free Splitting, and GRU Architecture

**Document Identifier:** `EXP-002-PIPELINE-AUDIT`  
**Governing Architecture Record:** `ADR-002`, `ADR-003`  
**Status:** `Accepted and Verified`  
**Scope:** `model-preparation` and `edge-aui-framework`

---

## 1. Context and Problem Statement

- The pipeline chains six stages: preprocessing, target generation, leak-free partitioning, sequence reconstruction, class weighting, and GRU training. Each stage must satisfy strict methodological constraints before the foundation model training experiment can proceed (Task 4.2).
- The audit identified 10 issues across three severity tiers. Two BLOCKER issues threatened data leakage across partitions, while three HIGH issues distorted class weighting, target causality, and label semantics.
- The corpus contains 112,865 windows across 2,776 sessions and 47 users. Class imbalance reaches 9,854:1 between `NO_OUTCOME` and `FORM_SUBMIT` when using exact inverse frequency weights.
- AdSERP contains zero DOM lifecycle events (`beforeunload`, `pagehide`, `unload`). The pipeline labelled stream exhaustion as `ABANDON` without explicit metadata.

---

## 2. Plan

- Audit all pipeline stages from raw kinematics to baseline evaluation. Trace every empirical observation to its source partition and verify its methodological status.
- Verify lookahead boundary enforcement at millisecond resolution. Verify earliest-event temporal causality before priority tie-breaking.
- Verify that partition assignment precedes overlapping sequence reconstruction. Verify zero cross-split identity leakage by user and session.
- Verify class weight computation on the training partition exclusively. Verify the exact formula $w_c = \frac{N + C\alpha}{C(N_c + \alpha)}$ without mean normalization.
- Verify that the pipeline emits metadata with `termination_source` and `observable_termination` flags for `ABANDON`. Enforce recording-termination proxy semantics in documentation.
- Evaluate a majority-class baseline on validation and test sets. Establish Macro-F1 as the primary evaluation metric for Task 4.2.
- Execute the full 50-test regression suite. Verify zero test failures across all components.

---

## 3. Decision

- Approve the pipeline to proceed with the foundation model training experiment (Task 4.2). This decision verifies methodological readiness, but does not verify the architecture as an empirically useful foundation representation.
- Enforce partition-first sequence reconstruction through `SplitResult` in [`microtensor_store.py`](file:///Users/user/Workspace/MivaCS/FYP/model-preparation/src/microtensor_store.py). The `reconstruct_sequences_from_parquet` function filters by partition IDs before sequence window slicing.
- Derive class weights strictly from training partition targets in [`target_generation.py`](file:///Users/user/Workspace/MivaCS/FYP/model-preparation/src/target_generation.py). Reject mean normalization to preserve the exact inverse frequency formula.
- Enforce earliest-event temporal selection with priority as a tie-breaker only. An earlier low-priority event takes precedence over a later high-priority event.
- Document `ABANDON` as an empirical recording-termination proxy in AdSERP. The pipeline emits `termination_source="stream_exhaustion"` and `observable_termination=False` metadata.
- Adopt Macro-F1 as the primary evaluation metric for Task 4.2. The majority baseline achieves Macro-F1 of 0.0833 on validation and 0.0890 on test.

---

## 4. Execution

- The team updated [`microtensor_store.py`](file:///Users/user/Workspace/MivaCS/FYP/model-preparation/src/microtensor_store.py) to accept `SplitResult` in `reconstruct_sequences_from_parquet`. The function filters partitions before sequence slicing (ISSUE-01).
- The team modified `train_foundation_model` in [`training.py`](file:///Users/user/Workspace/MivaCS/FYP/model-preparation/src/training.py). The function derives weights from `dataset.Y` in the training partition only (ISSUE-02).
- The team restored the exact class weight formula in `compute_class_weights` in [`target_generation.py`](file:///Users/user/Workspace/MivaCS/FYP/model-preparation/src/target_generation.py). The team removed the `weights / weights.mean()` normalization (ISSUE-03).
- The team added `termination_source` and `observable_termination` metadata to `extract_lookahead_outcome`. Documentation records proxy semantics (ISSUE-04).
- The team enforced two-tier temporal resolution in `extract_lookahead_outcome`. The code finds the minimum event timestamp, then applies priority for simultaneous ties only (ISSUE-05).
- The team defined `INVALID_USER_PLACEHOLDERS` and relaxed the user threshold to `>1` valid user. The team introduced `SplitResult` with the `split_by` field (ISSUE-06).
- The team added the `enforce_temporal_continuity` parameter with `max_step_gap_ms=300` to `reconstruct_sequences_from_parquet` (ISSUE-07).
- The team implemented a 4-way distribution audit across corpus, train, val, and test partitions (ISSUE-08).
- The team implemented `evaluate_majority_baseline`. The function reports Accuracy, Macro-F1, Weighted-F1, and per-class metrics (ISSUE-09).
- The team added a behavioural gradient backpropagation test. The test verifies zero gradients on the frozen backbone and active gradients on the unfrozen backbone (ISSUE-10).
- The team executed the full test suite with `python -m unittest discover tests -v`. All 50 tests passed in 3.183 seconds.

---

## 5. Observation and Result

- **Experiment Readiness:** The audit approves the pipeline to run the foundation model training experiment (Task 4.2). The team resolved and verified all 10 issues, while empirical validation of the architecture remains the goal of that upcoming experiment.
- **Partition Coverage:** The training partition contains 84,179 windows across 33 users. Validation contains 11,887 windows across 7 users, and the test partition contains 16,799 windows across 7 users.
- **Missing Class in Test:** The test partition contains zero instances of `BACKTRACK`. Only 3 users exhibited backtracks, and none landed in the test split.
- **Single-User Minorities:** Only 1 user per partition exhibits `FORM_SUBMIT`. The model cannot learn general motor features from a single subject.
- **Class Weight Diagnostics (Training, α=0):** Exact inverse weights yield a maximum of 4,008.52 (`FORM_SUBMIT`) and a minimum of 0.4068 (`NO_OUTCOME`). The ratio between maximum and minimum weights is 9,854:1.
- **Class Weight Diagnostics (Training, α=100):** Smoothed weights yield a maximum of 117.72 (`FORM_SUBMIT`) and a minimum of 0.4088 (`NO_OUTCOME`). The ratio between maximum and minimum weights is 288:1.
- **Majority Baseline (Validation):** The baseline yields 41.14% Accuracy, 0.0833 Macro-F1, and 0.2398 Weighted-F1. Per-class F1 is 0.5829 for `NO_OUTCOME` and 0.0000 for all other classes.
- **Majority Baseline (Test):** The baseline yields 36.41% Accuracy, 0.0890 Macro-F1, and 0.1944 Weighted-F1.
- **Test Suite:** The regression suite passed all 50 tests across 6 test files. The run produced zero failures and zero skipped tests.
- **Artifact Locations:** Directory [`2/before/`](before/) contains baseline diagnostic artifacts. Directory [`2/after/`](after/) contains post-audit diagnostic plots.
