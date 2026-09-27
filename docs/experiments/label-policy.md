# Scripted Intervention-Target Label Policy Report

**Phase:** B — Substitute Data and Labels (Task 10.3)  
**Policy Version:** `1.0.0`  
**Dataset Path:** `.data/processed/target_intervention_dataset.parquet`  
**Label Type:** Scripted Policy Output (Deterministic)  
**Schema Version:** `1.1.0`  
**Date:** 2026-09-27  

---

## 1. ADR-013 Mandatory Claim Boundary (Verbatim)

> This dataset represents substitute interaction traces and scripted label assignments. It does not represent human participant behavior, and a model trained on this dataset must never be reported as demonstrating superior intervention selection capability relative to the scripted policy.

Furthermore, per ADR-013 §6:
> Results from this dataset may be reported as: 'the dataset -> preparation -> training -> export -> runtime path executes end to end', 'the learned head loads in the browser and produces logits of the expected shape', and engineering and pipeline findings. Results from this dataset may NOT be reported as: evidence about human participants; evidence about usability, task performance, or intervention benefit; or evidence that learned intervention prediction outperforms the deterministic policy — a model trained on labels produced by that policy is being compared against its own teacher.

---

## 2. Policy Definition and Versioning

The scripted label policy deterministically maps `(observable_outcome, UIContext)` to one of five discrete intervention classes defined in `src/config.yaml`:

```
Intervention Vocabulary:
  0: no_op
  1: highlight_primary_action
  2: simplify_options
  3: expand_tooltip
  4: offer_assistance
```

### Deterministic Mapping Rules (Policy v1.0.0)

1. **`HOVER_DWELL`**:
   - If `primaryActionAvailable == True` and `taskProgress >= 0.5`:
     Assigns `highlight_primary_action` (ID: 1) to guide task step completion.
   - Otherwise:
     Assigns `expand_tooltip` (ID: 3) to provide contextual field guidance.
2. **`RAPID_SCROLL`**:
   - Assigns `simplify_options` (ID: 2) to reduce visual clutter and density.
3. **`BACKTRACK` or `ABANDON`**:
   - Assigns `offer_assistance` (ID: 4) to mitigate hesitation or task derailment.
4. **`CLICK`, `FORM_SUBMIT`, `NO_OUTCOME`**:
   - Assigns `no_op` (ID: 0) representing default, non-disruptive system behavior.

### Configuration-Driven Policy Override Contract

The policy supports an optional custom mapping dictionary (e.g., for ablation experiments or runtime policy updates). When a custom policy provides an override for an outcome, the override takes precedence.

---

## 3. Dataset Assembly and Ingestion

The dataset assembles model-ready training examples combining:
- **Temporal Sequences:** Consecutive MicroTensor windows ($T=8, D=18$) representing motor kinematics and modality masks.
- **Context Vectors:** Normalized $\mathbb{R}^6$ UIContext vectors encoding route, capabilities, task progress, and action availability.
- **Target Labels:** Scripted intervention class and ID, with explicit provenance metadata.

### Strict Non-Imputation Rule
Missing or malformed UIContext is never zero-imputed. Sequences lacking valid context are rejected and increment `excluded_missing_context` in assembly telemetry.

---

## 4. Class Distribution & Majority Baseline

The target intervention dataset was compiled from 18 verified traces across baseline and adaptive conditions.

### Dataset Overview
- **Source Traces:** 18 (9 baseline, 9 adaptive)
- **Total Candidate Sequences ($T=8, D=18$):** 289
- **Retained Model-Ready Examples:** 289
- **Excluded Due to Missing/Invalid Context:** 0 (100% strict compliance)
- **Majority Baseline Accuracy:** 43.3% (`no_op`)

### 5-Class Empirical Distribution

| Intervention Class | ID | Count | Percentage | Primary Behavioral Drivers |
| :--- | :---: | :---: | :---: | :--- |
| `no_op` | 0 | 125 | 43.3% | `NO_OUTCOME`, `CLICK`, `FORM_SUBMIT` |
| `offer_assistance` | 4 | 100 | 34.6% | `ABANDON`, `BACKTRACK` |
| `expand_tooltip` | 3 | 42 | 14.5% | `HOVER_DWELL` during configuration ($p \le 0.5$) |
| `highlight_primary_action` | 1 | 13 | 4.5% | `HOVER_DWELL` near completion ($p > 0.5$) |
| `simplify_options` | 2 | 9 | 3.1% | `RAPID_SCROLL` over data table |
| **Total** | — | **289** | **100.0%** | All 5 intervention classes represented |

### Analysis
1. **Majority Baseline ($43.3\%$):** A trivial majority classifier that always predicts `no_op` achieves 43.3% accuracy. Any trained multi-class head must be evaluated against this baseline.
2. **Multi-Class Representation:** All 5 classes from the canonical intervention vocabulary have non-zero representation in the dataset, ensuring balanced loss computation during transfer learning (Phase D, Task 12.1).
3. **Non-Imputation Invariant:** Zero sequences had missing context imputed; all 289 examples have valid, non-zero $\mathbb{R}^6$ context vectors validated against `validate_context_vector`.
