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
Intervention Vocabulary (from src/config.yaml):
  0: simplify_options
  1: highlight_primary_action
  2: offer_assistance
  3: expand_tooltip
  4: no_op
```

### Deterministic Mapping Rules (Policy v1.0.0)

1. **`HOVER_DWELL`**:
   - If `primaryActionAvailable == True` and `taskProgress >= 0.5`:
     Assigns `highlight_primary_action` (ID: 1) to guide task step completion.
   - Otherwise:
     Assigns `expand_tooltip` (ID: 3) to provide contextual field guidance.
2. **`RAPID_SCROLL`**:
   - Assigns `simplify_options` (ID: 0) to reduce visual clutter and density.
3. **`BACKTRACK` or `ABANDON`**:
   - Assigns `offer_assistance` (ID: 2) to mitigate hesitation or task derailment.
4. **`CLICK`, `FORM_SUBMIT`, `NO_OUTCOME`**:
   - Assigns `no_op` (ID: 4) representing default, non-disruptive system behavior.

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
| `simplify_options` | 0 | 9 | 3.1% | `RAPID_SCROLL` over data table |
| `highlight_primary_action` | 1 | 13 | 4.5% | `HOVER_DWELL` near completion ($p > 0.5$) |
| `offer_assistance` | 2 | 100 | 34.6% | `ABANDON`, `BACKTRACK` |
| `expand_tooltip` | 3 | 42 | 14.5% | `HOVER_DWELL` during configuration ($p \le 0.5$) |
| `no_op` | 4 | 125 | 43.3% | `NO_OUTCOME`, `CLICK`, `FORM_SUBMIT` |
| **Total** | — | **289** | **100.0%** | All 5 intervention classes represented |

### Analysis
1. **Majority Baseline ($43.3\%$):** A trivial majority classifier that always predicts `no_op` achieves 43.3% accuracy. Any trained multi-class head must be evaluated against this baseline.
2. **Multi-Class Representation:** All 5 classes from the canonical intervention vocabulary have non-zero representation in the dataset, ensuring balanced loss computation during transfer learning (Phase D, Task 12.1).
3. **Non-Imputation Invariant:** Zero sequences had missing context imputed; all 289 examples have valid, non-zero $\mathbb{R}^6$ context vectors validated against `validate_context_vector`.

---

## 5. Session-Bounded Partitions (Phase C, Task 11.1)

To prevent severe temporal data leakage, splits are partitioned **strictly by session ID**. No session, and no task run within a session, appears in more than one partition:
$$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} = \emptyset, \quad \mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset, \quad \mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{test}} = \emptyset$$

### Partition Allocation (70% / 15% / 15%, Seed 42)
- **Train Partition:** 13 sessions (72.2%), 208 sequence examples (72.0%)
- **Validation Partition:** 3 sessions (16.7%), 51 sequence examples (17.6%)
- **Test Partition:** 2 sessions (11.1%), 30 sequence examples (10.4%)
- **Total:** 18 sessions, 289 sequence examples

### Per-Partition Empirical Class Distribution

| Intervention Class | ID | Train Count (%) | Val Count (%) | Test Count (%) | Overall (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `simplify_options` | 0 | 9 (4.3%) | 0 (0.0%) | 0 (0.0%) | 9 (3.1%) |
| `highlight_primary_action` | 1 | 13 (6.2%) | 0 (0.0%) | 0 (0.0%) | 13 (4.5%) |
| `offer_assistance` | 2 | 71 (34.1%) | 17 (33.3%) | 12 (40.0%) | 100 (34.6%) |
| `expand_tooltip` | 3 | 23 (11.1%) | 15 (29.4%) | 4 (13.3%) | 42 (14.5%) |
| `no_op` | 4 | 92 (44.2%) | 19 (37.3%) | 14 (46.7%) | 125 (43.3%) |
| **Total** | — | **208 (100%)** | **51 (100%)** | **30 (100%)** | **289 (100%)** |

---

## 6. Absent Classes per Partition

Per Task 11.1 acceptance criteria, classes absent from any partition are explicitly declared:

- **Train:** **None** — All 5 intervention classes are represented in the training partition, ensuring gradients exist for every output logit during backbone and head training.
- **Validation:** `simplify_options` (0 samples), `highlight_primary_action` (0 samples).
- **Test:** `simplify_options` (0 samples), `highlight_primary_action` (0 samples).

> [!NOTE]
> In the scripted testbed traces, rapid scrolling and near-completion hovering occurred in specific task trials that random session assignment allocated to the training split. Validation and test evaluation on these two minority classes will reflect zero support; macro-F1 must be computed over classes with non-zero support, and unweighted majority baseline metrics must be reported alongside model performance.

---

## 7. Majority-Class Baselines

Evaluated using the majority class derived **strictly from the training partition** (`no_op`, class ID 4):

| Partition | Majority Class | Accuracy | Macro-F1 (Supported) |
| :--- | :--- | :---: | :---: |
| **Train** | `no_op` | 44.2% | 0.1227 |
| **Validation** | `no_op` | 37.3% | 0.1810 |
| **Test** | `no_op` | 46.7% | 0.2121 |

---

## 8. Class Imbalance Strategy & Training Partition Weights

### Strategy Selection Rationale (Addressing Open Question 3)
1. **Oversampling/Resampling Rejected:** Duplicating sequences or creating synthetic micro-interactions would distort sequential temporal dependencies and give an illusion of increased sample size.
2. **Unweighted Loss ("Neither") Rejected:** An unweighted loss function allows the head to collapse into the two dominant classes (`no_op` at 44.2% and `offer_assistance` at 34.1%), ignoring the minority actionable classes `simplify_options` (4.3%) and `highlight_primary_action` (6.2%).
3. **Smoothed Inverse Class Frequency Selected:** Applies loss penalty weights $w_c = \frac{N + C \cdot \alpha}{C \cdot (N_c + \alpha)}$ with smoothing $\alpha = 10.0$ to prevent explosive gradient ratios while lifting minority-class representation.

### Weights Resolved on Training Partition ONLY (Leak-Free)
$$w_c = \frac{208 + 5 \times 10}{5 \times (N_c + 10)} = \frac{258}{5 \times (N_c + 10)}$$

| Intervention Class | Class ID | Train Count ($N_c$) | Resolved Weight ($w_c$) | Normalized Relative Scale |
| :--- | :---: | :---: | :---: | :---: |
| `simplify_options` | 0 | 9 | **2.7158** | 5.37× |
| `highlight_primary_action` | 1 | 13 | **2.2435** | 4.43× |
| `expand_tooltip` | 3 | 23 | **1.5636** | 3.09× |
| `offer_assistance` | 2 | 71 | **0.6370** | 1.26× |
| `no_op` | 4 | 92 | **0.5059** | 1.00× (reference) |

- **Max/Min Nonzero Ratio:** 5.37 (well within the numerical stability bound $< 15.0$).
- **Integrity Guarantee:** A unit test (`test_class_weights_resolved_on_training_partition_only`) verifies that contaminating or modifying the validation or test partitions yields bit-identical training weights, proving zero test-set balance leakage.

