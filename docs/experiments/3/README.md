# Experiment 3: Progressive Fine-Tuning and Evaluation of TargetInterventionHead

**Document Identifier:** `EXP-003-INTERVENTION-HEAD`  
**Governing Architecture Record:** `ADR-003`, `ADR-006`, `ADR-013`  
**Status:** `Accepted and Verified`  
**Scope:** `model-preparation` and `edge-aui-framework`

---

## 1. Context and Problem Statement

- The framework requires a lightweight recurrent model to predict discrete adaptive UI interventions directly from continuous client micro-interactions. The model maps 18-dimensional MicroTensors and 6-dimensional UIContext vectors to 5 intervention actions.
- Foundational GRU pre-training on AdSERP established a 64-dimensional latent behavioral representation $h_T$. The `TargetInterventionHead` projects this representation onto the target dashboard intervention vocabulary.
- The experiment trains the head using 289 scripted testbed examples from dataset `v1.0.0`. The dataset enforces session-bounded partitioning across 18 recording sessions.
- In accordance with ADR-013, the labels derive from the scripted deterministic policy. The learned model cannot be claimed to outperform its teacher.

---

## 2. Plan

- Execute three progressive transfer-learning experiments starting from the foundation checkpoint `models/foundational_gru.pth`. Run each experiment with fixed seed 42 on dataset `v1.0.0`.
- In Experiment E1, freeze all recurrent GRU backbone parameters and train exclusively the `TargetInterventionHead` with head learning rate 0.001. Verify that backbone weights remain bit-identical after training steps.
- In Experiment E2, unfreeze the terminal GRU layer while keeping lower layers frozen. Train with differential learning rates: GRU learning rate 0.0001 and head learning rate 0.001.
- In Experiment E3, unfreeze all GRU backbone layers and fine-tune jointly with learning rate 0.0005. Audit catastrophic forgetting of foundational outcome predictions on AdSERP validation sequences.
- Evaluate all models against the majority-class baseline on the held-out test partition. Report Macro-F1 across supported classes as the primary unweighted classification criterion.
- Document concrete failure cases, confusion pairs, and serialized artifact sizes under strict evidence discipline tags.

---

## 3. Decision

- **Candidate Selection for Integration:** Following researcher review of the multi-metric trade-off profile, **Experiment E3 (Full Fine-Tuning)** is maintained as the designated choice for Phase E edge ONNX graph compilation and edge-aui-framework integration. The selection prioritizes top-1 discriminative capability on the target task (Test Macro-F1: 0.5524, Test Accuracy: 76.7%, MRR: 0.8389), while accepting a controlled 7.2% relative degradation on foundational outcome representation (clearing the 15.0% threshold) and a 90.0% Test HR@3.
- **Reference Baselines Maintained:** **Experiment E1 (Strict Freezing)** and **Experiment E2 (Partial Fine-Tuning)** are maintained in the repository and experiment records as architectural reference baselines. They establish conservative bounds offering 100.0% top-3 candidate recall (HR@3) and near-zero foundation representation drift (0.0% in E1, 2.6% in E2).
- **Overfitting Diagnostics:** Training versus validation loss tracking confirmed that catastrophic overfitting did not occur for E3 over 5 epochs under the conservative learning rate ($\text{lr}=0.0005$), with training loss ($1.5525$, $52.40\%$ acc) tracking validation loss ($1.6057$, $50.98\%$ acc) without upward divergence.
- **Mandatory UIContext Conditioning:** Enforce mandatory UIContext tensor conditioning for all intervention head forwards. Missing or zero-imputed context vectors are strictly prohibited by architecture contract ADR-006.
- **Training-Partition Class Weighting:** Ingest training-partition class weights (`smoothed`, $\alpha=10.0$) directly into cross-entropy loss. Weights derived strictly on the training partition prevent test-set label balance from leaking into optimization.
- **Claim Boundary Restatement:** Retain the ADR-013 substitute data claim boundary in all experimental documentation. The experiment demonstrates end-to-end pipeline execution and inductive learning, but does not claim human participant generalization or outperforming the deterministic teacher policy.
- **Edge Storage Budget:** Confirm all three checkpoints satisfy the sub-200 KB edge payload budget. Each serialized PyTorch checkpoint occupies exactly 184.3 KB on disk.

---

## 4. Execution

- The engineering team implemented `TargetInterventionDataset`, `load_intervention_dataset`, and `get_intervention_class_weights` in [`src/target_dataset.py`](file:///Users/user/Workspace/MivaCS/FYP/model-preparation/src/target_dataset.py). The loader reads session-bounded Parquet datasets with full provenance metadata.
- The team implemented `train_intervention_model`, `evaluate_intervention_model`, and `evaluate_intervention_suite` in [`src/training.py`](file:///Users/user/Workspace/MivaCS/FYP/model-preparation/src/training.py). The module supports CLI execution for experiments E1, E2, E3, and comprehensive evaluation.
- The team trained Experiment E1 with command `python3 -m src.training --experiment e1 --dataset-version v1.0.0 --seed 42`. The run converged over 5 epochs on Apple Silicon MPS hardware.
- The team trained Experiment E2 with command `python3 -m src.training --experiment e2 --dataset-version v1.0.0 --seed 42`. The run configured disjoint parameter groups for terminal layer `_l1` and head.
- The team trained Experiment E3 with command `python3 -m src.training --experiment e3 --dataset-version v1.0.0 --seed 42`. The run completed full fine-tuning and executed the catastrophic forgetting retention audit.
- The team verified parameter freezing invariants and evaluation contracts in [`tests/test_intervention_head_training.py`](file:///Users/user/Workspace/MivaCS/FYP/model-preparation/tests/test_intervention_head_training.py). All 10 Phase D tests passed green in 5.37 seconds.
- The team executed [`docs/experiments/3/generate_experiment_plots.py`](file:///Users/user/Workspace/MivaCS/FYP/model-preparation/docs/experiments/3/generate_experiment_plots.py). The script produced four diagnostic plots in [`docs/experiments/3/after/`](file:///Users/user/Workspace/MivaCS/FYP/model-preparation/docs/experiments/3/after/).

---

## 5. Observation and Result

- **Evaluation Summary Across Regimes:**

| Architecture Regime | Val Macro-F1 | Val Accuracy | Test Macro-F1 | Test Accuracy | HR@3 | Payload Size | Evidence Tag |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Majority Baseline** | 0.1810 | 37.3% | 0.2121 | 46.7% | 60.0% | - | `Measured` |
| **E1: Strict Freezing** | 0.2873 | 41.2% | 0.4530 | 63.3% | 100.0% | 184.3 KB | `Measured` |
| **E2: Partial Fine-Tuning** | 0.2873 | 41.2% | 0.4768 | 66.7% | 100.0% | 184.3 KB | `Measured` |
| **E3: Full Fine-Tuning** | **0.3917** | **51.0%** | **0.5524** | **76.7%** | **90.0%** | **184.3 KB** | `Measured` |

- **Empirical Superiority of E3:** Full fine-tuning achieved 0.5524 Test Macro-F1, outperforming the majority baseline by +0.3403 and Experiment E1 by +0.0994 [`Measured`].
- **Bit-Identical Freeze Verification:** Unit tests confirmed all 16 backbone weight tensors in E1 remained bit-identical after backward optimization [`Verified by automated test`].
- **Disjoint Group Verification:** Unit tests confirmed E2 optimizer groups are disjoint, preserving frozen weights in layer `_l0` while updating `_l1` and head [`Verified by automated test`].
- **Catastrophic Forgetting Audit:** E3 foundation outcome Macro-F1 shifted from 0.3947 to 0.3663 on AdSERP validation sequences, representing a 7.2% relative degradation that clears the 15.0% retention threshold [`Measured`].
- **Dominant Test Confusion Pairs:** In E3, 5 misclassifications occurred where `offer_assistance` was predicted as `no_op`, and 2 where `expand_tooltip` was predicted as `no_op` [`Measured`].
- **Representative Failure Case:** Sample `ef41ca1f-b253-4da4-943b-c0e14def8a2c:w24` (Task T1, true `offer_assistance`) was predicted as `no_op` with 54.1% probability during rapid scrolling transitions [`Measured`].
- **Edge Budget Compliance:** Serialized PyTorch model payload is 184.3 KB, satisfying the sub-200 KB constraint [`Measured`]. Resident memory and browser inference latency remain deferred to Phase E ONNX profiling [`Deferred`].
- **Diagnostic Visualizations:** Empirical confusion matrices, ablation comparisons, per-class F1 bars, and retention audits are preserved in [`after/`](file:///Users/user/Workspace/MivaCS/FYP/model-preparation/docs/experiments/3/after/).
