# Tasks: Phase D — Target-Domain Intervention-Head Training Experiments

Implementation checklist for Phase D of [Implementation Plan 1](../docs/plan/1/model-preparation-intervention-head-plan.md).

## Task List

- [x] **Task D.1: Target Intervention PyTorch Dataset & Data Loader Integration**
  - Implement `TargetInterventionDataset(Dataset)` and `load_intervention_dataset()` in `src/target_dataset.py` / `src/training.py`
  - Ingest `.data/processed/v1.0.0/target_intervention_dataset.parquet` and yield `(sequence, context_vector, target_intervention_id)` batches
  - Integrate training-partition class weights into `CrossEntropyLoss`
  - Unit tests for batch indexing, shape alignment, and loss weighting
  - *Verification:* `pytest tests/test_intervention_head_training.py -k test_dataset_loader`

- [x] **Task D.2: Experiment E1 — Strict Freezing (Frozen Backbone + Trained Head) [Task 12.1]**
  - Load foundation weights from `models/foundational_gru.pth`
  - Freeze backbone using `EdgeAUIGRU.freeze_backbone()`; attach `TargetInterventionHead(hidden_dim=64, context_dim=6, num_classes=5)`
  - Train head with `head_lr: 0.001` (`ablation.strict_freezing`) from fixed seed 42
  - Save checkpoint `models/intervention_head_e1.pth`
  - Verify bit-identical backbone weights before and after optimizer step via unit test
  - *Verification:* `pytest tests/test_intervention_head_training.py -k test_e1_freeze_contract` && `.venv/bin/python -m src.training --experiment e1 --dataset-version v1.0.0 --seed 42`

- [x] **Task D.3: Experiment E2 — Partial Fine-Tuning (Terminal Layer + Head) [Task 12.2]**
  - Start fresh from `models/foundational_gru.pth`
  - Unfreeze terminal layer `_l1` via `unfreeze_terminal_layer()`
  - Set up disjoint optimizer parameter groups with differential learning rates (`gru_lr: 0.0001`, `head_lr: 0.001` from `ablation.partial_finetuning`)
  - Save checkpoint `models/intervention_head_e2.pth`
  - Verify parameter groups are disjoint and lower layer `_l0` remains bit-identical via unit test
  - *Verification:* `pytest tests/test_intervention_head_training.py -k test_e2_partial_finetune_contract` && `.venv/bin/python -m src.training --experiment e2 --dataset-version v1.0.0 --seed 42`

- [x] **Task D.4: Experiment E3 — Full Fine-Tuning & Retention Diagnostics [Task 12.3]**
  - Start fresh from `models/foundational_gru.pth`
  - Unfreeze all layers via `unfreeze_backbone()`
  - Train with small learning rate `lr: 0.0005` (`ablation.full_finetuning`)
  - Monitor training vs validation loss divergence (overfitting diagnostics)
  - Implement foundation outcome retention check (evaluating outcome prediction before vs after fine-tuning)
  - Save checkpoint `models/intervention_head_e3.pth`
  - *Verification:* `pytest tests/test_intervention_head_training.py -k test_e3_full_finetune_contract` && `.venv/bin/python -m src.training --experiment e3 --dataset-version v1.0.0 --seed 42`

- [x] **Task D.5: Multi-Metric Evaluation Suite, Experiment 3 Record & Diagnostic Plots [Task 12.4]**
  - Implement `evaluate_intervention_suite()` in `src/training.py`
  - Evaluate E1, E2, E3 on test partition; compare against majority-class baseline
  - Generate publication-quality diagnostic plots in `docs/experiments/3/after/` via `docs/experiments/3/generate_experiment_plots.py`
  - Document formal experiment report in `docs/experiments/3/README.md` conforming to `docs/experiments/skeleton.md`
  - Label all measurements (`Measured`, `Verified by automated test`, etc.) and document concrete failure cases
  - *Verification:* `.venv/bin/python -m src.training --evaluate-suite --dataset-version v1.0.0` && `.venv/bin/python docs/experiments/3/generate_experiment_plots.py`

- [x] **Task D.6: Interactive Reproducibility Notebook Generator & Verification (Plan 0 Parallel)**
  - Implement generator script `generate_intervention_experiments_nb.py`
  - Generate `notebooks/03_intervention_head_experiments.ipynb` with Colab badge pointing to `explore/pipeline/revision/1`
  - Include automated environment/accelerator detection (CUDA/MPS/CPU), interactive E1/E2/E3 execution, comparative metrics table, and inline plots
  - Verify notebook execution and schema validity
  - *Verification:* `.venv/bin/python generate_intervention_experiments_nb.py`

- [x] **Task D.7: Regression Testing & Phase D Checkpoint Sign-Off**
  - Run full repository test suite (`pytest tests`)
  - Update `docs/plan/1/tasks/12.1.md` through `12.4.md` task files
  - Update Checkpoint D in `docs/plan/1/model-preparation-intervention-head-plan.md`
  - *Verification:* `.venv/bin/pytest tests`

---

## Checkpoint D: Training Experiments Complete
- [x] E1/E2/E3 each run from a fixed seed (42) and fixed dataset version (v1.0.0)
- [x] No model selected on a single metric; Macro-F1 reported as primary unweighted criterion
- [x] Class distribution, class weighting, and failure cases recorded
- [x] Majority baseline reported beside every experiment
- [x] Bit-identical parameter freezing verified by unit tests
- [x] Reproducibility notebook generated and verified
- [x] Review with human researcher before proceeding to Phase E
