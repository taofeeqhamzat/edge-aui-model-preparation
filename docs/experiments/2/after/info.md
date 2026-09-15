# Pipeline Audit: Post-Audit Resolution State

This document records the pipeline state after the team resolved all 10 issues.

## 1. Resolved BLOCKER Issues

The team resolved and regression-tested both BLOCKER issues:

- ISSUE-01: `reconstruct_sequences_from_parquet` now accepts `SplitResult`. The function filters by partition before it slices sequence windows. Verified by `test_split_before_sequence_construction_zero_leakage`.
- ISSUE-02: `train_foundation_model` derives weights strictly from `dataset.Y` in the training partition. Verified by `test_train_only_weighting_invariant`.

## 2. Resolved HIGH Issues

The team resolved all three HIGH issues with targeted code changes:

- ISSUE-03: The team restored the exact formula $w_c = \frac{N + C\alpha}{C(N_c + \alpha)}$ without mean normalization. Verified by `test_compute_class_weights_exact_formula`.
- ISSUE-04: `extract_lookahead_outcome` now emits `termination_source` and `observable_termination` metadata. Verified by `test_observable_abandonment_vs_stream_exhaustion`.
- ISSUE-05: Two-tier temporal resolution enforces earliest event selection. Priority applies only for simultaneous ties. Verified by `test_earliest_event_wins_before_priority`.

## 3. Resolved MEDIUM and LOW Issues

The team resolved and verified all five remaining issues:

- ISSUE-06: The code defines `INVALID_USER_PLACEHOLDERS`. The threshold relaxed to `>1` valid user. `SplitResult` returns `split_by`. Verified by `test_split_result_contract_and_reproducibility`.
- ISSUE-07: `enforce_temporal_continuity` with `max_step_gap_ms=300` breaks sequences at time gaps. Verified by `test_temporal_continuity_enforcement`.
- ISSUE-08: A 4-way distribution audit now covers corpus, train, val, and test partitions.
- ISSUE-09: `evaluate_majority_baseline` reports Accuracy 41.14%, Macro-F1 0.0833 (validation) and Accuracy 36.41%, Macro-F1 0.0890 (test).
- ISSUE-10: A behavioural gradient backpropagation test verifies zero gradients on the frozen backbone. Verified by `test_behavioral_gradient_flow`.

## 4. Visual Artifacts in the Post-Audit Directory

The post-audit directory contains five diagnostic plots:

- `target_distribution_by_partition.png`: Grouped bar chart of the 7-class window distribution across Corpus, Train, Val, and Test partitions.
- `class_weight_comparison.png`: Side-by-side bar chart that compares exact ($\alpha=0$) and smoothed ($\alpha=100$) weights on a logarithmic scale.
- `majority_baseline_performance.png`: Horizontal bar chart of Accuracy, Macro-F1, and Weighted-F1 for validation and test baselines.
- `issue_resolution_summary.png`: Severity-coded table that summarises the 10 issues and their resolution status.
- `traceability_matrix.png`: Formatted table that renders the Data-Fact-vs-Modelling-Decision matrix.
