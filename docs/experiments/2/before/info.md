# Pipeline Audit: Pre-Audit Issue State

This document records the pipeline state before the team resolved the 10 issues.

## 1. BLOCKER Issues

Two BLOCKER issues threatened data leakage and weight contamination:

- ISSUE-01: Overlapping sequence reconstruction could cross unpartitioned datasets. The `reconstruct_sequences_from_parquet` function did not filter by partition before it sliced sequence windows.
- ISSUE-02: The `train_foundation_model` function accepted precomputed weights or the full dataset. Class weights could include validation and test targets.

## 2. HIGH Issues

Three HIGH issues distorted training signals and label semantics:

- ISSUE-03: Mean normalization with `weights / weights.mean()` compressed majority class weights to 0.0004. Extreme minority outliers increased the denominator.
- ISSUE-04: The pipeline treated stream exhaustion as behavioural abandonment. No metadata separated observable lifecycle events from recording termination.
- ISSUE-05: The priority hierarchy could select a later outcome over an earlier outcome. A high-priority event that occurred after a low-priority event violated temporal causality.

## 3. MEDIUM Issues

Four MEDIUM issues weakened partitioning logic and evaluation:

- ISSUE-06: The user threshold `len(valid_users) > 5` was arbitrary. Empty strings could pass as valid user identifiers.
- ISSUE-07: The extractor dropped inactive windows that contained fewer than 3 events. This created step gaps that exceeded 250 ms between consecutive windows.
- ISSUE-08: Exploratory analysis reported only window counts. Session-level and user-level concentration stayed hidden.
- ISSUE-09: The team evaluated GRU accuracy without a naive majority-class baseline. Class imbalance bias stayed unmeasured.

## 4. LOW Issues

One LOW issue addressed test coverage:

- ISSUE-10: The code tested parameter `requires_grad` flags but did not verify gradient backpropagation. The team did not test frozen backbone gradient isolation at the behavioural level.
