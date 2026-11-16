# Plan 3 — Semantic Task-Outcome Model Preparation

This directory is the model-preparation half of the framework's next-milestone plan. It is a
planning package only; no training, dataset, configuration, model or deployment change was made
while producing it.

## Documents

- [Next-milestone model-preparation plan](next-milestone-model-preparation-plan.md) — objective,
  current state, 4A/4B split, semantic taxonomy/migration, dataset/experiment/training/export,
  acceptance and Definition of Done.
- [Decision dependencies](decision-dependencies.md) — the framework ADRs that constrain this
  repository, and the model-preparation obligations that follow.
- [Workstream 40](workstream/40-model-preparation/work.md) — task index, checkpoints, exit criteria
  and verification.
- [Task 40.1](workstream/40-model-preparation/tasks/40.1-ingest-schema-1.4-semantic-outcomes.md) —
  1.4.0 ingestion and semantic fields.
- [Task 40.2](workstream/40-model-preparation/tasks/40.2-semantic-target-generation-and-dataset.md) —
  semantic target generation and dataset v2.
- [Task 40.3](workstream/40-model-preparation/tasks/40.3-window-horizon-splits-class-balance.md) —
  window/horizon experiment, splits and class balance.
- [Task 40.4](workstream/40-model-preparation/tasks/40.4-semantic-head-training-evaluation.md) —
  semantic head training/evaluation (blocked until participant traces).
- [Task 40.5](workstream/40-model-preparation/tasks/40.5-onnx-export-bundle-v2-handoff.md) — ONNX
  export, bundle v2 and framework handoff.

## Phase split

- **4A (40.1–40.3):** ingestion, semantic dataset, splits/class balance, window/horizon experiment.
  Can run on scripted captures for pipeline verification.
- **4B (40.4–40.5):** training/evaluation/export. Blocked until usable participant traces exist and
  Decision 6 is approved.

## Authoritative cross-repository sources

- [Framework next-milestone plan](../../../../edge-aui-framework/docs/plan/3/next-milestone-implementation-plan.md)
- [Task-diversification UI assessment](../../../../edge-aui-framework/docs/assessments/task-diversification-ui-assessment.md)
- [ADR-025](../../../../edge-aui-framework/docs/decisions/ADR-025-task-harness-and-state-verification-boundary.md)
- [ADR-026](../../../../edge-aui-framework/docs/decisions/ADR-026-additive-semantic-outcome-taxonomy.md)
- [ADR-027](../../../../edge-aui-framework/docs/decisions/ADR-027-event-canonicalization-and-outcome-derivation-ownership.md)
- [ADR-028](../../../../edge-aui-framework/docs/decisions/ADR-028-normalized-gate-prediction-contract.md)
- [ADR-029](../../../../edge-aui-framework/docs/decisions/ADR-029-trace-schema-1-4-task-versioning.md)

## Current model baseline

`models/bundles/v1.0.0/bundle.json` is the previous five-class intervention-head artifact. Its
recorded test Macro-F1 (`0.5524`) and accuracy (`0.767`) are historical and are **not** semantic
outcome performance. Preserve it; do not overwrite it.
