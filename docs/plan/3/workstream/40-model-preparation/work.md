# Workstream 40 — Model Preparation and Semantic Target Head

This is the model-preparation mirror of Workstream 40 in the framework milestone plan. The
authoritative cross-repository plan is
[`next-milestone-model-preparation-plan.md`](../../next-milestone-model-preparation-plan.md).

## Objective

Move the model-preparation pipeline from five scripted intervention classes and a generic
seven-class outcome target to a semantic task-outcome dataset, experiment and target head:

- ingest trace schema 1.4.0 and semantic outcomes additively;
- generate semantic future-outcome labels that mirror the framework rules;
- build a versioned semantic dataset with provenance, participant/session-bounded splits and
  training-only class weights;
- evaluate a small set of history-window/prediction-horizon configurations;
- train/evaluate a semantic target head on participant-held-out data;
- export a versioned ONNX/bundle artifact and hand it to the framework.

## Current state

- `src/trace_ingestion.py` accepts 1.3.0/1.2.0/1.1.0 only.
- `src/target_generation.py` derives the generic seven-class outcome.
- `src/target_dataset.py` labels examples with the five-class scripted intervention policy.
- `src/config.yaml` has `outcome_taxonomy` (7), `target_intervention_vocabulary` (5) and
  `training.target_classes: 5`.
- `src/training.py` defines `TargetInterventionHead` with a fixed five-class projection.
- `src/export.py` validates a five-class `intervention_logits` graph.
- The foundation GRU, session splits, class weights, provenance manifest, training/eval, ONNX
  export and `bundle/v1.0.0` are implemented and must be preserved.
- The scripted harness is being updated in framework Workstream 30; this plan consumes its
  captures.

## Task index and order

| Order | Task | Description | Depends on | Scope |
| --- | --- | --- | --- | --- |
| 1 | [40.1](tasks/40.1-ingest-schema-1.4-semantic-outcomes.md) | 1.4.0 ingestion + semantic/task fields | Framework 20.5, 30.2 | Medium |
| 2 | [40.2](tasks/40.2-semantic-target-generation-and-dataset.md) | Semantic target generation and dataset v2 | 40.1 | Medium |
| 3 | [40.3](tasks/40.3-window-horizon-splits-class-balance.md) | Window/horizon experiment, splits, class balance | 40.2 | Medium |
| 4 | [40.4](tasks/40.4-semantic-head-training-evaluation.md) | Semantic head training and evaluation | 40.3 + participant traces | Medium |
| 5 | [40.5](tasks/40.5-onnx-export-bundle-v2-handoff.md) | ONNX export, bundle v2, framework handoff | 40.4 | Medium |

## Phase split

- **4A (40.1–40.3):** can run on scripted captures; produces pipeline/engineering evidence only.
- **4B (40.4–40.5):** blocked until usable participant traces exist and Decision 6 is approved.

Do not train or report semantic-head performance on scripted data as participant evidence.

## Checkpoints

| After | Checkpoint |
| --- | --- |
| 40.3 | Semantic dataset, class distribution, splits and window/horizon report reproducible |
| 40.5 | Semantic head trained/evaluated/exported; bundle v2 loads; old artifacts untouched |

## Exit criteria

- [ ] 1.4.0 traces ingest with semantic outcomes/task version; legacy versions still readable.
- [ ] Semantic target generation is deterministic and matches TypeScript fixtures.
- [ ] Dataset v2 has provenance, participant/session-bounded splits and training-only class weights.
- [ ] Window/horizon experiment uses a small pre-declared set and reports per-outcome coverage/lead
      time.
- [ ] Semantic head is trained/evaluated with per-class/macro-F1/confusion/majority/held-out metrics.
- [ ] ONNX/bundle v2 records vocabulary, dataset, code, seed and metrics; browser loads it.
- [ ] Foundation checkpoint, five-class head and `bundle/v1.0.0` remain reproducible.

## Verification

```bash
cd model-preparation
python3 -m pytest tests -q
python3 -m src.target_dataset --help
python3 -m src.training --help
python3 -m src.export --help
```

Task-specific focused tests are listed in each task file.

## Risks

| Risk | Mitigation |
| --- | --- |
| Retraining starts before participant data | Hard gate on 40.4; scripted runs labelled engineering |
| Semantic classes too sparse | Coverage report; class weights; per-class support; absent-class reporting |
| Future leakage | Prediction-time inputs only; session/participant/task splits before windows |
| Old artifacts overwritten | New names/versions; freeze v1.0.0 |
| TS/Python label drift | Shared fixtures and a parity test |
| Claim inflation | Separate model metrics from user-experience evaluation; use evidence labels |
