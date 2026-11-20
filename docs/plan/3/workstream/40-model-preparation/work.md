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
- train/evaluate a semantic target head on participant-derived data, with the declared
  session-bounded held-out split (no participant-held-out claim until ADR-023 approves an ID);
- export a versioned ONNX/bundle artifact and hand it to the framework.

## Current state (2026-11-20)

- `src/trace_ingestion.py` accepts 1.4.0 plus 1.3.0/1.2.0/1.1.0; canonical rows carry semantic
  outcomes and task-definition version.
- `src/target_dataset.py` builds the five-class intervention labels and the semantic
  `target_task_outcome_id` labels; `v2.2.0-participant` has 9,686 examples from 77 verified
  participant sessions with session-bounded splits and training-only class weights.
- `src/semantic_targets.py` pins the approved 14-ID target vocabulary; `inspect_help` is excluded
  and counted.
- `src/training.py` defines `TargetInterventionHead` (5) and `TargetSemanticHead` (14).
- `src/semantic_training.py` trains/evaluates the semantic head (E1/E2/E3, per-class/macro/
  majority/confusion/task/lead-time/calibration metrics, no-future-input test).
- `src/export.py` exports the semantic graph, quantises it and writes bundle v2.
- The foundation GRU, the five-class head and `models/bundles/v1.0.0` are preserved unchanged.
- Recommended verification note: `python3 -m pytest tests -q` passes 204 tests here; three
  `tests/test_preprocessing.py` AdSERP cases fail on the untouched environment baseline and are
  unrelated to this workstream.

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
- **4B (40.4–40.5):** executed on the 2026-11-20 participant export (92 exported, 77 verified).
  It produces participant-derived **engineering** evidence, not participant-held-out evidence.

Do not train or report semantic-head performance on scripted data as participant evidence.

## Checkpoints

| After | Checkpoint | Result |
| --- | --- | --- |
| 40.3 | Semantic dataset, class distribution, splits and window/horizon report reproducible | **Passed (provisional)** — `v2.2.0-participant`, 77 sessions; configuration D re-selected |
| 40.5 | Semantic head trained/evaluated/exported; bundle v2 loads; old artifacts untouched | **Passed** — E2 head, ONNX/INT8, `bundle/v2.0.0`, framework manifest load; legacy hashes unchanged |

## Exit criteria

- [x] 1.4.0 traces ingest with semantic outcomes/task version; legacy versions still readable.
- [x] Semantic target generation is deterministic and matches TypeScript fixtures.
- [x] Dataset v2 has provenance, participant/session-bounded splits and training-only class weights.
- [x] Window/horizon experiment uses a small pre-declared set and reports per-outcome coverage/lead
      time.
- [x] Semantic head is trained/evaluated with per-class/macro-F1/confusion/majority/held-out metrics.
- [x] ONNX/bundle v2 records vocabulary, dataset, code, seed and metrics; browser loads it.
- [x] Foundation checkpoint, five-class head and `bundle/v1.0.0` remain reproducible.

## Verification

```bash
cd model-preparation
python3 -m pytest tests -q
python3 -m src.target_dataset --help
python3 -m src.semantic_training --help
python3 -m src.export --help

cd ../edge-aui-framework
npx vitest run tests/gate_vocabulary.test.ts tests/onnx_contract.test.ts
```

Task-specific focused tests are listed in each task file.

## 4B result (2026-11-20)

| Item | Result |
| --- | --- |
| Participant export | 92 sessions, schema 1.4.0; 44 `scripted` bodies corrected on the local copy under recorded authority; 77 verified |
| Dataset | `v2.2.0-participant`, 9,686 examples, 77 session-bounded splits |
| Configuration | D (12×500 ms, [500,1500]) re-selected, still provisional |
| Head | `TargetSemanticHead`, 14 classes, `(sequence, context)` |
| Ablation | E1 test macro-F1 0.1105; **E2 0.1297 (selected on validation)**; E3 0.0767; majority 0.0841 |
| Export | `task_outcome_head_v2.onnx` 189,907 B; INT8 177,583 B; FP32 parity 1.9e-6; INT8 top-1 agreement 1.0 |
| Bundle v2 | vocabulary/dataset/hash/seed/metrics/contract recorded; old v1.0.0 artifacts hashed unchanged |
| Framework | v2 bundle parses; real INT8 semantic graph loads with both inputs; model version/provider reported (55 tests) |
| Presentation | [`notebooks/04_semantic_target_head_retraining.ipynb`](../../../../notebooks/04_semantic_target_head_retraining.ipynb) |

## Risks

| Risk | Mitigation |
| --- | --- |
| Retraining starts before participant data | Hard gate on 40.4; scripted runs labelled engineering |
| Semantic classes too sparse | Coverage report; class weights; per-class support; absent-class reporting |
| Future leakage | Prediction-time inputs only; session/participant/task splits before windows |
| Old artifacts overwritten | New names/versions; freeze v1.0.0 |
| TS/Python label drift | Shared fixtures and a parity test |
| Claim inflation | Separate model metrics from user-experience evaluation; use evidence labels |
