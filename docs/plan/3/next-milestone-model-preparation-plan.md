# Next-Milestone Model-Preparation Plan — Semantic Task-Outcome Target Head

- **Date:** 2026-11-16
- **Status:** Planning complete — no training, dataset, configuration or model artifact was changed
- **Model-preparation baseline:** commit `cbab411`, branch `explore/pipeline/revision/1`
- **Framework baseline:** `edge-aui-framework` commit `25033a8`, branch `feat/refinement`
- **Authoritative framework plan:** [`next-milestone-implementation-plan.md`](../../../../edge-aui-framework/docs/plan/3/next-milestone-implementation-plan.md)
- **Authoritative task-diversification assessment:** [`task-diversification-ui-assessment.md`](../../../../edge-aui-framework/docs/assessments/task-diversification-ui-assessment.md)
- **Plan version:** `1.0.0`

This is the model-preparation half of the framework milestone plan. It covers Phase 4 only:
trace ingestion, semantic target generation, dataset/splits, the window/horizon experiment,
semantic target-head training/evaluation and ONNX/bundle handoff. It does not cover framework UI,
Fast Gate or deployment work except where those constrain the data contract.

**Evidence labels:** **Verified** (current source/tests), **Inferred** (needs confirmation),
**Proposed** (planning decision), **Open decision** (researcher/supervisor).

No model metric, dataset result or deployment measurement is asserted here unless it is already
recorded in the repository and labelled historical.

---

## 1. Executive Summary

### 1.1 Objective

Prepare the model-preparation pipeline to ingest the framework's semantic task-outcome traces,
build a leak-free semantic target dataset, evaluate history-window/prediction-horizon choices,
train and evaluate a target head on participant-held-out data, and export a versioned ONNX/bundle
artifact that the browser runtime can consume.

The target head must predict **task-oriented semantic outcomes**, not the current five scripted
intervention classes and not merely the most recent generic click.

### 1.2 Approved and inherited decisions

| Decision | Direction | Model-preparation consequence |
| --- | --- | --- |
| New task IDs | `TD1`–`TD4` | Dataset/task metadata use TD1–TD4 and `taskDefinitionVersion = TD-v2` |
| State-verified completion | Testbed-owned | Completion labels come from task state; they are evaluation metadata, not model targets |
| Timeout/failure | `task_fail`/`Failed` | Ingestion and dataset must accept/record timeout/failure; not a prediction target |
| Customer Segment | Removed | No segment feature/label; do not add one |
| TD5 hover task | Excluded | No `inspect_kpi`/`inspect_help` target work in this milestone |
| Semantic taxonomy | Additive during collection | Keep generic seven-class labels; add semantic task outcomes; retrain only after participant traces |
| Reports catalogue | Approved | Report-selection outcomes appear only if the testbed supplies them; model does not know report IDs |

### 1.3 Phase 4 split

| Phase | Tasks | Can run when | Result |
| --- | --- | --- | --- |
| **4A — Preparation** | 40.1 ingestion 1.4; 40.2 semantic target/dataset; 40.3 window/horizon/splits/class balance | After framework trace schema 1.4 and scripted exports | Runnable semantic dataset + experiment plan; engineering evidence only |
| **4B — Retraining** | 40.4 training/evaluation; 40.5 ONNX/bundle/handoff | After usable participant traces and Decision 6 approval | Evaluated semantic target head + browser artifact |

**Do not start 40.4 on scripted data.** Scripted captures may smoke-test the pipeline but cannot
support participant or usability claims. The previous five-class five-category artifact remains
frozen and reproducible.

### 1.4 Main deliverables

- 1.4.0 trace-ingestion support with semantic outcomes/task version and legacy compatibility.
- Semantic target generator matching the TypeScript semantic rules on shared fixtures.
- Versioned semantic dataset (`v2.0.0-semantic` or approved equivalent) with provenance,
  participant/session-bounded splits, class weights and class distribution.
- Window/horizon experiment report with a small pre-declared configuration set.
- Trained/evaluated semantic target head with per-class/macro-F1/confusion/majority/lead-time
  results.
- ONNX INT8 export + bundle v2 with vocabulary, dataset, code, seed and metrics.
- Framework handoff contract for vocabulary/output dimension/model version.
- Updated `docs/data_schemas.md` and experiment documentation.

### 1.5 Open decisions

1. **Decision 6:** exact semantic field names, vocabulary version string and migration point.
   Recommendation: `taskOutcome` + `taskOutcomeVocabularyVersion`, 14 IDs, additive now and retrain
   in 4B. The framework and model-preparation must agree before 40.1/40.2.
2. Exact semantic vocabulary (14 IDs) and whether `NO_OUTCOME` is the fallback class.
3. Whether the generic seven-class head remains an auxiliary/diagnostic path after semantic
   validation.
4. Whether the old five-class intervention head remains a production comparison source or only an
   ablation artifact.
5. Participant-held-out split policy (participant vs session vs task) once participant IDs exist.
6. Window/horizon selection criteria are pre-declared in 40.3; final choice is an experiment result,
   not an assumption.

---

## 2. Current-State Assessment

### 2.1 Implemented

| Area | Current implementation | State |
| --- | --- | --- |
| Trace ingestion | `src/trace_ingestion.py` accepts 1.3.0/1.2.0/1.1.0 with version-keyed allow-lists and loud rejection | Verified |
| Generic target generation | `src/target_generation.py` derives seven generic outcomes over `[+500, +1500] ms` | Verified |
| Target dataset | `src/target_dataset.py` builds `(sequence, context, intervention)` examples from traces | Verified |
| Splits/class weights | Session-bounded splits, majority baselines, training-partition-only weights | Verified |
| Training | Foundation GRU + `TargetInterventionHead` (5 classes), freeze/partial/full ablation | Verified |
| Evaluation | Per-class/macro-F1/confusion/ranking diagnostics and majority baseline | Verified |
| Export | ONNX INT8 intervention head, graph contract/parity/size/latency checks | Verified |
| Repro bundle | `models/bundles/v1.0.0/bundle.json` with hashes/contract/metrics | Verified |
| Context parity | Python `R^6` encoder and cross-repo fixture test | Verified |

**Historical metric note:** `bundle/v1.0.0` records test Macro-F1 `0.5524`, accuracy `0.767` for the
five-class *intervention* head on dataset `v1.0.0`. These are not semantic-outcome performance and
must not be reused as such.

### 2.2 Gaps this plan closes

| Gap | Closed by |
| --- | --- |
| Ingestion rejects semantic/task-version fields | 40.1 |
| No semantic target generator or semantic dataset | 40.2 |
| No semantic class distribution/splits/window-horizon evaluation | 40.3 |
| Target head has only five intervention classes | 40.4 |
| No participant-held-out semantic evaluation | 40.4 |
| Export contract is five-class intervention logits | 40.5 |
| Scripted harness still drives T1–T3 with old controls | Framework Workstream 30 (30.1); model side consumes its captures |
| Docs describe the five-class head/old task IDs | Update with 40.1–40.5 |

### 2.3 Constraints

- Preserve `foundational_gru.pth`; do not overwrite the previous head/bundle.
- `src/config.yaml` is the canonical source of truth; never hand-edit generated framework config.
- No future data in inputs; no task-lifecycle labels as prediction targets.
- No zero-imputation of missing modalities.
- No participant/usability claim from scripted data.
- No arbitrary participant count or statistical-adequacy claim.
- Report `NOT MEASURED`/`Blocked` rather than inventing metrics.

---

## 3. Architecture and Responsibility Boundary

```text
edge-aui-framework runtime
  trace 1.4.0: behaviour, MicroTensor, macro, semantic outcomes, task events,
               normalized predictions, provenance
        │
        ▼
model-preparation
  trace ingestion → canonical events
        │
        ├── generic outcome derivation (retained)
        └── semantic target generation (new; mirrors TypeScript rules)
        │
        ▼
  semantic dataset (sequence, context, semantic target, provenance)
        │
        ▼
  participant/session-bounded splits + class weights
        │
        ▼
  window/horizon experiment → selected config
        │
        ▼
  semantic target head training/evaluation
        │
        ▼
  ONNX INT8 + bundle v2 (vocabulary, dataset, code, seed, metrics)
        │
        ▼
  framework runtime loads manifest/config; maps logits by vocabulary
```

**Ownership:**

| Responsibility | Owner |
| --- | --- |
| Semantic outcome vocabulary/taxonomy source | `src/config.yaml` in model-preparation, approved by researcher |
| Trace schema author | edge-aui-framework runtime |
| Trace acceptance/version allow-list | model-preparation ingestion |
| Semantic derivation rules | Framework generic canonicalizer + testbed app rules; Python target generator mirrors them |
| Dataset/splits/class weights | model-preparation |
| Model architecture/training/evaluation/export | model-preparation |
| Runtime model loading/policy | edge-aui-framework |

---

## 4. Semantic Taxonomy and Migration

### 4.1 Proposed vocabulary

```text
NO_OUTCOME
navigate_overview
navigate_analytics
navigate_reports
select_report
open_filter_section
close_filter_section
select_region
select_date_range
select_category
apply_filter
reset_filter
export_report
backtrack
```

`task_complete`, `task_abandon` and `task_fail` are task-lifecycle/evaluation outcomes, **not**
ordinary model prediction targets.

### 4.2 Additive migration

| Phase | Runtime trace | Dataset/training |
| --- | --- | --- |
| Collection (framework Phases 3–4A) | Keep generic `outcome`; add `taskOutcome` + vocabulary version | Build semantic dataset alongside generic fields; train/evaluate nothing yet |
| Before 4B | Freeze vocabulary/version | Add semantic taxonomy + head dimension to `src/config.yaml`; create semantic dataset version |
| 4B | Runtime still records both | Train **new** semantic head; evaluate; export ONNX/bundle v2 |
| After validation | Framework may consume semantic predictions | Old seven-class and five-class artifacts remain reproducible |

**Exact migration point:** after participant traces exist and before the semantic training run. Do
not relabel the current head; do not claim the current seven-class model predicts semantic outcomes.

### 4.3 Configuration

Proposed additions to `src/config.yaml` (names pending Decision 6):

```yaml
semantic_target_generation:
  vocabulary_version: "semantic-v1"
  outcome_taxonomy:
    0: NO_OUTCOME
    1: navigate_overview
    # ... 14 entries
  lookahead_horizon_ms: [500, 1500]

training:
  semantic_target_classes: 14
  # foundation_classes remains 7
```

`outcome_taxonomy` (generic seven-class) and `target_intervention_vocabulary` remain for the
legacy path. `npm run sync-config` in the framework propagates shared values; do not hand-edit
`edge-aui-framework/src/config/pipelineConfig.json`.

---

## 5. Trace and Dataset Contracts

### 5.1 Trace 1.4.0 ingestion (40.1)

Accept:
- `OutcomeEvent.taskOutcome`, vocabulary version, derivation source;
- `TaskEvent.taskDefinitionVersion`, `task_fail`, timeout/replaced/condition/interrupted reasons;
- `metadata` task/semantic vocabulary versions;
- normalized prediction fields if supplied by the framework;
- legacy 1.3.0/1.2.0/1.1.0 with their documented limitations.

Reject unknown versions and unmapped keys loudly. Keep the generic `outcome` field.

### 5.2 Semantic dataset (40.2)

Versioned schema additions (names pending Decision 6):
- `target_task_outcome` (string) + `target_task_outcome_id` (int);
- `task_outcome_vocabulary_version`;
- `task_definition_version`;
- `label_provenance` (`trace_semantic_outcome` | `re_derived` | `legacy_generic`);
- retained `target_outcome`/`target_intervention`/`label_policy_version`/`is_scripted_policy` for
  legacy and ablation.

Every row keeps the existing provenance fields:
`session_id`, `experiment_id`, `condition_id`, `task_id`, `anchor_window_id`,
`source_event_ids`, `preprocessing_version`, `feature_schema_version`,
`target_generation_version`.

### 5.3 Label rules

Mirror the framework semantic layer on shared fixtures:
- committed `change` only when the value changed;
- one checkbox interaction → one selection;
- one form submit → one `apply_filter`, suppress duplicate click;
- `export_report` is not `FORM_SUBMIT`;
- `tooltip-region` is not `select_region`;
- report open is `select_report`, not `navigate_analytics`;
- row hover is not selection;
- backtrack is a route return seeded from the starting page, not blur/accordion/filter edits;
- system/programmatic actions are not user outcomes;
- label only the future horizon; no future state/inputs.

---

## 6. Phase 4A — Preparation Tasks

### 40.1 Ingest schema 1.4 and semantic outcomes

- Extend allow-lists and canonical output; add semantic/task/version fields; legacy tests.
- Verify with a real 1.4.0 scripted export, not only a hand-built fixture.
- Update `docs/data_schemas.md`.

### 40.2 Semantic target generation and dataset

- Add `semantic_target_generation.py` and dataset v2 support.
- Cross-check Python rules against TypeScript fixtures from framework 20.4.
- Keep generic labels/columns; add semantic labels/provenance.
- Build from scripted captures for pipeline verification only.

### 40.3 Window/horizon, splits and class balance

- Evaluate four small configurations:
  `8×500/250/[500,1500]`, `8×500/250/[1000,2000]`, `4×500/250/[500,1500]`,
  `12×500/250/[500,1500]`.
- Report per-outcome frequency, lead time, P/R/F1, macro-F1, imbalance, false early predictions,
  incomplete windows.
- Select using pre-declared criteria; do not select on one aggregate.
- Participant/session-bounded splits; training-only class weights; absent-class reporting.
- Freeze semantic dataset version/manifest.

**Phase 4A checkpoint:** semantic dataset, class distribution, splits and window/horizon report are
reproducible from a clean checkout; no training claim is made.

---

## 7. Phase 4B — Retraining and Export

### 40.4 Semantic target-head training and evaluation

- New head output dimension = approved vocabulary size (14 proposed); preserve backbone.
- Keep freeze/partial/full ablation; report foundation retention.
- Report per-class P/R/F1, support, macro-F1, weighted-F1, confusion matrix, majority baseline,
  participant-held-out/task-level performance, lead time, calibration where support permits.
- No future leakage; no lifecycle labels as targets.
- Old five-class head/bundle frozen.

**Blocked until usable participant traces exist.** Scripted smoke training may verify the code path
but cannot satisfy evaluation acceptance.

### 40.5 ONNX export, bundle v2 and handoff

- Inputs `(batch,T,18)` + `(batch,6)`; output semantic logits `(batch,14)`.
- INT8 quantization; parity; measured size/latency.
- Bundle v2: vocabulary/version, dataset version/hash, code commit, seed, environment, metrics,
  input/output contract, artifact hashes.
- New artifact name (for example `task_outcome_head_v2.onnx`); do not overwrite old artifacts.
- Framework reads vocabulary/order from manifest/config.

**Phase 4B checkpoint:** semantic head loads in the browser and produces the configured dimension;
held-out evaluation and bundle are documented; old artifacts remain reproducible.

---

## 8. Evaluation and Export Requirements

### 8.1 Minimum evaluation

- Per-class precision, recall, F1 and support.
- Macro-F1, weighted-F1, confusion matrix.
- Majority baseline and class distribution.
- Participant-held-out performance (or session-held-out until participant data exists).
- Task-level performance across TD1–TD4.
- Prediction lead time.
- Calibration/confidence reliability where support allows.
- Model size and browser inference latency (actual measurement; `NOT MEASURED` otherwise).

### 8.2 Separation of claims

- Model prediction performance is not user-experience evidence.
- A successful prediction does not prove an intervention helps.
- Scripted-data results are pipeline/engineering evidence only.
- Do not claim statistical adequacy or prescribe participant counts.

---

## 9. Testing and Acceptance Matrix

| ID | Requirement | Verification | Acceptance |
| --- | --- | --- | --- |
| M-01 | 1.4.0 ingestion | ingestion tests + real export | Semantic/task fields carried; 1.3 legacy readable; unknown rejected |
| M-02 | Semantic label rules | shared TS/Python fixtures | Committed change/submit/report/backtrack correct; hover/substring/system excluded |
| M-03 | Dataset schema/provenance | schema/manifest tests | All required fields; no silent column reuse |
| M-04 | Leakage prevention | split/sequence tests | No future data; session/participant/task disjointness |
| M-05 | Class balance | split report | Per-partition support; training-only weights; absent classes declared |
| M-06 | Window/horizon experiment | experiment report | Pre-declared configs/criteria; per-outcome metrics/lead time |
| M-07 | Semantic head architecture | architecture tests | Output dimension = vocabulary; context mandatory |
| M-08 | Training/evaluation | training tests + experiment report | Per-class/macro-F1/confusion/majority/held-out/lead time |
| M-09 | No leakage in training | dataset/input tests | Inputs only at/before prediction time; no lifecycle targets |
| M-10 | ONNX contract/parity | export tests | Inputs/output dimension/vocabulary; parity within tolerance |
| M-11 | Bundle reproducibility | bundle tests | Hashes/versions/metrics; old bundle untouched |
| M-12 | Framework handoff | framework ONNX/slow-gate tests | Manifest vocabulary/config loads; no hardcoded order |
| M-13 | Claim boundary | experiment review | Scripted vs participant evidence explicitly separated |

---

## 10. Dependencies and Parallelisation

| Work | Depends on | Notes |
| --- | --- | --- |
| 40.1 ingestion | framework 20.5 trace schema; 30.2 real export | Can start before participant collection |
| 40.2 semantic dataset | 40.1; framework 20.4 semantic fixtures | Pipeline can run on scripted captures |
| 40.3 window/horizon/splits | 40.2 | No participant-held-out selection |
| 40.4 training | 40.3 + participant traces + Decision 6 | Hard gate; do not start early |
| 40.5 export/handoff | 40.4 + framework 20.3 | New artifact; old remains |
| Framework policy/interventions | 40.4/40.5 + 50.3 | Outside this plan |

**Safe parallel work:** ingestion/schema, semantic target fixtures, dataset tooling, documentation.
**Sequential:** schema → ingestion; dataset → window/horizon → training; training → export → runtime
adoption.

---

## 11. Risks and Open Decisions

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Semantic classes too sparse | High | Coverage report; class weights; per-class support; no overclaim |
| Retraining before participant data | High | Hard gate on 40.4; scripted runs labelled engineering |
| TS/Python label drift | High | Shared fixtures + parity test |
| Future leakage | High | Prediction-time inputs; participant/session/task splits |
| Old artifacts overwritten | High | New names/versions; freeze v1.0.0 |
| Window/horizon over-search | Medium | Four pre-declared configs and selection criteria |
| Decision 6 name/vocabulary drift | High | Freeze before 40.1/40.2; version every field |
| Claim inflation | High | Evidence labels and separate model/UX evaluation |

### Open decisions

1. Decision 6 field names/version/migration.
2. Exact 14-ID vocabulary and fallback semantics.
3. Participant-held-out split policy.
4. Whether generic head remains diagnostic/auxiliary.
5. Whether old intervention head remains production comparison.
6. Window/horizon final choice (experiment result).
7. Participant study protocol/counterbalancing (deferred).

---

## 12. Definition of Done

- [ ] 1.4.0 traces ingest with semantic outcomes/task version; 1.3 remains readable.
- [ ] Semantic target generation matches the TypeScript rules on shared fixtures.
- [ ] Semantic dataset v2 has provenance, participant/session-bounded splits and training-only
      class weights.
- [ ] Window/horizon experiment is small, pre-declared and reports coverage/lead time.
- [ ] Semantic head is trained/evaluated on participant-held-out data with per-class/macro-F1/
      confusion/majority/lead-time results.
- [ ] No future data or lifecycle labels leak into inputs/targets.
- [ ] ONNX/bundle v2 records vocabulary, dataset, code, seed and metrics; browser loads it.
- [ ] Old foundation/head/bundle artifacts remain reproducible.
- [ ] Docs/config/schema index updated.
- [ ] Claims distinguish scripted pipeline evidence from participant evidence.

---

## Appendix A — Commands

```bash
cd model-preparation
python3 -m pytest tests -q
python3 -m src.target_dataset --help
python3 -m src.training --help
python3 -m src.export --help
python3 -m src.reproduce --help
```

Task-specific focused tests are in [workstream/40-model-preparation/work.md](workstream/40-model-preparation/work.md).

## Appendix B — Source references

| Area | Files |
| --- | --- |
| Ingestion | `src/trace_ingestion.py`, `src/data.py`, `tests/test_trace_ingestion.py` |
| Target generation | `src/target_generation.py`, new `src/semantic_target_generation.py` |
| Dataset/splits | `src/target_dataset.py`, `src/data_manager.py`, tests |
| Training/eval | `src/training.py`, `src/config.py`, `src/config.yaml`, tests |
| Export/bundle | `src/export.py`, `src/reproduce.py`, `models/`, tests |
| Schemas/docs | `docs/data_schemas.md`, experiment reports |
| Framework decisions | `../../../../edge-aui-framework/docs/decisions/ADR-025…ADR-029` |
| Framework plan | `../../../../edge-aui-framework/docs/plan/3/next-milestone-implementation-plan.md` |
