# Data Schemas and Wire Contracts Specification

Authoritative index and specifications for data formats, feature representations, and UI context contracts used in the `model-preparation` pipeline.

---

## 1. MicroTensor Feature Schema (18-D)

- **Owning File:** `src/preprocessing.py`, `src/microtensor_store.py`
- **Configuration Source:** `src/config.yaml` (`preprocessing`)
- **Dimension:** 18 continuous float values (`np.float32`)
- **Representation:** $\tilde{X} = [X \odot M, M]$ where $X \in \mathbb{R}^9$ contains normalized kinematic metrics and $M \in \{0, 1\}^9$ is the binary modality mask vector.

| Index | Feature Key | Unit | Normalization Scale | Description |
|---|---|---|---|---|
| `0` | `meanVelocity` | px/ms | `/ 10.0` | Mean Euclidean cursor velocity across 500 ms window |
| `1` | `maxVelocity` | px/ms | `/ 10.0` | Maximum instantaneous Euclidean velocity |
| `2` | `meanAcceleration` | px/ms² | `/ 1.0` | Mean cursor acceleration magnitude |
| `3` | `hesitationCount` | count | `/ 25.0` | Directional turns with angle deviation $\theta > 45^\circ$ |
| `4` | `totalTrajectoryLength` | px | `/ 2000.0` | Cumulative Euclidean distance traversed |
| `5` | `dwellTimeMs` | ms | `/ 500.0` | Time hovering over interactive DOM elements |
| `6` | `trajectoryEntropy` | bits | `/ 5.0` | Shannon entropy of cursor trajectory segments |
| `7` | `scrollDepthPercentage`| % [0, 1] | `/ 1.0` | Vertical scroll offset relative to max scroll |
| `8` | `scrollVelocity` | px/ms | `/ 5.0` | Rate of vertical viewport scroll change |
| `9–17`| `mask[0..8]` | binary | 1.0 (valid) / 0.0 (masked) | Modality masks corresponding to features 0–8 |

Missing telemetry modalities (e.g. scroll on fixed viewports) strictly clear the corresponding modality mask bit ($M_i = 0.0$); zero-imputation of continuous values without masking is forbidden.

---

## 2. UIContext $\mathbb{R}^6$ Vector Encoding

- **Owning Files:** `src/context_encoding.py` (Python), `edge-aui-framework/src/types/contextVector.ts` (TypeScript)
- **Dimension:** 6 continuous float values bounded in $[0.0, 1.0]$ (`np.float32`)
- **Parity Tolerance:** Verified equal within $1 \times 10^{-6}$ by `tests/test_context_encoding.py`

| Index | Component Name | Encoding Derivation | Normalization Range / Scale | Description |
|---|---|---|---|---|
| `0` | `route` | Index in `ROUTE_VOCABULARY` | `/ 5.0` | Normalized route category (`['Overview', 'Analytics', 'Reports', 'Customers', 'Settings']`). 0.0 if unknown. |
| `1` | `primaryActionAvailable` | Boolean flag | `1.0` if True, `0.0` if False | Presence of active primary action button |
| `2` | `helpAvailable` | Boolean flag | `1.0` if True, `0.0` if False | Contextual help or tooltip availability |
| `3` | `expandable` | Boolean flag | `1.0` if True, `0.0` if False | Presence of expandable accordion / filter surface |
| `4` | `taskProgress` | $(stepNumber - 1) / totalSteps$ | Bounded in $[0.0, 1.0]$ | Task progress based on step id token and task model (`TASK_STEP_COUNTS: T1=4, T2=2, T3=4`). 0.0 if idle. |
| `5` | `actionAvailability` | $uniqueActions / 7.0$ | Bounded in $[0.0, 1.0]$ | Ratio of currently available DOM actions normalized by `ACTION_VOCABULARY` size. |

---

## 3. Intervention Vocabulary and Target Head Output Contract

- **Owning Files:** `src/intervention_label_policy.py`, `src/training.py` (`TargetInterventionHead`)
- **Dimension:** 5 discrete logits
- **Vocabulary:**
  `0: simplify_options`
  `1: highlight_primary_action`
  `2: offer_assistance`
  `3: expand_tooltip`
  `4: no_op`
- **Methodological Claim:** Labels generated via `intervention_label_policy.py` are scripted outputs from a deterministic rule policy conditioned on observable outcome events and UIContext, and are explicitly bounded under ADR-013.

### 3.1 Semantic Target Vocabulary (Task 40.2, ADR-030)

- **Owning Files:** `src/config.yaml` (`semantic_targets`), `src/semantic_targets.py`
- **Target vocabulary version:** `testbed-semantic-target/1.0.0`
- **Dimension:** 14 discrete logits, ordered:
  `0 navigate_overview`, `1 navigate_analytics`, `2 navigate_reports`, `3 select_report`,
  `4 open_filter_section`, `5 close_filter_section`, `6 select_region`, `7 select_date_range`,
  `8 select_category`, `9 apply_filter`, `10 reset_filter`, `11 export_report`, `12 backtrack`,
  `13 NO_OUTCOME`.
- **Recorded vs target:** the runtime records 15 semantic IDs. The extra ID, `inspect_help`, is
  evidence produced by the tooltip exclusion rule (it keeps `tooltip-region` from becoming
  `select_region`) and is **not** a target. Dataset assembly excludes those windows and counts
  them (`excluded_recorded_only`); it never relabels them `NO_OUTCOME`.
- **Label sources:** each example records `target_label_source` as one of
  `trace_semantic_outcome` (the 1.4.0 runtime derivation, preferred), `trace_no_outcome` (the trace
  declares the semantic contract but the record carries no ID), `lookahead_no_outcome` (no semantic
  act inside the horizon), or `re_derived` / `re_derived_no_match` (legacy capture re-derived with
  the TypeScript-mirrored rules).
- **Causality:** labels come only from records whose source event lies inside
  `(anchor_end + 500 ms, anchor_end + 1500 ms]` by default. An anchor whose horizon closes past the
  session end is excluded as incomplete rather than labelled `NO_OUTCOME`.
- **How it is set:** change `semantic_targets.target_vocabulary` in `src/config.yaml`; the Python
  constants are loaded from that section. Do not hardcode a different class count in the head.

---

## 4. Provenance-Versioned Dataset Manifest Schema (Task 11.2)

- **Owning File:** `src/data_manager.py` (`create_dataset_manifest`, `validate_dataset_manifest`)
- **CLI Commands:**
  - Build: `python3 -m src.target_dataset --split --build-manifest --version v1.0.0`
  - Validate: `python3 -m src.data --validate-manifest .data/processed/v1.0.0/manifest.json`
- **Immutability Contract:** Once generated for Phase D training, the manifest and dataset version are **frozen**. Any modification requires incrementing the version string and re-running all experiments.

### Top-Level Manifest Schema (`manifest.json`)

```json
{
  "manifest_schema_version": "1.0.0",
  "dataset_version": "v1.0.0",
  "content_hash": "cd7290a733c4",
  "created_at": "2026-09-27T16:32:05.459895+00:00",
  "source_data_type": "scripted_substitute_testbed",
  "claim_boundary": "ADR-013: Substitute interaction traces and scripted label assignments...",
  "provenance_spec": {
    "brief_reference": "Brief Section 8 & Section 17",
    "required_fields": [
      "session_id",
      "experiment_id",
      "condition_id",
      "task_id",
      "anchor_window_id",
      "source_event_ids",
      "preprocessing_version",
      "feature_schema_version",
      "target_generation_version"
    ]
  },
  "code_versions": {
    "preprocessing_version": "1.1.0",
    "feature_schema_version": "1.1.0",
    "label_policy_version": "1.0.0",
    "trace_schema_version": "1.1.0"
  },
  "split_summary": { ... },
  "class_weights": { ... },
  "total_examples": 289,
  "examples": [ ... ]
}
```

> **Trace schema version.** `code_versions.trace_schema_version` mirrors
> `EXPERIMENT_TRACE_SCHEMA_VERSION` at manifest-creation time (currently `1.3.0`). The example above
> is the historical `v1.0.0` manifest, which recorded `1.1.0`. This field is informational: the
> validator's code-version lock checks `preprocessing_version`, `feature_schema_version`, and
> `label_policy_version` only.

### The 9 Mandatory Provenance Fields (Brief §8)

Every example in `manifest.json` (and every row in `target_intervention_dataset.parquet`) retains complete provenance back to its originating trace:

| # | Field Name | Data Type | Source in Runtime Trace | Description & Validation Rule |
|---|---|---|---|---|
| 1 | `session_id` | `string` | `trace.session.sessionId` | Unique identifier of recorded interaction session. Must match an extant trace. |
| 2 | `experiment_id` | `string` | `trace.session.experimentId` | Experiment deployment run token. Non-empty string. |
| 3 | `condition_id` | `string` | `trace.session.conditionId` | UI condition: `"baseline"` or `"adaptive"`. |
| 4 | `task_id` | `string` | `trace.task.currentTaskId` | Active testbed benchmark task (`"T1"`, `"T2"`, or `"T3"`). |
| 5 | `anchor_window_id`| `int64` | `trace.microTensors[-1].windowId` | Integer ID of the terminal window in the sequence. |
| 6 | `source_event_ids`| `list[string]` | `trace.behaviourEvents[i]` | Canonical event identifiers `"{session_id}:ev_{i}"` overlapping the 500 ms window sequence span. Every ID must exist in `trace.behaviourEvents`. |
| 7 | `preprocessing_version` | `string` | `PREPROCESSING_VERSION` | Fixed pipeline code version (`"1.1.0"`). |
| 8 | `feature_schema_version` | `string` | `FEATURE_SCHEMA_VERSION` | Fixed 18-D feature schema version (`"1.1.0"`). |
| 9 | `target_generation_version` | `string` | `LABEL_POLICY_VERSION` | Version of scripted outcome-to-intervention policy (`"1.0.0"`). |

---

## 5. Target Intervention Dataset Parquet Schema

- **File Path:** `.data/processed/v1.0.0/target_intervention_dataset.parquet`
- **Schema Name:** `TARGET_DATASET_SCHEMA` in `src/target_dataset.py`

| Column | Type | Nullable | Description |
|---|---|---|---|
| `session_id` | `string` | No | Source session UUID |
| `experiment_id` | `string` | No | Testbed experiment run ID |
| `condition_id` | `string` | No | Experimental condition (`baseline` / `adaptive`) |
| `task_id` | `string` | No | Task ID (`T1`, `T2`, `T3`) |
| `anchor_window_id` | `int64` | No | Terminal sequence window ID |
| `sequence` | `list<list<float32>>` | No | Shape $(T=8, D=18)$ MicroTensor matrix |
| `context_vector` | `list<float32>` | No | Shape $(6,)$ normalized $\mathbb{R}^6$ UIContext vector |
| `target_outcome` | `string` | No | Downstream behavioral outcome class |
| `target_outcome_id` | `int64` | No | Integer ID $[0..6]$ in outcome taxonomy |
| `target_intervention` | `string` | No | Scripted intervention label |
| `target_intervention_id` | `int64` | No | Integer ID $[0..4]$ in intervention vocabulary |
| `label_policy_version` | `string` | No | Policy version string (`1.0.0`) |
| `is_scripted_policy` | `bool` | No | Always `true` per ADR-013 claim boundary |
| `split` | `string` | No | Session partition: `"train"`, `"val"`, or `"test"` |
| `source_event_ids` | `list<string>` | No | List of canonical event IDs within sequence span |
| `preprocessing_version` | `string` | No | Preprocessing code version (`1.1.0`) |
| `feature_schema_version` | `string` | No | Feature schema code version (`1.1.0`) |
| `target_generation_version` | `string` | No | Target policy code version (`1.0.0`) |

### 5.1 Semantic Target Dataset v2 (`v2.0.0-semantic`, Task 40.2)

- **File Path:** `.data/processed/v2.0.0-semantic/target_semantic_dataset.parquet`
- **Builder:** `python3 -m src.target_dataset --split --build-manifest --version v2.0.0-semantic`
  (`--target` defaults to semantic when the version contains `semantic`), or
  `python3 -m src.window_horizon --write-dataset` after the 40.3 experiment.
- **Retained columns:** every column of the intervention dataset above, including
  `target_outcome`/`target_outcome_id` and `target_intervention`/`target_intervention_id`, for
  backward comparison and ablation.
- **Added columns:**

| Column | Type | Nullable | Description |
|---|---|---|---|
| `target_task_outcome` | `string` | No (semantic rows) | Future semantic outcome ID |
| `target_task_outcome_id` | `int64` | No (semantic rows) | Class ID $[0..13]$ in the target vocabulary |
| `target_vocabulary_version` | `string` | No (semantic rows) | `testbed-semantic-target/1.0.0` |
| `recorded_vocabulary_version` | `string` | Yes | Runtime vocabulary version present in the trace |
| `task_definition_version` | `string` | Yes | `TD-v2` for the approved task set |
| `provenance` | `string` | No | `scripted` or `participant`; never mixed in one dataset |
| `target_label_source` | `string` | No (semantic rows) | Derivation provenance (see 3.1) |
| `target_event_timestamp` | `float64` | Yes | Source-event time of the labelled act |
| `lead_time_ms` | `float64` | Yes | `target_event_timestamp - anchor_window_end` |

- **Exclusions are counted, not hidden:** `excluded_incomplete_lookahead`,
  `excluded_recorded_only` (`inspect_help`), `excluded_missing_context` and `label_sources` are
  recorded in the parquet metadata and the manifest's `target_semantics` block.

---

## 6. Manifest Validation Invariants

The validator `validate_dataset_manifest` in `src/data_manager.py` executes 6 strict checks and raises `ManifestValidationError` (never warns):

1. **Header Completeness:** All required metadata fields must be present and non-null.
2. **Code Version Locking:** `code_versions` must equal active constants in Python code.
3. **Session-Bounded Disjointness:** $\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} = \emptyset$, $\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset$, $\mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{test}} = \emptyset$.
4. **Leak-Free Class Weights:** `class_weights.derived_on` must strictly equal `"train_partition_only"`.
5. **Full Example Provenance:** Every example must retain non-empty values for all 9 Brief §8 provenance fields.
6. **Source Event Traceability:** Every `sourceEventId` in every example must resolve to a valid event in the corresponding source trace file (`0 <= ev_idx < len(trace.behaviourEvents)`).
7. **Semantic Target Consistency (when `target_semantics` is present):** the header declares a
   target vocabulary and version, `split_unit` is `session` (ADR-031), every example's
   `target_task_outcome` is in that vocabulary, its ID matches the declared order, and its
   `target_vocabulary_version` matches the header. A recorded-only ID such as `inspect_help` is
   rejected as a label.

---

## 7. Trace Schema Versions, Provenance, and Configuration Caveat

### Supported ExperimentTrace versions

- **Owning File:** `src/trace_ingestion.py`
- **Current version:** `1.4.0` (`EXPERIMENT_TRACE_SCHEMA_VERSION`).
- **Supported legacy versions:** `1.3.0`, `1.2.0`, `1.1.0` (`SUPPORTED_EXPERIMENT_TRACE_SCHEMA_VERSIONS`).
- **Per-version key contract:** four version-keyed allow-lists now validate the trace body:
  `TRACE_KEY_ALLOWLIST` (top level), `BEHAVIOUR_EVENT_KEY_ALLOWLIST` (`behaviourEvents[]`),
  `OUTCOME_RECORD_KEY_ALLOWLIST` (`outcomes[]`) and `TASK_EVENT_KEY_ALLOWLIST` (`taskEvents[]`).
  Every record field outside its version's set is rejected with a named error. `1.4.0` adds the
  semantic outcome keys (`taskOutcome`, its vocabulary version, source and rule id) to outcome
  records and `taskDefinitionVersion`/`reason` to task events. `1.4.0` task-event reasons are
  restricted to `reset`, `replaced`, `condition_change`, `interrupted`, `timeout`, `wrong_action`,
  `state_mismatch`, `step_exhausted`.
- **No silent skips:** `ingest_trace_directory` reports every file whose `schemaVersion` falls outside the allow-list and then raises `UnsupportedTraceSchemaError`, which carries a machine-readable `.skipped` list of `{path, schema_version, reason}` entries. Passing `strict=False` (CLI: `--allow-unsupported-trace-versions`) ingests the supported subset only, and still reports what it skipped on stderr.

### Canonical semantic and task columns (Task 40.1)

`CANONICAL_EVENT_SCHEMA` (`src/data.py`) also carries the semantic outcome of the row's window and
a causal task snapshot:

- `task_outcome`, `task_outcome_vocabulary_version`, `task_outcome_source`,
  `task_outcome_rule_id` come from the `outcomes[]` record whose `windowId` matches the row's
  `window_id`. Legacy rows leave them null; nothing is invented.
- `semantic_outcome_vocabulary_version` and `task_definition_version` come from `metadata`.
- `task_status` and `task_failure_reason` are the last task-event status/reason at or before the
  row's timestamp for that task. This preserves `task_fail` (for example `timeout`) causally: a row
  never sees a future failure, and a reason is cleared when the task returns to `In Progress`.

Prediction/provenance fields are **deliberately deferred**: `predictions[]` and `policyDecisions[]`
are allow-listed but are not threaded into the canonical Parquet yet.

### Canonical event `provenance` column

`CANONICAL_EVENT_SCHEMA` (`src/data.py`) carries a `provenance` string column populated from `trace.session.provenance` (`"scripted"` or `"participant"`), defaulting to `"scripted"` for legacy traces that predate the field. Because `provenance` is a single word, its snake_case and CamelCase spellings coincide, so one column serves both alias conventions. The column exists so scripted substitute captures (ADR-013) and participant captures can never be pooled by accident.

### Deliberately deferred

`policyDecisions` is accepted (allow-listed) but **not** yet threaded into the MicroTensor, sequence, or target-dataset Parquet schemas or the training pipeline.

### Configuration caveat (effective values are hard-coded defaults)

`src/config.yaml` is not loaded on the trace-ingestion path: `load_config` is imported by `src/microtensor_store.py` and `src/target_dataset.py` but never called, and `src/trace_ingestion.py` never reads config at all. The effective values there are the hard-coded module defaults — including the bare `ts_ms // 250` window-correlation fallback in `parse_experiment_trace`. Do not treat `src/config.yaml` as authoritative on that path until the wiring is fixed. The semantic target vocabulary *is* read from `src/config.yaml` (`semantic_targets` section).

---

## 8. Window/Horizon Experiment (Task 40.3)

- **Owning File:** `src/window_horizon.py`
- **Report:** [`docs/experiments/window-horizon/window-horizon-report.md`](experiments/window-horizon/window-horizon-report.md)
- **Configurations (pre-declared, no search):** A `8x500 ms / [500,1500]`; B `8x500 ms /
  [1000,2000]`; C `4x500 ms / [500,1500]`; D `12x500 ms / [500,1500]`. Stride is 250 ms.
- **Selection criteria (declared before running):** maximise training classes with at least
  `MIN_CLASS_SUPPORT = 5` examples; minimise the largest training class share; minimise median lead
  time; then the declared A–D order.
- **Metrics per configuration:** training per-outcome frequency, time between semantic acts,
  lead-time percentiles, per-class precision/recall/F1 and macro-F1 of the **majority-class
  baseline** (no head is trained), class imbalance, false early predictions (must be 0), incomplete
  windows and excluded recorded-only windows, plus per-partition task coverage.
- **Claim boundary:** scripted captures only. The selection is a label-feasibility result and must
  be re-confirmed on participant data before the semantic head is trained; no semantic-model
  performance is claimed.
- **Selected dataset:** `python3 -m src.window_horizon --write-dataset` writes
  `.data/processed/<version>/` and records the chosen configuration in
  `window_horizon_selection.json` and the manifest's `target_semantics`.
- **Provenance scoping:** every run detects the declared provenance of its input and refuses a
  mixed directory; `--require-provenance participant` refuses scripted evidence. The report's
  evidence scope and caveats change with the scope.
- **Scripted result:** `.data/processed/v2.0.0-semantic/` (selected C) is pipeline evidence
  only. **Participant result:** `.data/processed/v2.1.0-participant/` (selected D,
  **provisional**) is the evidence for the Phase 4B configuration. The participant export,
  local provenance retrofit and verification gate are recorded in
  [`docs/experiments/participant-evidence/README.md`](experiments/participant-evidence/README.md).
- **Participant-derived pattern candidates:** `python3 -m src.participant_patterns` writes
  [`docs/experiments/participant-patterns/`](experiments/participant-patterns/participant-pattern-candidates.md)
  for Task 50.2. It reads participant traces only and refuses scripted input.

