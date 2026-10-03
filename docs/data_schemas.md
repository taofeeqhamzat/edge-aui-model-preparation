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

---

## 6. Manifest Validation Invariants

The validator `validate_dataset_manifest` in `src/data_manager.py` executes 6 strict checks and raises `ManifestValidationError` (never warns):

1. **Header Completeness:** All required metadata fields must be present and non-null.
2. **Code Version Locking:** `code_versions` must equal active constants in Python code.
3. **Session-Bounded Disjointness:** $\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} = \emptyset$, $\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset$, $\mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{test}} = \emptyset$.
4. **Leak-Free Class Weights:** `class_weights.derived_on` must strictly equal `"train_partition_only"`.
5. **Full Example Provenance:** Every example must retain non-empty values for all 9 Brief §8 provenance fields.
6. **Source Event Traceability:** Every `sourceEventId` in every example must resolve to a valid event in the corresponding source trace file (`0 <= ev_idx < len(trace.behaviourEvents)`).

---

## 7. Trace Schema Versions, Provenance, and Configuration Caveat

### Supported ExperimentTrace versions

- **Owning File:** `src/trace_ingestion.py`
- **Current version:** `1.3.0` (`EXPERIMENT_TRACE_SCHEMA_VERSION`).
- **Supported legacy versions:** `1.2.0`, `1.1.0` (`SUPPORTED_EXPERIMENT_TRACE_SCHEMA_VERSIONS`).
- **Per-version key contract:** `TRACE_KEY_ALLOWLIST` and `BEHAVIOUR_EVENT_KEY_ALLOWLIST` declare the exact permitted top-level and `behaviourEvents[]` keys. `1.3.0` additionally permits the `policyDecisions` array and the `ariaExpanded` event key. The new `metadata`/`session` fields (`clock`, `provenance`, `applicationVersion`, `modelVersion`, `executionProvider`, `policyVersion`, `mining`, `evictions`, `integrityWarnings`, `participantId`) are read but are not themselves key-restricted, matching the pre-existing behaviour.
- **No silent skips:** `ingest_trace_directory` reports every file whose `schemaVersion` falls outside the allow-list and then raises `UnsupportedTraceSchemaError`, which carries a machine-readable `.skipped` list of `{path, schema_version, reason}` entries. Passing `strict=False` (CLI: `--allow-unsupported-trace-versions`) ingests the supported subset only, and still reports what it skipped on stderr.

### Canonical event `provenance` column

`CANONICAL_EVENT_SCHEMA` (`src/data.py`) carries a `provenance` string column populated from `trace.session.provenance` (`"scripted"` or `"participant"`), defaulting to `"scripted"` for legacy traces that predate the field. Because `provenance` is a single word, its snake_case and CamelCase spellings coincide, so one column serves both alias conventions. The column exists so scripted substitute captures (ADR-013) and participant captures can never be pooled by accident.

### Deliberately deferred

`policyDecisions` is accepted (allow-listed) but **not** yet threaded into the MicroTensor, sequence, or target-dataset Parquet schemas or the training pipeline.

### Configuration caveat (effective values are hard-coded defaults)

`src/config.yaml` is not loaded on the trace-ingestion path: `load_config` is imported by `src/microtensor_store.py` and `src/target_dataset.py` but never called, and `src/trace_ingestion.py` never reads config at all. The effective values there are the hard-coded module defaults — including the bare `ts_ms // 250` window-correlation fallback in `parse_experiment_trace`. Do not treat `src/config.yaml` as authoritative until that wiring is fixed.

