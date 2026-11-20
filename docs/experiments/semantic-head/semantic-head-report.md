# Semantic Target-Head Retraining — Task 40.4/40.5 Report

**Date:** 2026-11-20  
**Repository:** `model-preparation`  
**Dataset:** `v2.2.0-participant` (`7ebb11cac86d`)  
**Selected arm:** `E2` (validation macro-F1)  
**Bundle:** `models/bundles/v2.0.0/bundle.json`

> **Evidence boundary.** Participant-declared traces used as participant-derived engineering
> evidence. No participant identifier is approved (ADR-023); splits are session-bounded
> (ADR-031); the sample is small and sparse. No participant-held-out generalisation is
> claimed. Model accuracy is not evidence of usability, task performance or intervention
> benefit.

## 1. Collection, provenance and the verifier gate

| Item | Result |
| --- | --- |
| Supabase export | 92 sessions, schema 1.4.0, no truncation |
| Body declared `scripted` | 44 |
| Body declared `participant` | 48 |
| Local retrofit | 44 corrected, 48 already correct |
| Verifier passed | 77 |
| Verifier excluded | 15 |

The database and the exported source were not modified. Only `session.provenance` and
`metadata.provenance` were rewritten on the local analysis copy, under the researcher
authority recorded in `retrofit-manifest.json`; `participantId` remains null. The framework
verifier decided the analysis input: a failed trace is retained as a record but is not
evidence. Eleven exclusions are `prediction_window_attribution`, three are
`structural_validity` and one is `task_lifecycle_consistency` — capture defects, not
provenance relabelling.

## 2. Dataset

| Field | Value |
| --- | --- |
| Version | `v2.2.0-participant` |
| Content hash | `7ebb11cac86d` |
| Rows | 9,686 |
| Sessions | 77 (train 54, val 12, test 11) |
| Split unit | `session` |
| Label key | `target_task_outcome` |
| Target vocabulary | 14 IDs, `testbed-semantic-target/1.0.0` |
| `inspect_help` windows excluded | 0 |
| Incomplete-lookahead windows excluded | 408 |
| Source data type | `participant_capture` |

Class weights were resolved on the training partition only. `NO_OUTCOME` is
82.6% of training rows. `backtrack` has zero support in every partition; `navigate_overview` and
`close_filter_section` have support only in training. Those classes are named as absent in
the evaluation rather than scored with a fabricated F1.

## 3. Window/horizon configuration

The four pre-declared configurations were re-run on the 77 verified sessions. The
pre-declared criteria selected **D** again (12 × 500 ms, horizon [500, 1500] ms). The
selection remains provisional (`provisional=True`): the margin to
the runner-up is inside sampling noise. It is used for this retraining run as the
re-confirmed configuration, not as a final claim.

| ID | History | Lookahead (ms) | Examples | Train classes ≥5 | Largest train share | Median lead (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| A | 8×500 ms | 500–1500 | 9,994 | 12 | 82.9% | 1025 |
| B | 8×500 ms | 1000–2000 | 9,845 | 12 | 82.7% | 1517 |
| C | 4×500 ms | 500–1500 | 10,302 | 12 | 83.3% | 1025 |
| D | 12×500 ms | 500–1500 | 9,686 | 12 | 82.6% | 1023 |

## 4. Head and training protocol

`TargetSemanticHead` outputs **14** classes (the approved target
vocabulary, ADR-030) and accepts `(sequence, context)`. The GRU backbone is loaded from
`models/foundational_gru.pth` and is never overwritten. E1/E2/E3 are strict freezing,
terminal-layer partial fine-tuning and full fine-tuning. For every arm the epoch with the
highest **validation** macro-F1 is checkpointed, then the test partition is evaluated once.

| Arm | Best epoch | Val macro-F1 | Test accuracy | Test macro-F1 | Test weighted-F1 | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | 22 | 0.1681 | 0.1053 | 0.1105 | 0.0802 | 0.2796 |
| E2 | 30 | 0.1836 | 0.1271 | 0.1297 | 0.1102 | 0.2913 |
| E3 | 10 | 0.1627 | 0.1093 | 0.0767 | 0.0987 | 0.2603 |

Training configuration for the selected arm:

| Field | Value |
| --- | --- |
| Head class count | 14 |
| Target vocabulary | `testbed-semantic-target/1.0.0` |
| Dataset | `v2.2.0-participant` / `7ebb11cac86d` |
| Split unit | `session` |
| History / horizon | 12 × 500 ms / [500.0, 1500.0] ms |
| Seed / epochs | 42 / 50 |
| Learning-rate scheme | `{"scheme": "partial_finetuning", "backbone_terminal_layer": "_l1", "gru_lr": 0.0001, "head_lr": 0.001, "terminal_params_count": 24960, "head_params_count": 5454}` |
| Class weights | training partition only (`train_partition_only`) |

## 5. Held-out evaluation (selected arm)

Test partition: **1,747 examples**, session-held-out. Macro-F1 over supported
classes, because three classes have zero test support. The majority baseline is the most
frequent training class applied to test.

| Metric | Model | Majority baseline |
| --- | --- | --- |
| Accuracy | 0.1271 | 0.8603 |
| Macro-F1 (supported) | 0.1297 | 0.0841 |
| Weighted-F1 | 0.1102 | 0.7957 |
| Macro-F1 (all 14) | 0.1019 | 0.0661 |
| Top-3 accuracy | 0.7590 | — |
| MRR | 0.4724 | — |
| ECE | 0.2913 | — |
| Brier | 0.9347 | — |

The model trades majority-class accuracy (0.86 for always predicting `NO_OUTCOME`) for
rare-class recall. Macro-F1 rises from 0.084 to 0.130; accuracy falls from 0.860 to 0.127.
This is the expected consequence of the declared training-only class weights on a severe
imbalance, and it is a weak model in absolute terms.

### Per-class results

| Outcome | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| `navigate_overview` | — | — | — | 0 (absent) |
| `navigate_analytics` | 0.072 | 1.000 | 0.135 | 7 |
| `navigate_reports` | 0.094 | 0.417 | 0.154 | 12 |
| `select_report` | 0.135 | 0.789 | 0.231 | 19 |
| `open_filter_section` | 0.125 | 0.317 | 0.179 | 60 |
| `close_filter_section` | — | — | — | 0 (absent) |
| `select_region` | 0.035 | 0.625 | 0.067 | 16 |
| `select_date_range` | 0.112 | 0.890 | 0.199 | 73 |
| `select_category` | 0.133 | 0.652 | 0.221 | 23 |
| `apply_filter` | 0.013 | 0.083 | 0.023 | 24 |
| `reset_filter` | 0.000 | 0.000 | 0.000 | 3 |
| `export_report` | 0.068 | 0.429 | 0.118 | 7 |
| `backtrack` | — | — | — | 0 (absent) |
| `NO_OUTCOME` | 0.871 | 0.054 | 0.102 | 1503 |

Absent in test: `navigate_overview`, `close_filter_section`, `backtrack`.

### Task-level results

| Task | Examples | Accuracy | Macro-F1 |
| --- | --- | --- | --- |
| TD1 | 313 | 0.093 | 0.135 |
| TD2 | 804 | 0.118 | 0.145 |
| TD3 | 373 | 0.158 | 0.149 |
| TD4 | 257 | 0.152 | 0.269 |

### Prediction lead time

Median lead time is measured from the anchor window end to the labelled future act. It is
reported only where labels carry a lead time; `NO_OUTCOME`-style labels do not.

| Outcome | Labels | Median (ms) | P25 | P75 |
| --- | --- | --- | --- | --- |
| `navigate_analytics` | 7 | 932 | 796 | 1284 |
| `navigate_reports` | 12 | 986 | 779 | 1192 |
| `select_report` | 19 | 1033 | 740 | 1240 |
| `open_filter_section` | 60 | 999 | 750 | 1249 |
| `select_region` | 16 | 1022 | 793 | 1251 |
| `select_date_range` | 73 | 958 | 706 | 1193 |
| `select_category` | 23 | 1013 | 793 | 1224 |
| `apply_filter` | 24 | 1137 | 914 | 1330 |
| `reset_filter` | 3 | 1282 | 1240 | 1365 |
| `export_report` | 7 | 1315 | 1178 | 1372 |

### Calibration

ECE is **0.291**. Confidence exceeds accuracy in every populated bin
above 0.2: the head is over-confident. Sparse classes and the weighted objective make this
expected; the reliability table is in the evaluation JSON and the notebook.

## 6. ONNX, INT8 and bundle v2

| Item | Measured value |
| --- | --- |
| FP32 artifact | `task_outcome_head_v2.onnx`, 189,907 B |
| INT8 artifact | `task_outcome_head_v2_int8.onnx`, 177,583 B |
| FP32 latency | 0.04 ms avg / 0.04 ms p95 (CPU ORT) |
| INT8 latency | 0.04 ms avg / 0.04 ms p95 (CPU ORT) |
| FP32/ONNX parity | max abs diff 1.91e-06 |
| INT8 parity | top-1 agreement 1.000, max logit diff 0.161 |
| ONNX hash | `e81a1888f467361a…` |
| INT8 hash | `0b1f3d4fa3a03c8d…` |

These are measurements of the exported graph in this environment, not claims about the
browser path. **Browser-path latency is NOT MEASURED**; the 0.04 ms figure is ONNX Runtime CPU
execution of the exported graph. The <500 KB combined payload, <20 MB memory and <50 ms
browser-inference targets are not met; see ADR-008 and ADR-016. The INT8 graph is only 6% smaller than FP32
because dynamic quantisation leaves the GRU matmuls largely untouched.

Bundle v2 records the 14-ID target vocabulary, both vocabulary versions, the dataset
version/content hash, split unit, code commit, seed, environment, input/output contract,
test metrics and SHA-256 hashes. `models/bundles/v1.0.0` and the five-class
`intervention_head*.onnx` artifacts were hashed before and after export and are unchanged.

## 7. Framework handoff

- `parseBundleManifest` detects `task_outcome_vocabulary` + `task_outcome_logits`, maps the
  graph into the foundation slot, and takes the INT8 hash as its model version.
- `OnnxSlowGate` supplies `sequence_input` and `context_input` to a two-input foundation
  session, maps the index through the manifest vocabulary and reports the semantic artifact
  version and provider.
- Committed framework copies: `public/models/task_outcome_head_v2_int8.onnx` and
  `public/models/task_outcome_head_v2.bundle.json`.
- `npx vitest run tests/gate_vocabulary.test.ts tests/onnx_contract.test.ts tests/intervention_head_contract.test.ts`:
  **55 passed**, including a real INT8 graph load that feeds both inputs and reports the
  semantic artifact version.

## 8. Claims and limitations

- **Supported:** the 92-trace participant export → provenance gate → dataset v2.2.0 →
  14-class head → held-out evaluation → ONNX/INT8 → bundle v2 → framework manifest load
  path executes end to end.
- **Supported:** on the session-held-out test partition the selected arm has higher
  macro-F1 than the majority baseline (0.130 vs 0.084) and much lower accuracy (0.127 vs
  0.860).
- **Not supported:** participant-held-out generalisation, statistical adequacy, or any
  usability/intervention-benefit claim. The model is weak in absolute terms and poorly
  calibrated; the sparse classes make the estimate unstable.
- The configuration selection remains provisional even though D was re-selected.
- The 15 verifier exclusions are capture defects; they are not analysed.
- `pytest tests -q` in this environment reports 204 passed and 3 pre-existing failures in
  `tests/test_preprocessing.py` (AdSERP fixture parsing), which also fail on the untouched
  baseline and are unrelated to this work.

## 9. Reproduction

```bash
cd model-preparation
# export -> retrofit -> verify -> dataset
python3 -m src.window_horizon --trace-dir .data/raw/participant-verified-2026-11-20 \
  --report-dir docs/experiments/window-horizon-participant-r2 \
  --require-provenance participant --write-dataset --version v2.2.0-participant
# train + evaluate all arms
python3 -m src.semantic_training --evaluate-suite --epochs 50 \
  --report docs/experiments/semantic-head/semantic-head-evaluation.json
# export + bundle
python3 -m src.export --head semantic --experiment e2 \
  --dataset-version v2.2.0-participant \
  --metrics docs/experiments/semantic-head/semantic-head-evaluation.json
```

The full presentation, with executed plots and tables, is
[`notebooks/04_semantic_target_head_retraining.ipynb`](../../notebooks/04_semantic_target_head_retraining.ipynb).
