# Window/Horizon Experiment — Semantic Target (Task 40.3)

**Evidence provenance:** `participant` (77 sessions)

> **Evidence scope:** Participant captures (77 sessions). These are participant-derived engineering evidence, not a completed study: participant identifiers are not approved (ADR-023), the splits are session-bounded (ADR-031), and the configuration selection is provisional until more participant traces are collected. No semantic head was trained, so no semantic-model performance is claimed.

Target vocabulary: `testbed-semantic-target/1.0.0` (14 IDs).

> **Follow-up (2026-11-20):** this report is the 40.3 label-feasibility experiment and claims no
> model performance. Task 40.4 subsequently trained the 14-class head on the selected dataset;
> its held-out metrics, limits and export are in
> [`../semantic-head/semantic-head-report.md`](../semantic-head/semantic-head-report.md).
Splits: session-bounded (ADR-031). Class weights: training partition only.

## Pre-declared selection criteria

1. maximise training classes with >= 5 examples
2. minimise the largest single-class share in training
3. minimise median lead time (earlier warning)
4. declared A, B, C, D order

## Configurations

| ID | History | Lookahead (ms) | Examples | Train coverage | Max class share | Lead time median (ms) | Majority macro-F1 | False early | Incomplete | Recorded-only |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 8x500 ms | 500–1500 | 9994 | 12 | 82.9% | 1025 | 0.0842 | 0 | 408 | 0 |
| B | 8x500 ms | 1000–2000 | 9845 | 12 | 82.7% | 1517 | 0.0841 | 0 | 557 | 0 |
| C | 4x500 ms | 500–1500 | 10302 | 12 | 83.3% | 1025 | 0.0843 | 0 | 408 | 0 |
| D | 12x500 ms | 500–1500 | 9686 | 12 | 82.6% | 1023 | 0.0841 | 0 | 408 | 0 |

**Selected configuration: D** (12x500 ms, [500, 1500] ms).

Configuration D was selected by the pre-declared criteria: 12 target classes have at least 5 training examples (highest), the largest training class holds 82.6% of examples, and the median lead time is 1023.3 ms. This selection is provisional: it is participant-derived engineering evidence with a small session count and must be re-confirmed after further collection before the semantic head training run.

> **Provisional:** this participant-derived selection must be re-confirmed after further participant collection before Phase 4B fixes the head configuration.

## Per-outcome frequency (training partition, selected configuration)

| Outcome | Count |
| --- | --- |
| `navigate_overview` | 4 |
| `navigate_analytics` | 55 |
| `navigate_reports` | 46 |
| `select_report` | 79 |
| `open_filter_section` | 305 |
| `close_filter_section` | 9 |
| `select_region` | 128 |
| `select_date_range` | 218 |
| `select_category` | 93 |
| `apply_filter` | 134 |
| `reset_filter` | 22 |
| `export_report` | 49 |
| `backtrack` | 0 |
| `NO_OUTCOME` | 5420 |

## Task-level coverage (selected configuration)

| Partition | Tasks present (examples) | Tasks absent |
| --- | --- | --- |
| train | TD1=1459, TD2=1574, TD3=2364, TD4=1165 | none |
| val | TD1=465, TD2=87, TD3=638, TD4=187 | none |
| test | TD1=313, TD2=804, TD3=373, TD4=257 | none |

## Caveats

- These are participant captures, not scripted engineering runs. They were
  retrofitted locally from an exported set that had been mislabelled; the
  retrofit manifest records the authority and the original declarations.
- Participant identifiers are not approved (ADR-023) and splits are
  session-bounded (ADR-031), so no participant-held-out claim is made.
- The example counts are small, so per-class support is sparse and the margin
  between the top configurations may be within sampling noise. The selection
  is provisional and must be re-confirmed after further collection.
