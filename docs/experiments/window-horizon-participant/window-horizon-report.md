# Window/Horizon Experiment — Semantic Target (Task 40.3)

**Evidence provenance:** `participant` (35 sessions)

> **Evidence scope:** Participant captures (35 sessions). These are participant-derived engineering evidence, not a completed study: participant identifiers are not approved (ADR-023), the splits are session-bounded (ADR-031), and the configuration selection is provisional until more participant traces are collected. No semantic head was trained, so no semantic-model performance is claimed.

Target vocabulary: `testbed-semantic-target/1.0.0` (14 IDs).
Splits: session-bounded (ADR-031). Class weights: training partition only.

## Pre-declared selection criteria

1. maximise training classes with >= 5 examples
2. minimise the largest single-class share in training
3. minimise median lead time (earlier warning)
4. declared A, B, C, D order

## Configurations

| ID | History | Lookahead (ms) | Examples | Train coverage | Max class share | Lead time median (ms) | Majority macro-F1 | False early | Incomplete | Recorded-only |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 8x500 ms | 500–1500 | 3743 | 12 | 81.3% | 1026 | 0.0834 | 0 | 193 | 0 |
| B | 8x500 ms | 1000–2000 | 3674 | 12 | 81.0% | 1506 | 0.0836 | 0 | 262 | 0 |
| C | 4x500 ms | 500–1500 | 3883 | 12 | 82.0% | 1026 | 0.0837 | 0 | 193 | 0 |
| D | 12x500 ms | 500–1500 | 3603 | 12 | 80.9% | 1026 | 0.0832 | 0 | 193 | 0 |

**Selected configuration: D** (12x500 ms, [500, 1500] ms).

Configuration D was selected by the pre-declared criteria: 12 target classes have at least 5 training examples (highest), the largest training class holds 80.9% of examples, and the median lead time is 1026.2 ms. This selection is provisional: it is participant-derived engineering evidence with a small session count and must be re-confirmed after further collection before the semantic head training run.

> **Provisional:** this participant-derived selection must be re-confirmed after further participant collection before Phase 4B fixes the head configuration.

## Per-outcome frequency (training partition, selected configuration)

| Outcome | Count |
| --- | --- |
| `navigate_overview` | 0 |
| `navigate_analytics` | 23 |
| `navigate_reports` | 12 |
| `select_report` | 24 |
| `open_filter_section` | 151 |
| `close_filter_section` | 9 |
| `select_region` | 61 |
| `select_date_range` | 67 |
| `select_category` | 47 |
| `apply_filter` | 60 |
| `reset_filter` | 7 |
| `export_report` | 15 |
| `backtrack` | 0 |
| `NO_OUTCOME` | 2017 |

## Task-level coverage (selected configuration)

| Partition | Tasks present (examples) | Tasks absent |
| --- | --- | --- |
| train | TD1=443, TD2=848, TD3=857, TD4=345 | none |
| val | TD1=140, TD4=177 | TD2, TD3 |
| test | TD1=259, TD2=51, TD3=406, TD4=77 | none |

## Caveats

- These are participant captures, not scripted engineering runs. They were
  retrofitted locally from an exported set that had been mislabelled; the
  retrofit manifest records the authority and the original declarations.
- Participant identifiers are not approved (ADR-023) and splits are
  session-bounded (ADR-031), so no participant-held-out claim is made.
- The example counts are small, so per-class support is sparse. The selection
  is provisional and must be re-confirmed after further collection.
- Macro-F1 is the declared majority-class baseline, not a trained model's score.
