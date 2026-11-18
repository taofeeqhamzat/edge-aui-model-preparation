# Window/Horizon Experiment — Semantic Target (Task 40.3)

> **Evidence scope:** scripted engineering captures only. This is a label-feasibility and majority-baseline report, not participant evidence. No semantic head was trained, so no semantic-model performance is claimed.

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
| A | 8x500 ms | 500–1500 | 50 | 2 | 29.0% | 1053 | 0.0556 | 0 | 48 | 0 |
| B | 8x500 ms | 1000–2000 | 36 | 1 | 28.6% | 1547 | 0.0750 | 0 | 62 | 0 |
| C | 4x500 ms | 500–1500 | 82 | 4 | 21.8% | 1120 | 0.0000 | 0 | 48 | 0 |
| D | 12x500 ms | 500–1500 | 23 | 0 | 33.3% | 1047 | 0.1071 | 0 | 43 | 0 |

**Selected configuration: C** (4x500 ms, [500, 1500] ms).

Configuration C was selected by the pre-declared criteria: 4 target classes have at least 5 training examples (highest), the largest training class holds 21.8% of examples, and the median lead time is 1120.3 ms.

## Per-outcome frequency (training partition, selected configuration)

| Outcome | Count |
| --- | --- |
| `navigate_overview` | 0 |
| `navigate_analytics` | 0 |
| `navigate_reports` | 0 |
| `select_report` | 7 |
| `open_filter_section` | 10 |
| `close_filter_section` | 0 |
| `select_region` | 3 |
| `select_date_range` | 3 |
| `select_category` | 12 |
| `apply_filter` | 10 |
| `reset_filter` | 2 |
| `export_report` | 4 |
| `backtrack` | 0 |
| `NO_OUTCOME` | 4 |

## Task-level coverage (selected configuration)

| Partition | Tasks present (examples) | Tasks absent |
| --- | --- | --- |
| train | TD1=8, TD2=17, TD3=19, TD4=11 | none |
| val | TD3=19 | TD1, TD2, TD4 |
| test | TD1=8 | TD2, TD3, TD4 |

## Caveats

- The captures are scripted engineering runs; they are not participant evidence.
- The example counts are small, so per-class support is sparse and the experiment
  cannot select a production configuration on its own. The selected configuration
  must be re-confirmed on participant data before the semantic head is trained.
- Macro-F1 is the declared majority-class baseline, not a trained model's score.
