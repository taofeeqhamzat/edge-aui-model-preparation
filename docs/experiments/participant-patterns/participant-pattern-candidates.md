# Participant-derived Fast Gate pattern candidates (Task 50.2)

**Evidence provenance:** `participant` (35 sessions)

> **Evidence scope:** Participant-derived macro sequences. Candidate patterns are evidence for Task 50.2, not the gate configuration; 50.1 PrefixSpan correctness remains open and the selection is provisional until more participant traces are collected.

## Method

- Reads participant macro sequences only; scripted traces are refused.
- Collapses consecutive duplicate symbols, because one participant act emits
  several macro events (for example three `OPEN_REPORT` records in 20 ms).
- Counts contiguous 2–4 symbol n-grams; support is the number of sessions that
  contain the pattern, not the number of occurrences.
- Minimum support: 3 sessions.
- Excludes symbols that are unreachable or misleading (`BACKTRACK`, `IDLE_DWELL`, `RAPID_SCROLL`, `SELECT_SEGMENT`, `TABLE_PAGE_NEXT`, `TABLE_PAGE_PREV`, `TABLE_ROW_SELECT`, `TABLE_SORT`); a pattern that
  cannot be produced cannot be a Fast Gate pattern.

## Participants and tasks

- Sessions: 35
- Sessions with no usable macro symbols: 0
- Task coverage: {'TD1': 11, 'TD2': 8, 'TD3': 8, 'TD4': 8}

## Candidates

| Pattern | Length | Support (sessions) | Occurrences | Tasks | Risk |
| --- | --- | --- | --- | --- | --- |
| `CLOSE_FILTERS` > `OPEN_FILTERS` | 2 | 27 | 56 | TD1, TD2, TD3 | none |
| `SELECT_REGION` > `CLOSE_FILTERS` > `OPEN_FILTERS` | 3 | 21 | 22 | TD1, TD2, TD3 | may include tooltip-region misclassification until the macro rule is fixed |
| `SELECT_REGION` > `CLOSE_FILTERS` | 2 | 21 | 44 | TD1, TD2, TD3 | may include tooltip-region misclassification until the macro rule is fixed |
| `CLOSE_FILTERS` > `SELECT_REGION` | 2 | 20 | 34 | TD1, TD2, TD3 | may include tooltip-region misclassification until the macro rule is fixed |
| `OPEN_FILTERS` > `SELECT_REGION` | 2 | 19 | 29 | TD1, TD2, TD3 | may include tooltip-region misclassification until the macro rule is fixed |
| `CLOSE_FILTERS` > `SELECT_REGION` > `CLOSE_FILTERS` > `OPEN_FILTERS` | 4 | 18 | 19 | TD1, TD3 | may include tooltip-region misclassification until the macro rule is fixed |
| `SELECT_REGION` > `CLOSE_FILTERS` > `OPEN_FILTERS` > `SELECT_REGION` | 4 | 18 | 19 | TD1, TD3 | may include tooltip-region misclassification until the macro rule is fixed |
| `CLOSE_FILTERS` > `OPEN_FILTERS` > `SELECT_REGION` | 3 | 18 | 19 | TD1, TD3 | may include tooltip-region misclassification until the macro rule is fixed |
| `CLOSE_FILTERS` > `SELECT_REGION` > `CLOSE_FILTERS` | 3 | 18 | 23 | TD1, TD3 | may include tooltip-region misclassification until the macro rule is fixed |
| `SELECT_DATE` > `CLOSE_FILTERS` > `OPEN_FILTERS` | 3 | 17 | 18 | TD1, TD2, TD3 | none |
| `SELECT_DATE` > `CLOSE_FILTERS` | 2 | 17 | 22 | TD1, TD2, TD3 | none |
| `SELECT_CATEGORY` > `CLOSE_FILTERS` > `OPEN_FILTERS` | 3 | 16 | 16 | TD2, TD3 | none |
| `OPEN_FILTERS` > `SELECT_DATE` | 2 | 16 | 20 | TD2, TD3 | none |
| `SELECT_CATEGORY` > `CLOSE_FILTERS` | 2 | 16 | 25 | TD2, TD3 | none |
| `SELECT_CATEGORY` > `CLOSE_FILTERS` > `OPEN_FILTERS` > `SELECT_CATEGORY` | 4 | 15 | 15 | TD2, TD3 | none |
| `SELECT_DATE` > `CLOSE_FILTERS` > `OPEN_FILTERS` > `SELECT_DATE` | 4 | 15 | 16 | TD2, TD3 | none |
| `CLOSE_FILTERS` > `OPEN_FILTERS` > `SELECT_CATEGORY` | 3 | 15 | 15 | TD2, TD3 | none |
| `CLOSE_FILTERS` > `OPEN_FILTERS` > `SELECT_DATE` | 3 | 15 | 17 | TD2, TD3 | none |
| `OPEN_FILTERS` > `SELECT_REGION` > `CLOSE_FILTERS` | 3 | 15 | 16 | TD1, TD2, TD3 | may include tooltip-region misclassification until the macro rule is fixed |
| `APPLY_FILTER` > `RESET_FILTERS` | 2 | 15 | 25 | TD2, TD3 | none |
| `OPEN_FILTERS` > `SELECT_CATEGORY` | 2 | 15 | 19 | TD2, TD3 | none |
| `CLOSE_FILTERS` > `OPEN_FILTERS` > `SELECT_REGION` > `CLOSE_FILTERS` | 4 | 14 | 14 | TD1, TD3 | may include tooltip-region misclassification until the macro rule is fixed |
| `OPEN_FILTERS` > `CLOSE_FILTERS` | 2 | 12 | 14 | TD1, TD2, TD3 | none |
| `SELECT_REGION` > `APPLY_FILTER` | 2 | 12 | 13 | TD1, TD3 | may include tooltip-region misclassification until the macro rule is fixed |
| `EXPORT_REPORT` > `APPLY_FILTER` > `EXPORT_REPORT` | 3 | 11 | 11 | TD1 | none |
| `APPLY_FILTER` > `EXPORT_REPORT` | 2 | 11 | 18 | TD1 | none |
| `EXPORT_REPORT` > `APPLY_FILTER` | 2 | 11 | 11 | TD1 | none |
| `HOVER_KPI` > `CLOSE_FILTERS` | 2 | 11 | 12 | TD1, TD2, TD3 | none |
| `OPEN_FILTERS` > `SELECT_DATE` > `OPEN_FILTERS` | 3 | 10 | 11 | TD2, TD3 | none |
| `SELECT_REGION` > `CLOSE_FILTERS` > `SELECT_REGION` | 3 | 10 | 12 | TD1, TD2, TD3 | may include tooltip-region misclassification until the macro rule is fixed |
| `APPLY_FILTER` > `SELECT_REGION` | 2 | 10 | 11 | TD1, TD2 | may include tooltip-region misclassification until the macro rule is fixed |
| `CLOSE_FILTERS` > `APPLY_FILTER` | 2 | 10 | 12 | TD1, TD2, TD3 | none |
| `NAV_ANALYTICS` > `CLOSE_FILTERS` | 2 | 10 | 10 | TD1 | none |
| `SELECT_DATE` > `OPEN_FILTERS` | 2 | 10 | 12 | TD2, TD3 | none |
| `CLOSE_FILTERS` > `OPEN_FILTERS` > `SELECT_DATE` > `OPEN_FILTERS` | 4 | 9 | 9 | TD2, TD3 | none |
| `CLOSE_FILTERS` > `SELECT_DATE` > `CLOSE_FILTERS` > `OPEN_FILTERS` | 4 | 9 | 9 | TD1, TD2 | none |
| `OPEN_FILTERS` > `SELECT_REGION` > `CLOSE_FILTERS` > `SELECT_REGION` | 4 | 9 | 9 | TD1, TD2, TD3 | may include tooltip-region misclassification until the macro rule is fixed |
| `APPLY_FILTER` > `SELECT_DATE` > `APPLY_FILTER` | 3 | 9 | 9 | TD2, TD3 | none |
| `APPLY_FILTER` > `SELECT_REGION` > `APPLY_FILTER` | 3 | 9 | 9 | TD1 | may include tooltip-region misclassification until the macro rule is fixed |
| `CLOSE_FILTERS` > `SELECT_DATE` > `CLOSE_FILTERS` | 3 | 9 | 9 | TD1, TD2 | none |
| `SELECT_CATEGORY` > `CLOSE_FILTERS` > `SELECT_CATEGORY` | 3 | 9 | 9 | TD2, TD3 | none |
| `APPLY_FILTER` > `SELECT_DATE` | 2 | 9 | 9 | TD2, TD3 | none |
| `CLOSE_FILTERS` > `SELECT_CATEGORY` | 2 | 9 | 16 | TD2, TD3 | none |
| `CLOSE_FILTERS` > `SELECT_DATE` | 2 | 9 | 18 | TD1, TD2 | none |
| `SELECT_DATE` > `APPLY_FILTER` | 2 | 9 | 9 | TD2, TD3 | none |
| `APPLY_FILTER` > `SELECT_DATE` > `APPLY_FILTER` > `RESET_FILTERS` | 4 | 8 | 8 | TD3 | none |
| `CLOSE_FILTERS` > `OPEN_FILTERS` > `SELECT_CATEGORY` > `CLOSE_FILTERS` | 4 | 8 | 8 | TD2, TD3 | none |
| `CLOSE_FILTERS` > `SELECT_CATEGORY` > `SELECT_DATE` > `CLOSE_FILTERS` | 4 | 8 | 8 | TD3 | none |
| `OPEN_FILTERS` > `SELECT_CATEGORY` > `CLOSE_FILTERS` > `SELECT_CATEGORY` | 4 | 8 | 8 | TD2, TD3 | none |
| `SELECT_CATEGORY` > `SELECT_DATE` > `CLOSE_FILTERS` > `OPEN_FILTERS` | 4 | 8 | 8 | TD3 | none |
| `APPLY_FILTER` > `RESET_FILTERS` > `APPLY_FILTER` | 3 | 8 | 10 | TD2, TD3 | none |
| `APPLY_FILTER` > `SELECT_CATEGORY` > `APPLY_FILTER` | 3 | 8 | 8 | TD2 | none |
| `CLOSE_FILTERS` > `APPLY_FILTER` > `SELECT_REGION` | 3 | 8 | 8 | TD1, TD2 | may include tooltip-region misclassification until the macro rule is fixed |
| `CLOSE_FILTERS` > `SELECT_CATEGORY` > `SELECT_DATE` | 3 | 8 | 8 | TD3 | none |
| `OPEN_FILTERS` > `SELECT_CATEGORY` > `CLOSE_FILTERS` | 3 | 8 | 8 | TD2, TD3 | none |
| `SELECT_CATEGORY` > `SELECT_DATE` > `CLOSE_FILTERS` | 3 | 8 | 8 | TD3 | none |
| `SELECT_DATE` > `APPLY_FILTER` > `RESET_FILTERS` | 3 | 8 | 8 | TD3 | none |
| `SELECT_REGION` > `CLOSE_FILTERS` > `APPLY_FILTER` | 3 | 8 | 8 | TD1, TD3 | may include tooltip-region misclassification until the macro rule is fixed |
| `APPLY_FILTER` > `SELECT_CATEGORY` | 2 | 8 | 8 | TD2 | none |
| `NAV_REPORTS` > `OPEN_REPORT` | 2 | 8 | 8 | TD4 | none |

## Caveats

- Candidate patterns are evidence, not the declared Fast Gate configuration.
  50.1 PrefixSpan correctness and boundedness remain open; 50.2 must validate
  each candidate (support, length, session isolation, no-match behaviour)
  before it enters the pattern map.
- `SELECT_REGION` is still produced by the macro `tooltip-region` substring rule
  (assessment limitation #6), so candidates containing it may include tooltip
  misclassifications; they are flagged, not silently accepted.
- The participant set is small and has no approved participant identifier
  (ADR-023). No participant-held-out generalisation is claimed.
- The selection is provisional until further participant traces are collected.
