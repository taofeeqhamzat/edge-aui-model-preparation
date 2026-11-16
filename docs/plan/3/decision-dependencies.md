# Model-Preparation Decision Dependencies — Milestone Plan 3

The authoritative architecture decision records live in the framework repository under
`edge-aui-framework/docs/decisions/`, because that repository owns the runtime trace contract and
the framework/application boundary. This file records **only the model-preparation obligations**
that follow from those decisions; it does not duplicate their rationale.

Do not create a second, drifting ADR set in this repository. If a decision changes, update the
authoritative ADR and this dependency table in the same change.

| Record | Relevance to model preparation | Model-preparation obligation |
| --- | --- | --- |
| [ADR-025](../../../../edge-aui-framework/docs/decisions/ADR-025-task-harness-and-state-verification-boundary.md) | Testbed owns TD1–TD4 predicates; timeout is `task_fail`/`Failed`; tasks carry `taskDefinitionVersion` | Ingest `taskDefinitionVersion` and `task_fail`; never use lifecycle labels as model targets; keep TD-v2 traces separate from legacy T1–T3 |
| [ADR-026](../../../../edge-aui-framework/docs/decisions/ADR-026-additive-semantic-outcome-taxonomy.md) | Additive semantic recording; keep generic seven-class taxonomy; retrain only after participant traces | Add `taskOutcome` fields without replacing `outcome`; keep old head/bundle; migrate taxonomy in 40.3/40.4 at the approved point |
| [ADR-027](../../../../edge-aui-framework/docs/decisions/ADR-027-event-canonicalization-and-outcome-derivation-ownership.md) | Framework owns canonicalization/backtracking mechanics; testbed owns semantic rules | Mirror the committed-change/submit/report/backtrack rules in Python; use shared fixtures and a parity test |
| [ADR-028](../../../../edge-aui-framework/docs/decisions/ADR-028-normalized-gate-prediction-contract.md) | Both gates return normalized outcome predictions; policy maps outcome→intervention | Export semantic outcome logits, not five intervention classes; manifest carries vocabulary; framework maps logits by config |
| [ADR-029](../../../../edge-aui-framework/docs/decisions/ADR-029-trace-schema-1-4-task-versioning.md) | Additive trace 1.4.0 with semantic outcome/task version/prediction fields; legacy readable | Add 1.4.0 to the ingestion allow-list; reject unknown versions; carry semantic/task fields; keep 1.3 support |

## Decision 6 — migration point

| Phase | Trace/dataset | Model |
| --- | --- | --- |
| 4A (preparation) | Record/ingest semantic outcomes additively; build semantic dataset v2; retain generic labels | Do not train a semantic head |
| Before 4B | Freeze field names/vocabulary/version; update `src/config.yaml` | Add semantic taxonomy/head dimension |
| 4B | Train on participant traces | New semantic head; old five-class head/bundle frozen |
| After 4B | Framework may consume semantic predictions | Old artifacts remain reproducible for comparison |

**Minimum decision before 40.1/40.2:** exact trace field names (`taskOutcome` proposed), vocabulary
version string, exact 14-ID vocabulary, and whether `NO_OUTCOME` is the fallback. Without that
approval, only schema/ingestion scaffolding can proceed; semantic dataset training cannot.

## Claim boundary

- Scripted traces are pipeline/engineering evidence only.
- A semantic head trained on labels derived from the framework's deterministic rules is not evidence
  that the learned head outperforms those rules.
- Model prediction metrics are not user-experience evidence.
- Report `NOT MEASURED`/`Blocked` rather than inferring a metric.
