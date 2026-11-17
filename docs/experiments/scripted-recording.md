# Scripted Testbed Trace Recording Report

**Workstream:** 30 — Collection validation (Task 30.1)  
**Recorded at:** 2026-11-17T10:32:03Z  
**Data source:** `scripted_testbed`  
**Provenance:** `scripted` — **not participant evidence**  
**Framework commit:** `5f5f734c454fd057a2b760cc704c791f56551f36`  
**Model-preparation commit:** `963ae54f1b1a2eecb229a18e39844fd6e726ab60`  
**Automation:** `agent-browser 0.36.0`  
**Browser:** `Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/157.0.0.0 Safari/537.36`  
**Viewport:** `1920×1080`  
**Environment:** `Darwin 27.3.0 (arm64)`

---

## 1. Claim boundary (ADR-013)

> Results from this dataset may be reported as: 'the dataset -> preparation -> training -> export -> runtime path executes end to end', 'the learned head loads in the browser and produces logits of the expected shape', and engineering and pipeline findings. Results from this dataset may NOT be reported as: evidence about human participants; evidence about usability, task performance, or intervention benefit; or evidence that learned intervention prediction outperforms the deterministic policy — a model trained on labels produced by that policy is being compared against its own teacher.

> Every trace produced by this harness is scripted engineering evidence. It is not participant evidence and must not be pooled with participant traces.

---

## 2. Execution summary

- **Conditions:** baseline, adaptive
- **Tasks:** TD1, TD2, TD3, TD4
- **Trials per condition/task:** 1
- **Captures recorded:** 8
- **Verified by `scripts/verify-trace.mjs`:** 8/8
- **Schema:** `1.4.0`

| Capture | Condition | Task | Status | Duration (ms) | Windows | Outcomes | Predictions | Applied interventions | Verified |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `scripted-TD1-baseline-trial1.json` | baseline | TD1 | Completed | 6328 | 17 | 17 | 4 | 0 | PASS |
| `scripted-TD2-baseline-trial1.json` | baseline | TD2 | Completed | 6138 | 17 | 17 | 3 | 0 | PASS |
| `scripted-TD3-baseline-trial1.json` | baseline | TD3 | Completed | 9385 | 28 | 28 | 4 | 0 | PASS |
| `scripted-TD4-baseline-trial1.json` | baseline | TD4 | Completed | 5273 | 13 | 13 | 3 | 0 | PASS |
| `scripted-TD1-adaptive-trial1.json` | adaptive | TD1 | Completed | 6007 | 17 | 17 | 4 | 0 | PASS |
| `scripted-TD2-adaptive-trial1.json` | adaptive | TD2 | Completed | 5810 | 18 | 18 | 4 | 0 | PASS |
| `scripted-TD3-adaptive-trial1.json` | adaptive | TD3 | Completed | 9009 | 28 | 28 | 4 | 0 | PASS |
| `scripted-TD4-adaptive-trial1.json` | adaptive | TD4 | Completed | 5046 | 16 | 16 | 4 | 0 | PASS |

### Semantic outcomes observed

| Task | Condition | `taskOutcome` values |
| :--- | :--- | :--- |
| TD1 | baseline | `open_filter_section`, `select_region`, `apply_filter`, `export_report` |
| TD2 | baseline | `select_date_range`, `open_filter_section`, `select_category`, `apply_filter` |
| TD3 | baseline | `select_region`, `open_filter_section`, `select_category`, `select_date_range`, `apply_filter`, `reset_filter` |
| TD4 | baseline | `navigate_reports`, `select_report`, `export_report` |
| TD1 | adaptive | `open_filter_section`, `select_region`, `apply_filter`, `export_report` |
| TD2 | adaptive | `select_date_range`, `open_filter_section`, `select_category`, `apply_filter` |
| TD3 | adaptive | `select_region`, `open_filter_section`, `select_category`, `select_date_range`, `apply_filter`, `reset_filter` |
| TD4 | adaptive | `navigate_reports`, `select_report`, `export_report` |

---

## 3. What the captures establish

1. Each task completes on its canonical path through the real task controls, with zero
   `task_error` events and the task-state evidence the completion predicate requires.
2. Each capture is a runtime-produced `1.4.0` trace carrying semantic outcomes, the
   task-definition version (`TD-v2`) and the semantic vocabulary version.
3. Baseline captures contain no applied intervention; every applied adaptive intervention
   belongs to the framework's configured intervention vocabulary.
4. The rows, filters and report state reported in each trace's final `task.uiState` match the
   approved task definitions.

## 4. What the captures do not establish

- Nothing about human participants, usability, task performance or intervention benefit.
- Nothing about a deployed origin or a live Supabase project; the run targets the local
  testbed origin.
- A trial whose first action precedes the first MicroTensor window has no semantic outcome for
  that action. The trace records the outcome windows the pipeline actually produced; it is not
  back-filled.

---

## 5. Verification

```bash
cd edge-aui-framework
node scripts/verify-trace.mjs ../model-preparation/.data/raw/scripted/<capture>.json
node scripts/compare-condition-traces.mjs --baseline <baseline.json> --adaptive <adaptive.json>
```
