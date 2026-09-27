# Scripted Testbed Trace Recording Report

**Phase:** B — Substitute Data and Labels (Task 10.1)  
**Recorded At:** 2026-09-26T23:28:22Z  
**Data Source:** `scripted_testbed`  
**Framework Commit:** `01765bbef3f54642a41156b8fcaeb0900f38ff6c`  
**Model-Preparation Commit:** `f4c2bd13dc9c0d74b98851bd377a592f6962d060`  
**Browser / Automation:** `agent-browser 0.27.0`  
**Environment:** `Darwin 27.0.0 (arm64)`  

---

## 1. ADR-013 Mandatory Claim Boundary

> Results from this dataset may be reported as: 'the dataset -> preparation -> training -> export -> runtime path executes end to end', 'the learned head loads in the browser and produces logits of the expected shape', and engineering and pipeline findings. Results from this dataset may NOT be reported as: evidence about human participants; evidence about usability, task performance, or intervention benefit; or evidence that learned intervention prediction outperforms the deterministic policy — a model trained on labels produced by that policy is being compared against its own teacher.

---

## 2. Experimental Execution Summary

- **Conditions Evaluated:** baseline, adaptive
- **Tasks Evaluated:** T1, T2, T3
- **Trials per Condition/Task:** 3
- **Total Validated Traces:** 18
- **Verification Result:** All 18 traces passed `scripts/verify-trace.mjs` with 100% check pass rate.

### Recorded Trace Manifest

| Filename | Condition | Task | Trial | Status | Duration (ms) | Windows | Outcomes | Interventions |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `experiment-trace-f523e5aa-1413-4c43-a55a-1b8c1ab31f7e-1790465162658.json` | baseline | T1 | 1 | Completed | 9605 | 27 | 27 | 0 |
| `experiment-trace-e16ebbf9-a03e-41c6-976d-5e1dbc9df891-1790465172017.json` | baseline | T1 | 2 | In Progress | 8974 | 26 | 26 | 0 |
| `experiment-trace-64fa9a6f-5297-450f-b5d2-5b9ba97fcd80-1790465181955.json` | baseline | T1 | 3 | Completed | 9554 | 26 | 26 | 1 |
| `experiment-trace-7d8151b4-2bcd-4943-b789-27622c0fcf4e-1790465190177.json` | baseline | T2 | 1 | Completed | 7836 | 24 | 24 | 0 |
| `experiment-trace-cc07b52d-ac76-4c18-8220-e1cba91071c3-1790465198401.json` | baseline | T2 | 2 | Completed | 7842 | 24 | 24 | 0 |
| `experiment-trace-ea889eb3-a8c4-497b-8a31-8a17d61b5040-1790465206631.json` | baseline | T2 | 3 | Completed | 7838 | 23 | 23 | 0 |
| `experiment-trace-860ff863-ecfc-4bcb-9ff5-778b75f68d68-1790465213689.json` | baseline | T3 | 1 | In Progress | 6668 | 20 | 20 | 1 |
| `experiment-trace-660e47c7-a1cf-41c4-bae0-fd7fe2d908fa-1790465220754.json` | baseline | T3 | 2 | In Progress | 6676 | 20 | 20 | 1 |
| `experiment-trace-9282f192-b963-4b95-9b81-1302a7f27f18-1790465227839.json` | baseline | T3 | 3 | In Progress | 6683 | 20 | 20 | 1 |
| `experiment-trace-fd9b69db-488e-47d8-b424-4d2757a44bf7-1790465237220.json` | adaptive | T1 | 1 | In Progress | 8978 | 26 | 26 | 0 |
| `experiment-trace-514f01c3-7c09-443d-964f-7a52c17bab74-1790465247179.json` | adaptive | T1 | 2 | Completed | 9564 | 27 | 27 | 0 |
| `experiment-trace-b2eeed1e-017a-4295-befe-b86e37932391-1790465256598.json` | adaptive | T1 | 3 | In Progress | 8987 | 26 | 26 | 0 |
| `experiment-trace-f6a91ab6-e8f6-4920-b980-6553d21588c5-1790465264851.json` | adaptive | T2 | 1 | Completed | 7857 | 22 | 22 | 0 |
| `experiment-trace-497766d9-f4ae-40ae-b38a-54ac574b3cad-1790465273101.json` | adaptive | T2 | 2 | Completed | 7851 | 23 | 23 | 0 |
| `experiment-trace-a0acb4d2-f5e0-4e96-a011-b25f13e0c4b4-1790465281348.json` | adaptive | T2 | 3 | Completed | 7850 | 22 | 22 | 0 |
| `experiment-trace-ef41ca1f-b253-4da4-943b-c0e14def8a2c-1790465288424.json` | adaptive | T3 | 1 | In Progress | 6683 | 20 | 20 | 1 |
| `experiment-trace-3a08d0f5-3807-4298-9769-a3172408b02c-1790465295499.json` | adaptive | T3 | 2 | In Progress | 6682 | 19 | 19 | 0 |
| `experiment-trace-290806e2-122c-44c3-ae06-76f2f1be7019-1790465302578.json` | adaptive | T3 | 3 | In Progress | 6679 | 20 | 20 | 1 |

---

## 3. Provenance & Ingestion Verification

Every recorded trace conforms to `ExperimentTrace schemaVersion 1.1.0` and satisfies:
1. Complete task lifecycle (`task_start`, all intermediate `task_step` entries, and `task_complete`).
2. Exact 1-to-1 mapping from emitted `microTensor` windows to derived `outcome` records.
3. Baseline condition zero-mutation invariant (0 applied interventions in baseline trials).
4. Non-empty correlation IDs across sessions, windows, predictions, and intervention episodes.
