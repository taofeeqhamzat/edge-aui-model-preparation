# Participant Evidence — Local Retrofit and Verifier Gate

- **Status:** participant-derived engineering evidence; **not** a completed study
- **Source:** Supabase project `bfntyqahujplvcowjdmh` (`edge-aui`)
- **Provenance:** `participant` (declared by researcher instruction; locally retrofitted)
- **Split unit:** session (ADR-031). No participant identifier is approved (ADR-023), so
  `participantId` is null and no participant-held-out generalisation is claimed.

This directory records how the participant sets were obtained, corrected and gated. The traces
themselves live under `.data/` and are not committed.

## Export rounds

| Round | Exported | Provenance retrofit | Verifier | Dataset | Head |
| --- | --- | --- | --- | --- | --- |
| 2026-11-19 | 39 | 39 corrected | 35 verified / 4 excluded | `v2.1.0-participant` | none |
| **2026-11-20** | **92** | **44 corrected, 48 already correct** | **77 verified / 15 excluded** | **`v2.2.0-participant`** | **`semantic_head_e2`** |

The 2026-11-20 round supersedes the 2026-11-19 round as analysis input. It uses the same
retrofit rule, the same verifier and the same session-bounded split construction. The 2026-11-19
sections below remain as the audit record of the first round.

## 2026-11-20 round (current analysis input)

- **Authority:** researcher instruction in session on 2026-11-20 while completing Workstream 40:
  the 92-session Supabase export is a participant capture set from the corrected deployment; the
  remaining `scripted` declarations are upload-mode oversights. The audit manifest records every
  original and corrected declaration and both file hashes.
- **Retrofit:** 44 trace bodies corrected, 48 already participant; database and source export
  untouched; `participantId` null.
- **Verification:** 77 verified; 15 excluded — `prediction_window_attribution` ×11,
  `structural_validity` ×3, `task_lifecycle_consistency` ×1.
- **Dataset:** `v2.2.0-participant`, 9,686 examples, 77 session-bounded splits; class weights
  from training only; `NO_OUTCOME` 82.6% of training. The manifest's source type is
  `participant_capture` and its claim boundary is participant-specific.
- **Configuration:** the 40.3 experiment re-selected D (12×500 ms, [500,1500]) provisionally.
- **Head:** E2 selected on validation macro-F1. Test macro-F1 0.1297 vs majority 0.0841;
  accuracy 0.1271 vs 0.8603. This is participant-derived engineering evidence, not a
  participant-held-out or user-benefit claim.
- **Artefacts:** [semantic-head report](../semantic-head/semantic-head-report.md),
  [dataset summary](dataset-summary-v2.2.0-participant.json),
  [window/horizon report](../window-horizon-participant-r2/window-horizon-report.md),
  [notebook](../../notebooks/04_semantic_target_head_retraining.ipynb).
- **Audit manifests:** [`retrofit-manifest-2026-11-20.json`](retrofit-manifest-2026-11-20.json),
  [`verification-manifest-2026-11-20.json`](verification-manifest-2026-11-20.json),
  [`export-manifest-2026-11-20.json`](export-manifest-2026-11-20.json).
- **Reproduction:**
  `python3 -m src.provenance_retrofit --source .data/raw/collected-2026-11-20
  --output .data/raw/participant-2026-11-20 --authority "<instruction>"`, then
  `python3 -m src.participant_evidence --source .data/raw/participant-2026-11-20
  --output .data/raw/participant-verified-2026-11-20 --framework-root ../edge-aui-framework`.

## 2026-11-19 round

## 1. Why a retrofit was needed

The deployed collection mode was corrected to `VITE_AUI_COLLECTION_MODE=all`, but every trace
collected before that correction was uploaded with `session.provenance` and
`metadata.provenance` set to `scripted` and `participantId` null. The database was **not**
modified. Instead:

1. the traces were exported read-only (`scripts/export-traces.mjs`);
2. a local, audited retrofit rewrote both provenance declarations on the exported copy;
3. the framework verifier decided which traces are analysis input.

**Authority:** researcher instruction, 2026-11-19 (session reference
`session-9046fc67-9827-408b-a1a6-498b4f03f54c`): every trace in that export is a participant
capture from the corrected deployment; the `scripted` label was an upload-mode oversight.

**Rule:** rewrite only `session.provenance` and `metadata.provenance`; leave `participantId`
null; never edit the source export; record the original declaration, the corrected declaration,
the authority and both file hashes per session — see
[`retrofit-manifest.json`](retrofit-manifest.json).

No provenance was inferred from behaviour, timing, device or environment.

## 2. Verification gate

The framework verifier (`edge-aui-framework/scripts/verify-trace.mjs --verbose`) ran on every
corrected trace. Only verified traces are analysis input; a failed trace remains a record but is
not evidence for the experiment. See
[`verification-manifest.json`](verification-manifest.json).

| Result | Count |
| --- | --- |
| Exported | 39 |
| Verified | 35 |
| Excluded | 4 |

Exclusions:

| Session | Task | Failed check | Reason |
| --- | --- | --- | --- |
| `44583aa5-…` | TD1 | `prediction_window_attribution` | predictions name/evaluate windows absent from `microTensors` |
| `647a305d-…` | TD2 | `structural_validity` | a task event is missing `taskDefinitionVersion` |
| `8088144a-…` | TD1 | `prediction_window_attribution` | predictions name/evaluate windows absent from `microTensors` |
| `f9c28d98-…` | TD4 | `prediction_window_attribution` | predictions name/evaluate windows absent from `microTensors` |

These are capture defects (dropped windows / incomplete task records), not retrofit artefacts.

## 3. Analysis results

| Artefact | Result |
| --- | --- |
| 40.3 participant experiment | [report](../window-horizon-participant/window-horizon-report.md) — selected **D (12×500 ms, [500,1500])**, **provisional** |
| Semantic dataset | `v2.1.0-participant`, 3,603 examples, manifest validated — [summary](dataset-summary-v2.1.0-participant.json) |
| 50.2 pattern candidates | [report](../participant-patterns/participant-pattern-candidates.md) — 159 candidates at ≥3-session support |

The participant run does **not** confirm the scripted selection (C). Under the pre-declared
criteria, D has the lowest training share of the largest class (80.9% vs 81.3%), but the margin
is 0.4 percentage points: the selection is provisional and must be re-confirmed after further
collection. No semantic head was trained.

## 4. Limitations

- 35 sessions, no approved participant identifier, session-bounded splits only.
- The whole export is declared participant by researcher authority. Provenance was not inferred
  from behaviour, timing or device; if a pre-correction scripted run was present in the export,
  it is inside the participant set by that declaration, not by an algorithmic judgement.
- `NO_OUTCOME` is about 80% of training examples; the target head will need class weighting and
  a coverage report, and 40.4 must not pretend the configuration is settled.
- Four traces are excluded; their labels remain unanalysed.
- The macro stream still carries assessment limitation #6 defects; `SELECT_REGION` candidates
  may include `tooltip-region` misclassifications and are flagged in the pattern report.

## 5. Reproduction

```bash
# 1. Export (service-role key is supplied for this command only; never committed)
SUPABASE_URL=https://<ref>.supabase.co \
SUPABASE_SERVICE_ROLE_KEY=<key> \
  node edge-aui-framework/scripts/export-traces.mjs --out model-preparation/.data/raw/collected-2026-11-19

# 2. Local, audited retrofit (source export untouched)
python3 -m src.provenance_retrofit \
  --source .data/raw/collected-2026-11-19 \
  --output .data/raw/participant \
  --authority "<researcher instruction and reason>"

# 3. Verify and stage the analysis input
python3 -m src.participant_evidence \
  --source .data/raw/participant \
  --output .data/raw/participant-verified \
  --framework-root ../edge-aui-framework

# 4. Re-run the experiment and the pattern derivation
python3 -m src.window_horizon \
  --trace-dir .data/raw/participant-verified \
  --report-dir docs/experiments/window-horizon-participant \
  --require-provenance participant --write-dataset --version v2.1.0-participant
python3 -m src.participant_patterns \
  --trace-dir .data/raw/participant-verified \
  --report-dir docs/experiments/participant-patterns
```
