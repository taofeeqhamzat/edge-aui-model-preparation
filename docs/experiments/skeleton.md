# Experiment Documentation Specification

This document defines the specification, format guidelines, and style constraints for experiment logs within `docs/experiments/`.

---

## 1. Directory Structure

Each experiment must reside in its own numbered directory under `docs/experiments/`:

- `docs/experiments/<N>/README.md`: Primary experiment document.
- `docs/experiments/<N>/before/`: Diagnostic logs, tables, and plots from the initial uncorrected state.
- `docs/experiments/<N>/after/`: Diagnostic logs, tables, and plots following corrective changes.

---

## 2. Document Header and Metadata

Every experiment document begins with a top-level heading followed by a standardized metadata block:

```markdown
# Experiment <N>: <Descriptive Title>

**Document Identifier:** `EXP-00X-<TOPIC>`  
**Governing Architecture Record:** `ADR-00X`  
**Status:** `Accepted and Verified` | `Accepted` | `Rejected` | `Deprecated` | `Staged`  
**Scope:** `model-preparation` and `edge-aui-framework`
```

### Metadata Field Requirements
- **Document Identifier:** Unique alphanumeric identifier formatted as `EXP-<NNN>-<TOPIC>` (for example, `EXP-001-PREPROCESSING`).
- **Governing Architecture Record:** Formal architecture decision record backing the experiment (for example, `ADR-001`).
- **Status:** Lifecycle state of the experiment (`Accepted and Verified`, `Accepted`, `Rejected`, `Deprecated`, or `Staged`).
- **Scope:** Repositories, packages, or submodules directly affected by the findings or implementation.

---

## 3. Standard Section Schema

Experiment records must follow the five standard numbered sections based on `docs/experiments/0/README.md`. Section titles may select one of the allowed naming options:

1. **`## 1. [GOAL] OR [CONTEXT] OR [PROBLEM STATEMENT]`**  
   Defines the motivation, underlying defect, or research hypothesis that the team investigates.
2. **`## 2. [PLAN]`**  
   Outlines the empirical methodology, investigation steps, and acceptance criteria.
3. **`## 3. [DECISION]`**  
   Records the adopted architectural, mathematical, or algorithmic decision.
4. **`## 4. [EXECUTION]`**  
   Documents code changes, configuration adjustments, and executed verification commands.
5. **`## 5. [OBSERVATION] OR [RESULT]`**  
   Reports quantified empirical findings, saturation audits, test suite outcomes, and artifact references.

---

## 4. Style Constraints

All experiment logs must strictly comply with the following technical writing constraints:

- **Sentence Length:** Maintain an average of approximately 10 words per sentence. Use direct, unambiguous phrasing.
- **Item Length:** Enforce a maximum of two sentences per list item.
- **Section Layout:** Provide exactly one single bullet list per section. Do not split sections into multiple disconnected lists or nested hierarchies.
- **Section Pruning:** Omit empty or unnecessary sections entirely. Never commit placeholder text or empty headers.
- **Empirical Rigor:** Always document precise numerical metrics, baseline values, and post-correction results.

---

## 5. Experiment Reference Template

Use the following template when staging or drafting a new experiment note:

```markdown
# Experiment <N>: <Title>

**Document Identifier:** `EXP-00X-<TOPIC>`  
**Governing Architecture Record:** `ADR-00X`  
**Status:** `Accepted` / `Rejected` / `Deprecated` / `Staged`  
**Scope:** `model-preparation` and `edge-aui-framework`

---

## 1. Context and Problem Statement

- State the observed defect, research goal, or operational challenge. Keep sentences short and clear.
- Identify the affected telemetry streams and baseline conditions.

---

## 2. Plan

- Outline the empirical investigation steps and validation protocol.
- Define explicit quantitative thresholds and success criteria.

---

## 3. Decision

- Record the adopted architectural or algorithmic selection. Reference the governing ADR for authoritative rationale.
- Summarize rejected alternatives and trade-offs directly.

---

## 4. Execution

- Detail file modifications, parameter updates, and implementation steps.
- List exact test commands and data processing pipelines run.

---

## 5. Observation and Result

- Quantify metric changes, boundary rates, and validation outcomes. Compare final results directly against baseline measurements.
- Reference diagnostic plots and saved visual artifacts.
```
