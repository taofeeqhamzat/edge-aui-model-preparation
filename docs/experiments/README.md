# Machine Learning Experiments and Empirical Audits

This directory contains empirical experiment logs, diagnostic visual plots, and decision records for the Edge-AUI model preparation pipeline.

## Experiment Index

- **[Experiment Documentation Specification](skeleton.md)**  
  Format standards, metadata schema, and style constraints for repository experiment logs.

- **[Experiment 0: Template / Staged Experiment](0/README.md)**  
  Staged reference template for upcoming experiments adhering to the documentation specification.

- **[Experiment 1: Behavioural Feature Normalization and Modality Masking](1/README.md)**  
  Resolves feature ceiling saturation and establishes sensor capability masking (ADR-001).
  - Diagnostic baseline plots: [`1/before/`](1/before/)
  - Diagnostic post-correction plots: [`1/after/`](1/after/)

- **[Experiment 2: Methodological Audit of Target Generation, Leak-Free Splitting, and GRU Architecture](2/README.md)**  
  Verifies training readiness across Tasks 3.1, 3.2, and 4.1. Resolves 10 pipeline issues (ADR-002, ADR-003).
  - Pre-audit issue state: [`2/before/`](2/before/)
  - Post-audit diagnostic plots: [`2/after/`](2/after/)

## Directory Structure

Each numbered experiment directory contains:

- `README.md`: Formal experimental report, theoretical basis, and architecture decisions.
- `before/`: Diagnostic logs, tables, and plots from the initial uncorrected state.
- `after/`: Diagnostic logs, tables, and plots after the engineering team applies corrections.
