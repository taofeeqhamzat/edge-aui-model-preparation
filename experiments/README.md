# Machine Learning Experiments and Empirical Audits

This directory contains empirical experiment logs, diagnostic visual plots, and decision records for the Edge-AUI model preparation pipeline.

## Experiment Index

- **[Experiment 1: Behavioural Feature Normalization and Modality Masking](file:///Users/user/Workspace/MivaCS/FYP/model-preparation/experiments/1/README.md)**  
  Resolves feature ceiling saturation and establishes sensor capability masking (ADR-001).
  - Diagnostic baseline plots: [`experiments/1/before/`](file:///Users/user/Workspace/MivaCS/FYP/model-preparation/experiments/1/before/)
  - Diagnostic post-correction plots: [`experiments/1/after/`](file:///Users/user/Workspace/MivaCS/FYP/model-preparation/experiments/1/after/)

## Directory Structure

Each numbered experiment directory contains:

- `README.md`: Formal experimental report, theoretical basis, and architecture decisions.
- `before/`: Diagnostic logs, tables, and plots from the initial uncorrected state.
- `after/`: Diagnostic logs, tables, and plots after the engineering team applies corrections.
