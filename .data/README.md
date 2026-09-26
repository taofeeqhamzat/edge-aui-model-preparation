---
license: cc-by-4.0
task_categories:
  - tabular-classification
  - time-series-forecasting
tags:
  - time-series-classification
  - sequence-modeling
  - micro-interactions
  - UI-adaptation
  - behavioral-modeling
  - human-computer-interaction
---

# Dataset Card: Edge-Native Adaptive UI Behavioral Logs

This repository stores aggregated behavioral interaction logs in raw and processed forms, primarily in the `.parquet` data storage format.
These logs support research on edge-native adaptive user interfaces.

## 1. Hosted Data

The repository hosts microtensor parquet files derived from public research datasets and runtime experiment sessions:

- AdSERP Search and Interaction Logs (Arapakis et al., 2025).
- High-Volume Trajectories (Mendeley Mouse Dynamics, 2026).
- Structured Human-Machine Interaction Logs (Carrera-Rivera et al., 2023).
- Client-Side Action Paths (Ou et al., 2021).
- Continuous Kinematics (Leiva and Arapakis, 2020).

### Interim & Sequence Datasets:
- **Canonical Traces (`interim/traces/canonical_traces.parquet`):** Ingested browser experiment telemetry traces (`schemaVersion 1.1.0`) with standard and extended provenance fields (`sessionId`, `experimentId`, `conditionId`, `taskId`, `windowId`, `sourceEventIds`, `preprocessingVersion`, `featureSchemaVersion`).
- **MicroTensor Sequences (`interim/sequences.parquet`):** Model-ready sequential interaction tensors ($T=8$, $D=18$) assembled with strict session and task boundary isolation and lookahead window alignment for `TargetInterventionHead` training.

## 2. References

For full details on the data preprocessing pipeline, MicroTensor feature schema, and target outcome taxonomy, refer to the following repositories:

- **Model Preparation Repository:** [taofeeqhamzat/edge-aui-model-preparation](https://github.com/taofeeqhamzat/edge-aui-model-preparation)
- **Framework Repository:** [taofeeqhamzat/edge-aui](https://github.com/taofeeqhamzat/edge-aui)

Please refer to the source repositories for detailed technical specifications and training guidelines.
