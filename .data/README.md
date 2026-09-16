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

This repository stores aggregated behavioral interaction logs in the microtensor parquet format.
These logs support research on edge-native adaptive user interfaces.

## 1. Hosted Data

The repository hosts microtensor parquet files derived from five public research datasets:
- Client-Side Action Paths (Ou et al., 2021).
- Structured Human-Machine Interaction Logs (Carrera-Rivera et al., 2023).
- Continuous Kinematics (Leiva and Arapakis, 2020).
- High-Volume Trajectories (Mendeley Mouse Dynamics, 2026).
- AdSERP Search and Interaction Logs (Arapakis et al., 2025).

## 2. References

For full details on the data preprocessing pipeline, MicroTensor feature schema, and target outcome taxonomy, refer to the following repositories:

- **Model Preparation Repository:** [taofeeqhamzat/edge-aui-model-preparation](https://github.com/taofeeqhamzat/edge-aui-model-preparation)
- **Framework Repository:** [taofeeqhamzat/edge-aui](https://github.com/taofeeqhamzat/edge-aui)

Please refer to the source repositories for detailed technical specifications and training guidelines.
