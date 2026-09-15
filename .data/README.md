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

## 1. Dataset Provenance and Attribution

This repository stores aggregated behavioral interaction logs. The dataset supports research on edge-native adaptive user interfaces.

The repository includes five public research datasets:

- **Client-Side Action Paths:** Ou et al. (2021). Granular client-side action paths to model sequential user interactions.
- **Structured Human-Machine Interaction Logs:** Carrera-Rivera et al. (2023). Interaction logs to identify macro-interactions and sequential behavioral patterns.
- **Continuous Kinematics:** Leiva and Arapakis (2020). Continuous cursor coordinates and kinematic metrics to record motor dynamics.
- **High-Volume Trajectories:** Mendeley Mouse Dynamics (2026). Continuous cursor trajectory sequences across heterogeneous tasks.
- **AdSERP Search and Interaction Logs:** Arapakis et al. (2025). Search engine result page interactions with cursor coordinates, fixations, and DOM elements.

## 2. Multi-Tiered Transfer Learning Strategy

The datasets support a two-stage training strategy for a lightweight Gated Recurrent Unit (GRU):

1. **Foundational Pre-Training:** The model learns cross-domain human motor behaviors from continuous kinematic streams. The recurrent layers model trajectory physics and temporal decay without interface-specific geometry.
2. **Target UI Fine-Tuning:** The system freezes the recurrent layers. The classification head trains on localized logs from the target application to map motor representations to downstream actions.

## 3. MicroTensor Feature Schema

The pipeline segments continuous pointer and scroll streams into 500 ms windows with a 250 ms stride. Each window outputs an 18-dimensional vector. The vector contains 9 kinematic features and 9 modality capability masks (ADR-001):

| Index | Field | Dimension Type | Unit / Range | Description |
|---|---|---|---|---|
| 0 | `mean_velocity` | Continuous | $px/ms$ | Mean Euclidean velocity |
| 1 | `max_velocity` | Continuous | $px/ms$ | Peak Euclidean velocity |
| 2 | `mean_acceleration` | Continuous | $px/ms^2$ | Mean Euclidean acceleration |
| 3 | `hesitation_count` | Discrete count | $\ge 0$ | Direction changes with angle $\theta > 45^\circ$ |
| 4 | `total_trajectory_length` | Continuous | $px$ | Accumulated Euclidean path length |
| 5 | `dwell_time_ms` | Continuous | $ms$ | Cumulative dwell time on interactive DOM targets |
| 6 | `trajectory_entropy` | Continuous | $[0.0, 1.0]$ | Spatial path efficiency |
| 7 | `scroll_depth_percentage` | Continuous | $[0.0, 1.0]$ | Viewport vertical scroll position |
| 8 | `scroll_velocity` | Continuous | $px/ms$ | Rate of vertical scroll displacement |
| 9-17 | `mask_*` | Binary flag | $\{0, 1\}$ | Modality capability indicator |

Binary masks record feature availability per dataset. The model avoids zero-imputation artifacts when a dataset lacks scroll or DOM telemetry.

## 4. Downstream Target Outcome Taxonomy

The self-supervised pipeline pairs each window with downstream interface outcomes in an adjacent lookahead horizon of 500 ms to 1500 ms (ADR-002):

- **Class 0 (`NO_OUTCOME`):** Inactivity or reading without a qualifying macro-interaction.
- **Class 1 (`CLICK`):** Pointer click on an interactive element.
- **Class 2 (`FORM_SUBMIT`):** Form submission action.
- **Class 3 (`BACKTRACK`):** Fast direction reversal or return navigation.
- **Class 4 (`RAPID_SCROLL`):** Fast scroll burst.
- **Class 5 (`HOVER_DWELL`):** Cursor pause over an interactive element.
- **Class 6 (`ABANDON`):** Observable window or session termination (`beforeunload`, `pagehide`, `unload`).

## 5. Partitioning and Leakage Prevention

The pipeline splits data by unique `user_id`. The pipeline falls back to `session_id` only when reliable user identifiers are missing.

Sequence windows reconstruct strictly within assigned split partitions. No sequence window spans multiple users or sessions.
