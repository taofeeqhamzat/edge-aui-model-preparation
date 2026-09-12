# Exploratory Analysis: Validated State After Corrections

This document records the data state after the team applied pipeline corrections.

## 1. Calibrated Normalization Scales

The team calibrated the feature scales to prevent artificial ceiling saturation:

- The scale for `meanAcceleration` increased from 0.1 to 1.0 pixels per square millisecond.
- Active ceiling saturation dropped from 24.36 percent to 1.34 percent.
- The scale for `hesitationCount` increased from 10.0 to 25.0 directional turns.
- Active ceiling saturation dropped from 14.71 percent to 1.22 percent.
- Natural zeros remain preserved for inactive windows (58.2 percent for scroll velocity).
- The calibrated scales preserve behavioural variance for recurrent neural network training.

## 2. Modality Capability Masking

The team aligned the mask vector with Architecture Decision Record 001:

- The mask vector strictly indicates sensor recording capability.
- The pipeline partitions modalities into three independent capability flags: pointer, DOM targets, and scroll.
- Pointer masks remain at 1.0000 throughout valid AdSERP sessions.
- Attentional DOM hover dwell functions independently from pointer movement.
- The mathematical formulation separates physical rest from unmonitored sensors.

## 3. Visual Artifacts in Validated Directory

The validated directory contains five diagnostic plots:

- `bound_and_sat_rate.png`: Shows boundary and saturation rates across all nine features.
- `dist.png`: Shows feature distributions with smooth exponential decay curves.
- `mask_act_freq.png`: Shows 100 percent mask activation for supported modalities.
- `win_session.png`: Shows session window counts.
- `Screenshot 2026-09-12 at 07.56.46.png`: Shows the executed Jupyter notebook audit table.
