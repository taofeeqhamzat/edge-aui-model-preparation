# Exploratory Analysis: Baseline State Before Corrections

This document records the baseline data state before the team applied pipeline corrections.

## 1. Modality Mask Inconsistencies

The initial pipeline applied dynamic window-level activity masking to pointer features:

- The pipeline set pointer masks to 1.0 only when a window contained two or more pointer events.
- When a user stopped mouse movement, pointer masks dropped to 0.0.
- The pipeline grouped attentional DOM dwell time (`dwellTimeMs`) into pointer features.
- Stationary pauses set dwell time to 0.0.
- In contrast, scroll features used a static dataset capability flag.
- This asymmetry caused a conflict between pointer and scroll modalities.

## 2. Incorrect Normalization Scaling and Ceiling Saturation

The initial pipeline used uncalibrated normalization constants:

- The scale for `meanAcceleration` was 0.1 pixels per square millisecond.
- This low threshold forced 24.36 percent of active windows to saturate at 1.0000.
- The scale for `hesitationCount` was 10.0 directional turns.
- High-frequency pointer jitter caused 14.71 percent of active windows to saturate at 1.0000.
- These narrow scales compressed natural behavioral variance.

## 3. Visual Artifacts in Baseline Directory

The baseline directory contains four diagnostic plots:

- `dist.png`: Shows feature distributions with severe ceiling clipping at 1.0.
- `mask_act_freq.png`: Shows pointer mask activation dropping to 72.6 percent because of stationary pauses.
- `win_session.png`: Shows window extraction counts across dataset sessions.
- `output.png`: Duplicate plot of baseline feature distributions.
