# Experiment 1: Behavioural Feature Normalization and Modality Masking

**Document Identifier:** `EXP-001-PREPROCESSING`  
**Governing Architecture Record:** `ADR-001`  
**Status:** `Accepted and Verified`  
**Scope:** `model-preparation` and `edge-aui-framework`

---

## 1. Context and Problem Statement

- The telemetry pipeline partitions user interactions into 500 ms sliding windows. Each window computes nine continuous behavioural metrics.
- The pipeline concatenates these metrics with a nine-dimensional binary modality mask. This produces an 18-dimensional vector for neural network ingestion.
- Legacy normalization scales caused severe ceiling saturation at 1.0. Acceleration saturated in 24.36 percent of active windows.
- Hesitation count saturated in 14.71 percent of active windows. Rapid pointer jitter during search tasks exceeded legacy thresholds.
- The pipeline applied dynamic activity masking to pointer features. Masks dropped to zero whenever the cursor paused.
- In contrast, scroll features evaluated static dataset-level capability flags. Furthermore, stationary hover cleared attentional dwell time to zero.
- This masking asymmetry conflated physical user stillness with missing hardware sensors. The resulting collinearity degraded representation learning.

---

## 2. Plan

- Audit unclipped metric percentiles across 13,846 AdSERP sliding windows. Identify empirical distributions across all nine continuous behavioural features.
- Differentiate natural behavioural zeros from artificial ceiling clipping. Verify zero-velocity scroll and direct cursor movements.
- Calibrate static normalization constants for acceleration and hesitation. Target active ceiling saturation rates below two percent.
- Formalize capability masking under ADR-001 to resolve modality asymmetry. Decouple physical user stillness from absent sensor hardware.
- Partition input modalities into pointer, DOM, and scroll capability channels. Ensure hover dwell operates independently from cursor movement.
- Verify pipeline modifications using unit tests and regenerated Parquet files. Inspect before and after diagnostic plots.

---

## 3. Decision

- Adopt ADR-001 capability masking to record hardware telemetry presence. Mask vectors strictly reflect sensor support rather than instantaneous movement.
- Partition input streams into three independent capability flags. These flags govern pointer support, DOM hover support, and scroll support.
- Decouple attentional DOM dwell time from cursor movement. Hover dwell extraction now operates even when pointer velocity is zero.
- Reject dynamic activity masking to eliminate collinearity in GRU linear projections. Neural weights now distinguish physical stillness from missing sensor channels.
- Match browser edge runtime where listeners monitor all channels continuously. Live inference supplies an all-ones capability mask vector.
- Calibrate the acceleration scale from 0.1 to 1.0. This change accommodates high-acceleration sweeps across desktop viewports.
- Calibrate the hesitation scale from 10.0 to 25.0 directional turns. This expansion captures high-frequency pointer jitter during search tasks.
- Retain velocity scales at 10.0, trajectory at 2000.0, and scroll velocity at 5.0. These bounds prevent clipping while maintaining natural distribution variance.

---

## 4. Execution

- Update normalization scale constants in `src/config.py` and `src/config.yaml`. Set acceleration scale to 1.0 and hesitation scale to 25.0.
- Refactor `src/preprocessing.py` to implement ADR-001 capability masking. Introduce dedicated flags for pointer support, DOM support, and scroll support.
- Update `generate_preprocessing_eda_nb.py` with boundary and saturation audit routines. Regenerate `notebooks/01_preprocessing_eda.ipynb` to verify empirical distributions.
- Execute unit test suite with `python -m unittest discover tests`. Ensure all feature extraction and masking tests pass cleanly.
- Regenerate the `adserp_microtensors.parquet` storage partition with calibrated parameters. Extract MicroTensors across all 300 AdSERP sessions.
- Generate diagnostic comparison plots for baseline and post-correction states. Save visual artifacts in `1/before/` and `1/after/`.

---

## 5. Observation and Result

- Active acceleration ceiling saturation dropped from 24.36 percent to 1.34 percent. This represents an eleven-fold reduction in ceiling truncation.
- Active hesitation ceiling saturation dropped from 14.71 percent to 1.22 percent. Calibrated scales successfully capture real pointer jitter variance.
- The audit verified legitimate natural zeros across inactive intervals. Scroll velocity displays 58.20 percent natural zeros during non-scrolling reading periods.
- Hesitation count displays 29.20 percent natural zeros during direct cursor movements. Trajectory entropy displays 17.80 percent natural zeros during straight paths.
- Pointer masks remain at 1.0000 across all 112,865 extracted AdSERP windows. Capability masking successfully decouples sensor hardware presence from physical user stillness.
- Attentional hover dwell now records stationary focus and does not reset to zero. Neural projection layers compute distinct parameter offsets for resting users.
- The automated test suite executed with zero failures across all 22 unit tests.
- Directory [`1/before/`](1/before/) contains baseline diagnostic plots. Directory [`1/after/`](1/after/) contains corrected diagnostic plots.
