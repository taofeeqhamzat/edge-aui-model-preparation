# Experiment 1: Behavioural Feature Normalization and Modality Masking

**Document Identifier:** `EXP-001-PREPROCESSING`  
**Governing Architecture Record:** `ADR-001`  
**Status:** Accepted and Verified  
**Scope:** `model-preparation` and `edge-aui-framework`

---

## 1. Context and Problem Statement

The Edge-AUI framework extracts interaction features from continuous browser telemetry. The pipeline partitions high-frequency pointer and scroll streams into 500 ms sliding windows with a 250 ms stride.

Each window extracts nine continuous behavioural metrics:
1. `meanVelocity`
2. `maxVelocity`
3. `meanAcceleration`
4. `hesitationCount`
5. `totalTrajectoryLength`
6. `dwellTimeMs`
7. `trajectoryEntropy`
8. `scrollDepthPercentage`
9. `scrollVelocity`

The pipeline concatenates these nine features with a nine-dimensional binary Modality Mask Vector $M \in \{0, 1\}^9$:
$$\widetilde{X}_t = \big[X_t \odot M, \ M\big] \in \mathbb{R}^{18}$$

During initial exploratory testing, the engineering team detected two major defects:
- Static normalization constants caused severe ceiling saturation on acceleration and hesitation.
- The pipeline applied dynamic window-level activity masking to pointer features, but static capability masking to scroll features.

This document records the mathematical basis, empirical audits, and architecture decisions that corrected both defects.

---

## 2. Part A: Feature Normalization Scaling

### 2.1 The Saturation Defect

Initial testing used narrow static normalization scales:
- `mean_acceleration_scale` was 0.1 pixels per square millisecond.
- `hesitation_scale` was 10.0 directional turns.

Under these legacy scales:
- **24.36 percent of active windows** hit the 1.0000 ceiling for `meanAcceleration`.
- **14.71 percent of active windows** hit the 1.0000 ceiling for `hesitationCount`.

High-frequency pointer jitter during web search tasks produces 15 to 25 directional changes. The legacy scales clipped real user variance and degraded representation learning.

### 2.2 Unclipped Distribution Metrics

The team calculated percentiles across 13,846 sliding windows from 300 AdSERP sessions:

| Metric | Raw $P_{95}$ | Raw $P_{99}$ | Legacy Saturation ($=1.0$) | Calibrated Saturation ($=1.0$) |
|:---|:---:|:---:|:---:|:---:|
| `meanVelocity` | $0.84\text{ px/ms}$ | $1.80\text{ px/ms}$ | 0.09% | 0.09% |
| `maxVelocity` | $2.78\text{ px/ms}$ | $6.25\text{ px/ms}$ | 0.37% | 0.37% |
| `meanAcceleration` | $0.44\text{ px/ms}^2$ | $1.21\text{ px/ms}^2$ | **24.36%** | **1.34%** |
| `hesitationCount` | 15.0 turns | 24.0 turns | **14.71%** | **1.22%** |
| `totalTrajectoryLength` | $339.7\text{ px}$ | $676.8\text{ px}$ | 0.00% | 0.00% |
| `dwellTimeMs` | $1640.0\text{ ms}$ | $2102.0\text{ ms}$ | 36.83% | 36.83% |
| `trajectoryEntropy` | 0.6223 | 0.7354 | 0.00% | 0.00% |
| `scrollDepthPercentage` | 1.0328 | 1.4265 | 5.60% | 5.60% |
| `scrollVelocity` | 4.4000 | 5.6000 | 2.73% | 2.73% |

### 2.3 Natural Zeros vs Artificial Clipping

The empirical audit verified legitimate behavioural zeros:
- `scrollVelocity` displays 58.20 percent natural zeros when users do not scroll.
- `hesitationCount` displays 29.20 percent natural zeros during direct cursor movements.
- `trajectoryEntropy` displays 17.80 percent natural zeros during straight ballistic paths.

These values represent valid physical states, not missing measurements.

### 2.4 Calibrated Configuration

The team updated the scales in `src/config.py` and `src/config.yaml`:

```python
scale_dict = {
    "velocity": 10.0,           # Accommodates fast sweeps
    "max_velocity": 10.0,       # Keeps symmetry with mean velocity
    "acceleration": 1.0,        # Reduces ceiling clipping eleven-fold
    "hesitation": 25.0,         # Captures pointer micro-jitter variance
    "trajectory": 2000.0,       # Accommodates diagonal canvas travel
    "scroll_velocity": 5.0      # Clips only extreme scroll flings
}
```

---

## 3. Part B: Modality Capability Masking (ADR-001)

### 3.1 The Inconsistency Defect

The initial code in `src/preprocessing.py` contained an asymmetry:
- Pointer features evaluated `len(pointer_events) >= 2`.
- When the cursor remained still, pointer masks dropped to 0.0.
- Scroll features evaluated a dataset-level boolean flag (`has_scroll_support`).
- When a user did not scroll, scroll masks remained 1.0.
- The pipeline grouped attentional DOM hover dwell (`dwellTimeMs`) under pointer masks.
- Stationary hover over target elements cleared the dwell signal to zero.

### 3.2 Theoretical Basis and Neural Gating Dynamics

The recurrent Gated Recurrent Unit receives input $\widetilde{X}_t \in \mathbb{R}^{18}$. Linear projections compute gate activations:
$$W_i \widetilde{X}_t = \sum_{j=1}^9 W_{i, j} (x_{t, j} \cdot M_j) + \sum_{j=1}^9 W_{i, 9+j} M_j$$

#### Case 1: Capability Masking (Adopted)
- When a sensor is active and the user is still ($x_{t, j} = 0, M_j = 1$):
  $$\phi_j = W_{i, 9+j}$$
- When a sensor is absent ($x_{t, j} = 0, M_j = 0$):
  $$\phi_j = 0$$
- The separation distance is:
  $$\Delta = W_{i, 9+j} \ne 0$$
The network learns dedicated parameters that identify verified physical stillness.

#### Case 2: Dynamic Activity Masking (Rejected)
- When a user pauses ($x_{t, j} = 0$), the mask drops to zero ($M_j = 0$).
- Both stillness and sensor absence produce identical zero vectors:
  $$\Delta = 0$$
The network cannot distinguish a contemplative pause from a missing sensor. This reintroduces collinearity.

### 3.3 Production Edge Inference Parity

In `edge-aui-framework`, the client browser attaches listeners for pointer, scroll, and DOM events. The browser monitors all channels continuously:
$$M_{\text{browser}} = \mathbf{1}_9 = [1, 1, 1, 1, 1, 1, 1, 1, 1]$$

If offline training deactivates $M$ during pauses, the model suffers acute distribution shift during live inference. The model fails to recognize stationary hover states.

### 3.4 Adopted Architecture Partition

The team formalized three independent capability flags:

1. **`has_pointer_support: bool = True`**  
   Governs features 0 through 4 and 6: velocity, acceleration, hesitation, trajectory length, and entropy.  
   Mask indices: `[0, 1, 2, 3, 4, 6]`.

2. **`has_dom_support: bool = True`**  
   Governs feature 5: `dwellTimeMs`.  
   Mask index: `[5]`.  
   Dwell extraction operates independently from pointer velocity.

3. **`has_scroll_support: bool = True`**  
   Governs features 7 and 8: scroll depth percentage and scroll velocity.  
   Mask indices: `[7, 8]`.

---

## 4. Verification and Empirical Results

The team verified all changes across the codebase and data storage:

1. **Unit Test Suite:**
   - Executed `python -m unittest discover tests`.
   - All 22 tests passed without failures.

2. **Parquet Store Regeneration:**
   - Re-extracted `adserp_microtensors.parquet` with the calibrated parameters.
   - Pointer mask activation is 1.0000 across all 112,865 windows.
   - Active acceleration ceiling saturation decreased from 24.36 percent to 1.34 percent.
   - Active hesitation ceiling saturation decreased from 14.71 percent to 1.22 percent.

3. **Diagnostic Image Records:**
   - Baseline records: `experiments/1/before/`
   - Post-correction records: `experiments/1/after/`
