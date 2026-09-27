# Data Schemas and Wire Contracts Specification

Authoritative index and specifications for data formats, feature representations, and UI context contracts used in the `model-preparation` pipeline.

---

## 1. MicroTensor Feature Schema (18-D)

- **Owning File:** `src/preprocessing.py`, `src/microtensor_store.py`
- **Configuration Source:** `src/config.yaml` (`preprocessing`)
- **Dimension:** 18 continuous float values (`np.float32`)
- **Representation:** $\tilde{X} = [X \odot M, M]$ where $X \in \mathbb{R}^9$ contains normalized kinematic metrics and $M \in \{0, 1\}^9$ is the binary modality mask vector.

| Index | Feature Key | Unit | Normalization Scale | Description |
|---|---|---|---|---|
| `0` | `meanVelocity` | px/ms | `/ 10.0` | Mean Euclidean cursor velocity across 500 ms window |
| `1` | `maxVelocity` | px/ms | `/ 10.0` | Maximum instantaneous Euclidean velocity |
| `2` | `meanAcceleration` | px/ms² | `/ 1.0` | Mean cursor acceleration magnitude |
| `3` | `hesitationCount` | count | `/ 25.0` | Directional turns with angle deviation $\theta > 45^\circ$ |
| `4` | `totalTrajectoryLength` | px | `/ 2000.0` | Cumulative Euclidean distance traversed |
| `5` | `dwellTimeMs` | ms | `/ 500.0` | Time hovering over interactive DOM elements |
| `6` | `trajectoryEntropy` | bits | `/ 5.0` | Shannon entropy of cursor trajectory segments |
| `7` | `scrollDepthPercentage`| % [0, 1] | `/ 1.0` | Vertical scroll offset relative to max scroll |
| `8` | `scrollVelocity` | px/ms | `/ 5.0` | Rate of vertical viewport scroll change |
| `9–17`| `mask[0..8]` | binary | 1.0 (valid) / 0.0 (masked) | Modality masks corresponding to features 0–8 |

Missing telemetry modalities (e.g. scroll on fixed viewports) strictly clear the corresponding modality mask bit ($M_i = 0.0$); zero-imputation of continuous values without masking is forbidden.

---

## 2. UIContext $\mathbb{R}^6$ Vector Encoding

- **Owning Files:** `src/context_encoding.py` (Python), `edge-aui-framework/src/types/contextVector.ts` (TypeScript)
- **Dimension:** 6 continuous float values bounded in $[0.0, 1.0]$ (`np.float32`)
- **Parity Tolerance:** Verified equal within $1 \times 10^{-6}$ by `tests/test_context_encoding.py`

| Index | Component Name | Encoding Derivation | Normalization Range / Scale | Description |
|---|---|---|---|---|
| `0` | `route` | Index in `ROUTE_VOCABULARY` | `/ 5.0` | Normalized route category (`['Overview', 'Analytics', 'Reports', 'Customers', 'Settings']`). 0.0 if unknown. |
| `1` | `primaryActionAvailable` | Boolean flag | `1.0` if True, `0.0` if False | Presence of active primary action button |
| `2` | `helpAvailable` | Boolean flag | `1.0` if True, `0.0` if False | Contextual help or tooltip availability |
| `3` | `expandable` | Boolean flag | `1.0` if True, `0.0` if False | Presence of expandable accordion / filter surface |
| `4` | `taskProgress` | $(stepNumber - 1) / totalSteps$ | Bounded in $[0.0, 1.0]$ | Task progress based on step id token and task model (`TASK_STEP_COUNTS: T1=4, T2=2, T3=4`). 0.0 if idle. |
| `5` | `actionAvailability` | $uniqueActions / 7.0$ | Bounded in $[0.0, 1.0]$ | Ratio of currently available DOM actions normalized by `ACTION_VOCABULARY` size. |

---

## 3. Intervention Vocabulary and Target Head Output Contract

- **Owning Files:** `src/intervention_label_policy.py`, `src/training.py` (`TargetInterventionHead`)
- **Dimension:** 5 discrete logits
- **Vocabulary:**
  `0: simplify_options`
  `1: highlight_primary_action`
  `2: offer_assistance`
  `3: expand_tooltip`
  `4: no_op`
- **Methodological Claim:** Labels generated via `intervention_label_policy.py` are scripted outputs from a deterministic rule policy conditioned on observable outcome events and UIContext, and are explicitly bounded under ADR-013.
