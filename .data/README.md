---
license: "cc-by-4.0"
task_categories:
  - time-series-classification
  - sequence-modeling
tags:
  - micro-interactions
  - UI-adaptation
  - behavioral-modeling
  - human-computer-interaction
---

# Dataset Card: Edge-Native Adaptive UI Behavioral Logs

## Dataset Provenance and Attribution

This dataset aggregates multiple open-source behavioral logs to support the project: **IMPLEMENTING A LIGHTWEIGHT AI-ASSISTED FRAMEWORK FOR BEHAVIOURAL PATTERN EXTRACTION AND ADAPTIVE UI RECOMMENDATIONS ON THE WEB**. 

The dataset integrates the following public research datasets, all of which are properly cited and attributed to their original authors:

* **Client-Side Action Paths:** Ou, C., Buschek, D., Eiband, M., & Butz, A. (2021). *This dataset provides granular client-side action paths necessary for modeling sequential user intent.*
* **Structured Human-Machine Interaction Logs:** Carrera-Rivera et al. (2023). *This dataset provides logs of structured interactions to help identify frequent macro-interactions and high-level behavioral patterns.*
* **Continuous Kinematics:** Leiva, L. A., & Arapakis, I. (2020). *Provides continuous cursor kinematics and physical movement features to understand low-level motor behaviors.*
* **High-Volume Trajectories:** Mendeley Mouse Dynamics (2026). *Provides high-volume cursor trajectory data for robust training of continuous behavioral sequences.*

## Multi-Tiered Transfer Learning Strategy

These datasets are specifically structured to support our multi-tiered transfer learning strategy for training a lightweight Gated Recurrent Unit (GRU):

1. **Foundational Pre-training:** The model first learns cross-domain human motor behaviors (kinematic features) using the large-scale public datasets (e.g., Continuous Kinematics by Leiva & Arapakis, and High-Volume Trajectories from Mendeley). This phase allows the foundational recurrent layers of the GRU to understand the underlying physics and biomechanics of human cursor movement, independent of any specific user interface.
2. **Target UI Fine-tuning:** The model's classification head is then fine-tuned on localized logs from a target testbed (such as a Security Operations Center dashboard). By combining the pre-trained kinematic understanding with specific structured interactions (like the Client-Side Action Paths and Human-Machine Interaction Logs), the model can accurately map physical movements to specific, localized UI outcomes. This step resolves the Out-of-Vocabulary (OOV) target action problem caused by domain mismatch.
