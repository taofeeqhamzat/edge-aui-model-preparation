# Edge-Native Adaptive UI: Model Preparation

This repository contains the data engineering pipeline and model training scripts for the Edge-Native Adaptive User Interface (AUI) framework.

## 1. Purpose

The code in this repository trains a lightweight Gated Recurrent Unit (GRU). The GRU operates as a behavioral-state encoder. It learns from continuous micro-interaction streams to predict observable user interface actions.

## 2. References

For full details on the architecture, datasets, and edge deployment, read the referenced repositories:

- **Framework Repository:** [taofeeqhamzat/edge-aui](https://github.com/taofeeqhamzat/edge-aui)
- **Dataset Repository:** [T40/edge-aui-framework-data](https://huggingface.co/datasets/T40/edge-aui-framework-data)

Please refer to the `AGENTS.md` file and `docs/` folder in this repository for pipeline instructions and architectural guidelines.

## 3. Planning

- [Plan 3 — semantic task-outcome model preparation](docs/plan/3/next-milestone-model-preparation-plan.md)
  covers the next milestone's ingestion, semantic target dataset, window/horizon experiment,
  target-head retraining and ONNX/bundle handoff.
- [Decision dependencies](docs/plan/3/decision-dependencies.md) records the framework ADRs that
  constrain this repository.

## 4. Semantic target-head retraining (Workstream 40.4–40.5)

```bash
# evaluate the 14-class head (E1/E2/E3 freeze / partial / full ablation)
python3 -m src.semantic_training --evaluate-suite --epochs 50 \
  --report docs/experiments/semantic-head/semantic-head-evaluation.json

# export ONNX + INT8 and write bundle v2
python3 -m src.export --head semantic --experiment e2 \
  --dataset-version v2.2.0-participant \
  --metrics docs/experiments/semantic-head/semantic-head-evaluation.json
```

- Report: [`docs/experiments/semantic-head/semantic-head-report.md`](docs/experiments/semantic-head/semantic-head-report.md)
- Presentation notebook:
  [`notebooks/04_semantic_target_head_retraining.ipynb`](notebooks/04_semantic_target_head_retraining.ipynb)
- Participant evidence and verifier gate:
  [`docs/experiments/participant-evidence/README.md`](docs/experiments/participant-evidence/README.md)
