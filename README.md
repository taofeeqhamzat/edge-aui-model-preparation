# Edge-Native Adaptive UI: Model Preparation

This repository contains the data engineering pipeline and model training scripts for the Edge-Native Adaptive User Interface (AUI) framework.

## 1. Purpose

The code in this repository trains a lightweight Gated Recurrent Unit (GRU). The GRU operates as a behavioral-state encoder. It learns from continuous micro-interaction streams to predict observable user interface actions.

## 2. References

For full details on the architecture, datasets, and edge deployment, read the referenced repositories:

- **Framework Repository:** [taofeeqhamzat/edge-aui](https://github.com/taofeeqhamzat/edge-aui)
- **Dataset Repository:** [T40/edge-aui-framework-data](https://huggingface.co/datasets/T40/edge-aui-framework-data)

Please refer to the `AGENTS.md` file and `docs/` folder in this repository for pipeline instructions and architectural guidelines.
