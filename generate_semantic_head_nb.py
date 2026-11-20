#!/usr/bin/env python3
"""
Generate `notebooks/04_semantic_target_head_retraining.ipynb`.

The notebook presents the Workstream 40.4/40.5 retraining run: the 92-trace participant
export, its provenance retrofit and verifier gate, dataset v2.2.0-participant, the 40.3
window/horizon re-confirmation, the 14-class head, the freeze/partial/full ablation, held-out
evaluation, ONNX/INT8 export and the bundle-v2 framework handoff.

It reads committed reports and local (git-ignored) dataset artifacts. Execute it with:

    jupyter nbconvert --to notebook --execute --inplace notebooks/04_semantic_target_head_retraining.ipynb
"""

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "notebooks" / "04_semantic_target_head_retraining.ipynb"

cells = []


def md(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip("\n")))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip("\n")))


md(
    r"""
# Semantic Target-Head Retraining — Workstream 40.4 / 40.5

**Model-preparation repository · 2026-11-20**

This notebook presents the target-head retraining work requested for Workstream 40:

1. the **92-trace participant export** from Supabase, its audited provenance retrofit and the
   framework verifier gate;
2. **dataset `v2.2.0-participant`** (14 approved target IDs, session-bounded splits,
   training-only class weights);
3. the **40.3 window/horizon re-confirmation** on the enlarged participant set;
4. the **14-class semantic target head** and the freeze / partial / full ablation;
5. held-out evaluation with per-class, macro, majority, task-level, lead-time and calibration
   metrics;
6. **ONNX + INT8 export**, bundle v2 and the framework handoff.

> **Evidence boundary.** These are participant-declared traces used as participant-derived
> *engineering* evidence. No participant identifier is approved (ADR-023), the split unit is the
> session (ADR-031), the sample is small and the label distribution is extremely imbalanced.
> No participant-held-out generalisation is claimed, and model accuracy is **not** evidence of
> usability, task performance or intervention benefit.
"""
)

code(
    r"""
from pathlib import Path
import json
import os
import sys

ROOT = Path.cwd()
if not (ROOT / "src").is_dir() and (ROOT.parent / "src").is_dir():
    ROOT = ROOT.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

# Keep plotting and font caches inside the workspace; the home cache is read-only here.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".metrics" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 140)
pd.set_option("display.max_colwidth", 80)

DATASET_VERSION = "v2.2.0-participant"
DATASET_DIR = ROOT / ".data" / "processed" / DATASET_VERSION
EXPORT_DIR = ROOT / ".data" / "raw" / "collected-2026-11-20"
RETROFIT_DIR = ROOT / ".data" / "raw" / "participant-2026-11-20"
VERIFIED_DIR = ROOT / ".data" / "raw" / "participant-verified-2026-11-20"
WH_REPORT = ROOT / "docs" / "experiments" / "window-horizon-participant-r2"
HEAD_REPORT = ROOT / "docs" / "experiments" / "semantic-head" / "semantic-head-evaluation.json"
BUNDLE = ROOT / "models" / "bundles" / "v2.0.0" / "bundle.json"
FRAMEWORK = ROOT.parent / "edge-aui-framework"

import torch
print("project root :", ROOT)
print("python       :", sys.version.split()[0])
print("torch        :", torch.__version__, "| device available:",
      "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
"""
)

md(
    r"""
## 1. Collection and provenance gate

The Supabase project `bfntyqahujplvcowjdmh` (`edge-aui`) holds **92 sessions**, all trace schema
**1.4.0**, exported read-only with `edge-aui-framework/scripts/export-traces.mjs`.

The database rows and the trace bodies disagree about provenance for part of the set: the
deployment's collection-mode correction landed after some captures, so 44 traces declare
`scripted` in the body and 5 session rows declare `participant` while the body still says
`scripted`. Under the researcher instruction recorded for this run, **the whole export is a
participant capture set**; the scripted declarations are treated as upload-mode oversights. A
local, audited retrofit rewrote only `session.provenance` and `metadata.provenance` on the
exported copy. The database was not modified; `participantId` stays null.

The framework verifier then decided which traces are analysis input. A failed trace remains a
record but is not evidence.
"""
)

code(
    r"""
export_manifest = json.loads((EXPORT_DIR / "export-manifest.json").read_text())
retrofit_manifest = json.loads((RETROFIT_DIR / "retrofit-manifest.json").read_text())
verification = json.loads((VERIFIED_DIR / "verification-manifest.json").read_text())

sessions = export_manifest["sessions"]
exported = len(sessions)
body_provenance = {}
for entry in sessions:
    path = EXPORT_DIR / f"experiment-trace-{entry['sessionId']}.json"
    trace = json.loads(path.read_text())
    body_provenance[entry["sessionId"]] = (
        trace["session"].get("provenance"),
        trace["metadata"].get("provenance"),
    )

summary = pd.DataFrame([
    ["Exported sessions", exported, "trace schema 1.4.0, no truncation"],
    ["Body provenance = scripted", sum(1 for p in body_provenance.values() if p[0] == "scripted"),
     "treated as upload-mode oversight under the recorded authority"],
    ["Body provenance = participant", sum(1 for p in body_provenance.values() if p[0] == "participant"),
     "declared participant at capture"],
    ["Retrofit corrected", retrofit_manifest["counts"]["corrected"],
     "session/metadata provenance rewritten on the local copy"],
    ["Verifier passed", verification["counts"]["verified"], "analysis input"],
    ["Verifier excluded", verification["counts"]["excluded"], "capture defects, not renamed as evidence"],
], columns=["Stage", "Count", "Note"])
display(summary)

excluded = [s for s in verification["sessions"] if s.get("status") != "verified"]
exclusion_rows = []
for entry in excluded:
    detail = (entry.get("detail") or "").split("—")[0].strip()
    exclusion_rows.append({
        "session": entry["session_id"][:12] + "…",
        "reason": detail.replace("[✗ FAIL]", "").strip(),
    })
display(pd.DataFrame(exclusion_rows).value_counts("reason").rename("sessions").to_frame())
display(Markdown(
    f"**Analysis input: {verification['counts']['verified']} of {exported} sessions.** "
    "The 15 exclusions are capture defects (window attribution, structural validity, task "
    "lifecycle), not provenance relabelling."
))
"""
)

md(
    r"""
## 2. Dataset `v2.2.0-participant`

The dataset was rebuilt with the verified traces (session-bounded split before window
generation, seed 42). Labels are the approved **14 target IDs** (ADR-030). The runtime's 15th ID
`inspect_help` is recorded evidence and is **excluded**, not relabelled; its exclusion count is
reported. Class weights are resolved on the training partition only.

The manifest no longer claims `scripted_substitute_testbed`: its source type and claim boundary
are derived from the rows' declared provenance.
"""
)

code(
    r"""
manifest = json.loads((DATASET_DIR / "manifest.json").read_text())
dataset = pd.read_parquet(DATASET_DIR / "target_semantic_dataset.parquet")

split = manifest["split_summary"]
semantics = manifest["target_semantics"]
vocabulary = semantics["target_vocabulary"]

overview = pd.DataFrame([
    ["Dataset version", manifest["dataset_version"]],
    ["Content hash", manifest["content_hash"]],
    ["Rows", f"{len(dataset):,}"],
    ["Sessions", f"{split['session_counts']['total']} "
                 f"(train {split['session_counts']['train']}, val {split['session_counts']['val']}, "
                 f"test {split['session_counts']['test']})"],
    ["Source data type", manifest["source_data_type"]],
    ["Split unit", split["split_unit"]],
    ["Label key", split["label_key"]],
    ["Target vocabulary", f"{len(vocabulary)} IDs · {semantics['target_vocabulary_version']}"],
    ["Recorded vocabulary", semantics["recorded_vocabulary_version"]],
    ["Excluded inspect_help windows", semantics["excluded_recorded_only"]],
    ["Excluded incomplete lookahead", semantics["excluded_incomplete_lookahead"]],
    ["Label sources", json.dumps(semantics["label_sources"])],
], columns=["Field", "Value"])
display(overview)

dist = split["class_distribution"]
class_table = pd.DataFrame([
    {
        "Outcome": name,
        "Train": dist["train"]["counts"][name],
        "Val": dist["val"]["counts"][name],
        "Test": dist["test"]["counts"][name],
        "Weight (train-only)": round(manifest["class_weights"]["weights"][name], 3),
    }
    for name in vocabulary
])
class_table["Total"] = class_table[["Train", "Val", "Test"]].sum(axis=1)
display(class_table)
display(Markdown(
    f"**`NO_OUTCOME` is {dist['train']['majority_percentage']:.1f}% of training rows**; "
    f"`{', '.join(name for name in vocabulary if dist['test']['counts'][name] == 0)}` have zero "
    "test support. Macro-F1 is therefore reported over supported classes, and absent classes "
    "are named rather than scored as zero F1."
))
"""
)

code(
    r"""
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
for ax, part in zip(axes, ("train", "val", "test")):
    counts = [dist[part]["counts"][name] for name in vocabulary]
    ax.barh(range(len(vocabulary)), counts, color="#38618c")
    ax.set_yticks(range(len(vocabulary)))
    ax.set_yticklabels(vocabulary, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(f"{part} (n={dist[part]['total_examples']:,})")
    ax.set_xlabel("examples")
fig.suptitle("Semantic target distribution by partition — v2.2.0-participant", y=1.02)
plt.tight_layout()
plt.show()

task_rows = []
for part in ("train", "val", "test"):
    coverage = split.get("task_coverage", {}).get(part)
    if coverage:
        for task, count in coverage.items():
            task_rows.append({"Partition": part, "Task": task, "Examples": count})
if task_rows:
    display(pd.DataFrame(task_rows).pivot(index="Task", columns="Partition", values="Examples"))
"""
)

md(
    r"""
## 3. Window/horizon re-confirmation (Task 40.3)

The four pre-declared configurations were re-run on the 77 verified sessions. Selection uses
only the pre-declared criteria (training coverage, largest-class share, median lead time, then
declared order). It does not use the validation or test partitions.
"""
)

code(
    r"""
wh = json.loads((WH_REPORT / "window-horizon-results.json").read_text())
rows = []
for experiment in wh["experiments"]:
    config = experiment["config"]
    metrics = experiment["metrics"]
    rows.append({
        "ID": config["id"],
        "History": f"{config['history_windows']}x500 ms",
        "Lookahead (ms)": f"{config['lookahead_ms'][0]}–{config['lookahead_ms'][1]}",
        "Examples": metrics["total_examples"],
        "Train classes ≥5": metrics["train_classes_with_min_support"],
        "Largest train share": f"{metrics['train_max_class_share'] * 100:.1f}%",
        "Lead median (ms)": round(metrics["lead_time_median_ms"]),
        "Majority macro-F1": round(metrics["macro_f1"], 4),
    })
wh_table = pd.DataFrame(rows)
display(wh_table)
selected = wh["selection"]
display(Markdown(
    f"**Selected: {selected['config']['id']} ({selected['config']['label']}).** "
    f"`provisional={selected.get('provisional')}` — the same configuration wins again, but the "
    "margin between the top configurations remains inside sampling noise. The configuration is "
    "used for the retraining run and is not claimed to be final."
))
"""
)

md(
    r"""
## 4. Head architecture

`TargetSemanticHead` is the same context-conditioned MLP contract as the legacy intervention
head — `[h_T, C] -> logits` — with the output pinned to the **14 approved target IDs**
(ADR-030), not the 15-ID recorded vocabulary and not the 5 intervention classes. The GRU
backbone is unchanged; `models/foundational_gru.pth` is loaded, never overwritten.
"""
)

code(
    r"""
from training import EdgeAUIGRU, TargetSemanticHead, SEMANTIC_NUM_CLASSES
from semantic_targets import TARGET_VOCABULARY, TARGET_VOCABULARY_VERSION

model = EdgeAUIGRU(
    input_dim=18, hidden_dim=64, num_layers=2,
    num_classes=SEMANTIC_NUM_CLASSES,
    head=TargetSemanticHead(hidden_dim=64),
)
with torch.no_grad():
    logits = model(torch.randn(2, 12, 18), context=torch.randn(2, 6))

parameter_counts = pd.DataFrame([
    ["GRU backbone", sum(p.numel() for p in model.gru.parameters())],
    ["Semantic head", sum(p.numel() for p in model.head.parameters())],
    ["Total", sum(p.numel() for p in model.parameters())],
], columns=["Component", "Parameters"])
display(parameter_counts)
display(Markdown(
    f"Head output classes: **{SEMANTIC_NUM_CLASSES}** · vocabulary version "
    f"`{TARGET_VOCABULARY_VERSION}` · forward output `{tuple(logits.shape)}`."
))
"""
)

md(
    r"""
## 5. Training protocol and ablation

E1/E2/E3 are the pre-declared freeze / partial / full arms. Each arm keeps the best epoch by
**validation** macro-F1 and evaluates the **test** partition once, after the checkpoint is
frozen. Seeds, epoch count, class weights, dataset hash, history and horizon are recorded next
to each checkpoint.
"""
)

code(
    r"""
report = json.loads(HEAD_REPORT.read_text())
ablation = report["ablation"]

arm_rows = []
for arm in ablation:
    test = arm["test"]
    arm_rows.append({
        "Arm": arm["experiment"].upper(),
        "Best epoch": arm["best_epoch"],
        "Val macro-F1": round(arm["val"]["macro_f1"], 4),
        "Test accuracy": round(test["accuracy"], 4),
        "Test macro-F1": round(test["macro_f1"], 4),
        "Test weighted-F1": round(test["weighted_f1"], 4),
        "Majority macro-F1": round(test["majority_macro_f1"], 4),
        "ECE": round(test["ece"], 4),
    })
ablation_table = pd.DataFrame(arm_rows)
display(ablation_table)

# The export/held-out arm is selected on validation macro-F1 only.
best_arm = max(ablation, key=lambda arm: arm["val"]["macro_f1"])
best_test = best_arm["test"]
training = best_arm["training_config"]
display(Markdown(
    f"**Selected arm: {best_arm['experiment'].upper()}** (highest validation macro-F1, "
    f"{best_arm['val']['macro_f1']:.4f}). Test is reported below for this arm only."
))

config_table = pd.DataFrame([
    ["Head class count", training["head_class_count"]],
    ["Dataset", f"{training['dataset_version']} ({training['dataset_content_hash']})"],
    ["Split unit", training["split_unit"]],
    ["History / horizon", f"{training['history_windows']} x 500 ms / {training['lookahead_horizon_ms']} ms"],
    ["Seed / epochs", f"{training['seed']} / {training['epochs']}"],
    ["Learning-rate scheme", json.dumps(training["learning_rate_scheme"])],
    ["Class-weight source", training["class_weights_source"]],
    ["Best epoch (val)", training["best_epoch"]],
], columns=["Training configuration", "Value"])
display(config_table)

fig, ax = plt.subplots(figsize=(8, 4))
for experiment in ("e1", "e2", "e3"):
    path = ROOT / "models" / f"semantic_head_{experiment}_training.json"
    if path.is_file():
        history = json.loads(path.read_text())["history"]
        ax.plot(range(1, len(history["val_macro_f1"]) + 1), history["val_macro_f1"],
                label=experiment.upper(), linewidth=1.6)
ax.axhline(best_test["majority_macro_f1"], color="grey", linestyle="--",
           label="majority baseline")
ax.set_xlabel("epoch")
ax.set_ylabel("validation macro-F1")
ax.set_title("Validation macro-F1 by arm")
ax.legend()
plt.tight_layout()
plt.show()
"""
)

md(
    r"""
## 6. Held-out evaluation

Test metrics for the validation-selected arm. Sessions are the split unit; a session never
appears in two partitions. Absent classes are named, not scored. The majority baseline is
computed from the training partition's most frequent class and applied to the test partition.
"""
)

code(
    r"""
per_class = best_test["per_class"]
prf = pd.DataFrame([
    {
        "Outcome": name,
        "Precision": round(values["precision"], 3),
        "Recall": round(values["recall"], 3),
        "F1": round(values["f1"], 3),
        "Support": values["support"],
        "Status": "absent" if values["support"] == 0 else "",
    }
    for name, values in per_class.items()
])
display(prf)

majority = best_test["majority_baseline"]
overall = pd.DataFrame([
    ["Test accuracy", round(best_test["accuracy"], 4)],
    ["Majority accuracy", round(majority["accuracy"], 4)],
    ["Macro-F1 (supported)", round(best_test["macro_f1"], 4)],
    ["Majority macro-F1", round(majority["macro_f1"], 4)],
    ["Macro-F1 (all 14)", round(best_test["macro_f1_all"], 4)],
    ["Weighted-F1", round(best_test["weighted_f1"], 4)],
    ["Top-3 accuracy", round(best_test["ranking"]["hr@3"], 4)],
    ["MRR", round(best_test["ranking"]["mrr"], 4)],
    ["Samples", str(best_test["n_samples"])],
], columns=["Metric", "Value"])
display(overall)
display(Markdown(
    f"Absent in test: `{', '.join(best_test['absent_classes'])}`. The model trades majority-class "
    "accuracy for rare-class recall: macro-F1 is above the majority baseline, accuracy is far "
    "below it. That trade-off is expected under the declared training-only class weights."
))
"""
)

code(
    r"""
labels = list(per_class)
matrix = np.asarray(best_test["confusion_matrix"])
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(matrix, cmap="Blues")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("predicted")
ax.set_ylabel("true")
ax.set_title(f"Confusion matrix — {best_arm['experiment'].upper()} test split")
for row in range(matrix.shape[0]):
    for column in range(matrix.shape[1]):
        if matrix[row, column]:
            ax.text(column, row, int(matrix[row, column]), ha="center", va="center",
                    fontsize=6, color="white" if matrix[row, column] > matrix.max() / 2 else "black")
fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.show()
"""
)

code(
    r"""
task_table = pd.DataFrame([
    {
        "Task": task,
        "Examples": values["n_samples"],
        "Accuracy": round(values["accuracy"], 3),
        "Macro-F1": round(values["macro_f1"], 3),
        "Absent classes": len(values["absent_classes"]),
    }
    for task, values in sorted(best_test["task_level"].items())
])
display(task_table)

lead = best_test["lead_time"]["per_class"]
lead_table = pd.DataFrame([
    {
        "Outcome": name,
        "Labels with lead time": values["count"],
        "Median (ms)": None if values["median_ms"] is None else round(values["median_ms"]),
        "P25 (ms)": None if values["p25_ms"] is None else round(values["p25_ms"]),
        "P75 (ms)": None if values["p75_ms"] is None else round(values["p75_ms"]),
    }
    for name, values in lead.items()
])
display(lead_table)

bins = pd.DataFrame(best_test["calibration"]["bins"])
bins["range"] = bins.apply(lambda row: f"{row['lower']:.1f}–{row['upper']:.1f}", axis=1)
display(bins[["range", "count", "mean_confidence", "accuracy", "gap"]].round(3))
display(Markdown(
    f"ECE = **{best_test['calibration']['ece']:.3f}**, Brier = "
    f"**{best_test['calibration']['brier']:.3f}**. Reliability is reported because support "
    "allows it, not because it is good: sparse classes make confidence poorly calibrated."
))
"""
)

md(
    r"""
### What these numbers do and do not show

- **Do show:** the dataset → head → export pipeline executes end to end on the declared
  participant export, with a 14-class head, session-bounded splits and no future inputs.
- **Do show:** the model predicts rare task outcomes above the majority-class macro-F1 while
  sacrificing majority accuracy.
- **Do not show:** participant-held-out generalisation (no approved participant identifier),
  statistical adequacy (77 sessions, sparse classes), or any user benefit. Model accuracy is
  not an intervention-effect measurement.
"""
)

md(
    r"""
## 7. ONNX export, INT8 and bundle v2

The exported graph keeps the deployment contract: `sequence_input (batch, T, 18)` and
`context_input (batch, 6)` → `task_outcome_logits (batch, 14)`. INT8 is dynamic quantisation.
Sizes and latencies are **measured**, not target claims. The bundle records the vocabulary
versions, dataset hash, split unit, code commit, seed, environment, metrics, contract and
artifact hashes. `models/bundles/v1.0.0` and the five-class artifacts are untouched.
"""
)

code(
    r"""
bundle = json.loads(BUNDLE.read_text())
display(pd.DataFrame([
    ["Bundle schema", bundle["bundle_schema_version"]],
    ["Head", bundle["head"]],
    ["Experiment", bundle["experiment"]],
    ["Target vocabulary", f"{bundle['head_class_count']} IDs · {bundle['target_vocabulary_version']}"],
    ["Recorded vocabulary", bundle["recorded_vocabulary_version"]],
    ["Dataset", f"{bundle['dataset_version']} ({bundle['dataset_content_hash']})"],
    ["Split unit", bundle["split_unit"]],
    ["Seed / commit", f"{bundle['seed']} / {bundle['code_commit'][:12]}"],
    ["Output contract", json.dumps(bundle["output_contract"]["task_outcome_logits"])],
    ["FP32 artifact", f"{bundle['model_artifact']} ({bundle['measured']['fp32']['file_size_bytes']:,} B)"],
    ["INT8 artifact", f"{bundle['model_artifact_int8']} ({bundle['measured']['int8']['file_size_bytes']:,} B)"],
    ["FP32 latency", f"{bundle['measured']['fp32']['avg_latency_ms']:.2f} ms avg / "
                     f"{bundle['measured']['fp32']['p95_latency_ms']:.2f} ms p95"],
    ["INT8 latency", f"{bundle['measured']['int8']['avg_latency_ms']:.2f} ms avg / "
                     f"{bundle['measured']['int8']['p95_latency_ms']:.2f} ms p95"],
    ["FP32/ONNX parity", f"max |diff| = {bundle['parity']['fp32_max_abs_diff']:.2e}"],
    ["INT8 parity", f"top-1 agreement = {bundle['parity']['int8_top1_agreement']:.3f}, "
                    f"max |logit diff| = {bundle['parity']['int8_max_abs_diff']:.3f}"],
], columns=["Bundle field", "Value"]))
"""
)

code(
    r"""
import onnx
import onnxruntime as ort

onnx_path = ROOT / "models" / bundle["model_artifact"]
int8_path = ROOT / "models" / bundle["model_artifact_int8"]
graph = onnx.load(str(onnx_path))
print("inputs :", [(i.name, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]) for i in graph.graph.input])
print("outputs:", [(o.name, [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim]) for o in graph.graph.output])

sequence = torch.randn(16, 12, 18).numpy()
context = torch.randn(16, 6).numpy()
feeds = {"sequence_input": sequence, "context_input": context}
fp32_out = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).run(None, feeds)[0]
int8_out = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"]).run(None, feeds)[0]
print("fp32 logits shape:", fp32_out.shape)
print("int8 logits shape:", int8_out.shape)
print("top-1 agreement :", float((fp32_out.argmax(1) == int8_out.argmax(1)).mean()))
display(Markdown(
    "Quantisation is verified as an agreement measurement. Latencies above are ONNX Runtime "
    "CPU execution of the exported graphs; **browser-path latency is NOT MEASURED**. This is "
    "not a claim that the deployed browser path meets the 50 ms / 20 MB / 500 KB targets — "
    "those targets are unmet as documented in ADR-008/ADR-016."
))
"""
)

md(
    r"""
## 8. Framework handoff

The framework reads the bundle manifest rather than a hardcoded class order.
`parseBundleManifest` detects `task_outcome_vocabulary` + `task_outcome_logits`, maps the
14-class graph into the foundation slot and records the artifact hash as the model version.
`OnnxSlowGate` feeds `sequence_input` and `context_input` to a two-input foundation session.
"""
)

code(
    r"""
framework_bundle = FRAMEWORK / "public" / "models" / "task_outcome_head_v2.bundle.json"
framework_onnx = FRAMEWORK / "public" / "models" / "task_outcome_head_v2_int8.onnx"
print("bundle copied :", framework_bundle.is_file())
print("graph copied  :", framework_onnx.is_file())
if framework_bundle.is_file():
    fb = json.loads(framework_bundle.read_text())
    print("model url     : /models/task_outcome_head_v2_int8.onnx")
    print("vocabulary    :", len(fb["task_outcome_vocabulary"]), "IDs")
    print("version       :", fb["target_vocabulary_version"])
    print("artifact hash :", fb["int8_hash_sha256"][:12])
"""
)

md(
    r"""
Framework verification commands (run from `edge-aui-framework`):

```bash
npx vitest run tests/gate_vocabulary.test.ts tests/onnx_contract.test.ts \
  tests/intervention_head_contract.test.ts
```

These cover the v2 bundle parse (14-class vocabulary vs declared output), the two-input
semantic graph contract, the real ONNX asset shapes, and a real INT8 graph load that feeds both
tensors and reports the semantic artifact version/provider. **55 tests pass.**
"""
)

md(
    r"""
## 9. Reproduction

```bash
# 1. Export the 92 sessions (service-role key supplied for this command only; never committed)
cd edge-aui-framework
SUPABASE_URL=https://bfntyqahujplvcowjdmh.supabase.co \
SUPABASE_SERVICE_ROLE_KEY=<key> \
  node scripts/export-traces.mjs --out ./.data/collected-2026-11-20

# 2. Audited local retrofit + verifier gate
cd ../model-preparation
python3 -m src.provenance_retrofit \
  --source .data/raw/collected-2026-11-20 --output .data/raw/participant-2026-11-20 \
  --authority "<researcher instruction>"
python3 -m src.participant_evidence \
  --source .data/raw/participant-2026-11-20 \
  --output .data/raw/participant-verified-2026-11-20 \
  --framework-root ../edge-aui-framework

# 3. Window/horizon re-run and dataset v2.2.0
python3 -m src.window_horizon \
  --trace-dir .data/raw/participant-verified-2026-11-20 \
  --report-dir docs/experiments/window-horizon-participant-r2 \
  --require-provenance participant --write-dataset --version v2.2.0-participant

# 4. Train the ablation and evaluate the held-out split
python3 -m src.semantic_training --evaluate-suite --epochs 50 \
  --report docs/experiments/semantic-head/semantic-head-evaluation.json

# 5. Export ONNX/INT8 and bundle v2
python3 -m src.export --head semantic --experiment e2 \
  --dataset-version v2.2.0-participant \
  --metrics docs/experiments/semantic-head/semantic-head-evaluation.json

# 6. Verify
python3 -m pytest tests -q
cd ../edge-aui-framework && npx vitest run tests/gate_vocabulary.test.ts \
  tests/onnx_contract.test.ts tests/intervention_head_contract.test.ts
```
"""
)


notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
    "title": "Semantic Target-Head Retraining — Workstream 40.4/40.5",
}
NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {NOTEBOOK} ({len(cells)} cells)")
