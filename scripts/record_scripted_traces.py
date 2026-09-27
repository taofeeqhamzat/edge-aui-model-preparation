#!/usr/bin/env python3
"""
Scripted Testbed Trace Recording Harness (Phase B, Task 10.1).

Executes scripted, repeatable runs of the edge-aui-framework testbed tasks (T1, T2, T3)
in both 'baseline' and 'adaptive' conditions, exports traces via the real in-browser
trace recorder, verifies each trace against the framework's completeness verifier,
and compiles a provenance manifest.

ADR-013 Claim Boundary:
Results from this dataset may be reported as:
- "the dataset -> preparation -> training -> export -> runtime path executes end to end";
- "the learned head loads in the browser and produces logits of the expected shape";
- engineering and pipeline findings.

Results from this dataset may NOT be reported as:
- evidence about human participants;
- evidence about usability, task performance, or intervention benefit;
- evidence that learned intervention prediction outperforms the deterministic policy.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ADR_013_CLAIM_BOUNDARY = (
    "Results from this dataset may be reported as: "
    "'the dataset -> preparation -> training -> export -> runtime path executes end to end', "
    "'the learned head loads in the browser and produces logits of the expected shape', and "
    "engineering and pipeline findings. "
    "Results from this dataset may NOT be reported as: "
    "evidence about human participants; "
    "evidence about usability, task performance, or intervention benefit; or "
    "evidence that learned intervention prediction outperforms the deterministic policy — "
    "a model trained on labels produced by that policy is being compared against its own teacher."
)

JS_RUNNER_SCRIPT = r"""
(async (taskId, conditionId) => {
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  const moveMouse = async (fromX, fromY, toX, toY, steps = 10, delayMs = 25) => {
    for (let i = 1; i <= steps; i++) {
      const curX = Math.round(fromX + ((toX - fromX) * i) / steps);
      const curY = Math.round(fromY + ((toY - fromY) * i) / steps);
      window.dispatchEvent(
        new MouseEvent('mousemove', {
          bubbles: true,
          clientX: curX,
          clientY: curY
        })
      );
      await sleep(delayMs);
    }
  };

  const clickEl = async (selector, dwellMs = 200) => {
    const el = document.querySelector(selector);
    if (!el) throw new Error(`Element not found for selector: ${selector}`);
    const rect = el.getBoundingClientRect();
    const cx = Math.round(rect.left + rect.width / 2);
    const cy = Math.round(rect.top + rect.height / 2);

    el.dispatchEvent(new MouseEvent('mouseover', { bubbles: true, clientX: cx, clientY: cy }));
    window.dispatchEvent(
      new MouseEvent('mousemove', {
        bubbles: true,
        clientX: cx,
        clientY: cy
      })
    );
    await sleep(dwellMs);

    if (dwellMs >= 300) {
      el.dispatchEvent(new MouseEvent('mouseover', { bubbles: true, clientX: cx, clientY: cy }));
      await sleep(50);
    }

    el.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, clientX: cx, clientY: cy }));
    await sleep(50);
    el.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, clientX: cx, clientY: cy }));
    el.dispatchEvent(new MouseEvent('click', { bubbles: true, clientX: cx, clientY: cy }));
    return { cx, cy };
  };

  // 1. Wait for runtime diagnostics
  let attempts = 0;
  while (!window.__EDGE_AUI__ && attempts < 50) {
    await sleep(100);
    attempts++;
  }
  const h = window.__EDGE_AUI__;
  if (!h) throw new Error('window.__EDGE_AUI__ not found after 5s');

  // 2. Start a clean trial with discrete session ID and clean buffer
  if (typeof h.startNewTrial === 'function') {
    await h.startNewTrial(conditionId);
    await sleep(300);
  } else {
    const condBtn = document.querySelector(`[data-aui-component="condition-${conditionId}"]`);
    if (condBtn && condBtn.getAttribute('aria-checked') !== 'true') {
      await clickEl(`[data-aui-component="condition-${conditionId}"]`, 100);
      await sleep(500);
    }
  }

  // 3. Start task trial
  let pos = await clickEl(`[data-aui-component="btn-start-${taskId}"]`, 100);
  await sleep(400);

  // 4. Execute task steps
  if (taskId === 'T1') {
    // Step T1-1: Navigate to Analytics
    pos = await clickEl('[data-aui-component="nav-Analytics"]', 200);
    await sleep(400);

    // Dwell over filter early in task (taskProgress < 0.5 -> expand_tooltip)
    await moveMouse(pos.cx, pos.cy, 300, 350, 15, 30);
    await sleep(600);
    const filterBtn = document.querySelector('[data-aui-component="filter-Region"]');
    if (filterBtn) {
      filterBtn.dispatchEvent(new MouseEvent('mouseover', { bubbles: true, clientX: 300, clientY: 350 }));
      await sleep(150);
      filterBtn.dispatchEvent(new MouseEvent('mouseover', { bubbles: true, clientX: 300, clientY: 350 }));
    }
    await sleep(600);

    // Step T1-2: Open Region Filter
    pos = await clickEl('[data-aui-component="filter-Region"]', 450);
    await sleep(500);

    // Step T1-3: Select Region
    const selectEl = document.querySelector('[data-aui-component="filter-Region-select"]');
    if (selectEl) {
      const sRect = selectEl.getBoundingClientRect();
      const sx = Math.round(sRect.left + sRect.width / 2);
      const sy = Math.round(sRect.top + sRect.height / 2);
      await moveMouse(pos.cx, pos.cy, sx, sy, 8, 25);
      selectEl.dispatchEvent(new MouseEvent('mouseover', { bubbles: true, clientX: sx, clientY: sy }));
      await sleep(350);
      selectEl.value = 'EMEA';
      selectEl.dispatchEvent(new Event('change', { bubbles: true }));
      pos = { cx: sx, cy: sy };
    }
    await sleep(500);

    // Step T1-4: Apply Filters (taskProgress >= 0.5 -> highlight_primary_action)
    await moveMouse(pos.cx, pos.cy, 400, 500, 10, 25);
    const applyBtn = document.querySelector('[data-aui-component="btn-apply-filters"]');
    if (applyBtn) {
      applyBtn.dispatchEvent(new MouseEvent('mouseover', { bubbles: true, clientX: 400, clientY: 500 }));
      await sleep(150);
      applyBtn.dispatchEvent(new MouseEvent('mouseover', { bubbles: true, clientX: 400, clientY: 500 }));
    }
    await sleep(600);
    pos = await clickEl('[data-aui-component="btn-apply-filters"]', 450);
    await sleep(500);
  } else if (taskId === 'T2') {
    // Step T2-1: Navigate to Analytics
    pos = await clickEl('[data-aui-component="nav-Analytics"]', 200);
    await sleep(400);

    // Browse table with cursor movement for ~2.5s (allows window buffer to reach 8+ windows)
    await moveMouse(pos.cx, pos.cy, 500, 300, 18, 40);
    await sleep(500);
    await moveMouse(500, 300, 600, 400, 15, 35);
    await sleep(400);

    // Rapid scroll over table (triggers RAPID_SCROLL -> simplify_options)
    for (let w = 0; w < 6; w++) {
      window.dispatchEvent(new WheelEvent('wheel', { bubbles: true, deltaY: 150, clientX: 600, clientY: 400 }));
      await sleep(50);
    }
    await sleep(600);

    // Step T2-2: Export Report
    await moveMouse(pos.cx, pos.cy, 700, 250, 10, 25);
    pos = await clickEl('[data-aui-component="btn-export"]', 350);
    await sleep(500);
  } else if (taskId === 'T3') {
    // Step T3-1: Navigate to Analytics
    pos = await clickEl('[data-aui-component="nav-Analytics"]', 200);
    await sleep(400);

    // Step T3-2: Open Product Category
    await moveMouse(pos.cx, pos.cy, 300, 400, 8, 25);
    pos = await clickEl('[data-aui-component="filter-Product Category"]', 350);
    await sleep(500);

    // Backtrack hesitation: focus date filter then blur away (triggers BACKTRACK -> offer_assistance)
    await moveMouse(pos.cx, pos.cy, 250, 200, 10, 30);
    const dateInput = document.querySelector('[data-aui-component="filter-date-input"]');
    if (dateInput) {
      dateInput.dispatchEvent(new FocusEvent('focus', { bubbles: true }));
      await sleep(200);
      dateInput.dispatchEvent(new FocusEvent('blur', { bubbles: true }));
      await sleep(300);
    }

    // Step T3-3: Change Segment
    const segSelect = document.querySelector('[data-aui-component="filter-segment-select"]');
    if (segSelect) {
      const r = segSelect.getBoundingClientRect();
      const sx = Math.round(r.left + r.width / 2);
      const sy = Math.round(r.top + r.height / 2);
      await moveMouse(pos.cx, pos.cy, sx, sy, 8, 25);
      await sleep(300);
      segSelect.value = 'Enterprise';
      segSelect.dispatchEvent(new Event('change', { bubbles: true }));
      pos = { cx: sx, cy: sy };
    }
    await sleep(500);

    // Step T3-4: Apply Filters
    await moveMouse(pos.cx, pos.cy, 400, 500, 10, 25);
    pos = await clickEl('[data-aui-component="btn-apply-filters"]', 300);
    await sleep(500);
  }

  // 5. Allow lookahead horizon (1500ms) and delayed window settlement (250ms) to complete
  await sleep(1800);

  // 6. Signal stream termination so any trailing edge window settles
  if (typeof h.flush === 'function') {
    h.flush();
  }
  await sleep(200);

  // 7. Return exported trace string
  return h.trace();
})
"""


def get_git_commit(cwd: Path) -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


def check_port_in_use(port: int) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://localhost:{port}/", timeout=1.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def ensure_page_ready(session_name: str, target_url: str) -> None:
    try:
        res = subprocess.run(
            ["agent-browser", "--session", session_name, "eval", "window.__EDGE_AUI__ ? 1 : 0"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0 and res.stdout.strip() == "1":
            return
    except Exception:
        pass

    # Reopen target url if not ready or on error
    subprocess.run(["agent-browser", "--session", session_name, "open", target_url], check=True)
    time.sleep(1.5)


def run_agent_browser_eval(session_name: str, task_id: str, condition_id: str, target_url: str) -> dict:
    ensure_page_ready(session_name, target_url)

    eval_call = f"({JS_RUNNER_SCRIPT})('{task_id}', '{condition_id}')"
    cmd = [
        "agent-browser",
        "--session",
        session_name,
        "eval",
        eval_call,
    ]

    for attempt in range(2):
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            break
        if attempt == 0:
            print(f"       [WARN] agent-browser eval failed ({res.stderr.strip()}), re-opening page and retrying...")
            ensure_page_ready(session_name, target_url)
            time.sleep(1.0)
        else:
            raise RuntimeError(f"agent-browser eval failed after retry: {res.stderr}\n{res.stdout}")

    output = res.stdout.strip()
    # The output of agent-browser eval is a JSON string of the returned value
    try:
        raw_json_str = json.loads(output)
        if isinstance(raw_json_str, str):
            return json.loads(raw_json_str)
        return raw_json_str
    except Exception as e:
        raise ValueError(f"Failed to parse trace JSON from agent-browser output: {e}\nOutput was: {output[:300]}...")


def verify_trace_with_framework(verify_script: Path, trace_file: Path) -> bool:
    cmd = ["node", str(verify_script), str(trace_file)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERROR] Trace verification failed for {trace_file.name}:\n{res.stdout}\n{res.stderr}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Record scripted testbed traces for Phase B.")
    parser.add_argument("--trials-per-condition", type=int, default=3, help="Number of trials per task per condition")
    parser.add_argument("--output", type=str, default=".data/raw/scripted", help="Output directory for traces")
    parser.add_argument("--port", type=int, default=5199, help="Vite server port")
    parser.add_argument(
        "--framework-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent.parent / "edge-aui-framework"),
        help="Path to edge-aui-framework directory",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    framework_dir = Path(args.framework_dir).resolve()
    output_dir = repo_root / args.output if not os.path.isabs(args.output) else Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    verify_script = framework_dir / "scripts" / "verify-trace.mjs"
    if not verify_script.exists():
        print(f"Error: Verifier script not found at {verify_script}")
        sys.exit(1)

    framework_commit = get_git_commit(framework_dir)
    model_prep_commit = get_git_commit(repo_root)

    print("=" * 70)
    print("Scripted Testbed Trace Recording Harness")
    print(f"Framework Dir:      {framework_dir}")
    print(f"Framework Commit:   {framework_commit}")
    print(f"Model-Prep Commit:  {model_prep_commit}")
    print(f"Output Directory:   {output_dir}")
    print(f"Trials per Task:    {args.trials_per_condition}")
    print("Conditions:         ['baseline', 'adaptive']")
    print("Tasks:              ['T1', 'T2', 'T3']")
    print("=" * 70)

    # Check or start Vite server
    server_process = None
    started_server = False
    if not check_port_in_use(args.port):
        print(f"Starting Vite dev server on port {args.port}...")
        server_process = subprocess.Popen(
            ["npx", "vite", "--port", str(args.port)],
            cwd=framework_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        started_server = True
        # Wait for server to come up
        for _ in range(30):
            time.sleep(0.5)
            if check_port_in_use(args.port):
                break
        else:
            print("Error: Timed out waiting for Vite server to start.")
            if server_process:
                server_process.kill()
            sys.exit(1)
        print("Vite server ready.")
    else:
        print(f"Reusing existing server on port {args.port}.")

    session_name = "scripted-rec"
    url = f"http://localhost:{args.port}/?auiDiagnostics=1"

    # Open browser session
    print(f"Opening browser session '{session_name}' at {url}...")
    subprocess.run(["agent-browser", "--session", session_name, "open", url], check=True)
    time.sleep(2.0)

    # Get browser version info
    browser_ver_res = subprocess.run(["agent-browser", "--version"], capture_output=True, text=True)
    browser_ver = browser_ver_res.stdout.strip() if browser_ver_res.returncode == 0 else "agent-browser"

    recorded_traces = []
    conditions = ["baseline", "adaptive"]
    tasks = ["T1", "T2", "T3"]

    total_runs = len(conditions) * len(tasks) * args.trials_per_condition
    run_idx = 0

    try:
        for condition in conditions:
            for task_id in tasks:
                for trial in range(1, args.trials_per_condition + 1):
                    run_idx += 1
                    print(f"[{run_idx}/{total_runs}] Recording condition='{condition}', task='{task_id}', trial={trial}...")

                    trace_data = run_agent_browser_eval(session_name, task_id, condition, url)

                    session_id = trace_data.get("session", {}).get("sessionId", f"unknown-{run_idx}")
                    timestamp_ms = int(time.time() * 1000)
                    filename = f"experiment-trace-{session_id}-{timestamp_ms}.json"
                    trace_path = output_dir / filename

                    with open(trace_path, "w", encoding="utf-8") as f:
                        json.dump(trace_data, f, indent=2)

                    # Verify trace
                    valid = verify_trace_with_framework(verify_script, trace_path)
                    if not valid:
                        print(f"[FATAL] Trace {filename} failed verification. Aborting.")
                        sys.exit(1)

                    metadata = trace_data.get("metadata", {})
                    task_info = trace_data.get("task", {})
                    entry = {
                        "filename": filename,
                        "sessionId": session_id,
                        "experimentId": trace_data.get("session", {}).get("experimentId"),
                        "conditionId": condition,
                        "taskId": task_id,
                        "trial": trial,
                        "status": task_info.get("status", metadata.get("finalTaskStatus")),
                        "durationMs": metadata.get("durationMs"),
                        "behaviourCount": metadata.get("behaviourCount"),
                        "microTensorCount": metadata.get("microTensorCount"),
                        "macroCount": metadata.get("macroCount"),
                        "outcomeCount": metadata.get("outcomeCount"),
                        "predictionCount": metadata.get("predictionCount"),
                        "interventionCount": metadata.get("interventionCount"),
                        "verified": True,
                    }
                    recorded_traces.append(entry)
                    print(f"       -> Saved & Verified: {filename} ({metadata.get('microTensorCount')} windows, {metadata.get('outcomeCount')} outcomes)")

        # Generate manifest
        manifest = {
            "schemaVersion": "1.0.0",
            "data_source": "scripted_testbed",
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "framework_commit": framework_commit,
            "model_preparation_commit": model_prep_commit,
            "browser_version": browser_ver,
            "os_environment": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "trials_per_task": args.trials_per_condition,
            "conditions": conditions,
            "tasks": tasks,
            "total_traces": len(recorded_traces),
            "claim_boundary": ADR_013_CLAIM_BOUNDARY,
            "traces": recorded_traces,
        }

        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print("\n" + "=" * 70)
        print(f"Recording Complete. Manifest written to {manifest_path}")
        print(f"Total Traces Recorded: {len(recorded_traces)}")
        print("All traces passed verification successfully.")
        print("=" * 70)

        # Generate markdown documentation
        doc_path = repo_root / "docs" / "experiments" / "scripted-recording.md"
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        generate_markdown_report(doc_path, manifest)
        print(f"Documentation report written to {doc_path}")

    finally:
        # Close agent-browser session
        try:
            subprocess.run(["agent-browser", "--session", session_name, "close"], capture_output=True)
        except Exception:
            pass

        # Stop Vite server if we started it
        if started_server and server_process:
            print("Stopping spawned Vite server...")
            server_process.terminate()
            server_process.wait()


def generate_markdown_report(doc_path: Path, manifest: dict):
    md_content = f"""# Scripted Testbed Trace Recording Report

**Phase:** B — Substitute Data and Labels (Task 10.1)  
**Recorded At:** {manifest['recorded_at']}  
**Data Source:** `{manifest['data_source']}`  
**Framework Commit:** `{manifest['framework_commit']}`  
**Model-Preparation Commit:** `{manifest['model_preparation_commit']}`  
**Browser / Automation:** `{manifest['browser_version']}`  
**Environment:** `{manifest['os_environment']}`  

---

## 1. ADR-013 Mandatory Claim Boundary

> {manifest['claim_boundary']}

---

## 2. Experimental Execution Summary

- **Conditions Evaluated:** {", ".join(manifest['conditions'])}
- **Tasks Evaluated:** {", ".join(manifest['tasks'])}
- **Trials per Condition/Task:** {manifest['trials_per_task']}
- **Total Validated Traces:** {manifest['total_traces']}
- **Verification Result:** All {manifest['total_traces']} traces passed `scripts/verify-trace.mjs` with 100% check pass rate.

### Recorded Trace Manifest

| Filename | Condition | Task | Trial | Status | Duration (ms) | Windows | Outcomes | Interventions |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for t in manifest["traces"]:
        md_content += (
            f"| `{t['filename']}` | {t['conditionId']} | {t['taskId']} | {t['trial']} | "
            f"{t['status']} | {t['durationMs']} | {t['microTensorCount']} | {t['outcomeCount']} | {t['interventionCount']} |\n"
        )

    md_content += """
---

## 3. Provenance & Ingestion Verification

Every recorded trace conforms to `ExperimentTrace schemaVersion 1.1.0` and satisfies:
1. Complete task lifecycle (`task_start`, all intermediate `task_step` entries, and `task_complete`).
2. Exact 1-to-1 mapping from emitted `microTensor` windows to derived `outcome` records.
3. Baseline condition zero-mutation invariant (0 applied interventions in baseline trials).
4. Non-empty correlation IDs across sessions, windows, predictions, and intervention episodes.
"""

    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(md_content)


if __name__ == "__main__":
    main()
