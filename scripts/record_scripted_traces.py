#!/usr/bin/env python3
"""
Scripted TD1–TD4 Trace Recording Harness (Task 30.1).

Drives the approved testbed task set (TD1–TD4) in the real browser runtime under both
`baseline` and `adaptive` conditions, exports the runtime-produced ExperimentTrace for each
finished trial, verifies it with the framework's verifier, and writes a provenance manifest.

What these captures are
-----------------------
Scripted runs are engineering evidence only. Every trace is `provenance: scripted`, and the
filenames and the manifest say so. They must never be described as, or pooled with,
participant evidence (ADR-013, ADR-018).

What each run does
------------------
- selects the condition through the trial controls (which rebuilds the runtime);
- starts the task through its `btn-start-<task>` control, never by calling task-manager
  internals, so the trial runs through the same path a participant's trial runs through;
- performs the canonical action sequence with real DOM events (pointer, mouse, click,
  input/change, focus), paced so each act lands in its own MicroTensor window;
- waits on observed state after every step (page, disclosure, value, row count) instead of
  fixed sleeps alone;
- fails loudly if an element or expected state is missing, and exits non-zero if any trial
  does not complete with the expected evidence;
- asks the diagnostics handle for the retained export of the finished trial, because
  completion rotates the session and clears the live recorder.

Claim boundary
--------------
See ADR_013_CLAIM_BOUNDARY in the manifest and in the generated report.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
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

PROVENANCE_NOTICE = (
    "Every trace produced by this harness is scripted engineering evidence. It is not "
    "participant evidence and must not be pooled with participant traces."
)

SCHEMA_VERSION = "1.4.0"

# ---------------------------------------------------------------------------
# Task definitions exercised by the driver.
#
# These mirror `edge-aui-framework/src/testbed/tasks/taskModel.ts` (definition version
# `TD-v2`). The driver keeps its own copy because it lives in the other repository; the
# step ids below are asserted against the recorded `taskEvents`, so a drift in the task model
# makes the run fail instead of producing a quietly different trial.
# ---------------------------------------------------------------------------
TASKS = {
    "TD1": {
        "starting_page": "Overview",
        "steps": ["TD1-1", "TD1-2", "TD1-3", "TD1-4", "TD1-5"],
        "required_semantic_outcomes": [
            "open_filter_section",
            "select_region",
            "apply_filter",
            "export_report",
        ],
        "ui": {"region": "Europe", "visibleRowCount": 18, "empty": False},
    },
    "TD2": {
        "starting_page": "Analytics",
        "steps": ["TD2-1", "TD2-2", "TD2-3", "TD2-4", "TD2-5"],
        "required_semantic_outcomes": ["select_date_range", "select_category", "apply_filter"],
        "ui": {
            "dateFrom": "2025-06-01",
            "categories": ["Books"],
            "visibleRowCount": 5,
            "empty": False,
        },
    },
    "TD3": {
        "starting_page": "Analytics",
        "steps": [f"TD3-{index}" for index in range(1, 9)],
        "required_semantic_outcomes": ["apply_filter", "reset_filter"],
        "ui": {
            "dateFrom": "",
            "region": "All",
            "categories": [],
            "visibleRowCount": 50,
            "empty": False,
        },
    },
    "TD4": {
        "starting_page": "Reports",
        "steps": ["TD4-1", "TD4-2", "TD4-3", "TD4-4"],
        "required_semantic_outcomes": ["select_report", "export_report"],
        "ui": {
            "region": "Europe",
            "categories": ["Books"],
            "visibleRowCount": 6,
            "empty": False,
            "currentReportId": "eu-books",
            "visitedReportIds": ["eu-electronics", "eu-books"],
        },
    },
}

CONDITIONS = ["baseline", "adaptive"]

# ---------------------------------------------------------------------------
# In-page runner.
#
# One async function drives one trial. It receives the task id and condition as arguments;
# nothing in the body depends on Python string interpolation, so quoting cannot corrupt it.
#
# `WINDOW_SETTLE_MS` is derived from the pipeline contract, not guessed: a 500 ms window
# closes at most 500 ms after an action, is emitted after the 250 ms settlement delay, and the
# tick loop runs every 125 ms. Waiting 900 ms therefore guarantees the action's window has
# been emitted, which is what makes it a separate labelled window instead of being merged with
# the next action.
# ---------------------------------------------------------------------------
JS_RUNNER = r"""
(async (taskId, conditionId) => {
  const WINDOW_SETTLE_MS = 900;
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const diagnostics = () => {
    const handle = window.__EDGE_AUI__;
    if (!handle) throw new Error('window.__EDGE_AUI__ is missing; is the page open with ?auiDiagnostics=1?');
    return handle;
  };
  const waitFor = async (predicate, label, timeoutMs = 15000, intervalMs = 50) => {
    const deadline = Date.now() + timeoutMs;
    let lastError;
    while (Date.now() < deadline) {
      try {
        const value = predicate();
        if (value) return value;
      } catch (err) {
        lastError = err;
      }
      await sleep(intervalMs);
    }
    throw new Error(
      `[${taskId}] Timed out waiting for ${label}` + (lastError ? ` (${lastError.message})` : '')
    );
  };
  // Every control is addressed by its `data-aui-component` id, which is the identifier the
  // task model and the semantic-outcome rules use. Building the selector here means a task
  // path cannot accidentally pass a raw id where a selector was expected.
  const selectorFor = (componentId) => `[data-aui-component="${componentId}"]`;
  const must = (componentId) => {
    const element = document.querySelector(selectorFor(componentId));
    if (!element) throw new Error(`[${taskId}] Missing required element: ${componentId}`);
    return element;
  };
  const settleWindow = () => sleep(WINDOW_SETTLE_MS);

  // Real DOM events, as the framework's observer sees a participant's input: the pointer
  // travels to the control, then mouseover/mousedown/mouseup/click are dispatched with
  // coordinates. Canonical events are never injected directly.
  const click = async (componentId) => {
    const element = must(componentId);
    element.scrollIntoView({ block: 'center' });
    const rect = element.getBoundingClientRect();
    const cx = Math.round(rect.left + rect.width / 2);
    const cy = Math.round(rect.top + rect.height / 2);
    const fromX = Math.round(window.innerWidth / 2);
    const fromY = Math.round(window.innerHeight / 2);
    for (let step = 1; step <= 6; step += 1) {
      window.dispatchEvent(
        new MouseEvent('mousemove', {
          bubbles: true,
          clientX: Math.round(fromX + ((cx - fromX) * step) / 6),
          clientY: Math.round(fromY + ((cy - fromY) * step) / 6)
        })
      );
      await sleep(15);
    }
    element.dispatchEvent(new MouseEvent('mouseover', { bubbles: true, clientX: cx, clientY: cy }));
    element.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, clientX: cx, clientY: cy }));
    await sleep(40);
    element.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, clientX: cx, clientY: cy }));
    element.dispatchEvent(new MouseEvent('click', { bubbles: true, clientX: cx, clientY: cy }));
    await sleep(80);
  };

  // React tracks the native value setter; assigning `.value` directly would update the DOM
  // without telling React, so the contributed state would silently keep the old value.
  const setNativeValue = (element, value) => {
    const descriptor = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(element), 'value');
    if (!descriptor || !descriptor.set) {
      throw new Error(`[${taskId}] Cannot set a native value on ${element.tagName}`);
    }
    descriptor.set.call(element, value);
    element.dispatchEvent(new Event('input', { bubbles: true }));
    element.dispatchEvent(new Event('change', { bubbles: true }));
  };

  const change = async (componentId, value) => {
    const element = must(componentId);
    setNativeValue(element, value);
    await waitFor(() => element.value === value, `${componentId} to become ${value}`);
    await settleWindow();
  };

  const check = async (componentId, label) => {
    const element = must(componentId);
    if (!element.checked) element.click();
    await waitFor(() => element.checked, label);
    await settleWindow();
  };

  const statusText = () => document.querySelector('[data-aui-trial-status]')?.textContent?.trim();
  const errorCount = () => Number(document.querySelector('[data-aui-trial-errors]')?.textContent ?? 0);
  const page = () =>
    document.querySelector('.app-container')?.getAttribute('data-aui-route') ?? 'unknown';
  const resultsCount = () => {
    const label = document.querySelector('[data-aui-component="results-count"]')?.textContent ?? '';
    const match = label.match(/\((\d+)\)/);
    if (!match) throw new Error(`[${taskId}] Results count is not readable: '${label}'`);
    return Number(match[1]);
  };
  const isExpanded = (componentId) =>
    must(componentId).getAttribute('aria-expanded') === 'true';
  const currentReportText = () =>
    document.querySelector('[data-aui-component="current-report"]')?.textContent?.trim() ?? '';
  const checkedCategories = () =>
    Array.from(document.querySelectorAll('[data-aui-component^="filter-category-"]'))
      .filter((element) => element.checked)
      .map((element) => element.getAttribute('data-aui-component').replace('filter-category-', ''));

  const act = async (componentId, expectation, label) => {
    await click(componentId);
    if (expectation) await waitFor(expectation, label || componentId);
    await settleWindow();
  };

  const startTrial = async () => {
    const conditionControl = `condition-${conditionId}`;
    if (must(conditionControl).getAttribute('aria-checked') !== 'true') {
      await click(conditionControl);
      await waitFor(
        () => must(conditionControl).getAttribute('aria-checked') === 'true',
        `the ${conditionId} condition to be selected`
      );
    }
    await waitFor(
      () => diagnostics().status().running && diagnostics().status().conditionId === conditionId,
      `a running ${conditionId} runtime`
    );

    const startControl = `btn-start-${taskId}`;
    await waitFor(() => !must(startControl).disabled, `${startControl} to be enabled`);
    await click(startControl);
    await waitFor(() => statusText() === 'In Progress', `task ${taskId} to start`);
    await waitFor(
      () => page() === TASK_STARTING_PAGE,
      `task ${taskId} to start on ${TASK_STARTING_PAGE}`
    );
  };

  const TASK_STARTING_PAGE = {
    TD1: 'Overview',
    TD2: 'Analytics',
    TD3: 'Analytics',
    TD4: 'Reports'
  }[taskId];
  if (!TASK_STARTING_PAGE) throw new Error(`[${taskId}] Unknown task`);

  const measurements = { sawEmptyResult: false };
  await startTrial();

  if (taskId === 'TD1') {
    await act('nav-Analytics', () => page() === 'Analytics', 'the Analytics page');
    await act(
      'filter-section-region',
      () => isExpanded('filter-section-region'),
      'the Region section to open'
    );
    await change('filter-Region-select', 'Europe');
    await act(
      'btn-apply-filters',
      () => resultsCount() === 18,
      '18 visible rows after the Europe filter'
    );
    await click('btn-export');
  } else if (taskId === 'TD2') {
    await act(
      'filter-section-date-range',
      () => isExpanded('filter-section-date-range'),
      'the Date Range section to open'
    );
    await change('filter-date-input', '2025-06-01');
    await act(
      'filter-section-product-category',
      () => isExpanded('filter-section-product-category'),
      'the Product Category section to open'
    );
    await check('filter-category-books', 'the Books category to be selected');
    await act(
      'btn-apply-filters',
      () => resultsCount() === 5,
      '5 visible rows after the date and category filter'
    );
  } else if (taskId === 'TD3') {
    await act(
      'filter-section-region',
      () => isExpanded('filter-section-region'),
      'the Region section to open'
    );
    await change('filter-Region-select', 'Europe');
    await act(
      'filter-section-product-category',
      () => isExpanded('filter-section-product-category'),
      'the Product Category section to open'
    );
    await check('filter-category-books', 'the Books category to be selected');
    await act(
      'filter-section-date-range',
      () => isExpanded('filter-section-date-range'),
      'the Date Range section to open'
    );
    await change('filter-date-input', '2025-09-01');
    await act('btn-apply-filters', () => {
      const empty = document.querySelector('[data-aui-component="results-empty"]');
      const emptyNow = resultsCount() === 0 && Boolean(empty);
      if (emptyNow) measurements.sawEmptyResult = true;
      return emptyNow;
    }, 'the empty result set');
    await act(
      'btn-reset-filters',
      () => resultsCount() === 50,
      '50 rows restored by Reset Filters'
    );
  } else if (taskId === 'TD4') {
    await act(
      'open-report-eu-electronics',
      () => page() === 'Analytics' && currentReportText().includes('Europe Electronics'),
      'Europe Electronics to open'
    );
    await act('nav-Reports', () => page() === 'Reports', 'the Reports page');
    await act(
      'open-report-eu-books',
      () => page() === 'Analytics' && currentReportText().includes('Europe Books'),
      'Europe Books to open'
    );
    await click('btn-export');
  }

  await waitFor(() => {
    const current = statusText();
    if (current === 'Completed') return true;
    if (current === 'Abandoned' || current === 'Failed') {
      throw new Error(`[${taskId}] Trial ended as ${current} with ${errorCount()} error(s)`);
    }
    return false;
  }, `task ${taskId} to reach Completed`, 20000);

  // Completion rotates the session and clears the live recorder. The retained snapshot is the
  // finished trial's export, and it becomes available once the last window has settled.
  const rawTrace = await waitFor(
    () => diagnostics().lastCompletedTrial(),
    `the retained export of the finished ${taskId} trial`,
    15000
  );
  const trace = JSON.parse(rawTrace);

  measurements.status = statusText();
  measurements.errors = errorCount();
  measurements.page = page();
  measurements.visibleRows = resultsCount();
  measurements.region = document.querySelector('[data-aui-component="filter-Region-select"]')?.value;
  measurements.dateFrom = document.querySelector('[data-aui-component="filter-date-input"]')?.value;
  measurements.categories = checkedCategories();
  measurements.currentReport = currentReportText();
  measurements.semanticOutcomes = Array.from(
    new Set(trace.outcomes.map((outcome) => outcome.taskOutcome).filter(Boolean))
  );
  measurements.viewport = { width: window.innerWidth, height: window.innerHeight };
  measurements.userAgent = navigator.userAgent;

  return JSON.stringify({ trace, measurements });
})
"""

# Errors that mean the browser transport failed, not that the trial failed. Only these are
# retried; a missing element or an incomplete task must fail loudly.
TRANSPORT_ERROR_MARKERS = (
    "CDP",
    "socket",
    "ECONN",
    "Broken pipe",
    "Target crashed",
    "Target closed",
    "not reachable",
    "net::",
    "daemon",
)


def get_git_commit(cwd: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        )
        return completed.stdout.strip()
    except Exception:
        return "UNKNOWN"


def get_git_branch(cwd: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()
    except Exception:
        return "UNKNOWN"


def port_in_use(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/", timeout=1.0) as response:
            return response.status == 200
    except Exception:
        return False


def browser_command(session: str, browser_args: list[str]) -> list[str]:
    command = ["agent-browser", "--session", session]
    if browser_args:
        command += ["--args", ",".join(browser_args)]
    return command


def run_browser(command: list[str], *, stdin: str | None = None, timeout: int = 180):
    return subprocess.run(
        command,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def open_page(command: list[str], url: str, viewport: tuple[int, int], download_dir: Path) -> None:
    opened = run_browser(
        command
        + [
            "--download-path",
            str(download_dir),
            "open",
            url,
        ]
    )
    if opened.returncode != 0:
        raise RuntimeError(f"Could not open {url}: {opened.stderr.strip() or opened.stdout.strip()}")
    viewport_call = run_browser(command + ["set", "viewport", str(viewport[0]), str(viewport[1])])
    if viewport_call.returncode != 0:
        raise RuntimeError(
            f"Could not set the viewport: {viewport_call.stderr.strip() or viewport_call.stdout.strip()}"
        )
    # Give the runtime time to boot and expose the diagnostics handle; the in-page runner
    # waits for it too, but failing here produces a clearer error than a 15 s timeout.
    deadline = time.time() + 15
    while time.time() < deadline:
        probe = run_browser(
            command + ["eval", "window.__EDGE_AUI__ ? 1 : 0"],
            timeout=30,
        )
        if probe.returncode == 0 and probe.stdout.strip() in ("1", '"1"'):
            return
        time.sleep(0.5)
    raise RuntimeError(
        f"The testbed runtime did not expose window.__EDGE_AUI__ at {url}. "
        "Open the page with ?auiDiagnostics=1 and check the console."
    )


def parse_eval_output(output: str) -> dict:
    """Decodes agent-browser's eval output.

    With `--json` the CLI prints `{"success":true,"data":{"result":<value>}}`; without it,
    the result is printed directly. The runner returns a JSON *string*, so the value still
    needs one more decode in both shapes.
    """
    text = output.strip()
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"agent-browser did not return JSON: {error}\n{text[:500]}")

    if isinstance(decoded, dict) and "success" in decoded:
        if not decoded.get("success"):
            raise RuntimeError(f"agent-browser eval failed: {decoded.get('error') or text[:500]}")
        result = (decoded.get("data") or {}).get("result")
        if result is None:
            raise RuntimeError(f"agent-browser eval returned no result: {text[:500]}")
    else:
        result = decoded

    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError as error:
            raise ValueError(f"Runner result was not JSON: {error}\n{result[:500]}")
    return result


def run_trial(command: list[str], task_id: str, condition: str) -> dict:
    call = f"({JS_RUNNER})({json.dumps(task_id)}, {json.dumps(condition)})"
    completed = run_browser(command + ["eval", "--stdin", "--json"], stdin=call)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(detail or "agent-browser eval exited non-zero")
    payload = parse_eval_output(completed.stdout)
    if "trace" not in payload or "measurements" not in payload:
        raise RuntimeError(f"Runner returned no trace/measurements: {str(payload)[:500]}")
    return payload


def is_transport_error(message: str) -> bool:
    return any(marker.lower() in message.lower() for marker in TRANSPORT_ERROR_MARKERS)


def verify_trace(verify_script: Path, trace_path: Path) -> tuple[bool, str]:
    completed = subprocess.run(
        ["node", str(verify_script), str(trace_path)], capture_output=True, text=True
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode == 0, output.strip()


def check_equals(failures: list[str], label: str, actual, expected) -> None:
    if actual != expected:
        failures.append(f"{label}: expected {expected!r}, got {actual!r}")


def check_true(failures: list[str], label: str, condition: bool, detail: str = "") -> None:
    if not condition:
        failures.append(f"{label}{': ' + detail if detail else ''}")


def assert_trial(trace: dict, measurements: dict, task_id: str, condition: str) -> list[str]:
    """Asserts the completion evidence a scripted run must show. Returns failure strings."""
    failures: list[str] = []
    spec = TASKS[task_id]

    check_equals(failures, "schemaVersion", trace.get("schemaVersion"), SCHEMA_VERSION)
    check_equals(
        failures, "metadata.provenance", trace.get("metadata", {}).get("provenance"), "scripted"
    )
    check_equals(
        failures, "session.provenance", trace.get("session", {}).get("provenance"), "scripted"
    )
    check_equals(
        failures, "session.conditionId", trace.get("session", {}).get("conditionId"), condition
    )
    check_equals(
        failures, "metadata.conditionId", trace.get("metadata", {}).get("conditionId"), condition
    )
    check_equals(
        failures, "metadata.taskDefinitionVersion",
        trace.get("metadata", {}).get("taskDefinitionVersion"), "TD-v2"
    )
    check_true(
        failures,
        "metadata.semanticOutcomeVocabularyVersion",
        bool(trace.get("metadata", {}).get("semanticOutcomeVocabularyVersion")),
        "missing from metadata",
    )

    task = trace.get("task") or {}
    check_equals(failures, "task.currentTaskId", task.get("currentTaskId"), task_id)
    check_equals(failures, "task.status", task.get("status"), "Completed")
    check_equals(failures, "task.definitionVersion", task.get("definitionVersion"), "TD-v2")
    check_equals(failures, "metadata.finalTaskStatus", trace.get("metadata", {}).get(
        "finalTaskStatus"
    ), "Completed")
    check_equals(failures, "task.errors", task.get("errors"), 0)

    task_events = trace.get("taskEvents") or []
    starts = [event for event in task_events if event.get("type") == "task_start"]
    completes = [event for event in task_events if event.get("type") == "task_complete"]
    errors = [event for event in task_events if event.get("type") == "task_error"]
    steps = [event.get("taskStepId") for event in task_events if event.get("type") == "task_step"]

    check_equals(failures, "taskEvents.task_start count", len(starts), 1)
    check_equals(failures, "taskEvents.task_complete count", len(completes), 1)
    check_equals(failures, "taskEvents.task_error count", len(errors), 0)
    for step_id in spec["steps"]:
        check_true(failures, f"taskEvents contains {step_id}", step_id in steps)
    for event in task_events:
        check_equals(
            failures,
            f"taskEvents[{event.get('type')}].taskDefinitionVersion",
            event.get("taskDefinitionVersion"),
            "TD-v2",
        )

    ui = task.get("uiState") or {}
    applied = ui.get("appliedFilters") or {}
    for key, expected in spec["ui"].items():
        # `dateFrom`, `region` and `categories` live under `appliedFilters`; the rest are
        # top-level fields of the UI-state snapshot.
        actual = applied.get(key) if key in ("dateFrom", "region", "categories") else ui.get(key)
        check_equals(failures, f"uiState.{key}", actual, expected)

    if task_id == "TD3":
        check_true(
            failures,
            "TD3 observed the empty result before reset",
            measurements.get("sawEmptyResult") is True,
            "the empty state was never observed",
        )

    outcomes = trace.get("outcomes") or []
    check_true(failures, "outcomes recorded", len(outcomes) > 0, "no outcome records")
    semantic = [outcome.get("taskOutcome") for outcome in outcomes if outcome.get("taskOutcome")]
    for outcome_id in spec["required_semantic_outcomes"]:
        check_true(
            failures,
            f"semantic outcome '{outcome_id}'",
            outcome_id in semantic,
            f"observed: {sorted(set(semantic))}",
        )
    for outcome in outcomes:
        if outcome.get("taskOutcome") and not outcome.get("taskOutcomeVocabularyVersion"):
            failures.append(
                f"outcomes[{outcome.get('windowId')}].taskOutcome has no vocabulary version"
            )

    windows = trace.get("microTensors") or []
    window_ids = [window.get("windowId") for window in windows]
    check_true(failures, "microTensor windows recorded", len(windows) > 0, "no windows")
    check_equals(
        failures, "microTensor windowId uniqueness", len(set(window_ids)), len(window_ids)
    )

    predictions = trace.get("predictions") or []
    policy = trace.get("policyDecisions") or []
    check_true(
        failures,
        "every prediction has a policy verdict",
        all(
            any(decision.get("predictionId") == prediction.get("predictionId") for decision in policy)
            for prediction in predictions
        ),
        "a prediction has no policy decision",
    )

    applied = [
        intervention
        for intervention in trace.get("interventions") or []
        if intervention.get("type") == "applied"
    ]
    if condition == "baseline":
        check_equals(failures, "baseline applied interventions", len(applied), 0)
    else:
        framework_interventions = {
            "no_op",
            "highlight_primary_action",
            "simplify_options",
            "expand_tooltip",
            "offer_assistance",
        }
        for intervention in applied:
            check_true(
                failures,
                "adaptive applied intervention is configured",
                intervention.get("intervention") in framework_interventions,
                f"unknown intervention {intervention.get('intervention')!r}",
            )

    check_equals(failures, "measured trial status", measurements.get("status"), "Completed")
    check_equals(failures, "measured trial errors", measurements.get("errors"), 0)
    check_equals(
        failures, "measured visible rows", measurements.get("visibleRows"),
        spec["ui"]["visibleRowCount"]
    )

    check_true(
        failures,
        "capture is versioned as 1.4.0",
        trace.get("schemaVersion") == SCHEMA_VERSION,
        f"got {trace.get('schemaVersion')!r}",
    )
    return failures


def write_manifest(output_dir: Path, manifest: dict) -> Path:
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path


def generate_markdown_report(doc_path: Path, manifest: dict) -> None:
    rows = []
    for entry in manifest["traces"]:
        checks = entry["checks"]
        rows.append(
            "| `{filename}` | {conditionId} | {taskId} | {status} | {duration:.0f} | "
            "{windows} | {outcomes} | {predictions} | {applied} | {verified} |".format(
                filename=entry["filename"],
                conditionId=entry["conditionId"],
                taskId=entry["taskId"],
                status=entry["status"],
                duration=float(entry.get("durationMs") or 0),
                windows=entry["counts"]["microTensors"],
                outcomes=entry["counts"]["outcomes"],
                predictions=entry["counts"]["predictions"],
                applied=entry["counts"]["interventionsApplied"],
                verified="PASS" if entry["verified"] else "FAIL",
            )
        )

    semantic_rows = []
    for entry in manifest["traces"]:
        semantic_rows.append(
            "| {taskId} | {conditionId} | {semantics} |".format(
                taskId=entry["taskId"],
                conditionId=entry["conditionId"],
                semantics=", ".join(f"`{value}`" for value in entry["semanticOutcomes"]) or "—",
            )
        )

    passed = sum(1 for entry in manifest["traces"] if entry["verified"])
    content = f"""# Scripted Testbed Trace Recording Report

**Workstream:** 30 — Collection validation (Task 30.1)  
**Recorded at:** {manifest['recorded_at']}  
**Data source:** `{manifest['data_source']}`  
**Provenance:** `{manifest['provenance']}` — **not participant evidence**  
**Framework commit:** `{manifest['framework_commit']}`  
**Model-preparation commit:** `{manifest['model_preparation_commit']}`  
**Automation:** `{manifest.get('agent_browser_version', 'unknown')}`  
**Browser:** `{manifest['browser_version']}`  
**Viewport:** `{manifest['viewport']['width']}×{manifest['viewport']['height']}`  
**Environment:** `{manifest['os_environment']}`

---

## 1. Claim boundary (ADR-013)

> {manifest['claim_boundary']}

> {manifest['provenance_notice']}

---

## 2. Execution summary

- **Conditions:** {", ".join(manifest['conditions'])}
- **Tasks:** {", ".join(manifest['tasks'])}
- **Trials per condition/task:** {manifest['trials_per_task']}
- **Captures recorded:** {len(manifest['traces'])}
- **Verified by `scripts/verify-trace.mjs`:** {passed}/{len(manifest['traces'])}
- **Schema:** `{manifest['trace_schema_version']}`

| Capture | Condition | Task | Status | Duration (ms) | Windows | Outcomes | Predictions | Applied interventions | Verified |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
{chr(10).join(rows)}

### Semantic outcomes observed

| Task | Condition | `taskOutcome` values |
| :--- | :--- | :--- |
{chr(10).join(semantic_rows)}

---

## 3. What the captures establish

1. Each task completes on its canonical path through the real task controls, with zero
   `task_error` events and the task-state evidence the completion predicate requires.
2. Each capture is a runtime-produced `1.4.0` trace carrying semantic outcomes, the
   task-definition version (`TD-v2`) and the semantic vocabulary version.
3. Baseline captures contain no applied intervention; every applied adaptive intervention
   belongs to the framework's configured intervention vocabulary.
4. The rows, filters and report state reported in each trace's final `task.uiState` match the
   approved task definitions.

## 4. What the captures do not establish

- Nothing about human participants, usability, task performance or intervention benefit.
- Nothing about a deployed origin or a live Supabase project; the run targets the local
  testbed origin.
- A trial whose first action precedes the first MicroTensor window has no semantic outcome for
  that action. The trace records the outcome windows the pipeline actually produced; it is not
  back-filled.

---

## 5. Verification

```bash
cd edge-aui-framework
node scripts/verify-trace.mjs ../model-preparation/.data/raw/scripted/<capture>.json
node scripts/compare-condition-traces.mjs --baseline <baseline.json> --adaptive <adaptive.json>
```
"""
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    with open(doc_path, "w", encoding="utf-8") as handle:
        handle.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record scripted TD1–TD4 testbed traces in the real browser runtime."
    )
    parser.add_argument(
        "--tasks", nargs="+", default=list(TASKS), choices=list(TASKS), help="Tasks to record"
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=CONDITIONS,
        choices=CONDITIONS,
        help="Experimental conditions to record",
    )
    parser.add_argument(
        "--trials-per-condition", type=int, default=1, help="Trials per task per condition"
    )
    parser.add_argument(
        "--output", type=str, default=".data/raw/scripted", help="Output directory for captures"
    )
    parser.add_argument("--port", type=int, default=5199, help="Vite dev-server port")
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Origin to drive (defaults to the local Vite server on --port)",
    )
    parser.add_argument(
        "--framework-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent.parent / "edge-aui-framework"),
        help="Path to the edge-aui-framework checkout",
    )
    parser.add_argument("--session", type=str, default="scripted-rec", help="agent-browser session")
    parser.add_argument(
        "--browser-args",
        type=str,
        default=os.environ.get("AUI_SCRIPTED_BROWSER_ARGS", ""),
        help="Comma-separated Chrome launch args, e.g. --no-sandbox (also AUI_SCRIPTED_BROWSER_ARGS)",
    )
    parser.add_argument("--viewport-width", type=int, default=1920, help="Browser viewport width")
    parser.add_argument("--viewport-height", type=int, default=1080, help="Browser viewport height")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Attempts per trial; only browser-transport failures are retried",
    )
    parser.add_argument(
        "--keep-browser", action="store_true", help="Leave the agent-browser session open"
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    framework_dir = Path(args.framework_dir).resolve()
    output_dir = repo_root / args.output if not os.path.isabs(args.output) else Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir = output_dir / ".downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    verify_script = framework_dir / "scripts" / "verify-trace.mjs"
    if not verify_script.exists():
        print(f"Error: verifier not found at {verify_script}", file=sys.stderr)
        sys.exit(1)

    framework_commit = get_git_commit(framework_dir)
    framework_branch = get_git_branch(framework_dir)
    model_preparation_commit = get_git_commit(repo_root)
    model_preparation_branch = get_git_branch(repo_root)

    url = args.url or f"http://localhost:{args.port}/?auiDiagnostics=1"
    browser_args = [value for value in args.browser_args.split(",") if value.strip()]

    print("=" * 78)
    print("Scripted TD1–TD4 Trace Recording Harness")
    print(f"Framework:          {framework_dir} @ {framework_branch}/{framework_commit[:12]}")
    print(f"Model-preparation:  {repo_root} @ {model_preparation_branch}/{model_preparation_commit[:12]}")
    print(f"Origin:             {url}")
    print(f"Output:             {output_dir}")
    print(f"Tasks:              {args.tasks}")
    print(f"Conditions:         {args.conditions}")
    print(f"Trials per pair:    {args.trials_per_condition}")
    print(f"Provenance:         scripted (not participant evidence)")
    print("=" * 78)

    # Start the Vite dev server when the default local origin is used and nothing is listening.
    server_process = None
    started_server = False
    if args.url is None and not port_in_use(args.port):
        print(f"Starting Vite dev server on port {args.port}…")
        server_process = subprocess.Popen(
            ["npx", "vite", "--port", str(args.port), "--strictPort"],
            cwd=framework_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        started_server = True
        for _ in range(60):
            time.sleep(0.5)
            if port_in_use(args.port):
                break
        else:
            print("Error: timed out waiting for the Vite server.", file=sys.stderr)
            server_process.kill()
            sys.exit(1)
        print("Vite server ready.")
    elif args.url is None:
        print(f"Reusing the server already listening on port {args.port}.")

    command = browser_command(args.session, browser_args)
    viewport = (args.viewport_width, args.viewport_height)

    manifest = {
        "schemaVersion": "2.0.0",
        "data_source": "scripted_testbed",
        "provenance": "scripted",
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "framework_commit": framework_commit,
        "framework_branch": framework_branch,
        "model_preparation_commit": model_preparation_commit,
        "model_preparation_branch": model_preparation_branch,
        "browser_version": "unknown",
        "agent_browser_version": "unknown",
        "os_environment": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "viewport": {"width": viewport[0], "height": viewport[1]},
        "origin": url,
        "trace_schema_version": SCHEMA_VERSION,
        "task_definition_version": "TD-v2",
        "trials_per_task": args.trials_per_condition,
        "conditions": args.conditions,
        "tasks": args.tasks,
        "claim_boundary": ADR_013_CLAIM_BOUNDARY,
        "provenance_notice": PROVENANCE_NOTICE,
        "traces": [],
        "failures": [],
    }

    # Remove this run's previous captures so the directory cannot mix two runs' artefacts.
    for stale in output_dir.glob("scripted-*.json"):
        stale.unlink()

    total = len(args.tasks) * len(args.conditions) * args.trials_per_condition
    index = 0
    try:
        open_page(command, url, viewport, download_dir)

        version = run_browser(command + ["--version"])
        cli_version = version.stdout.strip() if version.returncode == 0 else "agent-browser"
        manifest["agent_browser_version"] = cli_version

        for condition in args.conditions:
            for task_id in args.tasks:
                for trial in range(1, args.trials_per_condition + 1):
                    index += 1
                    label = f"{task_id}-{condition}-{trial}"
                    print(f"[{index}/{total}] Recording {label}…")
                    attempts = []
                    payload = None
                    for attempt in range(1, args.max_attempts + 1):
                        try:
                            # Every trial starts from a freshly loaded page. A trial boundary
                            # resets the recorder, task state and filter values, but browser
                            # UI state such as open filter sections and a half-finished task
                            # would otherwise leak into the next trial — and one failed trial
                            # would cascade into every trial after it.
                            open_page(command, url, viewport, download_dir)
                            payload = run_trial(command, task_id, condition)
                            attempts.append({"attempt": attempt, "result": "passed"})
                            break
                        except Exception as error:  # noqa: BLE001 - recorded, then judged
                            detail = str(error)
                            attempts.append(
                                {"attempt": attempt, "result": "failed", "error": detail[:2000]}
                            )
                            if attempt >= args.max_attempts or not is_transport_error(detail):
                                break
                            print(f"       transport failure, reopening the page: {detail[:200]}")

                    if payload is None:
                        manifest["failures"].append(
                            {"taskId": task_id, "conditionId": condition, "trial": trial,
                             "attempts": attempts}
                        )
                        print(f"       FAILED: {attempts[-1]['error'][:300]}")
                        continue

                    trace = payload["trace"]
                    measurements = payload["measurements"]
                    manifest["browser_version"] = (
                        measurements.get("userAgent") or manifest["browser_version"]
                    )
                    manifest["viewport"] = measurements.get("viewport") or manifest["viewport"]

                    filename = f"scripted-{task_id}-{condition}-trial{trial}.json"
                    trace_path = output_dir / filename
                    with open(trace_path, "w", encoding="utf-8") as handle:
                        json.dump(trace, handle, indent=2)

                    failures = assert_trial(trace, measurements, task_id, condition)
                    verified, verification_output = verify_trace(verify_script, trace_path)
                    if not verified:
                        failures.append(f"verify-trace.mjs failed:\n{verification_output}")
                    checks = {
                        "taskCompleted": (trace.get("task") or {}).get("status") == "Completed",
                        "zeroTaskErrors": (trace.get("task") or {}).get("errors") == 0,
                        "expectedRows": measurements.get("visibleRows"),
                        "semanticOutcomes": measurements.get("semanticOutcomes", []),
                        "baselineZeroMutations": (
                            condition != "baseline"
                            or not any(
                                intervention.get("type") == "applied"
                                for intervention in trace.get("interventions") or []
                            )
                        ),
                        "frameworkVerifier": verified,
                    }
                    metadata = trace.get("metadata") or {}
                    entry = {
                        "filename": filename,
                        "sessionId": (trace.get("session") or {}).get("sessionId"),
                        "experimentId": (trace.get("session") or {}).get("experimentId"),
                        "conditionId": condition,
                        "taskId": task_id,
                        "trial": trial,
                        "status": (trace.get("task") or {}).get("status"),
                        "schemaVersion": trace.get("schemaVersion"),
                        "taskDefinitionVersion": metadata.get("taskDefinitionVersion"),
                        "semanticOutcomeVocabularyVersion": metadata.get(
                            "semanticOutcomeVocabularyVersion"
                        ),
                        "durationMs": metadata.get("durationMs"),
                        "counts": {
                            "behaviourEvents": len(trace.get("behaviourEvents") or []),
                            "microTensors": len(trace.get("microTensors") or []),
                            "outcomes": len(trace.get("outcomes") or []),
                            "predictions": len(trace.get("predictions") or []),
                            "policyDecisions": len(trace.get("policyDecisions") or []),
                            "taskEvents": len(trace.get("taskEvents") or []),
                            "interventionsApplied": sum(
                                1
                                for intervention in trace.get("interventions") or []
                                if intervention.get("type") == "applied"
                            ),
                        },
                        "semanticOutcomes": measurements.get("semanticOutcomes", []),
                        "measurements": measurements,
                        "checks": checks,
                        "attempts": attempts,
                        "verified": not failures,
                        "failures": failures,
                    }
                    manifest["traces"].append(entry)
                    if failures:
                        manifest["failures"].append(
                            {
                                "taskId": task_id,
                                "conditionId": condition,
                                "trial": trial,
                                "failures": failures,
                            }
                        )
                        print(f"       FAILED verification:")
                        for failure in failures:
                            print(f"         - {failure}")
                    else:
                        counts = entry["counts"]
                        print(
                            f"       -> {filename}: Completed, {counts['microTensors']} windows, "
                            f"{counts['outcomes']} outcomes, {counts['predictions']} predictions, "
                            f"{checks['semanticOutcomes']}"
                        )

        manifest_path = write_manifest(output_dir, manifest)
        report_path = repo_root / "docs" / "experiments" / "scripted-recording.md"
        generate_markdown_report(report_path, manifest)

        print("\n" + "=" * 78)
        print(f"Manifest: {manifest_path}")
        print(f"Report:   {report_path}")
        print(f"Captures: {len(manifest['traces'])}/{total} recorded, "
              f"{sum(1 for entry in manifest['traces'] if entry['verified'])} verified")
        print("=" * 78)

        if manifest["failures"]:
            print(f"\nFAILED: {len(manifest['failures'])} trial(s) did not meet the acceptance "
                  "evidence.", file=sys.stderr)
            sys.exit(1)
        print("\nAll scripted captures completed, verified and manifested.")
    finally:
        if not args.keep_browser:
            try:
                run_browser(command + ["close"], timeout=30)
            except Exception:
                pass
        if started_server and server_process:
            print("Stopping the spawned Vite server…")
            server_process.terminate()
            server_process.wait()


if __name__ == "__main__":
    main()
