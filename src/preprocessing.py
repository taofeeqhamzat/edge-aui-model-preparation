"""
preprocessing.py
Modular Behavioral Preprocessing & MicroTensor Extraction Engine.
Implements:
- Layer A -> Layer B: Canonical Event Schema & Viewport Normalization ([0, 1] canvas coordinates)
- Layer B -> Layer C: MicroTensor Extraction with Binary Modality Masking (2D = 18 dimensions)
"""

import os
import glob
import math
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union

import numpy as np

# Optional pandas import for high-throughput DataFrame manipulation
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None  # type: ignore
    PANDAS_AVAILABLE = False

# Optional PyTorch import for Dataset integration
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    Dataset = object  # type: ignore
    DataLoader = None  # type: ignore
    TORCH_AVAILABLE = False

try:
    from data import find_project_root, resolve_adserp_dir
except ImportError:
    try:
        # pyrefly: ignore [missing-import]
        from src.data import find_project_root, resolve_adserp_dir
    except ImportError:
        find_project_root = lambda: Path(os.getcwd()).resolve()
        resolve_adserp_dir = lambda custom=None: None


# ============================================================================
# Feature Definitions & Outcome Taxonomy
# ============================================================================

FEATURE_NAMES = [
    "meanVelocity",
    "maxVelocity",
    "meanAcceleration",
    "hesitationCount",
    "totalTrajectoryLength",
    "dwellTimeMs",
    "trajectoryEntropy",
    "scrollDepthPercentage",
    "scrollVelocity"
]

CORE_FEATURE_NAMES = FEATURE_NAMES[:7]
CONTEXTUAL_FEATURE_NAMES = FEATURE_NAMES[7:]
NUM_FEATURES = len(FEATURE_NAMES)  # 9
MICROTENSOR_DIM = NUM_FEATURES * 2  # 18 with binary modality mask
INPUT_DIM = MICROTENSOR_DIM

FEATURE_COLUMN_NAMES = [
    "mean_velocity",
    "max_velocity",
    "mean_acceleration",
    "hesitation_count",
    "total_trajectory_length",
    "dwell_time_ms",
    "trajectory_entropy",
    "scroll_depth_percentage",
    "scroll_velocity"
]
MASK_COLUMN_NAMES = [f"mask_{c}" for c in FEATURE_COLUMN_NAMES]
ALL_MICROTENSOR_COLUMNS = FEATURE_COLUMN_NAMES + MASK_COLUMN_NAMES

LABEL_MAP = {
    "IDLE_ABANDON": 0,
    "CLICK": 1,
    "FORM_SUBMIT": 2,
    "BACKTRACK": 3,
    "RAPID_SCROLL": 4,
    "HOVER_DWELL": 5
}
LABEL_INV_MAP = {v: k for k, v in LABEL_MAP.items()}


# ============================================================================
# Layer A -> Layer B: Viewport Normalization & Canonical Event Schema
# ============================================================================

def parse_viewport_metadata(xml_path: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Parse source window/viewport and document dimensions from trial metadata XML.
    Returns ((viewport_w, viewport_h), (doc_w, doc_h)).
    Fails visibly (raises FileNotFoundError or ValueError) if XML is missing or malformed.
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Trial metadata XML file not found: {xml_path}")

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        raise ValueError(f"Failed to parse XML file {xml_path}: {e}")

    win_node = root.find("window")
    if win_node is None or not win_node.text or "x" not in win_node.text:
        raise ValueError(f"Metadata XML {xml_path} is missing a valid '<window>WxH</window>' element.")

    parts = win_node.text.strip().split("x")
    if len(parts) != 2:
        raise ValueError(f"Malformed <window> string '{win_node.text}' in {xml_path}; expected 'WIDTHxHEIGHT'.")

    try:
        viewport_w, viewport_h = float(parts[0]), float(parts[1])
    except ValueError as e:
        raise ValueError(f"Non-numeric <window> dimensions '{win_node.text}' in {xml_path}: {e}")

    if viewport_w <= 0 or viewport_h <= 0:
        raise ValueError(f"Non-positive <window> dimensions ({viewport_w}, {viewport_h}) in {xml_path}.")

    doc_node = root.find("document")
    if doc_node is None or not doc_node.text or "x" not in doc_node.text:
        raise ValueError(f"Metadata XML {xml_path} is missing a valid '<document>WxH</document>' element.")

    doc_parts = doc_node.text.strip().split("x")
    if len(doc_parts) != 2:
        raise ValueError(f"Malformed <document> string '{doc_node.text}' in {xml_path}; expected 'WIDTHxHEIGHT'.")

    try:
        doc_w, doc_h = float(doc_parts[0]), float(doc_parts[1])
    except ValueError as e:
        raise ValueError(f"Non-numeric <document> dimensions '{doc_node.text}' in {xml_path}: {e}")

    return (viewport_w, viewport_h), (doc_w, doc_h)


def resolve_session_files(session_path_or_id: str, raw_dir: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    Locate the CSV session file and companion XML trial metadata file.
    Accepts full path, filename with extension, or session ID without extension.
    """
    path_obj = Path(session_path_or_id)

    # 1. Direct path exists
    if path_obj.is_file():
        csv_path = str(path_obj.resolve())
        xml_candidate_1 = path_obj.with_suffix(".xml")
        xml_candidate_2 = path_obj.parent.parent / "trial-metadata" / f"{path_obj.stem}.xml"
        if xml_candidate_1.is_file():
            return csv_path, str(xml_candidate_1.resolve())
        elif xml_candidate_2.is_file():
            return csv_path, str(xml_candidate_2.resolve())
        return csv_path, None

    # 2. Search candidate AdSERP directories
    session_id = path_obj.stem
    csv_filename = f"{session_id}.csv"

    adserp_root = resolve_adserp_dir(raw_dir)
    search_dirs: List[Path] = []
    if raw_dir:
        r = Path(raw_dir).resolve()
        search_dirs.extend([r, r / "adserp-2025", r / "adserp"])
    if adserp_root:
        search_dirs.append(adserp_root)

    proj_root = find_project_root()
    search_dirs.extend([
        proj_root / ".data" / "raw" / "adserp-2025",
        proj_root / ".data" / "raw" / "adserp",
        proj_root / ".data" / "raw" / "raw" / "adserp-2025",
        proj_root.parent / "data-sources" / "adserp-2025",
        proj_root.parent / "data-sources" / "adserp",
        Path("/Users/user/Workspace/MivaCS/FYP/data-sources/adserp-2025")
    ])

    for base in search_dirs:
        # Check standard layout: base/mouse-movement-data/
        mouse_dir = base / "mouse-movement-data"
        meta_dir = base / "trial-metadata"
        candidate_csv = mouse_dir / csv_filename
        if candidate_csv.is_file():
            candidate_xml = meta_dir / f"{session_id}.xml"
            xml_path = str(candidate_xml.resolve()) if candidate_xml.is_file() else None
            return str(candidate_csv.resolve()), xml_path

        # Check direct placement in base/
        direct_csv = base / csv_filename
        if direct_csv.is_file():
            direct_xml = base / f"{session_id}.xml"
            xml_path = str(direct_xml.resolve()) if direct_xml.is_file() else None
            return str(direct_csv.resolve()), xml_path

    # 3. On-demand retrieval from hosted Hugging Face Hub if available
    try:
        from huggingface_hub import hf_hub_download
        csv_remote = f"raw/adserp-2025/mouse-movement-data/{csv_filename}"
        xml_remote = f"raw/adserp-2025/trial-metadata/{session_id}.xml"
        csv_downloaded = hf_hub_download(
            repo_id="T40/edge-aui-framework-data",
            repo_type="dataset",
            filename=csv_remote
        )
        xml_downloaded = hf_hub_download(
            repo_id="T40/edge-aui-framework-data",
            repo_type="dataset",
            filename=xml_remote
        )
        return csv_downloaded, xml_downloaded
    except Exception:
        pass

    return session_path_or_id, None


def parse_adserp_session(
    session_path_or_id: str,
    raw_dir: Optional[str] = None,
    normalize_reference: bool = False,
    reference_viewport: Tuple[float, float] = (1920.0, 1080.0),
    as_df: bool = False
) -> Union[List[Dict[str, Any]], Any]:
    """
    Parse raw AdSERP CSV streams and companion XML metadata.
    Extracts canonical events (timestamp_ms, event_type, x_norm, y_norm, xpath).
    Normalizes coordinates against observed source viewport (<window>WxH</window>)
    into canvas coordinates in [0, 1].
    """
    csv_path, xml_path = resolve_session_files(session_path_or_id, raw_dir)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"AdSERP session file not found: {session_path_or_id} (resolved: {csv_path})")

    # Observed source viewport dimensions (fail visibly if XML is missing)
    if not xml_path or not os.path.exists(xml_path):
        raise FileNotFoundError(f"Missing companion trial metadata XML for session: {session_path_or_id} (expected at: {xml_path})")

    (vp_w, vp_h), (doc_w, doc_h) = parse_viewport_metadata(xml_path)

    events: List[Dict[str, Any]] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            parts = line_str.split(",")
            if len(parts) >= 4:
                try:
                    ts_ms = int(parts[0])
                    x_raw = float(parts[1])
                    y_raw = float(parts[2])
                    ev_type = parts[3]
                    xpath = parts[4] if len(parts) > 4 else ""

                    # Canonical normalization relative to observed viewport
                    x_norm = max(0.0, min(1.0, x_raw / max(vp_w, 1.0)))
                    y_norm = max(0.0, min(1.0, y_raw / max(vp_h, 1.0)))

                    if normalize_reference:
                        ref_w, ref_h = reference_viewport
                        x_ref = x_norm * ref_w
                        y_ref = y_norm * ref_h
                    else:
                        x_ref = x_raw
                        y_ref = y_raw

                    events.append({
                        "timestamp_ms": ts_ms,
                        "event_type": ev_type,
                        "x_norm": float(x_norm),
                        "y_norm": float(y_norm),
                        "x_raw": x_ref,
                        "y_raw": y_ref,
                        "xpath": xpath,
                        "viewport_w": vp_w,
                        "viewport_h": vp_h,
                        "doc_w": doc_w,
                        "doc_h": doc_h
                    })
                except (ValueError, IndexError):
                    continue

    events.sort(key=lambda ev: ev["timestamp_ms"])

    if as_df and PANDAS_AVAILABLE:
        return pd.DataFrame(events)

    return events


# ============================================================================
# Layer B -> Layer C: MicroTensor Extraction with Modality Masking (2D=18)
# ============================================================================

def compute_window_microtensor(
    events: List[Dict[str, Any]],
    viewport: Tuple[float, float],
    document: Tuple[float, float],
    window_duration_ms: float = 500.0,
    has_scroll_support: bool = True,
    scales: Optional[Dict[str, float]] = None
) -> np.ndarray:
    r"""
    Extract a single 18-dimensional MicroTensor from a 500ms interaction window:
    X_t (9 features) concatenated with binary Modality Mask Vector M in {0, 1}^9:
    \widetilde{X}_t = [X_t \odot M, M] in R^18.
    All outputs strictly bounded in [0, 1].
    Fails visibly if viewport or document dimensions are non-positive.
    """
    vp_w, vp_h = viewport
    doc_w, doc_h = document

    if vp_w <= 0.0 or vp_h <= 0.0:
        raise ValueError(f"Invalid non-positive viewport dimensions: ({vp_w}, {vp_h})")
    if doc_w <= 0.0 or doc_h <= 0.0:
        raise ValueError(f"Invalid non-positive document dimensions: ({doc_w}, {doc_h})")

    scale_dict = {
        "velocity": 10.0,
        "max_velocity": 10.0,
        "acceleration": 0.1,
        "hesitation": 10.0,
        "trajectory": 2000.0,
        "scroll_velocity": 5.0
    }
    if scales:
        scale_dict.update(scales)

    mean_vel = 0.0
    max_vel = 0.0
    mean_accel = 0.0
    hesitation_cnt = 0.0
    total_traj_len = 0.0
    dwell_time_ms = 0.0
    trajectory_entropy = 0.0
    scroll_depth_pct = 0.0
    scroll_vel = 0.0

    mask = np.zeros(NUM_FEATURES, dtype=np.float32)

    pointer_events = [
        ev for ev in events
        if ev.get("event_type") in ("mousemove", "mouseover", "mousedown", "mouseup", "click")
    ]

    if len(pointer_events) >= 2:
        mask[:7] = 1.0

        t_arr = np.array([ev["timestamp_ms"] for ev in pointer_events], dtype=np.float64)
        x_norm = np.array([ev["x_norm"] for ev in pointer_events], dtype=np.float64)
        y_norm = np.array([ev["y_norm"] for ev in pointer_events], dtype=np.float64)

        if "x" in pointer_events[0] and pointer_events[0]["x"] is not None and "y" in pointer_events[0] and pointer_events[0]["y"] is not None:
            dx_px = np.diff(np.array([ev["x"] for ev in pointer_events], dtype=np.float64))
            dy_px = np.diff(np.array([ev["y"] for ev in pointer_events], dtype=np.float64))
        elif "x_raw" in pointer_events[0] and pointer_events[0]["x_raw"] is not None and "y_raw" in pointer_events[0] and pointer_events[0]["y_raw"] is not None:
            dx_px = np.diff(np.array([ev["x_raw"] for ev in pointer_events], dtype=np.float64))
            dy_px = np.diff(np.array([ev["y_raw"] for ev in pointer_events], dtype=np.float64))
        else:
            dx_px = np.diff(x_norm) * vp_w
            dy_px = np.diff(y_norm) * vp_h

        dt_ms = np.diff(t_arr)
        dt_ms = np.where(dt_ms <= 0, 1.0, dt_ms)

        distances = np.sqrt(dx_px**2 + dy_px**2)
        total_traj_len = float(np.sum(distances))

        velocities = distances / dt_ms
        if len(velocities) > 0:
            mean_vel = float(np.mean(velocities))
            max_vel = float(np.max(velocities))

        if len(velocities) >= 2:
            dt_acc = dt_ms[1:]
            dt_acc = np.where(dt_acc <= 0, 1.0, dt_acc)
            accels = np.abs(np.diff(velocities)) / dt_acc
            mean_accel = float(np.mean(accels))

        if len(dx_px) >= 2:
            angles = np.arctan2(dy_px, dx_px)
            angle_diffs = np.abs(np.diff(angles))
            angle_diffs = np.where(angle_diffs > np.pi, 2 * np.pi - angle_diffs, angle_diffs)
            hesitation_cnt = float(np.sum(angle_diffs > (np.pi / 4.0)))

            hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi), density=False)
            hist_sum = hist.sum()
            if hist_sum > 0:
                p = hist[hist > 0] / hist_sum
                trajectory_entropy = float(-np.sum(p * np.log2(p)) / 3.0)

    def _is_valid_dom_target(t: Any) -> bool:
        if t is None or isinstance(t, (float, int)):
            return False
        t_str = str(t).strip()
        return bool(t_str and t_str not in ("/", "/html", "nan", "None", "{}"))

    dwell_evs = [
        ev for ev in events
        if ev.get("event_type") == "mouseover"
        or _is_valid_dom_target(ev.get("target_id"))
        or _is_valid_dom_target(ev.get("xpath"))
    ]
    dwell_time_ms = min(float(len(dwell_evs) * 40.0), float(window_duration_ms))

    scroll_events = [ev for ev in events if ev.get("event_type") in ("scroll", "wheel")]
    if has_scroll_support:
        mask[7:] = 1.0
        scroll_count = len(scroll_events)
        scroll_vel = (scroll_count * 100.0) / max(window_duration_ms, 1.0)
        max_scrollable = max(doc_h - vp_h, 1.0)
        scroll_depth_pct = min(1.0, (scroll_count * 80.0) / max_scrollable)

    raw_features = np.array([
        np.clip(mean_vel / scale_dict["velocity"], 0.0, 1.0),
        np.clip(max_vel / scale_dict["max_velocity"], 0.0, 1.0),
        np.clip(mean_accel / scale_dict["acceleration"], 0.0, 1.0),
        np.clip(hesitation_cnt / scale_dict["hesitation"], 0.0, 1.0),
        np.clip(total_traj_len / scale_dict["trajectory"], 0.0, 1.0),
        np.clip(dwell_time_ms / window_duration_ms, 0.0, 1.0),
        np.clip(trajectory_entropy, 0.0, 1.0),
        np.clip(scroll_depth_pct, 0.0, 1.0),
        np.clip(scroll_vel / scale_dict["scroll_velocity"], 0.0, 1.0)
    ], dtype=np.float32)

    masked_features = raw_features * mask
    microtensor_18 = np.concatenate([masked_features, mask], axis=0).astype(np.float32)

    return microtensor_18


def extract_session_microtensors(
    events: List[Dict[str, Any]],
    window_size_ms: int = 500,
    stride_ms: int = 250,
    min_events_per_window: int = 3,
    has_scroll_support: bool = True
) -> np.ndarray:
    """
    Segment a canonical event stream into sliding 500ms windows with 250ms stride.
    Returns an ndarray of shape (num_windows, 18).
    """
    if len(events) < min_events_per_window:
        return np.empty((0, INPUT_DIM), dtype=np.float32)

    start_time = events[0]["timestamp_ms"]
    end_time = events[-1]["timestamp_ms"]

    if "viewport_w" not in events[0] or "viewport_h" not in events[0]:
        raise ValueError("Canonical event is missing required 'viewport_w' or 'viewport_h' metadata.")
    if "doc_w" not in events[0] or "doc_h" not in events[0]:
        raise ValueError("Canonical event is missing required 'doc_w' or 'doc_h' metadata.")

    vp_w = float(events[0]["viewport_w"])
    vp_h = float(events[0]["viewport_h"])
    doc_w = float(events[0]["doc_w"])
    doc_h = float(events[0]["doc_h"])

    if vp_w <= 0.0 or vp_h <= 0.0:
        raise ValueError(f"Invalid non-positive viewport dimensions in event stream: ({vp_w}, {vp_h})")
    if doc_w <= 0.0 or doc_h <= 0.0:
        raise ValueError(f"Invalid non-positive document dimensions in event stream: ({doc_w}, {doc_h})")

    ts_arr = np.array([ev["timestamp_ms"] for ev in events], dtype=np.int64)

    t_curr = start_time
    windows: List[np.ndarray] = []

    while t_curr + window_size_ms <= end_time:
        win_end = t_curr + window_size_ms
        i_start = int(np.searchsorted(ts_arr, t_curr, side="left"))
        i_end = int(np.searchsorted(ts_arr, win_end, side="left"))

        window_evs = events[i_start:i_end]
        if len(window_evs) >= min_events_per_window:
            tensor_18 = compute_window_microtensor(
                window_evs,
                viewport=(vp_w, vp_h),
                document=(doc_w, doc_h),
                window_duration_ms=float(window_size_ms),
                has_scroll_support=has_scroll_support
            )
            windows.append(tensor_18)

        t_curr += stride_ms

    if len(windows) == 0:
        return np.empty((0, INPUT_DIM), dtype=np.float32)

    return np.stack(windows, axis=0)


def extract_mock_microtensor(
    seq_len: int = 8,
    has_scroll: bool = True,
    random_seed: Optional[int] = 42
) -> np.ndarray:
    """
    Generates a synthetic sequence of MicroTensors for pipeline verification,
    smoke tests, and unit testing.
    Output shape is (seq_len, 18) with all values strictly bounded in [0, 1].
    """
    if random_seed is not None:
        rng = np.random.RandomState(random_seed)
    else:
        rng = np.random.RandomState()

    total_duration_ms = (seq_len - 1) * 250 + 500
    num_events = int(total_duration_ms / 16.0) + 10

    timestamps = np.linspace(1000.0, 1000.0 + total_duration_ms, num=num_events, dtype=np.int64)

    t_phase = np.linspace(0, 2 * np.pi, num_events)
    x_raw = 600.0 + 300.0 * np.cos(t_phase) + rng.normal(0, 5, num_events)
    y_raw = 400.0 + 200.0 * np.sin(t_phase * 2) + rng.normal(0, 5, num_events)

    events: List[Dict[str, Any]] = []
    for i in range(num_events):
        ev_type = "mousemove"
        if i % 15 == 0:
            ev_type = "mouseover"
        elif has_scroll and (i % 25 == 0):
            ev_type = "scroll"

        x_clamped = max(0.0, min(1422.0, float(x_raw[i])))
        y_clamped = max(0.0, min(1137.0, float(y_raw[i])))

        events.append({
            "timestamp_ms": int(timestamps[i]),
            "event_type": ev_type,
            "x_norm": x_clamped / 1422.0,
            "y_norm": y_clamped / 1137.0,
            "x_raw": x_clamped,
            "y_raw": y_clamped,
            "xpath": "//*[@id='target']" if ev_type == "mouseover" else "/",
            "viewport_w": 1422.0,
            "viewport_h": 1137.0,
            "doc_w": 1403.0,
            "doc_h": 2642.0
        })

    tensor_seq = extract_session_microtensors(
        events,
        window_size_ms=500,
        stride_ms=250,
        min_events_per_window=3,
        has_scroll_support=has_scroll
    )

    if len(tensor_seq) >= seq_len:
        return tensor_seq[:seq_len]

    if len(tensor_seq) == 0:
        single_window = compute_window_microtensor(
            events[:10],
            viewport=(1422.0, 1137.0),
            document=(1403.0, 2642.0),
            has_scroll_support=has_scroll
        )
        return np.repeat(single_window[np.newaxis, :], seq_len, axis=0)

    reps = int(math.ceil(seq_len / len(tensor_seq)))
    tiled = np.tile(tensor_seq, (reps, 1))
    return tiled[:seq_len]


# ============================================================================
# PyTorch Dataset Integration
# ============================================================================

if TORCH_AVAILABLE:
    class MicroInteractionSequenceDataset(Dataset):
        """
        PyTorch Dataset yielding continuous MicroTensor sequences (seq_len, 18)
        and associated outcome targets for foundation pre-training or fine-tuning.
        """
        def __init__(
            self,
            sequences: np.ndarray,
            targets: np.ndarray
        ):
            self.X = torch.tensor(sequences, dtype=torch.float32)
            self.Y = torch.tensor(targets, dtype=torch.long)

        def __len__(self) -> int:
            return len(self.Y)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            return self.X[idx], self.Y[idx]


# ============================================================================
# CLI Driver for Direct Execution
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <function_name> <session_or_arg>")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "parse_adserp_session":
        session_arg = sys.argv[2] if len(sys.argv) > 2 else "p004-b1-t1.csv"
        evs = parse_adserp_session(session_arg)
        print(f"[Preprocessing] Parsed events: {len(evs)}")
        if evs:
            print(f"Sample canonical event: {evs[0]}")
    elif cmd == "extract_mock_microtensor":
        t = extract_mock_microtensor()
        print(f"[Preprocessing] MicroTensor shape OK: {t.shape}")
        print(f"Bounds check: min={t.min():.4f}, max={t.max():.4f}")
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)