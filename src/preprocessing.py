"""
preprocessing.py
Sliding Window Feature Extraction & Ingestion Pipeline for Edge-AUI Framework.
Handles Continuous Kinematics, High-Volume Trajectories, HMI Sequences, and Action Paths.
"""

import os
import glob
import math
import json
import sys
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================================
# 1. Metadata and Raw Log Parsers
# ============================================================================

def parse_viewport_metadata(xml_path: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    viewport_w, viewport_h = 1366.0, 768.0 
    doc_w, doc_h = 1366.0, 2000.0

    if os.path.exists(xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            win_node = root.find("window")
            if win_node is not None and win_node.text and "x" in win_node.text:
                parts = win_node.text.strip().split("x")
                viewport_w, viewport_h = float(parts[0]), float(parts[1])

            doc_node = root.find("document")
            if doc_node is not None and doc_node.text and "x" in doc_node.text:
                parts = doc_node.text.strip().split("x")
                doc_w, doc_h = float(parts[0]), float(parts[1])
        except Exception:
            pass

    return (viewport_w, viewport_h), (doc_w, doc_h)


def load_raw_kinematics_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=" ", engine="python")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df



# ============================================================================
# 2. MicroTensor Vectorizer (500 ms Window Extraction)
# ============================================================================

FEATURE_NAMES = [
    "meanVelocity",
    "maxVelocity",
    "meanAcceleration",
    "hesitationCount",
    "totalTrajectoryLength",
    "dwellTimeMs",
    "scrollDepthPercentage",
    "scrollVelocity",
    "trajectoryEntropy"
]

LABEL_MAP = {
    "IDLE": 0,
    "CLICK": 1,
    "FORM_SUBMIT": 2,
    "BACKTRACK": 3,
    "RAPID_SCROLL": 4,
    "HOVER_DWELL": 5
}

def compute_window_microtensor(
    window_df: pd.DataFrame,
    viewport: Tuple[float, float] = (1366.0, 768.0),
    document: Tuple[float, float] = (1366.0, 2000.0),
    window_duration_ms: float = 500.0,
    dataset_type: str = "ck"
) -> np.ndarray:
    vp_w, vp_h = viewport
    doc_w, doc_h = document

    mean_vel = 0.0
    max_vel = 0.0
    mean_accel = 0.0
    hesitation_cnt = 0.0
    total_traj_len = 0.0
    dwell_time_ms = 0.0
    scroll_depth_pct = 0.0
    scroll_vel = 0.0
    entropy = 0.0

    # Dataset conditional block: High-Volume Trajectories (HVT)
    if dataset_type.lower() in ["hvt", "high-volume-trajectories-20226"] or ("velocity" in window_df.columns and "xpos" not in window_df.columns):
        # 1. Unit harmonization: convert raw timestamps from microseconds (us) to milliseconds (ms)
        t_raw = window_df["timestamp"].values.astype(np.float64)
        t_ms = t_raw / 1000.0

        # 2. Unit harmonization: convert raw velocity from px/s to px/ms
        if "velocity" in window_df.columns:
            vel_series = window_df["velocity"].dropna()
            vel_px_ms = vel_series.values.astype(np.float64) / 1000.0  # px/s -> px/ms
        else:
            vel_px_ms = np.array([], dtype=np.float64)

        if len(vel_px_ms) > 0:
            mean_vel = float(np.mean(vel_px_ms))
            max_vel = float(np.max(vel_px_ms))

        # Acceleration derived from velocity deltas and time intervals (px/ms^2)
        if len(vel_px_ms) >= 2:
            vel_indices = vel_series.index
            vel_t_ms = t_ms[window_df.index.get_indexer(vel_indices)]
            dt_ms = np.diff(vel_t_ms)
            dt_ms = np.where(dt_ms <= 0, 1.0, dt_ms)
            accels = np.abs(np.diff(vel_px_ms)) / dt_ms
            mean_accel = float(np.mean(accels))

        # Directional hesitation count and entropy from angles
        if "angle" in window_df.columns:
            angles = window_df["angle"].dropna().values.astype(np.float64)
            if len(angles) >= 2:
                angle_diffs = np.abs(np.diff(angles))
                angle_diffs = np.where(angle_diffs > 180.0, 360.0 - angle_diffs, angle_diffs)
                hesitation_cnt = float(np.sum(angle_diffs > 45.0))

                hist, _ = np.histogram(angles, bins=8, range=(-180.0, 180.0), density=False)
                hist = hist / max(hist.sum(), 1)
                hist = hist[hist > 0]
                if len(hist) > 0:
                    entropy = float(-np.sum(hist * np.log2(hist)) / math.log2(8))

        if "distance" in window_df.columns:
            total_traj_len = float(window_df["distance"].dropna().sum())

        if "duration" in window_df.columns:
            dwell_time_ms = float(window_df["duration"].dropna().sum())

        # Scroll fields are absent in HVT (modality mask padding)
        scroll_depth_pct = 0.0
        scroll_vel = 0.0

    else:
        # Continuous Kinematics (CK)
        mouse_events = window_df[
            window_df["event"].isin(["mousemove", "mouseover", "mousedown", "mouseup", "click"])
        ].copy()

        # Kinematic smoothing: enforce len(mouse_events) >= 3 for valid displacement & acceleration
        if len(mouse_events) >= 3:
            # pyrefly: ignore [unsupported-operation]
            x_norm = mouse_events["xpos"].values / max(vp_w, 1.0)
            # pyrefly: ignore [unsupported-operation]
            y_norm = mouse_events["ypos"].values / max(vp_h, 1.0)
            t_ms = mouse_events["timestamp"].values.astype(np.float64)

            dx_px = np.diff(x_norm) * vp_w
            dy_px = np.diff(y_norm) * vp_h
            # pyrefly: ignore [no-matching-overload]
            dt_ms = np.diff(t_ms)
            dt_ms = np.where(dt_ms <= 0, 1.0, dt_ms)

            distances = np.sqrt(dx_px**2 + dy_px**2)
            total_traj_len = float(np.sum(distances))

            velocities = distances / dt_ms
            mean_vel = float(np.mean(velocities))
            max_vel = float(np.max(velocities))

            if len(velocities) >= 2:
                accels = np.abs(np.diff(velocities)) / dt_ms[1:]
                mean_accel = float(np.mean(accels))

            angles = np.arctan2(dy_px, dx_px)
            angle_diffs = np.abs(np.diff(angles))
            angle_diffs = np.where(angle_diffs > np.pi, 2 * np.pi - angle_diffs, angle_diffs)
            hesitation_cnt = float(np.sum(angle_diffs > (np.pi / 4.0)))

            hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi), density=False)
            hist = hist / max(hist.sum(), 1)
            hist = hist[hist > 0]
            if len(hist) > 0:
                entropy = float(-np.sum(hist * np.log2(hist)) / math.log2(8))

        hover_events = window_df[window_df["event"] == "mouseover"]
        dwell_time_ms = float(len(hover_events) * 50.0)

        scroll_events = window_df[window_df["event"] == "scroll"]
        scroll_cnt = len(scroll_events)
        scroll_vel = (scroll_cnt * 100.0) / window_duration_ms
        max_scrollable = max(doc_h - vp_h, 1.0)
        scroll_depth_pct = min(100.0, (scroll_cnt * 50.0 / max_scrollable) * 100.0)

    # Symmetric scaling: divide both meanVelocity and maxVelocity by 10.0
    features = np.array([
        np.clip(mean_vel / 10.0, 0.0, 1.0),
        np.clip(max_vel / 10.0, 0.0, 1.0),
        np.clip(mean_accel / 0.1, 0.0, 1.0),
        np.clip(hesitation_cnt / 10.0, 0.0, 1.0),
        np.clip(total_traj_len / 2000.0, 0.0, 1.0),
        np.clip(dwell_time_ms / window_duration_ms, 0.0, 1.0),
        np.clip(scroll_depth_pct / 100.0, 0.0, 1.0),
        np.clip(scroll_vel / 5.0, 0.0, 1.0),
        np.clip(entropy, 0.0, 1.0)
    ], dtype=np.float32)

    return features

# ============================================================================
# 3. Temporal Window Ingestion & Outcome Label Extractor (Continuous Kinematics)
# ============================================================================

def process_ck_session(
    csv_path: str,
    window_size_ms: int = 500,
    stride_ms: int = 250,
    prediction_horizon_ms: int = 1500,
    min_events_per_window: int = 3
) -> List[Dict]:
    xml_path = csv_path.replace(".csv", ".xml")
    viewport, document = parse_viewport_metadata(xml_path)
    df = load_raw_kinematics_csv(csv_path)

    if len(df) == 0:
        return []

    start_time = df["timestamp"].min()
    end_time = df["timestamp"].max()

    samples = []
    curr_time = start_time

    while curr_time + window_size_ms <= end_time:
        win_end = curr_time + window_size_ms
        window_df = df[(df["timestamp"] >= curr_time) & (df["timestamp"] < win_end)]

        if len(window_df) >= min_events_per_window:
            micro_tensor = compute_window_microtensor(
                window_df, viewport, document, window_duration_ms=window_size_ms, dataset_type="ck"
            )

            horizon_end = win_end + prediction_horizon_ms
            future_df = df[(df["timestamp"] >= win_end) & (df["timestamp"] < horizon_end)]

            label = LABEL_MAP["IDLE"]
            if len(future_df) > 0:
                events = future_df["event"].values
                if "click" in events or "mousedown" in events:
                    label = LABEL_MAP["CLICK"]
                elif "scroll" in events and len(future_df[future_df["event"] == "scroll"]) > 4:
                    label = LABEL_MAP["RAPID_SCROLL"]
                elif "mouseover" in events:
                    label = LABEL_MAP["HOVER_DWELL"]
                elif "beforeunload" in events or "blur" in events:
                    label = LABEL_MAP["BACKTRACK"]

            samples.append({
                "window_start": curr_time,
                "window_end": win_end,
                "features": micro_tensor,
                "label": label
            })

        curr_time += stride_ms

    return samples

# ============================================================================
# 4. HVT (High Volume Trajectories) Processing
# ============================================================================

def process_hvt_session(
    csv_path: str,
    window_size_ms: int = 500,
    stride_ms: int = 250,
    max_rows: Optional[int] = 50000,
    max_windows: Optional[int] = None,
    min_events_per_window: int = 3,
    *args,
    **kwargs
) -> List[Dict]:
    try:
        df = pd.read_csv(csv_path, sep=";", nrows=max_rows)
    except Exception:
        return []

    if len(df) == 0:
        return []

    # Parse raw timestamp in microseconds (us) from local_date and local_time
    if "local_time" in df.columns:
        try:
            if "local_date" in df.columns:
                dt = pd.to_datetime(df["local_date"].astype(str) + " " + df["local_time"].astype(str), format="ISO8601")
            else:
                dt = pd.to_datetime(df["local_time"].astype(str), format="ISO8601")
            df["timestamp"] = (dt.astype("int64") // 1000).astype(np.float64)  # microseconds (us)
        except Exception:
            df["timestamp"] = np.arange(len(df)) * 50000.0  # fallback: 50ms in us
    elif "timestamp" not in df.columns:
        df["timestamp"] = np.arange(len(df)) * 50000.0  # fallback: 50ms in us

    df = df.sort_values("timestamp").reset_index(drop=True)
    ts_values = df["timestamp"].values

    window_size_us = window_size_ms * 1000.0
    stride_us = stride_ms * 1000.0

    curr_time = float(ts_values[0])
    end_time = float(ts_values[-1])

    samples = []
    while curr_time + window_size_us <= end_time:
        if max_windows is not None and len(samples) >= max_windows:
            break

        win_end = curr_time + window_size_us
        i_start = int(np.searchsorted(ts_values, curr_time, side="left"))
        i_end = int(np.searchsorted(ts_values, win_end, side="left"))

        if i_end - i_start >= min_events_per_window:
            window_df = df.iloc[i_start:i_end]
            micro_tensor = compute_window_microtensor(
                window_df,
                viewport=(1920.0, 1080.0),
                document=(1920.0, 1080.0),
                window_duration_ms=window_size_ms,
                dataset_type="hvt"
            )

            samples.append({
                "window_start": curr_time / 1000.0,
                "window_end": win_end / 1000.0,
                "features": micro_tensor,
                "label": LABEL_MAP["IDLE"]  # Auto-label mostly generic motion in HVT
            })
            curr_time += stride_us
        else:
            if i_start < len(ts_values) - 1:
                next_t = float(ts_values[i_start + 1])
                if next_t > curr_time + stride_us:
                    curr_time = next_t
                else:
                    curr_time += stride_us
            else:
                break

    return samples

# ============================================================================
# 5. Structural Grounding Processing (HMI & Action Paths)
# ============================================================================

def process_hmi_sequences(csv_path: str, max_records: Optional[int] = 500, *args, **kwargs) -> List[Dict]:
    try:
        df = pd.read_csv(csv_path, sep=";", nrows=max_records)
        if "epoch" not in df and "initepoch" not in df:
            df = pd.read_csv(csv_path, sep=",", nrows=max_records)
    except Exception:
        return []

    samples = []
    epoch_col = "epoch" if "epoch" in df else "initepoch" if "initepoch" in df else "timestamp" if "timestamp" in df else None
    
    if len(df) > 0 and epoch_col is not None:
        df = df.sort_values(epoch_col)
        # Just map structural sequence into the feature space using categorical placeholders 
        # or modality masking.
        # Here we pad the continuous features with zeroes to indicate structural discrete events.
        for idx in range(len(df) - 1):
            features = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
            features[0] = 1.0 # Signal discrete event
            
            samples.append({
                "features": features,
                "label": LABEL_MAP["CLICK"] # Assume structural interactions are clicks/submits
            })
            
    return samples

def process_action_paths(json_path: str, *args, **kwargs) -> List[Dict]:
    samples = []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        items = []
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict):
                    if "clickstream" in entry and isinstance(entry["clickstream"], list):
                        items.extend(entry["clickstream"])
                    elif "actions" in entry and isinstance(entry["actions"], list):
                        items.extend(entry["actions"])
                    else:
                        items.append(entry)
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, list):
                    items.extend(v)
                elif isinstance(v, dict):
                    items.append(v)

        for task in items:
            if not isinstance(task, dict):
                continue
            if "stay_seconds" in task:
                dwell_ms = float(task["stay_seconds"]) * 1000.0
                
                features = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
                features[5] = np.clip(dwell_ms / 500.0, 0.0, 1.0) # Map to dwellTimeMs
                
                samples.append({
                    "features": features,
                    "label": LABEL_MAP["BACKTRACK"] if "previous_url" in task else LABEL_MAP["IDLE"]
                })
    except Exception:
        pass
    return samples


# ============================================================================
# 6. PyTorch Dataset Implementation
# ============================================================================

try:
    from data_manager import ensure_dataset, find_dataset_dir, resolve_dataset_root
except ImportError:
    try:
        # pyrefly: ignore [missing-import]
        from src.data_manager import ensure_dataset, find_dataset_dir, resolve_dataset_root
    except ImportError:
        ensure_dataset = None
        find_dataset_dir = lambda root, name: os.path.join(str(root), name)
        resolve_dataset_root = lambda p: str(p)

if TORCH_AVAILABLE:
    class MicroInteractionSequenceDataset(Dataset):
        def __init__(
            self,
            data_root: Optional[str] = None,
            seq_len: int = 8,
            window_size_ms: int = 500,
            stride_ms: int = 250,
            prediction_horizon_ms: int = 1500,
            hf_repo_id: str = "T40/edge-aui-framework-data",
            hf_token: Optional[str] = None,
            max_sequences: Optional[int] = None,
            max_files_per_dataset: Optional[int] = None,
            allow_patterns: Optional[List[str]] = None
        ):
            self.seq_len = seq_len
            self.samples_X = []
            self.samples_Y = []

            # Resolve local data directory or auto-sync from Hugging Face Hub
            if ensure_dataset is not None:
                if data_root is None or not os.path.exists(data_root) or not any(os.scandir(data_root)):
                    data_root = ensure_dataset(
                        data_dir=data_root,
                        repo_id=hf_repo_id,
                        token=hf_token,
                        allow_patterns=allow_patterns
                    )
            elif data_root is None:
                data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".data", "raw"))

            data_root = resolve_dataset_root(data_root)
            self.data_root = data_root
            all_windows = []

            def _enough_windows():
                return max_sequences is not None and len(all_windows) >= max_sequences + seq_len

            # 1. Continuous Kinematics (CK)
            ck_dir = find_dataset_dir(data_root, "continuous-kinematics-2020")
            if os.path.exists(ck_dir) and not _enough_windows():
                ck_files = glob.glob(os.path.join(ck_dir, "**", "*.csv"), recursive=True)
                if max_files_per_dataset:
                    ck_files = ck_files[:max_files_per_dataset]
                for csv_path in ck_files:
                    windows = process_ck_session(csv_path, window_size_ms, stride_ms, prediction_horizon_ms)
                    all_windows.extend(windows)
                    if _enough_windows():
                        break
            
            # 2. High-Volume Trajectories (HVT)
            hvt_dir = find_dataset_dir(data_root, "high-volume-trajectories-20226")
            if os.path.exists(hvt_dir) and not _enough_windows():
                hvt_files = glob.glob(os.path.join(hvt_dir, "**", "*.csv"), recursive=True)
                if max_files_per_dataset:
                    hvt_files = hvt_files[:max_files_per_dataset]
                for csv_path in hvt_files:
                    windows = process_hvt_session(csv_path, window_size_ms, stride_ms)
                    all_windows.extend(windows)
                    if _enough_windows():
                        break
                    
            # Note: Discrete UI logs (structural-hmi-sequences-2023 and client-side-action-paths-2021)
            # are excluded from continuous MicroTensor sequence extraction to avoid category errors
            # and artificial collinearity. They are reserved for deterministic Fast Gate (PrefixSpan) benchmarking.

            if len(all_windows) >= seq_len:
                feature_matrix = np.stack([w["features"] for w in all_windows])
                labels = np.array([w["label"] for w in all_windows])

                for i in range(len(all_windows) - seq_len + 1):
                    seq_x = feature_matrix[i : i + seq_len]
                    target_y = labels[i + seq_len - 1]
                    self.samples_X.append(seq_x)
                    self.samples_Y.append(target_y)

            if len(self.samples_X) > 0:
                if max_sequences is not None and len(self.samples_X) > max_sequences:
                    self.samples_X = self.samples_X[:max_sequences]
                    self.samples_Y = self.samples_Y[:max_sequences]

                self.samples_X = torch.tensor(np.array(self.samples_X), dtype=torch.float32)
                self.samples_Y = torch.tensor(np.array(self.samples_Y), dtype=torch.long)
            else:
                self.samples_X = torch.empty((0, seq_len, len(FEATURE_NAMES)), dtype=torch.float32)
                self.samples_Y = torch.empty((0,), dtype=torch.long)

        def __len__(self) -> int:
            return len(self.samples_Y)

        # pyrefly: ignore [bad-override-param-name]
        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            return self.samples_X[idx], self.samples_Y[idx]

if __name__ == "__main__":
    if len(sys.argv) < 2:
            print("Usage: python preprocessing.py <function_name> <data_root> <optional function arguments>")
            sys.exit(1)
    else:
        arg = sys.argv[1]
        if arg == "parse_viewport_metadata":
            parsed_viewport_metadata = parse_viewport_metadata(sys.argv[2])
            print(f"viewport_metadata: {parsed_viewport_metadata}")
        elif arg == "process_action_paths":
            print(f"[Preprocessing] Processing action paths: {sys.argv[2]}")
            processed_action_paths = process_action_paths(sys.argv[2])
            print(f"processed_action_paths: {processed_action_paths}")
        elif arg == "process_ck_session":
            print(f"[Preprocessing] Processing ck session: {sys.argv[2]}")
            processed_ck_session = process_ck_session(sys.argv[2])
            print(f"processed_ck_session: {processed_ck_session}")
        elif arg == "process_hmi_sequences":
            print(f"[Preprocessing] Processing hmi sequences: {sys.argv[2]}")
            processed_hmi_sequences = process_hmi_sequences(sys.argv[2])
            print(f"processed_hmi_sequences: {processed_hmi_sequences}")
        elif arg == "process_hvt_session":
            print(f"[Preprocessing] Processing hvt session: {sys.argv[2]}")
            processed_hvt_session = process_hvt_session(sys.argv[2])
            print(f"processed_hvt_session: {processed_hvt_session}")
        else:
            print("Usage: python preprocessing.py <function_name> <data_root> <optional function arguments>")
            sys.exit(1)