"""
preprocessing.py
Sliding Window Feature Extraction & Ingestion Pipeline for Edge-AUI Framework.
Handles Continuous Kinematics, High-Volume Trajectories, HMI Sequences, and Action Paths.
"""

import os
import glob
import math
import json
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
    viewport: Tuple[float, float],
    document: Tuple[float, float],
    window_duration_ms: float = 500.0
) -> np.ndarray:
    vp_w, vp_h = viewport
    doc_w, doc_h = document

    mouse_events = window_df[
        window_df["event"].isin(["mousemove", "mouseover", "mousedown", "mouseup", "click"])
    ].copy()

    mean_vel = 0.0
    max_vel = 0.0
    mean_accel = 0.0
    hesitation_cnt = 0.0
    total_traj_len = 0.0
    entropy = 0.0

    if len(mouse_events) >= 2:
        x_norm = mouse_events["xpos"].values / max(vp_w, 1.0)
        y_norm = mouse_events["ypos"].values / max(vp_h, 1.0)
        t_ms = mouse_events["timestamp"].values.astype(np.float64)

        dx_px = np.diff(x_norm) * vp_w
        dy_px = np.diff(y_norm) * vp_h
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

    features = np.array([
        np.clip(mean_vel / 5.0, 0.0, 1.0),
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
    min_events_per_window: int = 2
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
                window_df, viewport, document, window_duration_ms=window_size_ms
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
    *args,
    **kwargs
) -> List[Dict]:
    try:
        df = pd.read_csv(csv_path, sep=";", nrows=max_rows)
    except Exception:
        return []

    if len(df) == 0:
        return []
    
    # Generate synthetic timeline since exact epoch is not always robustly parsed
    df["timestamp"] = np.arange(len(df)) * 50.0  # Approx 50ms per record
    
    start_time = df["timestamp"].min()
    end_time = df["timestamp"].max()
    
    samples = []
    curr_time = start_time
    
    while curr_time + window_size_ms <= end_time:
        if max_windows is not None and len(samples) >= max_windows:
            break
        win_end = curr_time + window_size_ms
        window_df = df[(df["timestamp"] >= curr_time) & (df["timestamp"] < win_end)]
        
        if len(window_df) >= 2:
            # Calculate features from available columns
            mean_vel = window_df["velocity"].mean() if "velocity" in window_df else 0.0
            max_vel = window_df["velocity"].max() if "velocity" in window_df else 0.0
            
            # HVT lacks absolute coordinates and scroll fields, apply modality mask padding
            # Pad scrolling and missing info with 0
            features = np.array([
                np.clip(mean_vel / 5.0, 0.0, 1.0),
                np.clip(max_vel / 10.0, 0.0, 1.0),
                0.0, # meanAccel
                0.0, # hesitationCount
                np.clip(window_df["distance"].sum() / 2000.0, 0.0, 1.0) if "distance" in window_df else 0.0,
                0.0, # dwellTimeMs
                0.0, # scrollDepth
                0.0, # scrollVelocity
                0.0  # entropy
            ], dtype=np.float32)
            
            samples.append({
                "features": features,
                "label": LABEL_MAP["IDLE"] # Auto-label mostly generic motion in HVT
            })
            
        curr_time += stride_ms
        
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
                    
            # 3. Structural HMI Sequences
            hmi_dir = find_dataset_dir(data_root, "structural-hmi-sequences-2023")
            if os.path.exists(hmi_dir) and not _enough_windows():
                hmi_files = glob.glob(os.path.join(hmi_dir, "**", "*.csv"), recursive=True)
                if max_files_per_dataset:
                    hmi_files = hmi_files[:max_files_per_dataset]
                for csv_path in hmi_files:
                    windows = process_hmi_sequences(csv_path)
                    all_windows.extend(windows)
                    if _enough_windows():
                        break

            # 4. Client-Side Action Paths
            ap_dir = find_dataset_dir(data_root, "client-side-action-paths-2021")
            if os.path.exists(ap_dir) and not _enough_windows():
                ap_files = glob.glob(os.path.join(ap_dir, "**", "*.json"), recursive=True)
                if max_files_per_dataset:
                    ap_files = ap_files[:max_files_per_dataset]
                for json_path in ap_files:
                    windows = process_action_paths(json_path)
                    all_windows.extend(windows)
                    if _enough_windows():
                        break

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

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            return self.samples_X[idx], self.samples_Y[idx]
