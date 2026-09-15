"""
microtensor_store.py
Flattened Columnar Storage & Sequence Reconstruction for 18-D MicroTensors.
Implements:
- 500ms sliding window extraction from Canonical Event Parquet.
- Flattened scalar Parquet storage (9 features + 9 masks + target outcome).
- Deterministic temporal sequence reconstruction (ORDER BY session_id, window_index).
- Leak-free train/val/test splitting grouped by user_id/session_id.
"""

import os
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union, Set
import numpy as np

INVALID_USER_PLACEHOLDERS: Set[str] = {
    "", "unknown", "anonymous", "null", "none", "nan", "undefined", "default", "user"
}


@dataclass
class SplitResult:
    train_ids: Set[str]
    val_ids: Set[str]
    test_ids: Set[str]
    split_by: str  # "user_id" | "session_id"
    seed: int

    def __iter__(self):
        yield self.train_ids
        yield self.val_ids
        yield self.test_ids

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    pa = None  # type: ignore
    pq = None  # type: ignore
    PYARROW_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None  # type: ignore
    PANDAS_AVAILABLE = False

try:
    from config import (
        load_config,
        PipelineConfig,
        MICROTENSOR_DIM,
        NUM_BEHAVIOURAL_FEATURES
    )
    from preprocessing import (
        compute_window_microtensor,
        FEATURE_COLUMN_NAMES,
        MASK_COLUMN_NAMES,
        ALL_MICROTENSOR_COLUMNS,
        find_project_root
    )
    from target_generation import (
        extract_lookahead_outcome,
        OUTCOME_TAXONOMY,
        OUTCOME_NAME_TO_ID
    )
except ImportError:
    # pyrefly: ignore [missing-import]
    from src.config import (
        load_config,
        PipelineConfig,
        MICROTENSOR_DIM,
        NUM_BEHAVIOURAL_FEATURES
    )
    # pyrefly: ignore [missing-import]
    from src.preprocessing import (
        compute_window_microtensor,
        FEATURE_COLUMN_NAMES,
        MASK_COLUMN_NAMES,
        ALL_MICROTENSOR_COLUMNS,
        find_project_root
    )
    # pyrefly: ignore [missing-import]
    from src.target_generation import (
        extract_lookahead_outcome,
        OUTCOME_TAXONOMY,
        OUTCOME_NAME_TO_ID
    )

# PyArrow schema for flattened MicroTensor Parquet
if PYARROW_AVAILABLE:
    MICROTENSOR_SCHEMA = pa.schema([
        ("dataset_id", pa.string()),
        ("session_id", pa.string()),
        ("user_id", pa.string()),
        ("window_index", pa.int32()),
        ("window_start_ms", pa.int64()),
        ("window_end_ms", pa.int64()),
        # 9 Continuous Features
        ("mean_velocity", pa.float32()),
        ("max_velocity", pa.float32()),
        ("mean_acceleration", pa.float32()),
        ("hesitation_count", pa.float32()),
        ("total_trajectory_length", pa.float32()),
        ("dwell_time_ms", pa.float32()),
        ("trajectory_entropy", pa.float32()),
        ("scroll_depth_percentage", pa.float32()),
        ("scroll_velocity", pa.float32()),
        # 9 Modality Masks
        ("mask_mean_velocity", pa.float32()),
        ("mask_max_velocity", pa.float32()),
        ("mask_mean_acceleration", pa.float32()),
        ("mask_hesitation_count", pa.float32()),
        ("mask_total_trajectory_length", pa.float32()),
        ("mask_dwell_time_ms", pa.float32()),
        ("mask_trajectory_entropy", pa.float32()),
        ("mask_scroll_depth_percentage", pa.float32()),
        ("mask_scroll_velocity", pa.float32()),
        # Downstream Target Outcome
        ("target_label", pa.int64()),
        ("target_name", pa.string())
    ])
else:
    MICROTENSOR_SCHEMA = None


def extract_microtensors_from_canonical(
    canonical_parquet_path: str,
    output_parquet_path: Optional[str] = None,
    window_size_ms: int = 500,
    stride_ms: int = 250,
    min_events_per_window: int = 3,
    lookahead_min_ms: int = 500,
    lookahead_max_ms: int = 1500,
    scales: Optional[Dict[str, float]] = None,
    force: bool = False
) -> str:
    """
    Extract windowed 18-D MicroTensors from a canonical event Parquet file and
    save them as flattened scalar columns in Parquet format.
    """
    if not PYARROW_AVAILABLE or MICROTENSOR_SCHEMA is None:
        raise RuntimeError("PyArrow is required for MicroTensor Parquet extraction.")

    in_path = Path(canonical_parquet_path).resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"Canonical Parquet file not found: {in_path}")

    if output_parquet_path is None:
        root = find_project_root()
        dataset_name = in_path.parent.name
        out_path = root / ".data" / "interim" / "microtensors" / f"{dataset_name}_microtensors.parquet"
    else:
        out_path = Path(output_parquet_path).resolve()

    if not force and out_path.is_file() and out_path.stat().st_size > 0:
        print(f"[MicroTensor] Using cached MicroTensor Parquet: {out_path}")
        return str(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[MicroTensor] Extracting MicroTensors from {in_path} -> {out_path}...")

    # Read canonical events table
    table = pq.read_table(str(in_path))
    df = table.to_pandas()

    dataset_id = str(df["dataset_id"].iloc[0]).lower()
    has_pointer = bool(dataset_id in ("adserp", "continuous_kinematics", "captcha_solve_30k", "video_cua", "target_testbed"))
    has_dom = bool(dataset_id in ("adserp", "target_testbed"))
    has_scroll = bool(dataset_id in ("adserp", "continuous_kinematics", "target_testbed"))

    # Group by session_id
    writer = pq.ParquetWriter(str(out_path), schema=MICROTENSOR_SCHEMA, compression="snappy")
    total_windows = 0
    batch_rows: List[Dict[str, Any]] = []

    try:
        session_groups = df.groupby("session_id", sort=False)
        for session_id, sess_df in session_groups:
            sess_df = sess_df.sort_values("timestamp_ms").reset_index(drop=True)
            if len(sess_df) < min_events_per_window:
                continue

            first_row = sess_df.iloc[0]
            dataset_id = str(first_row["dataset_id"])
            user_id = str(first_row["user_id"]) if pd.notna(first_row["user_id"]) else session_id

            vp_w = float(first_row["viewport_width"]) if pd.notna(first_row["viewport_width"]) and first_row["viewport_width"] > 0 else 1920.0
            vp_h = float(first_row["viewport_height"]) if pd.notna(first_row["viewport_height"]) and first_row["viewport_height"] > 0 else 1080.0
            doc_w = float(first_row["document_width"]) if pd.notna(first_row["document_width"]) and first_row["document_width"] > 0 else 1920.0
            doc_h = float(first_row["document_height"]) if pd.notna(first_row["document_height"]) and first_row["document_height"] > 0 else 2000.0

            events_records = sess_df.to_dict(orient="records")
            start_ts = events_records[0]["timestamp_ms"]
            end_ts = events_records[-1]["timestamp_ms"]
            ts_array = np.array([ev["timestamp_ms"] for ev in events_records], dtype=np.int64)

            t_curr = start_ts
            window_idx = 0

            while t_curr + window_size_ms <= end_ts:
                win_end = t_curr + window_size_ms
                i_start = int(np.searchsorted(ts_array, t_curr, side="left"))
                i_end = int(np.searchsorted(ts_array, win_end, side="left"))

                window_evs = events_records[i_start:i_end]
                if len(window_evs) >= min_events_per_window:
                    # Extract 18-D vector (9 features + 9 masks)
                    vec_18 = compute_window_microtensor(
                        window_evs,
                        viewport=(vp_w, vp_h),
                        document=(doc_w, doc_h),
                        window_duration_ms=float(window_size_ms),
                        has_pointer_support=has_pointer,
                        has_dom_support=has_dom,
                        has_scroll_support=has_scroll,
                        scales=scales
                    )

                    # Lookahead target outcome
                    lookahead_start = win_end + lookahead_min_ms
                    lookahead_end = win_end + lookahead_max_ms
                    i_look_start = int(np.searchsorted(ts_array, lookahead_start, side="left"))
                    i_look_end = int(np.searchsorted(ts_array, lookahead_end, side="left"))
                    future_evs = events_records[i_look_start:i_look_end]
                    session_terminated = bool(end_ts <= lookahead_end)
                    label_id, label_name = extract_lookahead_outcome(
                        future_evs,
                        session_terminated=session_terminated
                    )

                    row_dict: Dict[str, Any] = {
                        "dataset_id": dataset_id,
                        "session_id": str(session_id),
                        "user_id": str(user_id),
                        "window_index": int(window_idx),
                        "window_start_ms": int(t_curr),
                        "window_end_ms": int(win_end)
                    }

                    # Flatten 9 features
                    for f_idx, col_name in enumerate(FEATURE_COLUMN_NAMES):
                        row_dict[col_name] = float(vec_18[f_idx])

                    # Flatten 9 masks
                    for m_idx, mask_col in enumerate(MASK_COLUMN_NAMES):
                        row_dict[mask_col] = float(vec_18[9 + m_idx])

                    row_dict["target_label"] = int(label_id)
                    row_dict["target_name"] = str(label_name)

                    batch_rows.append(row_dict)
                    window_idx += 1

                t_curr += stride_ms

            if len(batch_rows) >= 20_000:
                table_batch = pa.Table.from_pylist(batch_rows, schema=MICROTENSOR_SCHEMA)
                writer.write_table(table_batch)
                total_windows += len(batch_rows)
                batch_rows = []

        if batch_rows:
            table_batch = pa.Table.from_pylist(batch_rows, schema=MICROTENSOR_SCHEMA)
            writer.write_table(table_batch)
            total_windows += len(batch_rows)
            batch_rows = []

    finally:
        writer.close()

    print(f"[MicroTensor] Successfully wrote {total_windows} window records to {out_path} ({out_path.stat().st_size / 1024 / 1024:.2f} MB)")
    return str(out_path)


def split_users_leak_free(
    parquet_path: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_seed: int = 42
) -> SplitResult:
    """
    Partition distinct user_ids (or session_ids if user_id is absent or unpopulated)
    across train, validation, and test sets to guarantee zero cross-split temporal leakage.

    Methodological Invariant:
    No user (and therefore no session) appears in more than one partition when reliable
    user_id exists. If reliable user_id is unavailable (all null, empty, or single placeholder),
    falls back cleanly to session_id.
    """
    table = pq.read_table(parquet_path, columns=["user_id", "session_id"])
    df = table.to_pandas()

    # Filter out invalid placeholder tokens
    raw_users = df["user_id"].dropna().astype(str).tolist()
    valid_users = sorted(list({
        u.strip() for u in raw_users
        if u.strip() and u.strip().lower() not in INVALID_USER_PLACEHOLDERS
    }))

    # Reliable user_id requires > 1 distinct non-placeholder user
    if len(valid_users) > 1:
        group_entities = valid_users
        split_by = "user_id"
    else:
        group_entities = sorted(df["session_id"].dropna().astype(str).unique().tolist())
        split_by = "session_id"

    rng = np.random.RandomState(random_seed)
    rng.shuffle(group_entities)

    n_total = len(group_entities)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))

    train_groups = set(group_entities[:n_train])
    val_groups = set(group_entities[n_train : n_train + n_val])
    test_groups = set(group_entities[n_train + n_val :])

    return SplitResult(
        train_ids=train_groups,
        val_ids=val_groups,
        test_ids=test_groups,
        split_by=split_by,
        seed=random_seed
    )


def reconstruct_sequences_from_parquet(
    parquet_path: str,
    seq_len: int = 8,
    stride: int = 1,
    filter_users: Optional[Set[str]] = None,
    filter_sessions: Optional[Set[str]] = None,
    split_result: Optional[SplitResult] = None,
    split_partition: Optional[str] = None,
    enforce_temporal_continuity: bool = False,
    max_step_gap_ms: int = 300
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct deterministic (T, 18) sequential tensors for PyTorch GRU training
    from flattened MicroTensor Parquet scalar columns.
    Enforces deterministic temporal ordering: ORDER BY session_id, window_index.

    Methodological Invariants:
    - Partition filtering (by SplitResult, filter_users, or filter_sessions) is applied
      strictly BEFORE sequence construction.
    - No sequence ever spans multiple sessions.
    - If enforce_temporal_continuity is True, sequences are segmented at time gaps > max_step_gap_ms.
      If False, sequences represent consecutive active windows within the session.
    Returns:
        (sequences: np.ndarray [N, seq_len, 18], targets: np.ndarray [N])
    """
    cols_to_read = [
        "session_id", "user_id", "window_index", "window_start_ms",
        "target_label"
    ] + ALL_MICROTENSOR_COLUMNS

    table = pq.read_table(parquet_path, columns=cols_to_read)
    df = table.to_pandas()

    # Apply SplitResult if provided
    if split_result is not None:
        target_partition = split_partition or "train"
        if target_partition == "train":
            p_ids = split_result.train_ids
        elif target_partition == "val":
            p_ids = split_result.val_ids
        elif target_partition == "test":
            p_ids = split_result.test_ids
        else:
            raise ValueError(f"Unknown split_partition: {target_partition}. Expected 'train', 'val', or 'test'.")

        if split_result.split_by == "user_id":
            filter_users = p_ids
        else:
            filter_sessions = p_ids

    # Apply user/session filter if provided
    if filter_users is not None:
        df = df[df["user_id"].isin(filter_users)]
    if filter_sessions is not None:
        df = df[df["session_id"].isin(filter_sessions)]

    if len(df) == 0:
        return np.empty((0, seq_len, MICROTENSOR_DIM), dtype=np.float32), np.empty((0,), dtype=np.int64)

    # Sort deterministically
    df = df.sort_values(["session_id", "window_index"]).reset_index(drop=True)

    feature_cols = ALL_MICROTENSOR_COLUMNS  # Exactly 18 columns
    feature_matrix = df[feature_cols].to_numpy(dtype=np.float32)
    target_vector = df["target_label"].to_numpy(dtype=np.int64)
    session_ids = df["session_id"].values
    window_starts = df["window_start_ms"].values

    sequences: List[np.ndarray] = []
    targets: List[int] = []

    # Segment by session boundary (and step gap if requested)
    sess_change = session_ids[:-1] != session_ids[1:]
    if enforce_temporal_continuity:
        gap_break = np.diff(window_starts) > max_step_gap_ms
        all_breaks = np.where(sess_change | gap_break)[0] + 1
    else:
        all_breaks = np.where(sess_change)[0] + 1

    slices = np.split(np.arange(len(df)), all_breaks)

    for idx_slice in slices:
        if len(idx_slice) >= seq_len:
            for s in range(0, len(idx_slice) - seq_len + 1, stride):
                window_indices = idx_slice[s : s + seq_len]
                seq_x = feature_matrix[window_indices]  # Shape: (seq_len, 18)
                target_y = target_vector[window_indices[-1]]  # Target of terminal window
                sequences.append(seq_x)
                targets.append(int(target_y))

    if not sequences:
        return np.empty((0, seq_len, MICROTENSOR_DIM), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X = np.stack(sequences, axis=0).astype(np.float32)
    Y = np.array(targets, dtype=np.int64)
    return X, Y


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Edge-AUI MicroTensor Parquet Extraction Driver")
    parser.add_argument("--canonical-path", type=str, default=".data/canonical/adserp/data.parquet", help="Path to canonical event Parquet")
    parser.add_argument("--output-path", type=str, default=".data/interim/microtensors/adserp_microtensors.parquet", help="Path to save MicroTensor Parquet")
    parser.add_argument("--force", action="store_true", help="Force re-extraction")
    args = parser.parse_args()

    extract_microtensors_from_canonical(
        canonical_parquet_path=args.canonical_path,
        output_parquet_path=args.output_path,
        force=args.force
    )
