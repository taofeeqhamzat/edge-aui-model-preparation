"""
benchmark_pipeline.py
Empirical benchmark comparing raw legacy CSV/XML ingestion vs. Canonical Parquet pipeline.
Measures:
1. Ingestion/conversion time
2. MicroTensor feature extraction time (legacy vs. canonical Parquet)
3. Sequence reconstruction & PyTorch DataLoader batch throughput
4. Disk storage footprint (bytes)
5. Peak RAM allocation (tracemalloc)
"""

import os
import sys
import time
import tracemalloc
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from data import (
    resolve_adserp_dir,
    convert_adserp_to_canonical,
    find_project_root
)
from preprocessing import (
    parse_adserp_session,
    extract_session_microtensors
)
from microtensor_store import (
    extract_microtensors_from_canonical,
    reconstruct_sequences_from_parquet
)


def run_benchmark(n_sessions: int = 50):
    raw_adserp = resolve_adserp_dir()
    if not raw_adserp or not raw_adserp.is_dir():
        print("AdSERP dataset not available for benchmark.")
        return

    mouse_dir = raw_adserp / "mouse-movement-data"
    meta_dir = raw_adserp / "trial-metadata"
    csv_files = sorted(list(mouse_dir.glob("*.csv")))[:n_sessions]

    print(f"================================================================")
    print(f" EDGE-AUI PIPELINE BENCHMARK: {n_sessions} SESSIONS")
    print(f"================================================================")

    # -------------------------------------------------------------------------
    # 1. Measure Raw Disk Footprint
    # -------------------------------------------------------------------------
    raw_bytes = 0
    for f in csv_files:
        raw_bytes += f.stat().st_size
        xml_f = meta_dir / f"{f.stem}.xml"
        if xml_f.is_file():
            raw_bytes += xml_f.stat().st_size

    print(f"[Storage] Raw CSV + XML size ({n_sessions} sessions): {raw_bytes / 1024:.1f} KB ({raw_bytes / 1024 / 1024:.2f} MB)")

    # -------------------------------------------------------------------------
    # 2. Benchmark BEFORE: Raw Ingestion -> MicroTensor Extraction
    # -------------------------------------------------------------------------
    tracemalloc.start()
    t0_legacy = time.perf_counter()
    legacy_windows_count = 0
    for csv_f in csv_files:
        sess_evs = parse_adserp_session(csv_f.name)
        tensors = extract_session_microtensors(sess_evs)
        legacy_windows_count += len(tensors)
    t_legacy_total = time.perf_counter() - t0_legacy
    current_mem, peak_legacy_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"[Before - Legacy Pipeline]")
    print(f"  - Total Ingestion + Feature Extraction Time: {t_legacy_total:.3f} s")
    print(f"  - Extracted Windows:                         {legacy_windows_count}")
    print(f"  - Peak RAM Allocation:                       {peak_legacy_mem / 1024 / 1024:.2f} MB")

    # -------------------------------------------------------------------------
    # 3. Benchmark AFTER - Step 1: One-time Canonical Conversion
    # -------------------------------------------------------------------------
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        canon_out = os.path.join(tmp_dir, "adserp_canon.parquet")
        micro_out = os.path.join(tmp_dir, "adserp_micro.parquet")

        tracemalloc.start()
        t0_conv = time.perf_counter()
        convert_adserp_to_canonical(
            raw_path=str(raw_adserp),
            output_path=canon_out,
            max_sessions=n_sessions,
            force=True
        )
        t_conv = time.perf_counter() - t0_conv
        _, peak_conv_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        canon_size = os.path.getsize(canon_out)
        print(f"\n[After - Parquet Pipeline: One-Time Canonicalization]")
        print(f"  - Canonical Parquet Conversion Time:         {t_conv:.3f} s")
        print(f"  - Canonical Parquet File Size:               {canon_size / 1024:.1f} KB ({canon_size / 1024 / 1024:.2f} MB)")
        print(f"  - Compression Ratio vs Raw:                 {raw_bytes / max(canon_size, 1):.2f}x")
        print(f"  - Peak RAM Allocation:                       {peak_conv_mem / 1024 / 1024:.2f} MB")

        # ---------------------------------------------------------------------
        # 4. Benchmark AFTER - Step 2: MicroTensor Extraction from Parquet
        # ---------------------------------------------------------------------
        tracemalloc.start()
        t0_micro = time.perf_counter()
        extract_microtensors_from_canonical(
            canonical_parquet_path=canon_out,
            output_parquet_path=micro_out,
            force=True
        )
        t_micro = time.perf_counter() - t0_micro
        _, peak_micro_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        micro_size = os.path.getsize(micro_out)
        print(f"\n[After - Parquet Pipeline: MicroTensor Extraction]")
        print(f"  - Feature Extraction from Parquet:           {t_micro:.3f} s")
        print(f"  - Flattened MicroTensor Parquet Size:        {micro_size / 1024:.1f} KB ({micro_size / 1024 / 1024:.2f} MB)")
        print(f"  - Peak RAM Allocation:                       {peak_micro_mem / 1024 / 1024:.2f} MB")

        # ---------------------------------------------------------------------
        # 5. Benchmark AFTER - Step 3: PyTorch Model Sequence Loading
        # ---------------------------------------------------------------------
        tracemalloc.start()
        t0_seq = time.perf_counter()
        X, Y = reconstruct_sequences_from_parquet(micro_out, seq_len=8)
        t_seq = time.perf_counter() - t0_seq
        _, peak_seq_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"\n[After - Parquet Pipeline: Sequence Loading & Reconstruction]")
        print(f"  - Reconstructed Sequences:                   {len(X)} sequences of shape {X.shape}")
        print(f"  - Sequence Loading Time:                     {t_seq:.3f} s ({len(X) / max(t_seq, 1e-4):,.0f} seq/s)")
        print(f"  - Peak RAM Allocation:                       {peak_seq_mem / 1024 / 1024:.2f} MB")

    # -------------------------------------------------------------------------
    # 6. Summary Comparison
    # -------------------------------------------------------------------------
    speedup = t_legacy_total / max(t_seq, 1e-4)
    print(f"\n================================================================")
    print(f" SUMMARY COMPARISON")
    print(f"================================================================")
    print(f" Iteration Preprocessing: Legacy {t_legacy_total:.3f}s vs. Cached Sequence Loading {t_seq:.3f}s")
    print(f" Iteration Speedup:       {speedup:.1f}x faster iteration during training cycles")
    print(f" Inode Reduction:         {n_sessions * 2} raw files -> 1 canonical + 1 interim Parquet file")
    print(f"================================================================\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Edge-AUI Pipeline Benchmark Driver")
    parser.add_argument("--sessions", type=int, default=50, help="Number of AdSERP sessions to benchmark")
    args = parser.parse_args()

    run_benchmark(n_sessions=args.sessions)
