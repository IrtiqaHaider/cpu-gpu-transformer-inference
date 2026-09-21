"""Preallocated pageable/pinned H2D and D2H copy-path microbenchmark.

CUDA event spans and synchronized host wall times are BOTH reported. Neither
should be advertised as a direct measurement of raw physical PCIe line rate.
The model engine itself intentionally uses ordinary blocking .to() copies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import pandas as pd
import torch


@torch.inference_mode()
def run_copy_benchmark(out_dir="results/copies", sizes_bytes=(3072, 98304, 393216, 6291456, 67108864),
                       repeats=50, warmup=5):
    if not torch.cuda.is_available():
        raise RuntimeError("The copy benchmark requires a CUDA GPU.")
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive and warmup nonnegative.")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "copy_summary.csv").exists():
        raise FileExistsError("Copy output exists; choose a new directory.")
    raw, summaries = [], []
    for nbytes in sizes_bytes:
        if nbytes <= 0 or nbytes % 4:
            raise ValueError("Each byte size must be positive and divisible by four.")
        for pinned in (False, True):
            host = torch.full((nbytes // 4,), 1.25, dtype=torch.float32,
                              pin_memory=pinned)
            gpu = torch.empty_like(host, device="cuda:0")
            for direction in ("H2D", "D2H"):
                src, dst = (host, gpu) if direction == "H2D" else (gpu, host)
                # H2D has initialized GPU before the D2H case. Buffers are not
                # mutated or reused until completion is confirmed.
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                # Initialize lazily allocated event resources before measurement.
                start_event.record(); end_event.record(); end_event.synchronize()
                for _ in range(warmup):
                    dst.copy_(src, non_blocking=True)
                    torch.cuda.synchronize(0)
                case = []
                for trial in range(repeats):
                    torch.cuda.synchronize(0)
                    start = time.perf_counter_ns()
                    start_event.record()
                    dst.copy_(src, non_blocking=True)
                    end_event.record()
                    end_event.synchronize()  # Required before a host reads D2H results.
                    wall_ms = (time.perf_counter_ns() - start) / 1e6
                    event_ms = start_event.elapsed_time(end_event)
                    row = {"bytes": nbytes, "pinned": pinned, "direction": direction,
                           "trial": trial, "wall_ms": wall_ms, "cuda_event_span_ms": event_ms,
                           "wall_effective_GB_per_s": nbytes / (wall_ms * 1e6),
                           "event_span_effective_GB_per_s": nbytes / (event_ms * 1e6) if event_ms else float("nan")}
                    raw.append(row); case.append(row)
                # Correctness check happens outside all timed regions.
                torch.testing.assert_close(dst.cpu(), src.cpu(), rtol=0, atol=0)
                summary = {"bytes": nbytes, "pinned": pinned, "direction": direction}
                for key in ("wall_ms", "cuda_event_span_ms", "wall_effective_GB_per_s", "event_span_effective_GB_per_s"):
                    summary[key + "_median"] = statistics.median(r[key] for r in case)
                summaries.append(summary)
            del host, gpu, src, dst
    pd.DataFrame(raw).to_csv(out / "copy_trials.csv", index=False)
    result = pd.DataFrame(summaries)
    result.to_csv(out / "copy_summary.csv", index=False)
    (out / "metadata.json").write_text(json.dumps({
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "repeats": repeats, "warmup": warmup,
        "buffers": "preallocated; no allocation or pinning inside timed interval",
        "timing_caveat": "Event spans can include submission/staging stalls, especially pageable memory. Wall spans include host/event/sync overhead. No overlap or raw wire-rate claim.",
    }, indent=2))
    print(result.to_string(index=False))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/copies")
    p.add_argument("--sizes-bytes", nargs="+", type=int, default=[3072, 98304, 393216, 6291456, 67108864])
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    a = p.parse_args()
    run_copy_benchmark(a.out, a.sizes_bytes, a.repeats, a.warmup)


if __name__ == "__main__":
    main()
