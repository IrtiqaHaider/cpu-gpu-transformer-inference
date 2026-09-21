"""One figure per metric; no fabricated data and no combined throughput units."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def make_plots(summary_csv, out_dir=None):
    source = Path(summary_csv)
    out = Path(out_dir) if out_dir else source.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(source)
    df = df[df.status == "ok"].copy()
    if df.empty:
        raise ValueError("There are no successful benchmark rows to plot.")
    df["gpu_peak_allocated_MiB"] = df.gpu_peak_allocated_bytes / 2**20
    metrics = {
        "prefill_ms_median": "Prefill latency (ms; last-position logits, cache built)",
        "prefill_input_tokens_per_s_median": "Prefill input tokens / second",
        "ttft_ms_median": "Time to first CPU-ready token (ms)",
        "decode_tpot_ms_median": "Cached decode time per batch step (ms)",
        "decode_generated_tokens_per_s_median": "Cached decode generated tokens / second",
        "generation_generated_tokens_per_s_median": "End-to-end generated tokens / second",
        "gpu_peak_allocated_MiB": "Peak PyTorch GPU allocation (MiB)",
    }
    paths = []
    # Separate figures for each B,S avoid unreadable 20-line plots.
    for (batch, seq), group in df.groupby(["batch_size", "sequence_length"]):
        for metric, label in metrics.items():
            fig, ax = plt.subplots(figsize=(8, 4.8))
            for layout, rows in group.groupby("layout"):
                rows = rows.sort_values("cpu_offload_ratio")
                ax.plot(100 * rows.cpu_offload_ratio, rows[metric], marker="o", label=layout)
            ax.set(xlabel="Transformer blocks executed on CPU (%)", ylabel=label,
                   title=f"B={batch}, prompt={seq} tokens | FP32 | eager attention")
            ax.grid(True, alpha=0.25)
            ax.legend(title="GPU-block layout")
            fig.tight_layout()
            target = out / f"B{batch}_S{seq}_{metric}.png"
            fig.savefig(target, dpi=150)
            plt.close(fig)
            paths.append(target)
    if "profile_prefill_activation_fraction" in df:
        for (batch, seq), group in df.groupby(["batch_size", "sequence_length"]):
            fig, ax = plt.subplots(figsize=(8, 4.8))
            for layout, rows in group.groupby("layout"):
                rows = rows.sort_values("cpu_offload_ratio")
                ax.plot(100 * rows.cpu_offload_ratio,
                        100 * rows.profile_prefill_activation_fraction, marker="o", label=layout)
            ax.set(xlabel="Transformer blocks executed on CPU (%)",
                   ylabel="Activation-copy share of serialized profile (%)",
                   title=f"Diagnostic profile ONLY | B={batch}, S={seq}")
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            target = out / f"B{batch}_S{seq}_profile_transfer_share.png"
            fig.savefig(target, dpi=150)
            plt.close(fig)
            paths.append(target)
    print(f"Wrote {len(paths)} figures to {out}")
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary_csv")
    p.add_argument("--out")
    a = p.parse_args()
    make_plots(a.summary_csv, a.out)


if __name__ == "__main__":
    main()
