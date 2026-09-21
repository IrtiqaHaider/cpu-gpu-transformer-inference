"""Repeatable latency/throughput sweep; raw trials and profiles stay separate."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from .engine import OffloadEngine, Trace
from .model import GPT2, GPT2Spec


def configure_cpu(threads: int, seed: int = 123) -> None:
    if threads < 1:
        raise ValueError("threads must be positive.")
    torch.set_num_threads(threads)
    # This is global and can be set only before parallel work begins.
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def metadata(engine: OffloadEngine, args: dict) -> dict:
    def command(argv):
        try:
            return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT,
                                           timeout=15).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            return f"unavailable: {exc}"
    packages = {}
    for package in ("torch", "numpy", "pandas", "safetensors", "huggingface-hub", "transformers"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "platform": platform.platform(), "python": platform.python_version(),
        "cpu_count_logical": os.cpu_count(), "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
        "nvidia_smi": command(["nvidia-smi"]), "lscpu": command(["lscpu"]),
        "packages": packages, "model": engine.model.source,
        "model_config": asdict(engine.model.cfg), "arguments": args,
        "dtype": "float32 on both devices", "attention": "explicit eager causal attention",
        "timing_scope": "Model work + input transfer + greedy selection + CPU token delivery; no tokenization, downloads or placement",
        "profile_scope": "Separately synchronized wall timings; diagnostic only, not raw PCIe DMA",
        "padding": "unsupported; equal-length unpadded synthetic token IDs",
        "eos_policy": "ignored; fixed number of output tokens for every run",
    }


@torch.inference_mode()
def timed_prefill(engine: OffloadEngine, ids: torch.Tensor) -> dict:
    engine.synchronize()
    start = time.perf_counter_ns()
    logits, cache = engine.forward(ids, use_cache=True)
    engine.synchronize()
    elapsed = (time.perf_counter_ns() - start) / 1e6
    # Default logits are [B,1,V]; all S tokens are processed by all blocks and
    # cached. Token selection / output-token delivery are NOT in this metric.
    return {"prefill_ms": elapsed,
            "prefill_input_tokens_per_s": ids.numel() * 1000.0 / elapsed}


@torch.inference_mode()
def timed_generation(engine: OffloadEngine, ids: torch.Tensor, new_tokens: int) -> dict:
    if new_tokens < 2:
        raise ValueError("At least two new tokens are needed to measure cached decode.")
    if ids.shape[1] + new_tokens - 1 > engine.model.cfg.n_positions:
        raise ValueError("Prompt + processed decode tokens exceed context length.")
    engine.synchronize()
    start = time.perf_counter_ns()
    logits, cache = engine.forward(ids, use_cache=True)
    token, host = engine.sample_greedy(logits)
    first_token_ready = time.perf_counter_ns()
    # host has completed its D2H transfer here. Retaining only these small token
    # tensors, not logits or per-step caches, avoids an artificial memory leak.
    output = [host]
    step_ms = []
    previous_ready = first_token_ready
    for _ in range(new_tokens - 1):
        logits, cache = engine.forward(token, cache, use_cache=True)
        token, host = engine.sample_greedy(logits)
        now = time.perf_counter_ns()
        step_ms.append((now - previous_ready) / 1e6)
        previous_ready = now
        output.append(host)
    engine.synchronize()
    end = time.perf_counter_ns()
    total_ms = (end - start) / 1e6
    ttft_ms = (first_token_ready - start) / 1e6
    decode_ms = total_ms - ttft_ms
    batch = ids.shape[0]
    kv = cache.bytes_by_device()
    return {
        "ttft_ms": ttft_ms,
        "generation_ms": total_ms,
        "decode_ms": decode_ms,
        "decode_tpot_ms": decode_ms / (new_tokens - 1),
        "decode_generated_tokens_per_s": batch * (new_tokens - 1) * 1000.0 / decode_ms,
        "generation_generated_tokens_per_s": batch * new_tokens * 1000.0 / total_ms,
        "decode_step_ms": step_ms,
        "final_kv_cpu_bytes": kv["cpu"], "final_kv_gpu_bytes": kv["cuda:0"],
        "final_cached_positions": cache.length,
    }


@torch.inference_mode()
def profile_case(engine: OffloadEngine, ids: torch.Tensor) -> tuple[list[dict], dict]:
    prefill = Trace(engine.uses_cuda)
    logits, cache = engine.forward(ids, trace=prefill)
    token, _ = engine.sample_greedy(logits, prefill)
    decode = Trace(engine.uses_cuda)
    logits, _ = engine.forward(token, cache, trace=decode)
    engine.sample_greedy(logits, decode)
    rows = []
    summary = {}
    for phase, trace in (("prefill", prefill), ("decode", decode)):
        rows.extend({**r, "phase": phase} for r in trace.rows)
        total = sum(r["wall_ms"] for r in trace.rows)
        for category in ("cpu_compute", "gpu_compute", "activation_copy", "token_copy"):
            summary[f"profile_{phase}_{category}_ms"] = trace.total(category)
        summary[f"profile_{phase}_activation_fraction"] = trace.total("activation_copy") / total if total else 0.0
        summary[f"profile_{phase}_activation_bytes"] = sum(
            r["bytes"] for r in trace.rows if r["category"] == "activation_copy")
        expected = engine.boundaries * ids.shape[0] * (ids.shape[1] if phase == "prefill" else 1) * engine.model.cfg.n_embd * 4
        if summary[f"profile_{phase}_activation_bytes"] != expected:
            raise AssertionError("Measured activation bytes disagree with placement boundaries.")
    return rows, summary


@torch.inference_mode()
def reference_outputs(engine: OffloadEngine, ids: torch.Tensor):
    engine.set_placement(0)
    logits, cache = engine.forward(ids, all_logits=True)
    next_ids = (ids[:, -1:] + 1) % engine.model.cfg.vocab_size
    decoded, _ = engine.forward(next_ids, cache, all_logits=True)
    return logits.cpu().clone(), decoded.cpu().clone(), next_ids


@torch.inference_mode()
def check_placement(engine, ids, references) -> dict:
    ref_prefill, ref_decode, next_ids = references
    actual, cache = engine.forward(ids, all_logits=True)
    decoded, _ = engine.forward(next_ids, cache, all_logits=True)
    actual = actual.cpu()
    decoded = decoded.cpu()
    # Absolute tolerance covers low-magnitude logits; the relative term helps
    # large logits. Always save maximum error instead of hiding behind allclose.
    torch.testing.assert_close(actual, ref_prefill, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(decoded, ref_decode, atol=1e-3, rtol=1e-3)
    return {
        "placement_prefill_max_abs_error": (actual - ref_prefill).abs().max().item(),
        "placement_decode_max_abs_error": (decoded - ref_decode).abs().max().item(),
        "placement_argmax_agreement": (actual.argmax(-1) == ref_prefill.argmax(-1)).float().mean().item(),
    }


def run_sweep(engine: OffloadEngine, out_dir: str | Path,
              gpu_layers=(0, 6, 12), batches=(1,), seq_lens=(32, 128),
              layouts=("prefix",), new_tokens=8, warmup=1, repeats=3,
              seed=123, do_profile=True, cli_args=None) -> pd.DataFrame:
    if repeats < 1 or warmup < 0 or new_tokens < 2:
        raise ValueError("Need repeats>=1, warmup>=0 and new_tokens>=2.")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.csv").exists():
        raise FileExistsError(f"{out}/summary.csv exists; use a new output directory.")
    cfg = engine.model.cfg
    check_ids = torch.randint(cfg.vocab_size, (2, min(8, cfg.n_positions - 1)),
                              generator=torch.Generator().manual_seed(seed))
    references = reference_outputs(engine, check_ids)
    args = cli_args or dict(gpu_layers=list(gpu_layers), batches=list(batches),
        seq_lens=list(seq_lens), layouts=list(layouts), new_tokens=new_tokens,
        warmup=warmup, repeats=repeats, seed=seed, do_profile=do_profile)
    (out / "metadata.json").write_text(json.dumps(metadata(engine, args), indent=2))
    configs = list(itertools.product(layouts, gpu_layers, batches, seq_lens))
    random.Random(seed).shuffle(configs)  # Reduce monotonic warm-up/clock drift bias.
    summaries, trials, profiles = [], [], []
    for case_index, (layout, k, batch, seq) in enumerate(configs):
        case = {"case_index": case_index, "layout": layout, "gpu_layers": k,
                "cpu_offload_ratio": 1.0 - k / cfg.n_layer,
                "batch_size": batch, "sequence_length": seq, "new_tokens": new_tokens}
        print(f"[{case_index + 1}/{len(configs)}] {case}", flush=True)
        if batch < 1 or seq < 1 or seq + new_tokens - 1 > cfg.n_positions:
            raise ValueError(f"Invalid experiment dimensions: {case}")
        try:
            gc.collect()
            engine.set_placement(k, layout)
            validation = check_placement(engine, check_ids, references)
            ids = torch.randint(cfg.vocab_size, (batch, seq), generator=
                                torch.Generator().manual_seed(seed + batch * 1009 + seq))
            if engine.uses_cuda:
                engine.synchronize()
                # Release allocations from placement validation BEFORE warmup.
                # Clearing after warmup would make the first timed allocator path cold.
                torch.cuda.empty_cache()
            for _ in range(warmup):
                timed_prefill(engine, ids)
                timed_generation(engine, ids, new_tokens)
            gc.collect()
            if engine.uses_cuda:
                engine.synchronize()
                baseline_allocated = torch.cuda.memory_allocated(0)
                torch.cuda.reset_peak_memory_stats(0)
            else:
                baseline_allocated = 0
            case_trials = []
            for trial in range(repeats):
                values = {**timed_prefill(engine, ids),
                          **timed_generation(engine, ids, new_tokens)}
                step_ms = values.pop("decode_step_ms")
                row = {**case, "trial": trial, **values,
                       "decode_step_ms_json": json.dumps(step_ms)}
                trials.append(row)
                case_trials.append(row)
            memory = {
                "gpu_baseline_allocated_bytes": baseline_allocated,
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(0) if engine.uses_cuda else 0,
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(0) if engine.uses_cuda else 0,
                **engine.resident_bytes(),
            }
            summary = {**case, "status": "ok", "activation_boundaries": engine.boundaries,
                       "dtype": "float32", **validation, **memory}
            for metric in ("prefill_ms", "prefill_input_tokens_per_s", "ttft_ms", "generation_ms",
                           "decode_ms", "decode_tpot_ms", "decode_generated_tokens_per_s",
                           "generation_generated_tokens_per_s"):
                values = [row[metric] for row in case_trials]
                summary[metric + "_median"] = float(np.median(values))
                summary[metric + "_p95"] = float(np.percentile(values, 95))
                summary[metric + "_min"] = float(min(values))
                summary[metric + "_max"] = float(max(values))
            for metric in ("final_kv_cpu_bytes", "final_kv_gpu_bytes", "final_cached_positions"):
                summary[metric] = case_trials[-1][metric]
            summary["gpu_peak_above_baseline_bytes"] = memory["gpu_peak_allocated_bytes"] - baseline_allocated
            if do_profile:
                rows, details = profile_case(engine, ids)
                profiles.extend({**case, **r} for r in rows)
                summary.update(details)
            summaries.append(summary)
        except torch.cuda.OutOfMemoryError as exc:
            # Record failures instead of silently dropping the expensive points.
            summaries.append({**case, "status": "cuda_oom", "error": str(exc)})
            gc.collect()
            torch.cuda.empty_cache()
        # Checkpoint CSVs after every case so a Colab disconnection loses less.
        pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
        pd.DataFrame(trials).to_csv(out / "trials.csv", index=False)
        if profiles:
            pd.DataFrame(profiles).to_csv(out / "profiles.csv", index=False)
    return pd.DataFrame(summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai-community/gpt2")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--gpu-layers", type=int, nargs="+", default=[0, 6, 12])
    parser.add_argument("--layouts", nargs="+", choices=["prefix", "suffix", "interleaved"], default=["prefix"])
    parser.add_argument("--batches", type=int, nargs="+", default=[1])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=min(2, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", default="results/quick")
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--tiny-random", action="store_true", help="Offline software smoke test, NOT a pretrained-model benchmark")
    args = parser.parse_args()
    configure_cpu(args.threads, args.seed)
    if args.tiny_random:
        model = GPT2(GPT2Spec(vocab_size=101, n_positions=128, n_embd=32, n_layer=4, n_head=4))
    else:
        model = GPT2.from_pretrained(args.model, args.revision)
    engine = OffloadEngine(model)
    run_sweep(engine, args.out, args.gpu_layers, args.batches, args.seq_lens,
              args.layouts, args.new_tokens, args.warmup, args.repeats, args.seed,
              not args.no_profile, vars(args))


if __name__ == "__main__":
    main()
