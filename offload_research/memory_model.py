"""GPT-2 FP32 allocated-memory predictor for the original eager prefix engine.

Exact parameter/buffer/final-cache payloads + a fitted nonnegative residual.
The residual also absorbs allocator/workspace and temporary-cache effects.
This is NOT a physical VRAM capacity guarantee or a serving admission controller.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from .cost_model import KEYS, Workload

MIB = 2**20
MEMORY_SCHEMA = "0.2.0"
MEMORY_RIDGE = 1e-6
FEATURE_NAMES = ["gpu_active", "prefill_hidden_mib", "prefill_attention_mib",
                 "final_gpu_kv_mib", "last_position_logits_mib"]


def check_config(config: dict) -> None:
    expected = {"n_layer": 12, "n_embd": 768, "n_head": 12,
                "vocab_size": 50257, "n_positions": 1024}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("Memory model is scoped to the original GPT-2 small engine/configuration.")
    if config.get("n_inner") not in (None, 3072):
        raise ValueError("Unsupported inner width.")


def components(w: Workload, config: dict) -> dict:
    """Payloads, not allocator-rounded sizes. Output weight is tied, counted once."""
    check_config(config)
    w.validate(config)
    h, p, v = config["n_embd"], config["n_positions"], config["vocab_size"]
    inner = config.get("n_inner") or 4*h
    # QKV+projection, 2 layer norms, MLP and all biases.
    block_params = 4*h*h + 2*h*inner + 9*h + inner
    endpoint_params = v*h + p*h + 2*h
    active, k = int(w.gpu_layers > 0), w.gpu_layers
    gpu_params = (k*block_params + active*endpoint_params)*4
    cpu_params = ((config["n_layer"]-k)*block_params + (1-active)*endpoint_params)*4
    # One [1,1,P,P] bool causal mask per block, element size 1.
    gpu_buffers = k*p*p
    final_positions = w.sequence_length + w.new_tokens - 1
    gpu_kv = 2*k*w.batch_size*final_positions*h*4
    return {"gpu_parameter_bytes": gpu_params, "cpu_parameter_bytes": cpu_params,
            "gpu_buffer_bytes": gpu_buffers, "final_kv_gpu_bytes": gpu_kv,
            "final_cached_positions": final_positions,
            "payload_floor_bytes": gpu_params + gpu_buffers + gpu_kv}


def features(w: Workload, config: dict) -> np.ndarray:
    c = components(w, config)
    active = int(w.gpu_layers > 0)
    b, s, h = w.batch_size, w.sequence_length, config["n_embd"]
    return np.asarray([active, active*b*s*h*4/MIB,
                       active*b*config["n_head"]*s*s*4/MIB,
                       c["final_kv_gpu_bytes"]/MIB,
                       active*b*config["vocab_size"]*4/MIB], dtype=float)


def workload_rows(frame: pd.DataFrame):
    return [Workload(**r) for r in frame[KEYS].to_dict("records")]


def fit_memory(frame: pd.DataFrame, metadata: dict) -> dict:
    if frame.empty or frame.duplicated(KEYS).any():
        raise ValueError("Need nonempty, unique calibration rows.")
    config = metadata["model_config"]
    check_config(config)
    rows = workload_rows(frame)
    all_x = np.stack([features(w, config) for w in rows])
    floors = np.asarray([components(w, config)["payload_floor_bytes"]/MIB for w in rows])
    actual = frame.gpu_peak_allocated_bytes.to_numpy(dtype=float)/MIB
    if not np.isfinite(actual).all() or (actual < floors-1e-6).any():
        raise ValueError("Invalid measured allocation or allocation below the payload floor.")
    gpu = frame.gpu_layers.to_numpy() > 0
    if not gpu.any():
        raise ValueError("GPU calibration rows are required.")
    if (actual[~gpu] != 0).any():
        raise ValueError("CPU-only workload must have zero modeled GPU allocation.")
    x, y = all_x[gpu], np.maximum(actual[gpu]-floors[gpu], 0)
    scales = np.maximum(x.max(axis=0), 1.0)
    xs = x/scales
    # Absolute-error NNLS, training-only scaling. Ridge is fixed, not CV-tuned.
    coef, _ = nnls(np.vstack([xs, np.sqrt(MEMORY_RIDGE)*np.eye(x.shape[1])]),
                   np.r_[y, np.zeros(x.shape[1])], maxiter=10000)
    return {"schema_version": MEMORY_SCHEMA, "model_config": config,
            "model_source": metadata["model"],
            "feature_names": FEATURE_NAMES,
            "feature_scales": scales.tolist(), "coefficients_mib": coef.tolist(),
            "design_rank": int(np.linalg.matrix_rank(xs)), "ridge": MEMORY_RIDGE,
            "training_rows": len(frame), "training_gpu_rows": int(gpu.sum()),
            "domain": {"batch_size_range": [int(frame.batch_size.min()),int(frame.batch_size.max())],
                       "sequence_length_range": [int(frame.sequence_length.min()),int(frame.sequence_length.max())],
                       "output_lengths_seen": sorted(int(n) for n in frame.new_tokens.unique()),
                       "gpu_counts_seen": sorted(int(k) for k in frame.gpu_layers.unique()),
                       "layout": "prefix"},
            "scope": "Peak PyTorch allocated bytes across the original timed case, NOT total device memory.",
            "residual": "Empirical temporary/workspace/allocator/cache-growth overhead; not a causal decomposition."}


def predict_memory_frame(model: dict, frame: pd.DataFrame) -> pd.DataFrame:
    config = model["model_config"]
    rows = workload_rows(frame)
    values = []
    for w in rows:
        c = components(w, config)
        overhead = float((features(w, config)/model["feature_scales"]) @ model["coefficients_mib"])
        peak = c["payload_floor_bytes"]/MIB + max(overhead, 0.0)
        values.append({**w.__dict__, **c, "predicted_overhead_mib": overhead,
                       "predicted_peak_mib": peak})
    return pd.DataFrame(values)


@dataclass(frozen=True)
class MemoryGuard:
    """Explicit heuristic buffer. NOT a confidence bound or OOM guarantee."""
    relative: float = 0.05
    extra_mib: float = 16.0

    def __post_init__(self):
        if not np.isfinite([self.relative, self.extra_mib]).all() or self.relative < 0 or self.extra_mib < 0:
            raise ValueError("Memory guard must be finite and nonnegative.")

    def apply(self, predicted_peak_mib: float, gpu_layers: int) -> float:
        if not np.isfinite(predicted_peak_mib) or predicted_peak_mib < 0:
            raise ValueError("Invalid predicted memory.")
        return 0.0 if gpu_layers == 0 else predicted_peak_mib*(1+self.relative)+self.extra_mib


class MemoryModel:
    def __init__(self, model: dict):
        if model.get("schema_version") != MEMORY_SCHEMA:
            raise ValueError("Unsupported memory schema.")
        check_config(model["model_config"])
        if model.get("feature_names") != FEATURE_NAMES:
            raise ValueError("Unexpected memory features.")
        for key in ("feature_scales", "coefficients_mib"):
            a = np.asarray(model[key], float)
            if a.shape != (len(FEATURE_NAMES),) or not np.isfinite(a).all() or (a < 0).any():
                raise ValueError("Invalid memory model arrays.")
        if (np.asarray(model["feature_scales"]) <= 0).any():
            raise ValueError("Invalid memory feature scales.")
        self.model = model

    @classmethod
    def load(cls, path: str | Path):
        return cls(json.loads(Path(path).read_text()))

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.model, indent=2, allow_nan=False)+"\n")

    def predict(self, *, batch_size: int, sequence_length: int, new_tokens: int,
                gpu_layers: int, layout: str = "prefix", allow_extrapolation: bool = False) -> dict:
        w = Workload(batch_size, sequence_length, new_tokens, gpu_layers, layout)
        w.validate(self.model["model_config"])
        d = self.model["domain"]
        outside = [k for k in ("batch_size", "sequence_length")
                   if not d[k+"_range"][0] <= getattr(w,k) <= d[k+"_range"][1]]
        if new_tokens not in d["output_lengths_seen"]:
            outside.append("new_tokens")
        if gpu_layers not in d["gpu_counts_seen"]:
            outside.append("gpu_layers")
        if outside and not allow_extrapolation:
            raise ValueError(f"Outside memory calibration support: {outside}.")
        if outside:
            warnings.warn(f"Unvalidated memory extrapolation: {outside}.", stacklevel=2)
        r = predict_memory_frame(self.model, pd.DataFrame([w.__dict__])).iloc[0].to_dict()
        return {**r, "extrapolation": bool(outside), "measured": False}
