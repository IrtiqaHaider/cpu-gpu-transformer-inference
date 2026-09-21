"""Interpretable, empirical latency predictors for the existing FP32 GPT-2 study.

These coefficients are predictive associations, not causal CPU/GPU/PCIe timings.
No model weights, CUDA execution, or measured timings are needed at prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import nnls

VERSION = "0.1.0"
RIDGE = 1e-6  # Fixed before fitting; not selected using cross-validation results.
TARGETS = {
    "prefill_ms": ("prefill_ms_median", "prefill"),
    "ttft_ms": ("ttft_ms_median", "prefill"),
    "decode_tpot_ms": ("decode_tpot_ms_median", "decode"),
}
KEYS = ["batch_size", "sequence_length", "new_tokens", "gpu_layers", "layout"]
GROUP_KEYS = ["batch_size", "sequence_length", "new_tokens"]
CORE_FEATURES = [
    "cpu_block_count", "gpu_block_count",
    "cpu_linear_work", "gpu_linear_work",
    "cpu_attention_work", "gpu_attention_work",
    "cpu_endpoint_batch", "gpu_endpoint_batch",
]
TRANSFER_FEATURES = ["activation_boundaries", "activation_payload_mib"]


@dataclass(frozen=True)
class Workload:
    batch_size: int
    sequence_length: int
    new_tokens: int
    gpu_layers: int
    layout: str = "prefix"

    def validate(self, config: dict) -> None:
        for name in ("batch_size", "sequence_length", "new_tokens", "gpu_layers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer, not {value!r}.")
        if self.batch_size < 1 or self.sequence_length < 1 or self.new_tokens < 2:
            raise ValueError("Need batch>=1, prompt>=1 and output>=2.")
        if not 0 <= self.gpu_layers <= int(config["n_layer"]):
            raise ValueError("GPU block count is outside the model.")
        if self.sequence_length + self.new_tokens - 1 > int(config["n_positions"]):
            raise ValueError("Prompt + processed decode tokens exceeds context window.")
        if self.layout != "prefix":
            raise ValueError("Step 1 is calibrated only for prefix placement; other layouts are unsupported.")


def feature_names(variant: str) -> list[str]:
    if variant not in ("compute", "compute_transfer"):
        raise ValueError("variant must be compute or compute_transfer")
    return CORE_FEATURES + (TRANSFER_FEATURES if variant == "compute_transfer" else [])


def features(workload: Workload, config: dict, phase: str, variant: str) -> np.ndarray:
    workload.validate(config)
    if phase not in ("prefill", "decode"):
        raise ValueError("phase must be prefill or decode")
    b, s, n, k = (workload.batch_size, workload.sequence_length,
                   workload.new_tokens, workload.gpu_layers)
    total = int(config["n_layer"])
    h = int(config["n_embd"])
    q = s if phase == "prefill" else 1
    # Cached steps process total lengths s+1,...,s+n-1, mean s+n/2.
    context = s if phase == "prefill" else s + n / 2.0
    cpu = total - k
    gpu_endpoint = int(k > 0)  # Preserve the engine's tied-weight endpoint policy.
    boundaries = 2 if 0 < k < total else 0
    values = [
        cpu, k,
        cpu * b * q, k * b * q,
        cpu * b * q * context, k * b * q * context,
        (1 - gpu_endpoint) * b, gpu_endpoint * b,
    ]
    if variant == "compute_transfer":
        values += [boundaries, boundaries * b * q * h * 4 / 2**20]
    feature_names(variant)  # Validate variant even when transfer features are absent.
    return np.asarray(values, dtype=np.float64)


def design(frame: pd.DataFrame, config: dict, phase: str, variant: str) -> np.ndarray:
    return np.stack([
        features(Workload(**{key: row[key] for key in KEYS}), config, phase, variant)
        for row in frame[KEYS].to_dict("records")
    ])


def fit_nonnegative(x: np.ndarray, y: np.ndarray) -> dict:
    """Relative-error weighted NNLS with small fixed ridge regularization.

    All scaling is fit on this call's training rows only. No response information
    is used at prediction. The ridge is a numerical stabilizer, not a tuned knob.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.shape != (len(x),) or len(x) < 1:
        raise ValueError("Invalid design or target dimensions.")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (x < 0).any() or (y <= 0).any():
        raise ValueError("Features must be finite/nonnegative and latencies positive.")
    scales = np.maximum(np.max(np.abs(x), axis=0), 1.0)
    scaled = x / scales
    target_scale = float(np.median(y))
    # Normalize the target too: fixed ridge must not change with latency units.
    weighted = scaled / (y / target_scale)[:, None]
    a = np.vstack([weighted, np.sqrt(RIDGE) * np.eye(x.shape[1])])
    rhs = np.concatenate([np.ones(len(y)), np.zeros(x.shape[1])])
    coef, _ = nnls(a, rhs, maxiter=10000)
    coef = coef * target_scale
    return {
        "coefficients_ms": coef.tolist(), "feature_scales": scales.tolist(),
        "target_scale_ms": target_scale, "training_rows": len(y), "design_rank": int(np.linalg.matrix_rank(scaled)),
        "feature_count": x.shape[1], "ridge": RIDGE,
    }


def predict_nonnegative(x: np.ndarray, fitted: dict) -> np.ndarray:
    prediction = (np.asarray(x) / np.asarray(fitted["feature_scales"])) @ np.asarray(fitted["coefficients_ms"])
    return np.maximum(prediction, 1e-9)


def fit_model(frame: pd.DataFrame, metadata: dict, variant: str) -> dict:
    if frame.empty:
        raise ValueError("No calibration rows.")
    config = metadata["model_config"]
    fits = {}
    for label, (column, phase) in TARGETS.items():
        fits[label] = fit_nonnegative(design(frame, config, phase, variant), frame[column].to_numpy())
    return {
        "schema_version": VERSION, "variant": variant,
        "feature_names": feature_names(variant), "model_config": config,
        "model_source": metadata["model"],
        "benchmark_environment": {key: metadata.get(key) for key in
            ("gpu_name", "torch_threads", "torch_interop_threads", "cpu_count_logical", "lscpu", "packages", "cuda_runtime", "dtype", "attention")},
        "domain": {
            "batch_size_range": [int(frame.batch_size.min()), int(frame.batch_size.max())],
            "sequence_length_range": [int(frame.sequence_length.min()), int(frame.sequence_length.max())],
            "output_lengths_seen": sorted(int(x) for x in frame.new_tokens.unique()),
            "gpu_counts_seen": sorted(int(x) for x in frame.gpu_layers.unique()),
            "layout": "prefix",
        },
        "targets": fits,
        "scope": "Retrospective empirical model; not causal profiling, a hardware simulator, or a tail-latency guarantee.",
    }


def predict_frame(model: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """Internal batch prediction, including deliberate CV extrapolation folds."""
    result = frame[KEYS].copy().reset_index(drop=True)
    for label, (_, phase) in TARGETS.items():
        x = design(frame, model["model_config"], phase, model["variant"])
        result["predicted_" + label] = predict_nonnegative(x, model["targets"][label])
    result["predicted_generation_ms"] = result.predicted_ttft_ms + (result.new_tokens - 1) * result.predicted_decode_tpot_ms
    return result


def save_model(model: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(model, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class CostModel:
    def __init__(self, model: dict):
        if model.get("schema_version") != VERSION:
            raise ValueError("Unsupported model schema version.")
        self.model = model

    @classmethod
    def load(cls, path: str | Path) -> "CostModel":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def predict(self, *, batch_size: int, sequence_length: int, new_tokens: int,
                gpu_layers: int, layout: str = "prefix", allow_extrapolation: bool = False) -> dict:
        w = Workload(batch_size, sequence_length, new_tokens, gpu_layers, layout)
        w.validate(self.model["model_config"])
        domain = self.model["domain"]
        outside = []
        for name in ("batch_size", "sequence_length"):
            lo, hi = domain[name + "_range"]
            if not lo <= getattr(w, name) <= hi:
                outside.append(name)
        if new_tokens not in domain["output_lengths_seen"]:
            outside.append("new_tokens")
        if outside and not allow_extrapolation:
            raise ValueError(f"Outside calibration support: {outside}. Set allow_extrapolation=True only for a declared experiment.")
        if outside:
            warnings.warn(f"Unvalidated extrapolation in {outside}; not a measured performance result.", stacklevel=2)
        row = predict_frame(self.model, pd.DataFrame([w.__dict__])).iloc[0]
        return {
            **w.__dict__,
            **{key: float(value) for key, value in row.items() if key.startswith("predicted_")},
            "extrapolation": bool(outside),
            "gpu_count_seen_in_calibration": gpu_layers in domain["gpu_counts_seen"],
            "measured": False,
        }
