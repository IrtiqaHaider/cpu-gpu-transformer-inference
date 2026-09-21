"""Prediction-only placement selection: no measured latency/memory inputs.

All methods share the same predicted-memory feasibility filter, including the
max-GPU baseline. The candidate set is intentionally the calibrated prefix grid.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .cost_model import CostModel
from .memory_model import MemoryModel, MemoryGuard

CANDIDATES = (0, 3, 6, 9, 12)
POLICIES = ("max_gpu", "compute", "compute_transfer")
BUDGETS_MIB = (256, 320, 384, 448, 512, 640, 768, 1024)


def validate_candidates(candidates):
    c = tuple(candidates)
    if not c or len(set(c)) != len(c):
        raise ValueError("Candidate list must be nonempty and unique.")
    if any(isinstance(k, bool) or not isinstance(k, (int,np.integer)) or k not in CANDIDATES for k in c):
        raise ValueError("Only calibrated GPU counts 0,3,6,9,12 are supported in Step 2.")
    return tuple(sorted(int(k) for k in c))


def choose(table: pd.DataFrame, budget_mib: float, policy: str) -> dict | None:
    if policy not in POLICIES:
        raise ValueError("Unknown placement policy.")
    if isinstance(budget_mib, bool) or not np.isfinite(budget_mib) or budget_mib < 0:
        raise ValueError("GPU budget must be finite and nonnegative.")
    required = ["gpu_layers", "guarded_peak_mib"]
    if policy != "max_gpu": required += ["predicted_generation_ms_"+policy]
    if set(required)-set(table.columns):
        raise ValueError("Incomplete prediction table.")
    if not np.isfinite(table[required].to_numpy(float)).all() or (table[required] < 0).any().any():
        raise ValueError("Invalid prediction table values.")
    if table.gpu_layers.duplicated().any():
        raise ValueError("Duplicate candidates.")
    feasible = table[table.guarded_peak_mib <= budget_mib]
    if feasible.empty:
        return None
    if policy == "max_gpu":
        ranked = feasible.sort_values("gpu_layers", ascending=False)
    else:
        # Deterministic tie break: smaller modeled allocation, then fewer blocks.
        ranked = feasible.sort_values(["predicted_generation_ms_"+policy,
                                       "guarded_peak_mib", "gpu_layers"])
    return ranked.iloc[0].to_dict()


class PlacementPlanner:
    def __init__(self, memory: MemoryModel, compute: CostModel, compute_transfer: CostModel,
                 guard: MemoryGuard | None = None):
        self.memory, self.latency = memory, {"compute":compute, "compute_transfer":compute_transfer}
        self.guard = guard or MemoryGuard()
        for name, latency in self.latency.items():
            if latency.model["variant"] != name:
                raise ValueError("Latency model variant mismatch.")
            for key in ("model_config", "model_source"):
                if latency.model[key] != memory.model[key]:
                    raise ValueError("Memory and latency models belong to different configurations/checkpoints.")
            a = latency.model.get("calibration_input_sha256")
            b = memory.model.get("calibration_input_sha256")
            if a is not None and b is not None and a != b:
                raise ValueError("Memory and latency calibration inputs differ.")

    def candidates(self, *, batch_size: int, sequence_length: int, new_tokens: int,
                   candidates=CANDIDATES, allow_extrapolation: bool = False) -> pd.DataFrame:
        rows = []
        for k in validate_candidates(candidates):
            args = dict(batch_size=batch_size, sequence_length=sequence_length,
                        new_tokens=new_tokens, gpu_layers=k, allow_extrapolation=allow_extrapolation)
            m = self.memory.predict(**args)
            r = {"gpu_layers": k, "predicted_peak_mib": m["predicted_peak_mib"],
                 "guarded_peak_mib": self.guard.apply(m["predicted_peak_mib"], k),
                 "payload_floor_mib": m["payload_floor_bytes"]/2**20,
                 "extrapolation": m["extrapolation"]}
            for variant, model in self.latency.items():
                p = model.predict(**args)
                r["predicted_generation_ms_"+variant] = p["predicted_generation_ms"]
                r["extrapolation"] = r["extrapolation"] or p["extrapolation"]
            rows.append(r)
        return pd.DataFrame(rows)

    def recommend(self, *, batch_size: int, sequence_length: int, new_tokens: int,
                  gpu_budget_mib: float, policy: str = "compute_transfer",
                  candidates=CANDIDATES, allow_extrapolation: bool = False) -> dict:
        table = self.candidates(batch_size=batch_size, sequence_length=sequence_length,
                                new_tokens=new_tokens, candidates=candidates,
                                allow_extrapolation=allow_extrapolation)
        selected = choose(table, gpu_budget_mib, policy)
        return {"status": "predicted_feasible" if selected is not None else "no_predicted_feasible_candidate",
                "policy":policy, "gpu_budget_mib":float(gpu_budget_mib),
                "batch_size":batch_size, "sequence_length":sequence_length, "new_tokens":new_tokens,
                "selected":selected, "measured":False,
                "constraint": "Heuristic peak PyTorch allocation budget, NOT physical GPU capacity.",
                "cpu_ram_assumption": "Sufficient CPU RAM; CPU RAM and other processes are not constrained."}
