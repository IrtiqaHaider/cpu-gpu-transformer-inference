"""Step 1: validate the existing main sweep, fit latency models, audit grouped CV.

Usage: python -m offload_research.fit --archive run_export.zip --out research_run
This does not import torch, load a checkpoint, or execute a GPU benchmark.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path, PurePosixPath
import platform
import time
import zipfile

import numpy as np
import pandas as pd

from .cost_model import (GROUP_KEYS, KEYS, TARGETS, VERSION, Workload,
                         fit_model, predict_frame, save_model)

METRICS = ["prefill_ms", "prefill_input_tokens_per_s", "ttft_ms", "generation_ms",
           "decode_ms", "decode_tpot_ms", "decode_generated_tokens_per_s",
           "generation_generated_tokens_per_s"]
EVAL_TARGETS = ["prefill_ms", "ttft_ms", "decode_tpot_ms", "generation_ms"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_main(archive_path: str | Path) -> tuple[pd.DataFrame, dict, dict]:
    path = Path(archive_path)
    require(path.is_file(), f"Archive not found: {path}")
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        require(len(names) == len(set(names)), "Duplicate ZIP members are not supported.")
        matches = [name for name in names if PurePosixPath(name).parts[-2:] == ("main", "summary.csv")]
        require(len(matches) == 1, "Expected exactly one main/summary.csv. Upload the completed main-sweep export.")
        folder = str(PurePosixPath(matches[0]).parent)
        blobs = {}
        for name in ("summary.csv", "trials.csv", "metadata.json"):
            member = folder + "/" + name
            require(member in names, f"Missing {member}")
            require(z.getinfo(member).file_size <= 64 * 2**20, f"Oversized input member: {member}")
            blobs[name] = z.read(member)
    metadata = json.loads(blobs["metadata.json"])
    summary = pd.read_csv(io.BytesIO(blobs["summary.csv"]))
    trials = pd.read_csv(io.BytesIO(blobs["trials.csv"]))
    config = metadata.get("model_config", {})
    require(config.get("n_layer") == 12 and config.get("n_embd") == 768,
            "Step 1 supports this project's GPT-2 small configuration only.")
    require(metadata.get("dtype") == "float32 on both devices", "Expected the FP32 baseline.")
    require(metadata.get("attention") == "explicit eager causal attention", "Unexpected attention backend.")
    require(metadata.get("model", {}).get("model_id") == "openai-community/gpt2", "Unexpected checkpoint family.")
    revision = metadata.get("model", {}).get("revision", "")
    require(len(revision) == 40 and all(c in "0123456789abcdef" for c in revision), "Missing resolved checkpoint SHA.")
    require(not metadata.get("arguments", {}).get("tiny_random", False), "Random-model smoke runs are not calibration data.")
    require(metadata.get("torch_threads") == 2, "This first baseline model is scoped to the existing two-thread run.")
    require(not summary.empty and not trials.empty, "Empty results.")
    for frame, extra in ((summary, ["case_index", "status"]), (trials, ["case_index", "trial"] + METRICS)):
        missing = set(KEYS + extra) - set(frame.columns)
        require(not missing, f"Missing columns: {sorted(missing)}")
        for key in ["case_index"] + KEYS[:-1]:
            values = pd.to_numeric(frame[key], errors="raise").to_numpy(dtype=float)
            require(np.isfinite(values).all() and np.equal(values, np.floor(values)).all(), f"Invalid integer column: {key}")
            frame[key] = values.astype(np.int64)
    require((summary.status == "ok").all(), "Failed cases exist; do not silently drop them. Inspect before fitting.")
    require((summary.layout == "prefix").all(), "Use the main prefix-only sweep, not layouts/.")
    require(not summary.case_index.duplicated().any(), "Duplicate case indices.")
    require(not summary.duplicated(KEYS).any(), "Repeated workload/placement rows need an explicit session model.")
    require(not trials.duplicated(["case_index", "trial"]).any(), "Duplicate trial IDs.")
    require(set(trials.case_index) == set(summary.case_index), "Trial/summary cases differ.")
    repeats = int(metadata["arguments"]["repeats"])
    rows = []
    for record in summary.to_dict("records"):
        Workload(**{key: record[key] for key in KEYS}).validate(config)
        group = trials[trials.case_index == record["case_index"]].sort_values("trial")
        require(len(group) == repeats and set(group.trial) == set(range(repeats)), "Incomplete trial block.")
        for key in KEYS:
            require((group[key] == record[key]).all(), f"Trial workload mismatch in {key}")
        values = group[METRICS].to_numpy(dtype=float)
        require(np.isfinite(values).all() and (values > 0).all(), "Nonpositive or nonfinite timing/rate.")
        # Recompute all saved summary statistics; never delete slow trials.
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            statistics = {"median": np.median(values), "p95": np.percentile(values, 95),
                          "min": np.min(values), "max": np.max(values)}
            for suffix, actual in statistics.items():
                key = metric + "_" + suffix
                require(key in record and np.isclose(actual, record[key], rtol=1e-8, atol=1e-7),
                        f"Summary mismatch: case {record['case_index']} {key}")
        b, s, n = record["batch_size"], record["sequence_length"], record["new_tokens"]
        checks = [
            (group.generation_ms, group.ttft_ms + group.decode_ms),
            (group.decode_tpot_ms, group.decode_ms / (n - 1)),
            (group.prefill_input_tokens_per_s, b * s * 1000 / group.prefill_ms),
            (group.decode_generated_tokens_per_s, b * (n - 1) * 1000 / group.decode_ms),
            (group.generation_generated_tokens_per_s, b * n * 1000 / group.generation_ms),
        ]
        require(all(np.allclose(a, b_, rtol=1e-8, atol=1e-7) for a, b_ in checks), "Token/time accounting mismatch.")
        if "final_cached_positions" in group:
            require((group.final_cached_positions == s + n - 1).all(), "Unexpected cache length.")
        if "decode_step_ms_json" in group:
            for value in group.decode_step_ms_json:
                steps = np.asarray(json.loads(value), dtype=float)
                require(steps.shape == (n - 1,) and np.isfinite(steps).all() and (steps >= 0).all(), "Invalid decode steps.")
        row = {key: record[key] for key in ["case_index"] + KEYS}
        for metric in METRICS:
            row[metric + "_median"] = float(np.median(group[metric]))
        row["repeats"] = len(group)
        row["generation_p95_over_median"] = float(np.percentile(group.generation_ms, 95) / np.median(group.generation_ms))
        row["pooled_generation_tokens_per_s"] = float(len(group) * b * n * 1000 / group.generation_ms.sum())
        row["workload_group"] = f"B{b}_S{s}_N{n}"
        rows.append(row)
    data = pd.DataFrame(rows).sort_values(KEYS).reset_index(drop=True)
    expected_grid = {
        (b, s, n, k, "prefix") for b in (1, 2, 4) for s in (32, 128, 512)
        for n in (32,) for k in (0, 3, 6, 9, 12)
    }
    actual_grid = set(data[KEYS].itertuples(index=False, name=None))
    require(actual_grid == expected_grid, "Step 1 expects the original 45-case main sweep; this archive has a different grid.")
    manifest = {
        "input_filename": path.name, "input_sha256": sha256(path.read_bytes()),
        "source_folder": folder,
        "member_sha256": {name: sha256(blob) for name, blob in blobs.items()},
        "configuration_rows": len(data), "raw_trials_checked": len(trials),
        "workload_groups": int(data.workload_group.nunique()),
        "all_trial_rows_retained_in_medians": True,
        "summary_statistics_verified": True, "token_accounting_verified": True,
        "layouts_used_for_fitting": False,
        "profiles_used_as_features": False,
        "memory_measurements_used_as_latency_features": False,
        "explanation": "Only the 45-case main sweep is used; the 12 layout records remain separate, not a new blind test.",
    }
    return data, metadata, manifest


def group_folds(data: pd.DataFrame):
    groups = data[GROUP_KEYS].astype(str).agg("/".join, axis=1)
    for group in sorted(groups.unique()):
        train = np.flatnonzero((groups != group).to_numpy())
        test = np.flatnonzero((groups == group).to_numpy())
        require(len(train) > 0 and len(test) > 0, "At least two workload groups are needed.")
        yield group, train, test


def errors(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    ae = np.abs(predicted - actual)
    ape = ae / actual * 100
    return {"n": len(actual), "mae_ms": float(np.mean(ae)),
            "median_ape_pct": float(np.median(ape)), "mean_ape_pct": float(np.mean(ape)),
            "worst_ape_pct": float(np.max(ape)), "wape_pct": float(ae.sum() / actual.sum() * 100)}


def score(data: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    return {key: errors(data[key + "_median"].to_numpy(), predictions["predicted_" + key].to_numpy())
            for key in EVAL_TARGETS}


def run(archive: str | Path, out: str | Path) -> dict:
    started = time.perf_counter()
    out = Path(out)
    require(not out.exists(), f"Output already exists: {out}. Choose a new directory; results are never overwritten.")
    data, metadata, manifest = load_main(archive)
    all_predictions, fold_records, final_models, reports, coefficients = [], [], {}, {}, []
    for variant in ("compute", "compute_transfer"):
        parts = []
        for fold_index, (name, tr, te) in enumerate(group_folds(data)):
            train, test = data.iloc[tr], data.iloc[te]
            fitted = fit_model(train, metadata, variant)
            prediction = predict_frame(fitted, test)
            prediction["source_row"] = te
            prediction["case_index"] = test.case_index.to_numpy()
            prediction["fold"] = fold_index
            prediction["held_out_group"] = name
            prediction["variant"] = variant
            for key in EVAL_TARGETS:
                prediction["actual_" + key] = test[key + "_median"].to_numpy()
                prediction["ape_" + key] = abs(prediction["predicted_" + key] - prediction["actual_" + key]) / prediction["actual_" + key] * 100
            parts.append(prediction)
            fold_records.append({"variant": variant, "fold": fold_index, "held_out_group": name,
                "training_case_indices": [int(x) for x in train.case_index],
                "validation_case_indices": [int(x) for x in test.case_index]})
        cv = pd.concat(parts, ignore_index=True).sort_values("source_row").reset_index(drop=True)
        final = fit_model(data, metadata, variant)
        final_models[variant] = final
        reports[variant] = {
            "retrospective_grouped_cv": score(data, cv),
            "training_fit_diagnostic_not_validation": score(data, predict_frame(final, data)),
            "feature_count": len(final["feature_names"]),
            "rank_by_target": {key: model["design_rank"] for key, model in final["targets"].items()},
        }
        for target, model in final["targets"].items():
            for name, scale, coef in zip(final["feature_names"], model["feature_scales"], model["coefficients_ms"]):
                coefficients.append({"variant": variant, "target": target, "feature": name,
                                     "training_scale": scale, "scaled_coefficient_ms": coef,
                                     "raw_feature_coefficient_ms": coef / scale})
        all_predictions.append(cv)
    report = {
        "schema_version": VERSION, "analysis_type": "retrospective_calibration_and_grouped_validation",
        "configuration_rows": len(data), "raw_trials_checked": manifest["raw_trials_checked"],
        "folds_per_model": int(data.workload_group.nunique()),
        "group_definition": GROUP_KEYS, "input_sha256": manifest["input_sha256"],
        "new_gpu_benchmarks_run": False,
        "models": reports,
        "limitations": [
            "All historical results have already been inspected; this is not a new blind test.",
            "Entire workload groups, not individual trial rows, are withheld in each fold.",
            "Scaling and coefficients are fitted on each fold's training rows only.",
            "This version predicts latency only: no memory-budget planner is implemented yet.",
            "Prefix-only, GPT-2 small, FP32, recorded two-thread T4 environment.",
            "Transfer features are boundary/payload proxies, not measured or causal PCIe timings.",
            "Correlated features make individual coefficients non-identifiable as physical costs.",
            "Some folds extrapolate beyond training batch/sequence extrema; pooled errors mix interpolation and extrapolation.",
            "Prediction targets are medians; this is not a tail-latency or uncertainty model.",
            "Predicted generation = predicted TTFT + (N-1)*predicted average decode step. Medians need not add exactly.",
            "CPU variability, model error and drift can all contribute to validation error.",
            "Results do not establish a placement advantage or inference speedup.",
        ],
    }
    out.mkdir(parents=True)
    (out / "models").mkdir()
    for variant, fitted in final_models.items():
        fitted["calibration_input_sha256"] = manifest["input_sha256"]
        save_model(fitted, out / "models" / f"{variant}.json")
    data.to_csv(out / "calibration.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(out / "retrospective_cv_predictions.csv", index=False)
    pd.DataFrame(coefficients).to_csv(out / "coefficients.csv", index=False)
    snapshot = out / "code_snapshot"
    snapshot.mkdir()
    code_hashes = {}
    for path in sorted(Path(__file__).parent.glob("*.py")):
        content = path.read_bytes()
        (snapshot / path.name).write_bytes(content)
        code_hashes[path.name] = sha256(content)
    manifest["code_sha256"] = code_hashes
    manifest["fitting_environment"] = {
        "python": platform.python_version(),
        **{pkg: importlib.metadata.version(pkg) for pkg in ("numpy", "pandas", "scipy")},
    }
    report["cpu_fit_and_validation_wall_seconds"] = time.perf_counter() - started
    for filename, payload in (("summary.json", report), ("input_manifest.json", manifest),
                              ("source_metadata.json", metadata), ("cv_folds.json", fold_records)):
        (out / filename).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Research step 1: retrospective cost-model baseline", "",
             f"Validated {len(data)} main-sweep configurations and {manifest['raw_trials_checked']} raw trials.",
             f"Used {data.workload_group.nunique()} leave-one-workload-out folds per model.",
             "No new GPU inference was executed. Historical exploratory data is not a new blind test.", "",
             "## Median absolute percentage error on withheld workload groups", "",
             "| Model | Prefill | TTFT | Decode step | Generation |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for variant in reports:
        vals = [reports[variant]["retrospective_grouped_cv"][key]["median_ape_pct"] for key in EVAL_TARGETS]
        lines.append("| " + variant + " | " + " | ".join(f"{x:.2f}%" for x in vals) + " |")
    lines += ["", "Lower is better. Do not interpret 100 minus percentage error as model accuracy.",
              "The summary JSON also reports mean/worst errors; inspect per-case failures, not only the median.",
              "", "## Scope and limitations", ""] + ["- " + x for x in report["limitations"]]
    lines += ["", "## Next checkpoint", "",
              "Inspect this baseline before adding memory prediction and a budget-aware planner.",
              "Do not run new held-out GPU experiments until the candidate policies and evaluation protocol are fixed."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:17]))
    print(f"\nSaved: {out.resolve()}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    run(args.archive, args.out)


if __name__ == "__main__":
    main()
