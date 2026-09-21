"""Audit Step 1; fit memory; evaluate prediction-only policies on grouped folds.

python -m offload_research.step2 --step1 checkpoint.zip --baseline run.zip --out output
No torch import or new GPU inference. Original benchmark files are never changed.
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
from .cost_model import KEYS, Workload, CostModel, fit_model, predict_frame
from .fit import load_main, require, sha256, group_folds, EVAL_TARGETS
from .memory_model import MIB, MemoryModel, MemoryGuard, components, fit_memory, predict_memory_frame
from .planner import CANDIDATES, POLICIES, BUDGETS_MIB, PlacementPlanner, choose

STEP1_HASHES = {
    "__init__.py":"a21491ca194a2505b49d8f9ca5bc0b391b1534c0fdc7582bf9afb29e1c6c666a",
    "cost_model.py":"26b55e90071884733dcc10092652d843315e3fe96792cb85547892c5b1035ab5",
    "fit.py":"1f535115d86323c8efdaaad6ca3e6aac9a43294e0595ed7caf2661fd19c373e9",
}
MEMORY_COLS = ["gpu_peak_allocated_bytes", "gpu_parameter_bytes", "gpu_buffer_bytes",
               "cpu_parameter_bytes", "final_kv_gpu_bytes", "final_cached_positions"]


def json_write(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def read_members(path: Path, required: list[str]) -> dict[str,bytes]:
    """Read bounded ZIP members only; never extract or execute uploaded code."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        require(len(names) == len(set(names)), "Duplicate ZIP members.")
        result = {}
        for name in required:
            require(name in names, f"Missing archive member: {name}")
            require(z.getinfo(name).file_size <= 64*MIB, f"Oversized ZIP member: {name}")
            result[name] = z.read(name)
    return result


def load_memory_data(archive: Path):
    data, metadata, manifest = load_main(archive)  # Recheck original raw timings.
    require(metadata.get("gpu_name") == "Tesla T4", "Step 2 targets the measured Tesla T4 baseline.")
    folder = manifest["source_folder"]
    blobs = read_members(archive, [folder+"/summary.csv", folder+"/trials.csv"])
    summary = pd.read_csv(io.BytesIO(blobs[folder+"/summary.csv"]))
    trials = pd.read_csv(io.BytesIO(blobs[folder+"/trials.csv"]))
    require(not set(MEMORY_COLS)-set(summary.columns), "Memory columns missing from original summary.")
    for column in MEMORY_COLS:
        values = summary[column].to_numpy(float)
        require(np.isfinite(values).all() and (values >= 0).all() and np.equal(values,np.floor(values)).all(),
                "Invalid memory integer column: "+column)
    combined = data.merge(summary[["case_index"]+MEMORY_COLS], on="case_index", how="left", validate="one_to_one")
    for row in combined.to_dict("records"):
        w = Workload(**{k:row[k] for k in KEYS})
        exact = components(w, metadata["model_config"])
        for key in MEMORY_COLS[1:]:
            require(exact[key] == row[key], f"Exact payload mismatch: case {row['case_index']} {key}")
        require(row["gpu_peak_allocated_bytes"] >= exact["payload_floor_bytes"], "Peak below live payload floor.")
        if w.gpu_layers == 0:
            require(row["gpu_peak_allocated_bytes"] == 0, "Unexpected CPU-only GPU allocation.")
        for key in ("final_kv_gpu_bytes", "final_cached_positions"):
            require(key in trials, f"Missing per-trial {key}")
            require((trials.loc[trials.case_index == row["case_index"], key] == exact[key]).all(),
                    "Raw trial cache accounting mismatch.")
    return combined, metadata, manifest


def audit_step1(path: Path, data: pd.DataFrame, metadata: dict, baseline_manifest: dict):
    required = ["summary.json", "input_manifest.json", "calibration.csv", "source_metadata.json",
                "retrospective_cv_predictions.csv", "cv_folds.json", "test_output.txt"]
    required += ["code_snapshot/"+k for k in STEP1_HASHES]
    required += ["models/"+v+".json" for v in ("compute", "compute_transfer")]
    blobs = read_members(path, required)
    manifest = json.loads(blobs["input_manifest.json"])
    report = json.loads(blobs["summary.json"])
    require(manifest["input_sha256"] == baseline_manifest["input_sha256"], "Step 1 refers to a different baseline ZIP.")
    require(manifest["member_sha256"] == baseline_manifest["member_sha256"], "Baseline member hashes differ.")
    require(json.loads(blobs["source_metadata.json"]) == metadata, "Source metadata mismatch.")
    for name, digest in STEP1_HASHES.items():
        require(sha256(blobs["code_snapshot/"+name]) == digest, "Unrecognized Step 1 code snapshot: "+name)
        require(sha256((Path(__file__).parent/name).read_bytes()) == digest, "Bundled Step 1 code was altered.")
    saved_data = pd.read_csv(io.BytesIO(blobs["calibration.csv"]))
    require(len(saved_data) == len(data) and not saved_data.duplicated(KEYS).any(), "Invalid Step 1 calibration table.")
    a = saved_data.sort_values(KEYS).reset_index(drop=True)
    b = data.sort_values(KEYS).reset_index(drop=True)
    require(list(a[KEYS].itertuples(index=False,name=None)) == list(b[KEYS].itertuples(index=False,name=None)),
            "Step 1 calibration keys differ.")
    for key in a.columns:
        require(key in b, "Unexpected Step 1 calibration column.")
        if pd.api.types.is_numeric_dtype(a[key]):
            require(np.allclose(a[key],b[key],rtol=1e-8,atol=1e-7), "Calibration mismatch: "+key)
        else:
            require(a[key].equals(b[key]), "Calibration text mismatch: "+key)
    cv = pd.read_csv(io.BytesIO(blobs["retrospective_cv_predictions.csv"]))
    require(len(cv) == 2*len(data), "Incomplete Step 1 CV.")
    folds_saved = json.loads(blobs["cv_folds.json"])
    require(len(folds_saved) == 18, "Incomplete fold manifest.")
    final = {}
    for variant in ("compute", "compute_transfer"):
        part = cv[cv.variant == variant]
        require(len(part) == len(data) and not part.case_index.duplicated().any(), "Duplicate/missing CV cases.")
        for i, (name, tr, te) in enumerate(group_folds(data)):
            entries = [f for f in folds_saved if f["variant"] == variant and f["fold"] == i]
            require(len(entries) == 1, "Duplicate/missing fold definition.")
            f = entries[0]
            require(f["held_out_group"] == name and f["training_case_indices"] == data.iloc[tr].case_index.tolist()
                    and f["validation_case_indices"] == data.iloc[te].case_index.tolist(), "Fold grouping mismatch.")
            pred = predict_frame(fit_model(data.iloc[tr],metadata,variant),data.iloc[te])
            old = part.set_index("case_index").loc[data.iloc[te].case_index].reset_index()
            for target in EVAL_TARGETS:
                require(np.allclose(old["actual_"+target],data.iloc[te][target+"_median"],rtol=1e-8,atol=1e-6),
                        "CV actual-target mismatch.")
                require(np.allclose(old["predicted_"+target],pred["predicted_"+target],rtol=1e-4,atol=1e-4),
                        "CV prediction failed reproduction.")
        for target in EVAL_TARGETS:
            actual, predicted = part["actual_"+target].to_numpy(), part["predicted_"+target].to_numpy()
            ape = abs(predicted-actual)/actual*100
            r = report["models"][variant]["retrospective_grouped_cv"][target]
            require(np.isclose(np.median(ape),r["median_ape_pct"],rtol=1e-7), "Reported median error mismatch.")
            require(np.isclose(max(ape),r["worst_ape_pct"],rtol=1e-7), "Reported worst error mismatch.")
        model = json.loads(blobs["models/"+variant+".json"])
        require(model["calibration_input_sha256"] == manifest["input_sha256"], "Saved latency provenance mismatch.")
        require(model["model_source"] == metadata["model"] and model["model_config"] == metadata["model_config"],
                "Saved latency model scope differs.")
        refit = fit_model(data,metadata,variant)
        p1, p2 = predict_frame(model,data), predict_frame(refit,data)
        for target in EVAL_TARGETS:
            require(np.allclose(p1["predicted_"+target],p2["predicted_"+target],rtol=1e-4,atol=1e-4),
                    "Saved final latency model failed reproduction.")
        final[variant] = model
    return final, {"step1_archive_sha256":sha256(path.read_bytes()),
                   "step1_filename":path.name,
                   "baseline_identity_checked":True, "code_identity_checked":True,
                   "calibration_recomputed":True, "grouped_predictions_reproduced":True,
                   "saved_models_reproduced":True,
                   "step1_recorded_test_output":blobs["test_output.txt"].decode().strip(),
                   "recorded_tests_are_not_new_GPU_tests":True,
                   "step1_report":report}


def score_selection(selected: dict | None, actual: pd.DataFrame, budget: float) -> dict:
    """Evaluation only. Actuals are deliberately not accepted by planner.choose."""
    feasible = actual[actual.gpu_peak_allocated_bytes/MIB <= budget]
    oracle = None if feasible.empty else feasible.sort_values(["generation_ms_median","gpu_layers"]).iloc[0]
    result = {"oracle_gpu_layers":None if oracle is None else int(oracle.gpu_layers),
              "oracle_generation_ms":None if oracle is None else float(oracle.generation_ms_median),
              "selected_gpu_layers":None if selected is None else int(selected["gpu_layers"]),
              "actual_selected_peak_mib":None, "actual_selected_generation_ms":None,
              "regret_pct":None, "budget_excess_mib":0.0, "matches_oracle":False}
    if selected is None:
        return {**result, "status":"abstain"}
    match = actual[actual.gpu_layers == int(selected["gpu_layers"])]
    require(len(match)==1, "Selected candidate was not measured exactly once.")
    row = match.iloc[0]
    excess = max(0.0,float(row.gpu_peak_allocated_bytes/MIB-budget))
    result.update(actual_selected_peak_mib=float(row.gpu_peak_allocated_bytes/MIB),
                  actual_selected_generation_ms=float(row.generation_ms_median),budget_excess_mib=excess)
    if excess > 0:
        return {**result,"status":"budget_violation"}  # Never award low regret to an infeasible plan.
    require(oracle is not None, "Feasible selection but no measured feasible candidate.")
    result.update(status="feasible",regret_pct=float(100*(row.generation_ms_median/oracle.generation_ms_median-1)),
                  matches_oracle=bool(int(row.gpu_layers)==int(oracle.gpu_layers)))
    return result


def policy_stats(records: pd.DataFrame):
    reports = {}
    for policy, part in records.groupby("policy",sort=False):
        good = part[part.status == "feasible"]
        reports[policy] = {"decisions":len(part), "feasible":len(good),
                           "budget_violations":int((part.status=="budget_violation").sum()),
                           "abstentions":int((part.status=="abstain").sum()),
                           "oracle_matches":int(part.matches_oracle.sum()),
                           "median_regret_pct_among_feasible":float(good.regret_pct.median()) if len(good) else None,
                           "worst_regret_pct_among_feasible":float(good.regret_pct.max()) if len(good) else None,
                           "worst_budget_excess_mib":float(part.budget_excess_mib.max())}
    return reports


def protocol(guard: MemoryGuard):
    return {"version":"step2-v0.2.0", "evaluation":"retrospective leave-one-workload-out",
            "candidate_gpu_layers":list(CANDIDATES),"layout":"prefix",
            "policies":list(POLICIES),"budgets_mib":list(BUDGETS_MIB),
            "objective":"median full generation latency within modeled peak allocated-memory budget",
            "memory_guard":{"relative":guard.relative,"extra_mib":guard.extra_mib,
                            "kind":"fixed heuristic, not a confidence bound"},
            "budget_selection":"Fixed round-number experimental constraints; not actual device limits.",
            "training":"Both memory residuals and latency models refit on 40 rows; 5 workload rows withheld.",
            "oracle":"Best measured feasible among the SAME five prefix candidates, no guard applied to actuals.",
            "failure_rule":"Budget violations and abstentions counted; regret only for actually feasible selections.",
            "no_leakage":"Neither measured latency nor measured memory of a held-out workload enters planning.",
            "historical_data_previously_inspected":True, "new_gpu_inference":False,
            "untested_layouts_and_layer_counts":"rejected rather than silently generalized"}


def run(step1: str | Path, baseline: str | Path, out: str | Path) -> dict:
    started = time.perf_counter()
    step1,baseline,out = Path(step1),Path(baseline),Path(out)
    require(not out.exists(), "Output exists. Use a new timestamped directory.")
    data,metadata,base_manifest = load_memory_data(baseline)
    final_latency, audit = audit_step1(step1,data,metadata,base_manifest)
    guard = MemoryGuard()
    specification = protocol(guard)
    all_mem, all_policies, all_candidates, fold_rows = [],[],[],[]
    timings = []
    for fold,(name,tr,te) in enumerate(group_folds(data)):
        train,test=data.iloc[tr],data.iloc[te]
        t=time.perf_counter()
        mem=fit_memory(train,metadata)
        latency={v:fit_model(train,metadata,v) for v in ("compute","compute_transfer")}
        fit_seconds=time.perf_counter()-t
        mem_pred=predict_memory_frame(mem,test)
        mem_pred["case_index"]=test.case_index.to_numpy()
        mem_pred["actual_peak_mib"]=test.gpu_peak_allocated_bytes.to_numpy()/MIB
        mem_pred["guarded_peak_mib"]=[guard.apply(p,k) for p,k in zip(mem_pred.predicted_peak_mib,mem_pred.gpu_layers)]
        mem_pred["fold"]=fold
        mem_pred["held_out_group"]=name
        all_mem.extend(mem_pred.to_dict("records"))
        planner=PlacementPlanner(MemoryModel(mem),CostModel(latency["compute"]),CostModel(latency["compute_transfer"]),guard)
        w=test.iloc[0]
        t=time.perf_counter()
        # Some outer folds are out of the fold training range. Label them explicitly.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",UserWarning)
            table=planner.candidates(batch_size=int(w.batch_size),sequence_length=int(w.sequence_length),
                                     new_tokens=int(w.new_tokens),allow_extrapolation=True)
        table_ms=(time.perf_counter()-t)*1000
        for r in table.to_dict("records"):
            all_candidates.append({"fold":fold,"held_out_group":name,**r})
        for budget in BUDGETS_MIB:
            for policy in POLICIES:
                t=time.perf_counter()
                selected=choose(table,budget,policy)
                selection_ms=(time.perf_counter()-t)*1000
                all_policies.append({"fold":fold,"held_out_group":name,
                                     "batch_size":int(w.batch_size),"sequence_length":int(w.sequence_length),
                                     "new_tokens":int(w.new_tokens),"budget_mib":budget,"policy":policy,
                                     "prediction_extrapolates_from_fold_range":bool(table.extrapolation.any()),
                                     **score_selection(selected,test,budget)})
                timings.append({"fold":fold,"policy":policy,"budget_mib":budget,
                                "fit_models_seconds":fit_seconds,"candidate_prediction_ms":table_ms,
                                "selection_only_ms":selection_ms})
        fold_rows.append({"fold":fold,"held_out_group":name,
                          "training_case_indices":train.case_index.tolist(),
                          "held_out_case_indices":test.case_index.tolist(),
                          "memory_model":mem,"latency_models":latency})
    memory_cv=pd.DataFrame(all_mem)
    policies=pd.DataFrame(all_policies)
    gpu=memory_cv[memory_cv.gpu_layers>0]
    abs_err=abs(gpu.predicted_peak_mib-gpu.actual_peak_mib)
    ape=100*abs_err/gpu.actual_peak_mib
    stats={"gpu_validation_rows":len(gpu),"cpu_zero_rows":int((memory_cv.gpu_layers==0).sum()),
           "median_ape_pct_gpu_only":float(ape.median()),"worst_ape_pct_gpu_only":float(ape.max()),
           "mae_mib_gpu_only":float(abs_err.mean()),
           "largest_underprediction_mib":float(np.maximum(gpu.actual_peak_mib-gpu.predicted_peak_mib,0).max()),
           "guard_misses_rows":int((gpu.actual_peak_mib>gpu.guarded_peak_mib).sum()),
           "guard_is_not_a_statistical_guarantee":True}
    pivot=policies.pivot(index=["fold","budget_mib"],columns="policy",values="selected_gpu_layers")
    comparisons={v:{"same_choice_as_max_gpu":int((pivot[v]==pivot.max_gpu).sum()),"decisions":len(pivot)}
                 for v in ("compute","compute_transfer")}
    final_mem=fit_memory(data,metadata)
    final_mem["calibration_input_sha256"]=base_manifest["input_sha256"]
    final=PlacementPlanner(MemoryModel(final_mem),CostModel(final_latency["compute"]),CostModel(final_latency["compute_transfer"]),guard)
    # This previously unmeasured workload is a demo only, not a fresh validation result.
    demo=[]
    for budget in BUDGETS_MIB:
        for policy in POLICIES:
            d=final.recommend(batch_size=3,sequence_length=64,new_tokens=32,gpu_budget_mib=budget,policy=policy)
            s=d["selected"] or {}
            demo.append({"batch_size":3,"sequence_length":64,"new_tokens":32,"budget_mib":budget,
                         "policy":policy,"status":d["status"],"selected_gpu_layers":s.get("gpu_layers"),
                         "predicted_peak_mib":s.get("predicted_peak_mib"),"guarded_peak_mib":s.get("guarded_peak_mib"),
                         "predicted_generation_ms":s.get("predicted_generation_ms_"+policy) if policy!="max_gpu" else None,
                         "measured":False})
    limitations=[
        "Historical results were already inspected; grouped diagnostics are not a new blind test.",
        "Only five prefix GPU counts and the original GPT-2 small/FP32/two-thread/T4 environment are supported.",
        "Memory target is the case maximum across measured prefill and generation calls, not per-trial median memory.",
        "Payload floor is exact for the supplied engine; residual coefficients are empirical, not causal terms.",
        "GPU allocation is not reserved/process/device memory; CPU RAM, co-tenants and placement transients are unconstrained.",
        "The fixed 5% plus 16 MiB guard is a heuristic, not an OOM or confidence guarantee.",
        "Regret uses the same five measured candidates. It is not global optimality or a speedup against GPU-only.",
        "Budget grid repeats the same nine workloads; 216 policy records are not 216 independent experiments.",
        "Model fits on all historical rows are for future predictions; evaluation refits both model types per fold.",
        "CPU timing spikes remain in targets. No fresh GPU inference or physical budget-enforcement experiment is run.",
        "Changed attention backend, precision, engine cache allocation, hardware or model size requires revalidation.",
    ]
    report={"schema_version":"0.2.0","analysis_type":"retrospective_memory_and_policy_evaluation",
            "configurations":len(data),"original_trials_audited":base_manifest["raw_trials_checked"],
            "folds":len(fold_rows),"policies":len(POLICIES),"budgets_per_workload":len(BUDGETS_MIB),
            "policy_decision_records":len(policies),"new_gpu_inference":False,
            "memory":stats,"policy_metrics":policy_stats(policies),"comparison_to_max_gpu":comparisons,
            "limitations":limitations,"analysis_wall_seconds":time.perf_counter()-started}
    out.mkdir(parents=True)
    (out/"models").mkdir()
    for v,m in final_latency.items():
        json_write(out/"models"/(v+".json"),m)
    json_write(out/"models"/"memory.json",final_mem)
    memory_cv.to_csv(out/"memory_cv.csv",index=False)
    policies.to_csv(out/"policy_cv.csv",index=False)
    pd.DataFrame(all_candidates).to_csv(out/"policy_candidate_predictions.csv",index=False)
    pd.DataFrame(demo).to_csv(out/"demo_plans_NOT_MEASUREMENTS.csv",index=False)
    pd.DataFrame(timings).to_csv(out/"cpu_analysis_timings_NOT_INFERENCE.csv",index=False)
    data.to_csv(out/"calibration_with_memory.csv",index=False)
    code_dir=out/"code_snapshot"
    code_dir.mkdir()
    hashes={}
    for path in sorted(Path(__file__).parent.glob("*.py")):
        (code_dir/path.name).write_bytes(path.read_bytes())
        hashes[path.name]=sha256(path.read_bytes())
    manifest={"baseline":base_manifest,"step1_filename":step1.name,"step1_sha256":sha256(step1.read_bytes()),
              "code_sha256":hashes,"created_utc":datetime.now(timezone.utc).isoformat(),
              "protocol_sha256":sha256(json.dumps(specification,sort_keys=True).encode()),
              "analysis_environment":{"python":platform.python_version(),
                   **{p:importlib.metadata.version(p) for p in ("numpy","pandas","scipy")}}}
    for name,payload in [("summary.json",report),("step1_audit.json",audit),("protocol.json",specification),
                         ("input_manifest.json",manifest),("source_metadata.json",metadata),("cv_models_and_folds.json",fold_rows)]:
        json_write(out/name,payload)
    lines=["# Step 2: GPU-memory estimation and budget-aware placement", "",
           "Step 1 artifact, source identity and saved predictions passed reproduction checks.",
           f"Audited {len(data)} configurations and {base_manifest['raw_trials_checked']} original trials.",
           "No GPU inference was executed.","", "## Retrospective memory prediction (GPU rows only)",
           f"Median error: {stats['median_ape_pct_gpu_only']:.2f}%; worst error: {stats['worst_ape_pct_gpu_only']:.2f}%.",
           f"Heuristic-guard misses: {stats['guard_misses_rows']}/{len(gpu)} allocation rows.","",
           "## Policy evaluation", "", "| Policy | Decisions | Budget violations | Median feasible regret | Worst feasible regret |",
           "| --- | ---: | ---: | ---: | ---: |"]
    for p,r in report["policy_metrics"].items():
        lines.append(f"| {p} | {r['decisions']} | {r['budget_violations']} | {r['median_regret_pct_among_feasible']:.2f}% | {r['worst_regret_pct_among_feasible']:.2f}% |")
    lines += ["","Shared memory filter; oracle uses actual allocations without a heuristic margin.",
              "Violation cases are counted, not rewarded with deceptively low regret.","", "## Scope",""]+["- "+x for x in limitations]
    (out/"summary.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines[:20]))
    print("Saved:",out.resolve())
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--step1",type=Path,required=True)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.step1,a.baseline,a.out)

if __name__=="__main__": main()
