# Evidence index and provenance

## Authoritative archives

`data/archives/step4_export.zip` is the original Step 4 upload. Its `inputs/step3_export.zip` contains Step 3, whose `inputs/step2_export.zip` contains Step 2, the September 19 baseline archive and Step 1. No nested archive was edited. `data/archives/initial_checks_export.zip` separately preserves the September 18 validation, quick run and copy microbenchmark. It is supplementary evidence, not part of the later runtime's measurements.

`release_checks/artifact_manifest.json` records immutable source/data hashes. `release_checks/step4_audit.json` records the Step 4 input hash and analysis reproduction. Hashes show internal consistency and detect changes; they do not constitute public preregistration or external proof of when a measurement occurred.

## Report data references

| ID | Files | Relevant evidence |
| --- | --- | --- |
| D0 | `data/baseline/main/{summary,trials,profiles}.csv`, `metadata.json` | 450 main trials, allocation/latency tradeoff, CPU-stage diagnostics |
| D1 | `data/baseline/layouts/{summary,trials,profiles}.csv` | 120 layout trials; 12/72 MiB boundary payload contrast |
| D2 | `data/step1/summary.json`, `retrospective_cv_predictions.csv`; `data/step2/{summary.json,memory_cv.csv,policy_cv.csv}` | Historical grouped validation, model fitting and policy baseline |
| D3 | `data/step3/analysis/{all_trials,observed_cases,policy_scores}.csv`, `summary.json`; `data/step3/frozen/` | 230 new/control trials; frozen model predictions and decisions |
| D4 | `data/step4/analysis/{all_trials,observed_cases,thread_comparison,session_replication,policy_scores}.csv`, `summary.json` | 260 replication/thread trials, 26 case/thread summaries, same-guard regret |
| D5 | `data/step4/reference_threads_{1,2}.json`, `runtime_preflight.json`, `test_output.txt` | 26 numerical gates, changed runtime boot identity, recorded 240-test pass |
| D6 | `data/step4/step3_exploratory/margin_{sensitivity,summary}.csv` | Post-hoc guard alternatives on Step 3 only; not deployed validation |
| D7 | `data/initial_checks/validation.json`, `copies/` | Earlier separately dated pretrained validation and isolated transfers |

## Headline calculations

Baseline headline: filter D0 to batch=1, sequence_length=128, new_tokens=32; compare gpu_layers=6 and 12. Memory saving = 1 - split_peak/full_gpu_peak. Decode penalty = split_decode_tpot/full_gpu_decode_tpot. Do not substitute end-to-end throughput for cached decoding.

Frozen-prediction errors: D3 primary cases only (20). Memory errors exclude CPU-only cases (16 GPU-positive). D4 prediction errors use the ten batch-three primary cases separately per thread count; memory errors use eight GPU-positive cases per thread count. Do not directly compare these medians as if the cohorts were identical.

Guard example: D3/D4 `B3_S256_N32`, budget 640, selected GPU blocks 9, best measured feasible blocks 12. D4 uses 10 trial medians per candidate/thread condition. Predicted GPU-only peak=614.497973 MiB; guarded=661.222872 MiB. Actual two-thread peak=619.239746 MiB. Two-thread latency ratio=1114.286924/340.701586=3.270566. Step 3 ratio=3.395478.

Thread ratio: D4 `generation_ratio_one_vs_two` is median(one thread)/median(two threads), not a throughput speedup. Slow-trial counts exceed 1.5× the corresponding case/thread median, including three historical-control cases. Count=1/130 for one thread, 17/130 for two threads. The threshold is descriptive and data-dependent, not a predeclared pass/fail criterion.

Main-study total: 450 + 120 + 230 + 260 = 1,060 measured generations. It excludes the earlier 18 smoke-test generations, calibration refitting, warmups, tests and repeated policy-table rows. Do not call it 1,060 independent workloads.

## Recorded versus newly executed verification

Uploaded Step 4: 240 tests passed; 26 pretrained gate checks passed; 260 generation trials completed on T4. During release assembly: the same core source ran 231 CPU-compatible tests, nine CUDA tests skipped; historical refitting and 18 Step 3/4 CSV analyses reproduced on CPU. Any release packaging tests are separate from these counts. No new GPU execution happened during report preparation.
