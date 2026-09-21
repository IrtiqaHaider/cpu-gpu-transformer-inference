# Research step 1: retrospective cost-model baseline

Validated 45 main-sweep configurations and 450 raw trials.
Used 9 leave-one-workload-out folds per model.
No new GPU inference was executed. Historical exploratory data is not a new blind test.

## Median absolute percentage error on withheld workload groups

| Model | Prefill | TTFT | Decode step | Generation |
| --- | ---: | ---: | ---: | ---: |
| compute | 5.46% | 6.08% | 6.40% | 6.25% |
| compute_transfer | 4.97% | 6.07% | 7.06% | 6.18% |

Lower is better. Do not interpret 100 minus percentage error as model accuracy.
The summary JSON also reports mean/worst errors; inspect per-case failures, not only the median.

## Scope and limitations

- All historical results have already been inspected; this is not a new blind test.
- Entire workload groups, not individual trial rows, are withheld in each fold.
- Scaling and coefficients are fitted on each fold's training rows only.
- This version predicts latency only: no memory-budget planner is implemented yet.
- Prefix-only, GPT-2 small, FP32, recorded two-thread T4 environment.
- Transfer features are boundary/payload proxies, not measured or causal PCIe timings.
- Correlated features make individual coefficients non-identifiable as physical costs.
- Some folds extrapolate beyond training batch/sequence extrema; pooled errors mix interpolation and extrapolation.
- Prediction targets are medians; this is not a tail-latency or uncertainty model.
- Predicted generation = predicted TTFT + (N-1)*predicted average decode step. Medians need not add exactly.
- CPU variability, model error and drift can all contribute to validation error.
- Results do not establish a placement advantage or inference speedup.

## Next checkpoint

Inspect this baseline before adding memory prediction and a budget-aware planner.
Do not run new held-out GPU experiments until the candidate policies and evaluation protocol are fixed.
