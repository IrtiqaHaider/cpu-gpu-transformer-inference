# Step 4 audit and completion decision

**Decision: the planned measurement stage is complete. No additional GPU run is required for this release.**

Input: `step4_20260919T211303466524Z_export.zip`  
SHA-256: `4e5918a8382d0487c5e5f4c80a8318f41a175c629882ae561de96619179bfae9`

The audit reproduced every saved Step 4 analysis CSV from raw trials, the full Step 4 summary, the parent Step 3 predictions/decisions/analysis, and frozen model bytes. It independently reran the unchanged software suite on CPU (231 passed; nine CUDA-specific skipped). The uploaded test log records 240 passed on its GPU runtime. Release preparation did not run GPU inference.

## Completion and numerical gates

52 case/block measurements completed, covering 13 cases at two thread settings in four blocks. Each case/block has five measured trials: 260 complete generations. Runtime, freeze and numerical-gate hashes agree. The recorded boot identifier differs from Step 3, supporting a distinct runtime boot, not necessarily a different physical host.

All 26 Hugging Face comparisons passed: 13 cases at each CPU-thread setting, each checking prefill logits and three teacher-forced cached decode steps. The largest recorded prefill absolute difference is 0.0001220703125; cached-decode difference is 0.00006103515625. Argmax agreement is 1.0 on those checks. This is not 100% task accuracy or validation of every benchmark-generated token.

## Frozen prediction and selection results

| Condition | Generation median / worst error, compute+transfer | GPU-positive memory median / worst error |
| --- | --- | --- |
| One CPU thread | 17.47% / 29.44% | 0.88% / 1.87% |
| Two CPU threads | 14.82% / 29.80% | 0.82% / 1.87% |

All policies choose the same placement for each workload/budget. There are 16 such decisions per policy per thread, with zero observed budget violations; these reuse candidate measurements and are not independent experiments. The unchanged margin's largest two-thread regret is 227.06%, or 3.27× the best measured feasible latency. The one-thread maximum is 215.15%, or 3.15×.

## CPU-thread and session observations

Across all 13 cases, one-thread/two-thread median generation ratios span 0.905–1.078. One thread is not uniformly faster. It has fewer trials above 1.5× each case/thread median (1/130 versus 17/130); that descriptive result comes from one ABBA-ordered VM and does not isolate a universal causal mechanism.

Two-thread Step 4/Step 3 median latency ratios span 0.948–1.164 over the 13 shared cases. The runtime shift is not used to rescale predictions. Step 4 is a post-hoc selected replication subset, not a second unseen-workload test.

## Implication for the report

Lead with a reproducible resource tradeoff, frozen-model evaluation, no selection advantage over a simple baseline, and a replicated conservative-margin penalty. Do not present these results as a novel faster serving engine, real-capacity rescue, stable production p95, or proof that one CPU thread is universally best.
