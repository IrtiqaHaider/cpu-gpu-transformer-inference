# Step 3: fresh-workload evaluation

Completed 20 primary configurations and 3 historical controls. 230 measured generations across two blocks.

## Frozen latency predictions

| Predictor | Target | Median error | Worst error |
|---|---|---:|---:|
| compute | prefill_ms | 10.41% | 24.72% |
| compute | ttft_ms | 13.86% | 46.03% |
| compute | decode_tpot_ms | 10.68% | 49.82% |
| compute | generation_ms | 11.69% | 42.79% |
| compute_transfer | prefill_ms | 10.75% | 24.68% |
| compute_transfer | ttft_ms | 13.57% | 46.01% |
| compute_transfer | decode_tpot_ms | 10.05% | 49.85% |
| compute_transfer | generation_ms | 11.09% | 42.84% |

## Policy decisions

| Policy | Decisions | Violations | Worst feasible regret |
|---|---:|---:|---:|
| max_gpu | 32 | 0 | 239.54777807305683% |
| compute | 32 | 0 | 239.54777807305683% |
| compute_transfer | 32 | 0 | 239.54777807305683% |

## Interpretation limits

- Only four new prompt/batch workload shapes at output length 32.
- Two blocks are not independent sessions; ten trials per setting do not establish stable p95.
- Historical controls diagnose environment changes but never rescale frozen predictions.
- Peak allocated-memory budgets are synthetic, not enforced physical device limits.
- Best measured feasible reference is restricted to the five supported prefix candidates.
- New measurements are now test data; any subsequent model tuning must use another future test.

Inspect historical_controls.csv and each block runtime.json before attributing error to the predictor. Do not retune the margin, drop slow trials, or claim a policy improvement unless the measured comparison supports it.
