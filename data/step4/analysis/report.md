# Step 4: replication and CPU-thread sensitivity

260 measured generations. Frozen model and guard unchanged.

## Prediction errors (new session only)

| Threads | Predictor | Target | Median error | Worst error |
|---:|---|---|---:|---:|
| 1 | compute | generation | 15.94% | 29.40% |
| 1 | compute_transfer | generation | 17.47% | 29.44% |
| 2 | compute | generation | 14.78% | 29.76% |
| 2 | compute_transfer | generation | 14.82% | 29.80% |

## Policy results

| Threads | Policy | Violations | Worst feasible regret |
|---:|---|---:|---:|
| 1 | max_gpu | 0 | 215.15237564314646% |
| 1 | compute | 0 | 215.15237564314646% |
| 1 | compute_transfer | 0 | 215.15237564314646% |
| 2 | max_gpu | 0 | 227.0565708490714% |
| 2 | compute | 0 | 227.0565708490714% |
| 2 | compute_transfer | 0 | 227.0565708490714% |

## Interpretation

Read thread_comparison.csv and session_replication.csv separately. A ratio >1 means the numerator condition was slower.
Do not pool old and new sessions as exchangeable trials, or turn these results into an optimization win without support.

- Post-hoc selected replication subset, not a second blind workload evaluation.
- Only one additional VM; no universal latency or tail guarantee.
- Budget constraint is peak allocated tensor memory, not a hard physical VRAM limit.
- No claim that CPU thread count alone causes all timing changes.
- All three policies share the same unchanged 5% + 16 MiB guard.
- Whole-case telemetry includes warmup, placement and checks; it is not per-trial attribution.
