# Methods and interpretation rules

## System

GPT-2 small, checkpoint `607a30d783dfa663caf39e06633721c8d4cfcd7e`; 12 blocks, width 768, 12 heads, vocabulary 50,257, context limit 1,024. FP32 is used on CPU and GPU with explicit eager causal attention. Input prompts are equal-length unpadded synthetic token IDs. EOS is ignored to keep output length fixed. GPU work uses one T4 per runtime, with PyTorch 2.11.0+cu128 and Transformers 4.48.3 recorded in the measured runs.

Blocks are placed once. Only activations cross CPU/GPU boundaries during a forward pass. Each layer's KV cache resides with that layer. When any block is on GPU, embeddings and output-side modules also reside on GPU; the shared embedding/output weight is counted once. Mixed contiguous placement therefore has two activation crossings, not one. All-CPU moves the endpoints to CPU too. This is not a controlled comparison of block placement alone across the CPU-only endpoint.

## Timing

Prefill processes the whole prompt, builds KV caches and computes final-position logits. First-token time also includes greedy selection and host token delivery. For N=32 outputs, prefill selects the first output and only 31 cached decode forward passes are executed. End-to-end time excludes downloads, tokenization, model placement and planner overhead. Warmups are excluded. GPU synchronization surrounds completion timing. Serialized stage profiles are separately instrumented diagnostics, not pure PCIe bandwidth or an exact decomposition of median generation time.

Decode throughput is B*(N-1)/decode_seconds. Generated throughput is B*N/generation_seconds. Prefill throughput counts input tokens B*S; it is not generated output. With an even trial count, median(rate) need not equal tokens/median(time). Peak memory is the measured maximum PyTorch allocated-tensor counter, not allocator reservation or total device/process memory. See PyTorch sources in REFERENCES.md.

## Cost models

`cost_model.py` uses relative-error-weighted nonnegative least squares with a fixed 1e-6 ridge stabilizer. Feature scales and target normalization are fitted only on training rows. Targets are prefill, TTFT and average cached decode-step latency. Generation is predicted as TTFT + (N-1)*average_decode_step.

Compute features include CPU/GPU block counts, linear-work and attention-work proxies, and endpoint batch terms. The transfer-aware variant adds activation-boundary count and total activation payload. For prefill, q=S and context=S; for decode, q=1 and mean processed context=S+N/2. Coefficients are associations, not independently measured hardware costs. Decode design matrices are rank deficient (rank 7/8 or 9/10 in full fits), so individual coefficients are not uniquely identifiable without the imposed stabilization.

The memory predictor adds fitted nonnegative temporary-allocation terms to exact parameter, buffer and final KV payloads. It is explicitly scoped to the original GPT-2 small implementation. Final cache length is S+N-1. Predicted feasibility uses 1.05*point_peak + 16 MiB for GPU-active candidates; CPU-only is zero for the GPU metric. The margin is unchanged throughout the primary prospective and replication experiments.

## Policy evaluation

Candidates are prefix placements with 0/3/6/9/12 GPU blocks. Maximum-GPU chooses the highest feasible count. The other policies minimize their predicted complete-generation latency over the same feasibility filter. The retrospective and prospective policies do not consume measured test latencies when selecting. Their best-measured-feasible reference is restricted to those five candidates and uses actual allocations without a guard.

Regret = 100*(selected_median_time / best_measured_feasible_median_time - 1). Violations are counted separately; a violating choice is not rewarded with low regret. Each budget/policy row reuses underlying candidate measurements and is not an independent GPU experiment. Budgets are 256/320/384/448/512/640/768/1,024 MiB; these are synthetic allocations, not enforced T4 capacity limits.

For fixed workload, the mixed-placement compute feature terms are affine in GPU count; the two added transfer features are constant across mixed prefix counts. This restricted model/candidate family is one reason not to generalize the observed equality of decisions to arbitrary schedules.

## Experimental separation

Historical baseline: 45 main records plus 12 layout records, 10 trials each, 3 warmups. Four layout-prefix records duplicate main workload/placement settings. Predictors train on 45 main rows. Steps 1/2 use nine leave-one-workload-out groups; all five placements of the held-out workload remain out of training. The already-inspected history is development validation, not a blind test.

Step 3: four new shapes (B=1/3, S=64/256), all five candidates, 200 timed generations; three historical controls add 30. Two reverse-order blocks use 5 warmups and 5 trials per case. Models, predictions and choices were locally hash-sealed before collection. This is a local integrity record, not trusted external timestamping or public preregistration.

Step 4: selected after inspecting Step 3, so it is a post-hoc robustness subset, not a second blind shape test. B=3 and S=64/256 plus three historical controls; 13 cases at one and two CPU threads. Four blocks use 2→1→1→2, reversing case order within each thread pair. There are 260 timed generations in one new runtime. Changed boot IDs record runtime separation, not physical-host independence. No refitting, rescaling or margin tuning occurred.

## Correctness and uncertainty

Step 3 records 20 independent Hugging Face comparisons; Step 4 records 26, checking prompt logits plus three teacher-forced decode steps for each configured shape/placement/thread case. All pass their numerical tolerances. This is not validation of all 31 decoded tokens or task-quality accuracy. CUDA unit tests and reference gates are distinct from performance trials.

Keep session and CPU-thread strata separate. The slow-trial heuristic (>1.5 times that case/thread median) is descriptive, not a formal outlier test. ABBA partially balances ordering; it does not randomize away contention or establish a causal throttle diagnosis. Ten trials per case and one additional runtime do not establish stable production tails. No token-level pseudoreplication, significance claim or discarded slow trials is used in the report.

Planner/calibration timings exist as CPU-only development diagnostics in Step 2, but online amortized overhead and end-to-end serving benefit were not established. No external inference framework, second model family, output-length extrapolation or quality benchmark is evaluated.
