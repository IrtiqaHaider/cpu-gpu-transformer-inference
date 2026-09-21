# Selected primary references

These sources support background and definitions. The study's numerical results come from the supplied archives, not these papers. This is a selected positioning bibliography, not an exhaustive novelty search.

1. Sheng, Y., et al. **FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU.** arXiv:2303.06865, 2023. https://arxiv.org/abs/2303.06865  
   Context: heterogeneous GPU/CPU/disk resources and optimization-based tensor placement for throughput-oriented inference. This artifact is not a FlexGen reproduction.
2. Zhao, X., et al. **HeteGen: Heterogeneous Parallel Inference for Large Language Models on Resource-Constrained Devices.** arXiv:2403.01164, 2024. https://arxiv.org/abs/2403.01164  
   Context: heterogeneous parallel computation and asynchronous overlap. This artifact's dependent layers execute sequentially.
3. Jiang, X., et al. **NEO: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference.** arXiv:2411.01142, 2024. https://arxiv.org/abs/2411.01142  
   Context: attention/KV offloading, asymmetric pipelining and load-aware online serving. Neither NEO nor an online serving baseline was run here.
4. PyTorch documentation. **PyTorch Benchmark.** https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html  
   Context: warmup, synchronization, thread settings and measurement variability. Consulted September 20, 2026.
5. PyTorch 2.11 documentation. **torch.cuda.memory.max_memory_allocated.** https://docs.pytorch.org/docs/2.11/generated/torch.cuda.memory.max_memory_allocated.html  
   Defines peak GPU memory occupied by tensors; not all device memory. Consulted September 20, 2026.
6. SciPy documentation. **scipy.optimize.nnls.** https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html  
   Nonnegative least-squares solver used by the predictors. Consulted September 20, 2026.

