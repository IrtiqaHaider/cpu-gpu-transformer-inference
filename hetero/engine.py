"""Manual, static layer placement with device-local KV caches.

No weight moves occur inside forward(). No Accelerate, device_map, pipeline
parallelism, quantization, or asynchronous copy/compute overlap is used.
All parameters, activations and KV tensors are float32 in this first version.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, TypeVar

import torch
from torch.nn import functional as F
from .model import GPT2

T = TypeVar("T")


def make_plan(n_layers: int, gpu_layers: int, layout: str = "prefix") -> tuple[str, ...]:
    if n_layers < 1 or not 0 <= gpu_layers <= n_layers:
        raise ValueError("gpu_layers must be between zero and n_layers.")
    if layout == "prefix":
        selected = set(range(gpu_layers))
    elif layout == "suffix":
        selected = set(range(n_layers - gpu_layers, n_layers))
    elif layout == "interleaved":
        # Exactly k approximately evenly spaced GPU blocks; k=N/2 alternates.
        selected = {i for i in range(n_layers)
                    if (i + 1) * gpu_layers // n_layers > i * gpu_layers // n_layers}
    else:
        raise ValueError("layout must be prefix, suffix or interleaved.")
    return tuple("cuda:0" if i in selected else "cpu" for i in range(n_layers))


def count_boundaries(plan: tuple[str, ...], anchor: str) -> int:
    path = (anchor,) + plan + (anchor,)
    return sum(a != b for a, b in zip(path, path[1:]))


class Trace:
    """SERIALIZED diagnostic wall timings, not an end-to-end benchmark.

    Each stage is synchronized separately. Copy durations include PyTorch
    allocation/dispatch/host staging; they are NOT pure PCIe DMA durations.
    """
    def __init__(self, uses_cuda: bool):
        self.uses_cuda = uses_cuda
        self.rows: list[dict] = []

    def sync(self) -> None:
        if self.uses_cuda:
            torch.cuda.synchronize(0)

    def call(self, name: str, category: str, fn: Callable[[], T],
             src: str = "", dst: str = "", nbytes: int = 0) -> T:
        self.sync()
        start = time.perf_counter_ns()
        value = fn()
        self.sync()
        elapsed = (time.perf_counter_ns() - start) / 1e6
        self.rows.append({"name": name, "category": category, "src": src,
                          "dst": dst, "bytes": nbytes, "wall_ms": elapsed})
        return value

    def total(self, category: str) -> float:
        return sum(r["wall_ms"] for r in self.rows if r["category"] == category)


@dataclass(frozen=True)
class KVCache:
    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    length: int
    batch_size: int
    owner: object

    def bytes_by_device(self) -> dict[str, int]:
        result = {"cpu": 0, "cuda:0": 0}
        for pair in self.layers:
            for tensor in pair:
                key = str(tensor.device)
                result[key] = result.get(key, 0) + tensor.numel() * tensor.element_size()
        return result


class OffloadEngine:
    def __init__(self, model: GPT2, gpu_layers: int = 0, layout: str = "prefix"):
        self.model = model
        self._cache_owner = object()
        self.set_placement(gpu_layers, layout)

    def set_placement(self, gpu_layers: int, layout: str = "prefix") -> None:
        """Setup only; clear all caller-owned caches before changing placement."""
        plan = make_plan(self.model.cfg.n_layer, gpu_layers, layout)
        if gpu_layers and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Select a Colab GPU runtime or use gpu_layers=0.")
        self.plan = plan
        self.gpu_layers = gpu_layers
        self.layout = layout
        self.uses_cuda = gpu_layers > 0
        # Keep embedding + output weight as ONE physical tensor. At k=0 all
        # modules are CPU; otherwise these endpoint modules are on CUDA.
        self.anchor = torch.device("cuda:0" if self.uses_cuda else "cpu")
        self.model.wte.to(device=self.anchor, dtype=torch.float32)
        self.model.wpe.to(device=self.anchor, dtype=torch.float32)
        self.model.ln_f.to(device=self.anchor, dtype=torch.float32)
        for block, device in zip(self.model.blocks, self.plan):
            block.to(device=device, dtype=torch.float32)
        self.model.eval()
        self._cache_owner = object()  # Old caches are now explicitly invalid.
        self.synchronize()

    def synchronize(self) -> None:
        if self.uses_cuda:
            torch.cuda.synchronize(0)

    @property
    def boundaries(self) -> int:
        return count_boundaries(self.plan, str(self.anchor))

    def resident_bytes(self) -> dict[str, int]:
        sizes = {"cpu_parameter_bytes": 0, "gpu_parameter_bytes": 0,
                 "cpu_buffer_bytes": 0, "gpu_buffer_bytes": 0}
        for label, values in (("parameter", self.model.parameters()),
                              ("buffer", self.model.buffers())):
            for value in values:
                side = "gpu" if value.device.type == "cuda" else "cpu"
                sizes[f"{side}_{label}_bytes"] += value.numel() * value.element_size()
        return sizes

    def _move(self, x: torch.Tensor, device: torch.device | str, name: str,
              category: str, trace: Trace | None) -> torch.Tensor:
        device = torch.device(device)
        if x.device == device:
            return x
        fn = lambda: x.to(device, non_blocking=False)
        if trace is None:
            return fn()
        return trace.call(name, category, fn, str(x.device), str(device),
                          x.numel() * x.element_size())

    @staticmethod
    def _compute(name: str, device: torch.device | str, fn: Callable[[], T],
                 trace: Trace | None) -> T:
        if trace is None:
            return fn()
        category = "gpu_compute" if torch.device(device).type == "cuda" else "cpu_compute"
        return trace.call(name, category, fn, dst=str(device))

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, cache: KVCache | None = None,
                *, use_cache: bool = True, all_logits: bool = False,
                trace: Trace | None = None) -> tuple[torch.Tensor, KVCache | None]:
        """Equal-length, UNPADDED inputs only. Returns [B,1,V] by default.

        Token IDs may begin on CPU or the endpoint device. A cache is valid only
        for this engine's current placement. No padding/attention-mask API is
        exposed deliberately; silently treating padding as text would be wrong.
        """
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a rank-2 torch.long tensor.")
        batch, seq = input_ids.shape
        if batch < 1 or seq < 1:
            raise ValueError("Batch and sequence dimensions must be nonempty.")
        past_length = 0 if cache is None else cache.length
        if past_length + seq > self.model.cfg.n_positions:
            raise ValueError("Prompt + cached/new input exceeds GPT-2's context window.")
        if cache is not None:
            if cache.owner is not self._cache_owner:
                raise ValueError("Cache belongs to another engine or an old placement.")
            if cache.batch_size != batch or len(cache.layers) != len(self.plan):
                raise ValueError("Cache batch or layer count does not match.")
            for pair, device in zip(cache.layers, self.plan):
                if any(str(t.device) != device or t.shape[-2] != cache.length for t in pair):
                    raise ValueError("Cache shape/device mismatch; caches must stay layer-local.")

        ids = self._move(input_ids, self.anchor, "input_ids", "token_copy", trace)
        def embed():
            positions = torch.arange(past_length, past_length + seq, device=self.anchor)
            return self.model.wte(ids) + self.model.wpe(positions)[None, :, :]
        hidden = self._compute("embeddings", self.anchor, embed, trace)
        presents = []
        for i, (block, device) in enumerate(zip(self.model.blocks, self.plan)):
            hidden = self._move(hidden, device, f"into_block_{i}", "activation_copy", trace)
            past = None if cache is None else cache.layers[i]
            hidden, present = self._compute(
                f"block_{i}", device, lambda: block(hidden, past), trace)
            if use_cache:
                presents.append(present)
            del present
        # Keep the full hidden sequence through the final boundary, intentionally
        # making boundary bytes comparable across layouts. Last-token slicing is
        # performed only at the vocabulary projection; optimizing this is a v2 idea.
        hidden = self._move(hidden, self.anchor, "into_final_norm", "activation_copy", trace)
        hidden = self._compute("final_norm", self.anchor, lambda: self.model.ln_f(hidden), trace)
        logits = self._compute("lm_head", self.anchor, lambda: F.linear(
            hidden if all_logits else hidden[:, -1:, :], self.model.wte.weight), trace)
        result_cache = KVCache(tuple(presents), past_length + seq, batch, self._cache_owner) if use_cache else None
        return logits, result_cache

    @torch.inference_mode()
    def sample_greedy(self, logits: torch.Tensor,
                      trace: Trace | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        token = self._compute("argmax", self.anchor,
                              lambda: logits[:, -1, :].argmax(-1, keepdim=True), trace)
        # Host copy gives a well-defined user-visible token-ready timing boundary.
        host = self._move(token, "cpu", "output_token", "token_copy", trace)
        return token, host

    @torch.inference_mode()
    def generate_fixed(self, input_ids: torch.Tensor, new_tokens: int = 16) -> torch.Tensor:
        """Fixed-length greedy generation; deliberately does not stop at EOS."""
        if new_tokens < 1:
            raise ValueError("new_tokens must be positive.")
        if input_ids.shape[1] + new_tokens - 1 > self.model.cfg.n_positions:
            raise ValueError("Generation would exceed the context window.")
        logits, cache = self.forward(input_ids)
        generated = []
        for i in range(new_tokens):
            token, host = self.sample_greedy(logits)
            generated.append(host)
            if i + 1 < new_tokens:
                logits, cache = self.forward(token, cache)
        return torch.cat(generated, dim=1)
