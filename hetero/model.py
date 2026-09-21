"""Inference-only GPT-2 in ordinary PyTorch; no automatic device dispatch.

The checkpoint loader converts Hugging Face GPT-2 Conv1D [in, out] matrices
into nn.Linear [out, in] matrices. The output projection uses wte.weight
DIRECTLY, preserving weight tying without a second copy.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GPT2Spec:
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: int | None = None
    layer_norm_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        if min(self.vocab_size, self.n_positions, self.n_embd,
               self.n_layer, self.n_head) <= 0:
            raise ValueError("All model dimensions must be positive.")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head.")
        if self.n_inner is not None and self.n_inner <= 0:
            raise ValueError("n_inner must be positive.")

    @classmethod
    def from_hf_config(cls, cfg: Mapping) -> "GPT2Spec":
        if cfg.get("model_type", "gpt2") != "gpt2":
            raise ValueError("This implementation supports standard GPT-2 only.")
        if cfg.get("activation_function", "gelu_new") != "gelu_new":
            raise ValueError("Only GPT-2's gelu_new activation is implemented.")
        for flag in ("add_cross_attention", "scale_attn_by_inverse_layer_idx",
                     "reorder_and_upcast_attn"):
            if cfg.get(flag, False):
                raise ValueError(f"Unsupported GPT-2 variant: {flag}=True")
        if not cfg.get("scale_attn_weights", True):
            raise ValueError("Attention scaling must be enabled.")
        if not cfg.get("tie_word_embeddings", True):
            raise ValueError("Untied embedding checkpoints are not supported.")
        return cls(**{k: cfg[k] for k in cls.__dataclass_fields__ if k in cfg})


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class Attention(nn.Module):
    def __init__(self, cfg: GPT2Spec) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.register_buffer(
            "causal", torch.ones(cfg.n_positions, cfg.n_positions,
                                 dtype=torch.bool).tril()[None, None],
            persistent=False,
        )

    def forward(self, x: torch.Tensor,
                past: tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, query_len, hidden = x.shape
        q, k, v = self.qkv(x).split(hidden, dim=-1)
        # Contiguous K/V own their storage instead of retaining a QKV view.
        q, k, v = [t.view(batch, query_len, self.n_head, self.head_dim)
                   .transpose(1, 2).contiguous() for t in (q, k, v)]
        if past is not None:
            k = torch.cat((past[0], k), dim=-2)
            v = torch.cat((past[1], v), dim=-2)
        key_len = k.shape[-2]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # The offset is essential for cached decoding and multi-token chunks.
        mask = self.causal[:, :, key_len - query_len:key_len, :key_len]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, dim=-1)
        out = (probs @ v).transpose(1, 2).contiguous().view(batch, query_len, hidden)
        return self.proj(out), (k, v)


class Block(nn.Module):
    def __init__(self, cfg: GPT2Spec) -> None:
        super().__init__()
        inner = cfg.n_inner or 4 * cfg.n_embd
        self.ln1 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.fc = nn.Linear(cfg.n_embd, inner)
        self.proj = nn.Linear(inner, cfg.n_embd)

    def forward(self, x: torch.Tensor,
                past: tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attended, present = self.attn(self.ln1(x), past)
        x = x + attended
        x = x + self.proj(gelu_new(self.fc(self.ln2(x))))
        return x, present


class GPT2(nn.Module):
    """Weights and blocks; execution and placement live in OffloadEngine.

    Dropout is intentionally omitted: this is an inference-only implementation.
    """
    def __init__(self, cfg: GPT2Spec) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.n_positions, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.source = {"model_id": "random-initialization", "revision": None}
        self.apply(self._init_weights)
        self.requires_grad_(False)
        self.eval()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def checkpoint_mapping(self):
        """Yield (local parameter name, HF name without transformer., transpose)."""
        yield "wte.weight", "wte.weight", False
        yield "wpe.weight", "wpe.weight", False
        for i in range(self.cfg.n_layer):
            for local, remote in (("ln1", "ln_1"), ("ln2", "ln_2"),
                                  ("attn.qkv", "attn.c_attn"),
                                  ("attn.proj", "attn.c_proj"),
                                  ("fc", "mlp.c_fc"), ("proj", "mlp.c_proj")):
                for part in ("weight", "bias"):
                    transpose = part == "weight" and not local.startswith("ln")
                    yield f"blocks.{i}.{local}.{part}", f"h.{i}.{remote}.{part}", transpose
        yield "ln_f.weight", "ln_f.weight", False
        yield "ln_f.bias", "ln_f.bias", False

    @torch.no_grad()
    def load_hf_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        prefix = "transformer." if "transformer.wte.weight" in state else ""
        params = dict(self.named_parameters())
        for local, remote, transpose in self.checkpoint_mapping():
            key = prefix + remote
            if key not in state:
                raise KeyError(f"Missing checkpoint parameter: {key}")
            value = state[key].T if transpose else state[key]
            if value.shape != params[local].shape:
                raise ValueError(f"Shape mismatch for {key}: {value.shape} vs {params[local].shape}")
            params[local].copy_(value)
        if "lm_head.weight" in state and not torch.equal(
                state["lm_head.weight"], state[prefix + "wte.weight"]):
            raise ValueError("Checkpoint output head is not tied to its embeddings.")

    @classmethod
    def from_pretrained(cls, model_id: str = "openai-community/gpt2",
                        revision: str = "main") -> "GPT2":
        # Imports are lazy: offline unit tests need PyTorch only.
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        cfg_path = hf_hub_download(model_id, "config.json", revision=revision)
        # HF cache paths contain the resolved commit SHA. Load all later files
        # from this exact snapshot, even if 'main' changes during a download.
        resolved = Path(cfg_path).parent.name
        if len(resolved) != 40:
            raise RuntimeError("Could not resolve a pinned HF snapshot revision.")
        cfg = GPT2Spec.from_hf_config(json.loads(Path(cfg_path).read_text()))
        model = cls(cfg)
        path = hf_hub_download(model_id, "model.safetensors", revision=resolved)
        state = load_file(path, device="cpu")
        model.load_hf_state_dict(state)
        del state
        model.source = {"model_id": model_id, "revision": resolved}
        return model
