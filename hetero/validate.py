"""Independent Hugging Face logit check before collecting pretrained results.

This optional integration gate needs requirements-reference.txt. The inference
engine itself does not depend on transformers. Do not interpret a skipped HF
check or a tiny/random smoke test as a pretrained-model correctness result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .benchmark import configure_cpu
from .engine import OffloadEngine
from .model import GPT2, GPT2Spec


@torch.inference_mode()
def validate(model_id="openai-community/gpt2", revision="main", tiny=False,
             out="results/validation.json") -> dict:
    # Public model API only; the engine does not call HF transformer blocks.
    from transformers import GPT2Config, GPT2LMHeadModel
    import transformers

    torch.manual_seed(123)
    if tiny:
        config = GPT2Config(vocab_size=101, n_positions=64, n_embd=32,
                            n_layer=4, n_head=4, attn_pdrop=0, resid_pdrop=0,
                            embd_pdrop=0, activation_function="gelu_new")
        config._attn_implementation = "eager"
        reference = GPT2LMHeadModel(config).eval().float()
        model = GPT2(GPT2Spec.from_hf_config(config.to_dict()))
        model.load_hf_state_dict(reference.state_dict())
    else:
        model = GPT2.from_pretrained(model_id, revision)
        reference = GPT2LMHeadModel.from_pretrained(
            model_id, revision=model.source["revision"], use_safetensors=True,
            attn_implementation="eager", torch_dtype=torch.float32,
        ).cpu().eval()
    engine = OffloadEngine(model)
    # Test prompt and teacher-forced cached chunks separately. Never feed each
    # model its own generated tokens when testing numerical equivalence.
    prompt = torch.randint(model.cfg.vocab_size, (2, 11))
    continuation = torch.randint(model.cfg.vocab_size, (2, 3))
    hf_prompt = reference(prompt, use_cache=True)
    hf_decode = reference(continuation, past_key_values=hf_prompt.past_key_values,
                          use_cache=True)
    expected_prefill = hf_prompt.logits.cpu()
    expected_decode = hf_decode.logits.cpu()
    # Reference model must not contaminate later GPU-memory measurements.
    del reference, hf_prompt, hf_decode
    n = model.cfg.n_layer
    plans = [(0, "prefix")]
    if torch.cuda.is_available():
        plans += [(n, "prefix"), (n // 2, "prefix"),
                  (n // 2, "suffix"), (n // 2, "interleaved")]
    results = []
    for k, layout in plans:
        engine.set_placement(k, layout)
        actual_prefill, cache = engine.forward(prompt, all_logits=True)
        actual_decode, cache = engine.forward(continuation, cache, all_logits=True)
        actual_prefill, actual_decode = actual_prefill.cpu(), actual_decode.cpu()
        atol = 5e-4 if k == 0 else 1e-3
        rtol = 1e-3
        for actual, expected in ((actual_prefill, expected_prefill),
                                 (actual_decode, expected_decode)):
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        entry = {
            "gpu_layers": k, "layout": layout, "atol": atol, "rtol": rtol,
            "prefill_max_abs_error": (actual_prefill - expected_prefill).abs().max().item(),
            "cached_chunk_max_abs_error": (actual_decode - expected_decode).abs().max().item(),
            "prefill_argmax_agreement": (actual_prefill.argmax(-1) == expected_prefill.argmax(-1)).float().mean().item(),
            "cached_chunk_argmax_agreement": (actual_decode.argmax(-1) == expected_decode.argmax(-1)).float().mean().item(),
            "passed": True,
        }
        print(entry, flush=True)
        results.append(entry)
        del cache, actual_prefill, actual_decode
    report = {"model": model.source, "tiny_random": tiny,
              "transformers": transformers.__version__, "torch": torch.__version__,
              "cuda_available": torch.cuda.is_available(), "checks": results}
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="openai-community/gpt2")
    p.add_argument("--revision", default="main")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", default="results/validation.json")
    a = p.parse_args()
    configure_cpu(a.threads)
    validate(a.model, a.revision, a.tiny, a.out)


if __name__ == "__main__":
    main()
