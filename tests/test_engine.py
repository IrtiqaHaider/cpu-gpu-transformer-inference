"""Offline software tests; CUDA-specific cases skip without a CUDA device."""
import math

import pytest
import torch
from torch.nn import functional as F

from hetero.engine import OffloadEngine, Trace, make_plan, count_boundaries
from hetero.model import GPT2, GPT2Spec
from hetero.benchmark import timed_prefill, timed_generation, profile_case, run_sweep


@pytest.fixture
def engine():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    return OffloadEngine(GPT2(GPT2Spec(vocab_size=97, n_positions=32,
                                      n_embd=24, n_layer=4, n_head=4)))


def tokens(batch=2, seq=7):
    return torch.randint(97, (batch, seq), generator=torch.Generator().manual_seed(73))


@pytest.mark.parametrize("layout", ["prefix", "suffix", "interleaved"])
@pytest.mark.parametrize("k", [0, 1, 3, 6, 11, 12])
def test_plan_count(k, layout):
    plan = make_plan(12, k, layout)
    assert len(plan) == 12
    assert plan.count("cuda:0") == k


def test_boundary_count():
    assert count_boundaries(make_plan(12, 0), "cpu") == 0
    assert count_boundaries(make_plan(12, 12), "cuda:0") == 0
    assert count_boundaries(make_plan(12, 6), "cuda:0") == 2
    assert count_boundaries(make_plan(12, 6, "suffix"), "cuda:0") == 2
    assert count_boundaries(make_plan(12, 6, "interleaved"), "cuda:0") == 12


@pytest.mark.parametrize("k", [-1, 13])
def test_bad_layer_count(k):
    with pytest.raises(ValueError):
        make_plan(12, k)


def test_bad_layout():
    with pytest.raises(ValueError):
        make_plan(12, 6, "unknown")


@torch.inference_mode()
def test_last_logits_match_full(engine):
    all_logits, cache = engine.forward(tokens(), all_logits=True)
    last, _ = engine.forward(tokens())
    assert all_logits.shape == (2, 7, 97)
    assert last.shape == (2, 1, 97)
    torch.testing.assert_close(last, all_logits[:, -1:, :], atol=1e-6, rtol=1e-5)
    assert cache.length == 7


@pytest.mark.parametrize("chunk", [1, 2, 3])
@torch.inference_mode()
def test_cache_equals_recomputed_prefix(engine, chunk):
    ids = tokens(seq=12)
    _, cache = engine.forward(ids[:, :5])
    for start in range(5, 11, chunk):
        end = min(start + chunk, 12)
        actual, cache = engine.forward(ids[:, start:end], cache, all_logits=True)
        expected, _ = engine.forward(ids[:, :end], use_cache=False, all_logits=True)
        torch.testing.assert_close(actual, expected[:, start:end], atol=2e-6, rtol=1e-5)
        assert cache.length == end


@torch.inference_mode()
def test_future_tokens_cannot_affect_past(engine):
    ids = tokens()
    other = ids.clone()
    other[:, 4:] = (other[:, 4:] + 7) % 97
    a, _ = engine.forward(ids, all_logits=True)
    b, _ = engine.forward(other, all_logits=True)
    torch.testing.assert_close(a[:, :4], b[:, :4], atol=0, rtol=0)


@torch.inference_mode()
def test_attention_against_independent_sdpa(engine):
    attn = engine.model.blocks[0].attn
    x = torch.randn(2, 5, 24)
    actual, _ = attn(x)
    q, k, v = attn.qkv(x).chunk(3, dim=-1)
    q, k, v = [t.view(2, 5, 4, 6).transpose(1, 2) for t in (q, k, v)]
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
    expected = attn.proj(expected.transpose(1, 2).contiguous().view(2, 5, 24))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)


@torch.inference_mode()
def test_generation_equals_no_cache(engine):
    ids = tokens()
    actual = engine.generate_fixed(ids, 5)
    prefix = ids.clone()
    expected = []
    for _ in range(5):
        logits, cache = engine.forward(prefix, use_cache=False)
        assert cache is None
        tok, _ = engine.sample_greedy(logits)
        expected.append(tok)
        prefix = torch.cat((prefix, tok), dim=1)
    assert torch.equal(actual, torch.cat(expected, dim=1))


def test_cache_rejected_after_relocation(engine):
    _, cache = engine.forward(tokens())
    engine.set_placement(0)
    with pytest.raises(ValueError, match="old placement"):
        engine.forward(tokens(seq=1), cache)


def test_cache_rejected_by_other_engine(engine):
    _, cache = engine.forward(tokens())
    other = OffloadEngine(GPT2(engine.model.cfg))
    with pytest.raises(ValueError, match="another engine"):
        other.forward(tokens(seq=1), cache)


@pytest.mark.parametrize("bad", [torch.zeros(2, 3), torch.zeros(3, dtype=torch.long),
                                  torch.empty(1, 0, dtype=torch.long)])
def test_bad_inputs(engine, bad):
    with pytest.raises(ValueError):
        engine.forward(bad)


def test_context_and_batch_validation(engine):
    with pytest.raises(ValueError, match="context"):
        engine.forward(tokens(seq=33))
    _, cache = engine.forward(tokens())
    with pytest.raises(ValueError, match="batch"):
        engine.forward(tokens(batch=1, seq=1), cache)
    with pytest.raises(ValueError, match="context"):
        engine.generate_fixed(tokens(seq=30), 4)


def test_hf_parameter_mapping_and_tied_head(engine):
    state = {}
    params = dict(engine.model.named_parameters())
    for local, remote, transpose in engine.model.checkpoint_mapping():
        value = params[local].detach().clone()
        state["transformer." + remote] = value.T if transpose else value
    state["lm_head.weight"] = state["transformer.wte.weight"].clone()
    other = GPT2(engine.model.cfg)
    other.load_hf_state_dict(state)
    for (_, a), (_, b) in zip(engine.model.named_parameters(), other.named_parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert not any(name.startswith("lm_head") for name, _ in other.named_parameters())
    state["lm_head.weight"][0, 0] += 1
    with pytest.raises(ValueError, match="not tied"):
        other.load_hf_state_dict(state)


def test_trace_and_cache_bytes(engine):
    trace = Trace(False)
    logits, cache = engine.forward(tokens(), trace=trace)
    engine.sample_greedy(logits, trace)
    assert trace.total("cpu_compute") > 0
    assert trace.total("activation_copy") == 0
    expected = 2 * 2 * 7 * 24 * 4 * 4  # K+V, B, S, H, bytes/float, layers
    assert cache.bytes_by_device() == {"cpu": expected, "cuda:0": 0}
    rows, report = profile_case(engine, tokens())
    assert report["profile_prefill_activation_bytes"] == 0
    assert {r["phase"] for r in rows} == {"prefill", "decode"}


def test_timing_accounting(engine):
    pre = timed_prefill(engine, tokens())
    run = timed_generation(engine, tokens(), 5)
    assert pre["prefill_ms"] > 0
    assert run["generation_ms"] > run["ttft_ms"] > 0
    assert run["final_cached_positions"] == 7 + 5 - 1
    assert len(run["decode_step_ms"]) == 4
    assert math.isclose(run["decode_generated_tokens_per_s"], 2 * 4 * 1000 / run["decode_ms"])
    assert math.isclose(run["generation_generated_tokens_per_s"], 2 * 5 * 1000 / run["generation_ms"])


def test_sweep_writes_real_rows(engine, tmp_path):
    df = run_sweep(engine, tmp_path, gpu_layers=[0], batches=[1], seq_lens=[4],
                   new_tokens=3, warmup=1, repeats=2)
    assert len(df) == 1
    assert df.iloc[0].status == "ok"
    for name in ("metadata.json", "summary.csv", "trials.csv", "profiles.csv"):
        assert (tmp_path / name).is_file()
    assert df.iloc[0].gpu_parameter_bytes == 0
    with pytest.raises(FileExistsError):
        run_sweep(engine, tmp_path, gpu_layers=[0], batches=[1], seq_lens=[4], new_tokens=3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("layout", ["prefix", "suffix", "interleaved"])
@pytest.mark.parametrize("k", [1, 2, 4])
@torch.inference_mode()
def test_gpu_equivalence_bytes_and_kv_locality(engine, layout, k):
    ids = tokens()
    expected, cache = engine.forward(ids, all_logits=True)
    expected_dec, _ = engine.forward(ids[:, :1], cache, all_logits=True)
    del cache
    engine.set_placement(k, layout)
    actual, cache = engine.forward(ids, all_logits=True)
    actual_dec, _ = engine.forward(ids[:, :1], cache, all_logits=True)
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(actual_dec.cpu(), expected_dec, atol=2e-5, rtol=1e-4)
    for pair, device in zip(cache.layers, engine.plan):
        assert all(str(t.device) == device for t in pair)
    _, profile = profile_case(engine, ids)
    assert profile["profile_prefill_activation_bytes"] == engine.boundaries * 2 * 7 * 24 * 4
    assert profile["profile_decode_activation_bytes"] == engine.boundaries * 2 * 24 * 4
    assert engine.resident_bytes()["gpu_parameter_bytes"] > 0
