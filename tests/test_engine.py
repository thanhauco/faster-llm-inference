import numpy as np
import pytest

from specdec.bench.workloads import WORKLOADS
from specdec.engine import SpecStats, autoregressive_generate, speculative_generate

DRAFTERS = ["ngram", "draft-model", "eagle1", "eagle3", "dflash", "dspark"]


@pytest.mark.parametrize("name", DRAFTERS)
@pytest.mark.parametrize("workload", ["code-edit", "reasoning", "synthetic-random"])
def test_greedy_output_identical_to_target(target, drafters, name, workload):
    prompt = WORKLOADS[workload].prompts(target, 1, seed=5)[0]
    ref = autoregressive_generate(target, prompt, 60).tokens
    for k in (1, 4):
        out = speculative_generate(target, drafters[name], prompt, 60, k=k)
        assert out.tokens == ref


@pytest.mark.parametrize("name", DRAFTERS)
def test_stats_invariants(target, drafters, name):
    prompt = WORKLOADS["rag-qa"].prompts(target, 1, seed=1)[0]
    r = speculative_generate(target, drafters[name], prompt, 80, k=5, temperature=0.7,
                             rng=np.random.default_rng(0))
    s = r.stats
    assert len(r.tokens) == 80
    assert sum(s.acceptance_histogram) == s.num_steps
    assert s.num_accepted_tokens <= s.num_draft_tokens <= 5 * s.num_steps
    assert 1.0 <= s.mean_acceptance_length <= 6.0
    # Emitted tokens = 1 (prefill) + sum(accepted + 1), clipped at max_new_tokens.
    assert 1 + s.num_steps + s.num_accepted_tokens >= 80
    rates = s.per_position_acceptance()
    assert all(0.0 <= x <= 1.0 for x in rates)


def test_truncated_stats_match_definition():
    s = SpecStats(4)
    for drafted, accepted in [(4, 4), (4, 1), (2, 2), (0, 0), (4, 3)]:
        s.record(drafted, accepted)
    t = s.truncated(2)
    assert t.num_draft_tokens == 2 + 2 + 2 + 0 + 2
    assert t.num_accepted_tokens == 2 + 1 + 2 + 0 + 2
    assert t.acceptance_histogram == [1, 1, 3]
    with pytest.raises(ValueError):
        s.truncated(5)


def test_merge_adds_up():
    a, b = SpecStats(3), SpecStats(3)
    a.record(3, 2)
    b.record(3, 0)
    m = a.merge(b)
    assert (m.num_steps, m.num_accepted_tokens, m.acceptance_histogram) == (2, 2, [1, 0, 1, 0])
    assert m.mean_acceptance_length == pytest.approx(2.0)


def test_adaptive_verification_shortens_drafts(target, drafters):
    prompt = WORKLOADS["open-chat"].prompts(target, 1, seed=2)[0]
    d = drafters["dspark"]
    full = speculative_generate(target, d, prompt, 80, k=6)
    adaptive = speculative_generate(target, d, prompt, 80, k=6, confidence_threshold=0.5)
    assert adaptive.tokens == full.tokens  # still lossless
    assert adaptive.stats.mean_draft_length < full.stats.mean_draft_length


def test_k_must_be_positive(target, drafters):
    with pytest.raises(ValueError):
        speculative_generate(target, drafters["ngram"], [1, 2, 3], 5, k=0)


def test_learned_drafters_beat_chance(target, drafters):
    """Trained heads should agree with the target far more often than 1/V."""
    prompt = WORKLOADS["reasoning"].prompts(target, 1, seed=3)[0]
    for name in ("eagle3", "dspark", "draft-model"):
        s = speculative_generate(target, drafters[name], prompt, 100, k=3).stats
        assert s.draft_acceptance_rate > 0.2, name


def test_eagle_requires_fit(target):
    from specdec.drafters import EagleDrafter

    with pytest.raises(RuntimeError):
        speculative_generate(target, EagleDrafter(target), [1, 2, 3], 5)
