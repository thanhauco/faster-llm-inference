"""Speculative sampling must be lossless: same output distribution as the target alone."""

import itertools

import numpy as np
import pytest

from specdec.drafters import DraftModelDrafter, NgramDrafter
from specdec.engine import speculative_generate
from specdec.models import ToyLM, make_draft_model
from specdec.sampling import apply_temperature, total_variation
from specdec.verify import verify_draft


def test_greedy_accepts_matching_prefix():
    p = np.eye(4)[[2, 1, 3, 0]]  # target argmax: 2, 1, 3, 0
    r = verify_draft([2, 1, 0], None, p, np.random.default_rng(0), greedy=True)
    assert (r.num_accepted, r.next_token) == (2, 3)


def test_greedy_all_accepted_gives_bonus_token():
    p = np.eye(4)[[2, 1, 3, 0]]
    r = verify_draft([2, 1, 3], None, p, np.random.default_rng(0), greedy=True)
    assert (r.num_accepted, r.next_token, r.emitted) == (3, 0, 4)


def test_empty_draft_samples_from_target():
    p = np.array([[0.0, 1.0, 0.0]])
    r = verify_draft([], None, p, np.random.default_rng(0))
    assert (r.num_accepted, r.next_token) == (0, 1)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        verify_draft([1, 2], None, np.ones((2, 4)) / 4, np.random.default_rng(0))


@pytest.mark.parametrize("deterministic", [False, True])
def test_single_position_acceptance_rate_is_one_minus_tv(deterministic):
    rng = np.random.default_rng(0)
    p = np.array([0.5, 0.3, 0.15, 0.05])
    q = np.array([0.25, 0.25, 0.25, 0.25])
    n, accepted, out = 20000, 0, np.zeros(4)
    for _ in range(n):
        if deterministic:
            tok, qrow = 1, None
        else:
            tok = int(rng.choice(4, p=q))
            qrow = q[None]
        r = verify_draft([tok], qrow, np.stack([p, p]), rng)
        accepted += r.num_accepted
        out[tok if r.num_accepted else r.next_token] += 1
    expected = p[1] if deterministic else 1 - total_variation(p, q)
    assert abs(accepted / n - expected) < 0.015
    # First emitted token is distributed exactly as p.
    assert total_variation(out / n, p) < 0.015


def _exact_sequence_probs(model: ToyLM, prompt, length, temperature):
    probs = {}
    V = model.vocab_size
    for seq in itertools.product(range(V), repeat=length):
        sess = model.session(prompt)
        pr = 1.0
        for tok in seq:
            pr *= apply_temperature(sess.next_probs(), temperature)[tok]
            sess.extend([tok])
        probs[seq] = pr
    return probs


@pytest.mark.parametrize("drafter_kind", ["draft-model", "ngram"])
def test_speculative_sampling_matches_target_distribution(drafter_kind):
    """Empirical distribution of 3-token continuations == exact target distribution."""
    target = ToyLM(vocab_size=5, hidden_size=8, head_scale=1.0, seed=3)
    prompt = [1, 2, 1, 2, 1]  # repetitive so the n-gram drafter actually proposes
    if drafter_kind == "draft-model":
        drafter = DraftModelDrafter(make_draft_model(target, noise=1.0, seed=9))
    else:
        drafter = NgramDrafter(1, 3)
    temperature, length, n = 0.9, 3, 6000
    exact = _exact_sequence_probs(target, prompt, length, temperature)
    counts = dict.fromkeys(exact, 0)
    rng = np.random.default_rng(42)
    rejections = 0
    for _ in range(n):
        r = speculative_generate(target, drafter, prompt, length, k=3, temperature=temperature, rng=rng)
        counts[tuple(r.tokens)] += 1
        rejections += r.stats.num_draft_tokens - r.stats.num_accepted_tokens
    assert rejections > 100, "test is only meaningful if drafts actually get rejected"
    emp = np.array([counts[s] / n for s in exact])
    ref = np.array(list(exact.values()))
    # Sampling noise for 125 outcomes and n=6000 keeps TV well below 0.05.
    assert total_variation(emp, ref) < 0.05
