import pytest

from specdec.cost_model import HARDWARE, MODELS, CostModel, expected_acceptance_length
from specdec.drafters.base import DraftCost


@pytest.fixture
def cm():
    return CostModel(HARDWARE["h100"], MODELS["8b"])


def test_small_batch_verification_is_nearly_free(cm):
    # Memory-bound: scoring 6 tokens costs about the same as 1.
    assert cm.forward_ms(1, 6) == pytest.approx(cm.forward_ms(1, 1), rel=0.01)


def test_large_batch_verification_costs_compute(cm):
    small = cm.forward_ms(1, 6) / cm.forward_ms(1, 1)
    large = cm.forward_ms(256, 6) / cm.forward_ms(256, 1)
    assert large > 1.5 and large > small + 0.5


def test_expected_acceptance_length():
    assert expected_acceptance_length(0.0, 5) == pytest.approx(1.0)
    assert expected_acceptance_length(1.0, 5) == pytest.approx(6.0)
    assert expected_acceptance_length(0.5, 2) == pytest.approx(1.75)


def test_best_k_shrinks_with_batch(cm):
    cost = lambda k: DraftCost(passes=k, tokens_per_pass=1, rel_params=0.04)  # noqa: E731
    best = []
    for b in (1, 16, 64, 256):
        speedups = {k: cm.speedup(b, expected_acceptance_length(0.7, k), cost(k), k + 1) for k in range(1, 9)}
        best.append(max(speedups, key=speedups.get))
    assert best == sorted(best, reverse=True)
    assert best[0] > best[-1]


def test_free_drafter_with_zero_acceptance_is_not_a_speedup(cm):
    sp = cm.speedup(1, 1.0, DraftCost(0, 0, 0.0, cpu_ms=0.05), 1)
    assert sp < 1.0


def test_block_drafter_cost_independent_of_k(cm):
    one = cm.draft_step_ms(1, DraftCost(passes=1, tokens_per_pass=3, rel_params=0.1))
    many = cm.draft_step_ms(1, DraftCost(passes=1, tokens_per_pass=9, rel_params=0.1))
    autoreg = cm.draft_step_ms(1, DraftCost(passes=8, tokens_per_pass=1, rel_params=0.1))
    assert many == pytest.approx(one, rel=0.01)
    assert autoreg > 5 * many
