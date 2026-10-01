import numpy as np

from specdec.drafters import NgramDrafter
from specdec.ngram import find_suffix_match, propose_continuation


def test_longest_match_wins():
    toks = [5, 1, 2, 3, 9, 7, 2, 3, 4, 1, 2, 3]
    # suffix [1, 2, 3] matches at index 1..3 -> next token is 9
    assert find_suffix_match(toks, 1, 4) == (3, 4)
    assert propose_continuation(toks, 3, 1, 4) == [9, 7, 2]


def test_most_recent_occurrence_wins_for_equal_length():
    toks = [1, 2, 8, 1, 2, 9, 1, 2]
    n, end = find_suffix_match(toks, 2, 2)
    assert n == 2 and toks[end] == 9


def test_min_n_is_respected():
    toks = [4, 7, 1, 2, 7]
    assert find_suffix_match(toks, 2, 4) == (0, -1)
    assert find_suffix_match(toks, 1, 4)[0] == 1


def test_no_match_returns_empty_proposal():
    assert propose_continuation([1, 2, 3, 4], 5) == []
    assert propose_continuation([1], 5) == []


def test_proposal_truncated_at_context_end():
    toks = [1, 2, 3, 1, 2]
    assert propose_continuation(toks, 5, 1, 2) == [3, 1, 2]


def test_ngram_drafter_matches_post_config(target):
    d = NgramDrafter.vllm_default()
    assert (d.prompt_lookup_min, d.prompt_lookup_max) == (1, 4)
    sess = target.session([3, 4, 5, 6, 7, 3, 4, 5])
    draft = d.propose(sess, 5, 0.0, np.random.default_rng(0))
    assert draft.tokens == [6, 7, 3, 4, 5] and draft.probs is None
    assert d.cost(5).passes == 0
