"""Suffix n-gram matching, the primitive behind prompt-lookup decoding.

The same lookup powers two things in this repo:

* :class:`specdec.drafters.NgramDrafter` – proposes the tokens that followed
  an earlier occurrence of the current suffix (vLLM ``method="ngram"``).
* :class:`specdec.models.ToyLM` – the toy target's "induction head" copy
  mechanism, which is what makes copy-heavy workloads (code edits, RAG)
  behave like they do on real LLMs.
"""

from __future__ import annotations

from typing import Sequence


def find_suffix_match(
    tokens: Sequence[int],
    min_n: int = 1,
    max_n: int = 4,
) -> tuple[int, int]:
    """Find the longest suffix n-gram (``min_n <= n <= max_n``) that occurred earlier.

    Returns ``(n, end)`` where ``tokens[end - n:end]`` equals the last ``n``
    tokens and ``end`` is the index of the token that *followed* that earlier
    occurrence (so ``tokens[end]`` is the copy candidate). The most recent
    earlier occurrence wins. Returns ``(0, -1)`` when nothing matches.
    """
    length = len(tokens)
    if length < 2 or max_n < 1:
        return 0, -1
    max_n = min(max_n, length - 1)
    for n in range(max_n, min_n - 1, -1):
        suffix = list(tokens[length - n:])
        last = suffix[-1]
        # Scan backwards so the most recent occurrence is found first. ``end`` must
        # be < length so there is at least one token to copy.
        for end in range(length - 1, n - 1, -1):
            if tokens[end - 1] != last:
                continue
            if list(tokens[end - n:end]) == suffix:
                return n, end
    return 0, -1


def propose_continuation(
    tokens: Sequence[int],
    k: int,
    min_n: int = 1,
    max_n: int = 4,
) -> list[int]:
    """Prompt-lookup proposal: up to ``k`` tokens following the best suffix match."""
    n, end = find_suffix_match(tokens, min_n=min_n, max_n=max_n)
    if n == 0:
        return []
    return list(tokens[end:end + k])
