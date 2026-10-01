"""Lossless verification of draft tokens (speculative sampling).

Given ``n`` draft tokens ``d_1..d_n`` sampled from draft distributions
``q_1..q_n`` and the target's distributions ``p_1..p_{n+1}`` (computed in a
*single* target forward pass over the drafts), accept ``d_i`` with probability
``min(1, p_i(d_i) / q_i(d_i))``. On the first rejection, emit a token sampled
from the residual ``norm(max(p_i - q_i, 0))`` and stop; if every draft is
accepted, emit a "bonus" token from ``p_{n+1}``.

This procedure (Leviathan et al. 2023; Chen et al. 2023) makes the output
distribution *identical* to sampling from the target alone, which is why every
method in the post (n-gram, EAGLE, DFlash/DSpark, MTP) is lossless.

With ``temperature == 0`` it reduces to: accept while the draft equals the
target's argmax.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from specdec.sampling import sample


@dataclass
class VerifyResult:
    num_accepted: int
    next_token: int  # correction token on rejection, bonus token otherwise

    @property
    def emitted(self) -> int:
        """Tokens produced by this verification step (accepted + 1)."""
        return self.num_accepted + 1


def verify_draft(
    draft_tokens: list[int],
    draft_probs: np.ndarray | None,
    target_probs: np.ndarray,
    rng: np.random.Generator,
    greedy: bool = False,
) -> VerifyResult:
    """Verify drafts against already-tempered target distributions.

    Args:
        draft_tokens: proposed tokens ``d_1..d_n``.
        draft_probs: ``(n, V)`` tempered draft distributions the drafts were
            sampled from, or ``None`` for deterministic drafters (n-gram), in
            which case ``q_i`` is the one-hot on ``d_i``.
        target_probs: ``(n + 1, V)`` tempered target distributions.
        greedy: exact argmax matching (temperature 0).
    """
    n = len(draft_tokens)
    if target_probs.shape[0] != n + 1:
        raise ValueError(f"need {n + 1} target distributions, got {target_probs.shape[0]}")

    if greedy:
        for i, tok in enumerate(draft_tokens):
            best = int(np.argmax(target_probs[i]))
            if tok != best:
                return VerifyResult(i, best)
        return VerifyResult(n, int(np.argmax(target_probs[n])))

    for i, tok in enumerate(draft_tokens):
        p = target_probs[i]
        if draft_probs is None:
            q_tok = 1.0
        else:
            q_tok = float(draft_probs[i, tok])
        p_tok = float(p[tok])
        if q_tok > 0.0 and rng.random() < min(1.0, p_tok / q_tok):
            continue
        # Rejected: sample from the residual distribution max(p - q, 0).
        if draft_probs is None:
            residual = p.copy()
            residual[tok] = 0.0
        else:
            residual = np.maximum(p - draft_probs[i], 0.0)
        if residual.sum() <= 0.0:  # p == q numerically; any sample from p is exact
            residual = p
        return VerifyResult(i, sample(residual, rng))
    return VerifyResult(n, sample(target_probs[n], rng))
