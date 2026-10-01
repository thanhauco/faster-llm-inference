"""The speculative decoding loop: draft -> verify in one target pass -> accept.

::

    prefill(prompt) -> first token (pending)
    repeat:
        draft  = drafter.propose(context, k)               # cheap guesses
        p      = target(context + draft)                    # ONE forward over k+1 positions
        n, tok = verify_draft(draft, q, p)                  # lossless accept/reject
        context += draft[:n] + [tok]                        # always >= 1 new token

Every step emits ``n + 1`` tokens for the price of one target forward pass
(plus drafting). The mean of ``n + 1`` is the *acceptance length* (tau),
the single number that decides whether speculation pays off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from specdec.drafters.base import Draft, Drafter
from specdec.models import ToyLM
from specdec.sampling import apply_temperature, sample
from specdec.verify import verify_draft


@dataclass
class SpecStats:
    """Acceptance statistics in the same shape vLLM reports them.

    ``acceptance_histogram[j]`` counts steps that accepted exactly ``j`` drafts
    (bonus token excluded), matching vLLM's per-request spec-decode metrics.
    """

    num_spec_tokens: int
    num_steps: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_verified_tokens: int = 0  # target positions scored (drafts + 1 per step)
    acceptance_histogram: list[int] = field(default_factory=list)
    accepted_per_position: list[int] = field(default_factory=list)
    drafted_per_position: list[int] = field(default_factory=list)
    step_log: list[tuple[int, int]] = field(default_factory=list)  # (drafted, accepted) per step

    def __post_init__(self):
        k = self.num_spec_tokens
        self.acceptance_histogram = self.acceptance_histogram or [0] * (k + 1)
        self.accepted_per_position = self.accepted_per_position or [0] * k
        self.drafted_per_position = self.drafted_per_position or [0] * k

    def record(self, num_drafted: int, num_accepted: int) -> None:
        self.num_steps += 1
        self.num_draft_tokens += num_drafted
        self.num_accepted_tokens += num_accepted
        self.num_verified_tokens += num_drafted + 1
        self.acceptance_histogram[num_accepted] += 1
        self.step_log.append((num_drafted, num_accepted))
        for i in range(num_drafted):
            self.drafted_per_position[i] += 1
            if i < num_accepted:
                self.accepted_per_position[i] += 1

    def merge(self, other: "SpecStats") -> "SpecStats":
        if other.num_spec_tokens != self.num_spec_tokens:
            raise ValueError("cannot merge stats with different k")
        out = SpecStats(self.num_spec_tokens)
        for name in ("num_steps", "num_draft_tokens", "num_accepted_tokens", "num_verified_tokens"):
            setattr(out, name, getattr(self, name) + getattr(other, name))
        for name in ("acceptance_histogram", "accepted_per_position", "drafted_per_position"):
            setattr(out, name, [a + b for a, b in zip(getattr(self, name), getattr(other, name))])
        out.step_log = self.step_log + other.step_log
        return out

    def truncated(self, k: int) -> "SpecStats":
        """Estimate the stats a max draft length of ``k`` would give, from steps drafted at a larger k.

        Verification of a draft prefix only depends on that prefix, so a step
        that drafted ``d`` and accepted ``a`` would have drafted ``min(d, k)``
        and accepted ``min(a, k)``. Lets one run cover a whole sweep over k.
        """
        if k > self.num_spec_tokens:
            raise ValueError("can only truncate to a smaller k")
        out = SpecStats(k)
        for drafted, accepted in self.step_log:
            out.record(min(drafted, k), min(accepted, k))
        return out

    @property
    def mean_acceptance_length(self) -> float:
        """Tokens emitted per verification step, bonus included: ``1 + accepted / steps``."""
        return 1.0 + self.num_accepted_tokens / self.num_steps if self.num_steps else 1.0

    @property
    def draft_acceptance_rate(self) -> float:
        return self.num_accepted_tokens / self.num_draft_tokens if self.num_draft_tokens else 0.0

    @property
    def mean_draft_length(self) -> float:
        return self.num_draft_tokens / self.num_steps if self.num_steps else 0.0

    @property
    def mean_verified_tokens(self) -> float:
        return self.num_verified_tokens / self.num_steps if self.num_steps else 1.0

    def per_position_acceptance(self) -> list[float]:
        """Conditional acceptance at each draft position (given it was drafted)."""
        return [a / d if d else 0.0 for a, d in zip(self.accepted_per_position, self.drafted_per_position)]

    def to_dict(self) -> dict:
        return {
            "num_spec_tokens": self.num_spec_tokens,
            "num_steps": self.num_steps,
            "num_draft_tokens": self.num_draft_tokens,
            "num_accepted_tokens": self.num_accepted_tokens,
            "mean_acceptance_length": round(self.mean_acceptance_length, 4),
            "draft_acceptance_rate": round(self.draft_acceptance_rate, 4),
            "mean_verified_tokens": round(self.mean_verified_tokens, 4),
            "acceptance_histogram": list(self.acceptance_histogram),
            "per_position_acceptance": [round(x, 4) for x in self.per_position_acceptance()],
        }


@dataclass
class GenerationResult:
    tokens: list[int]  # generated tokens only (prompt excluded)
    stats: SpecStats | None = None


def _adaptive_length(draft: Draft, threshold: float | None) -> int:
    """Adaptive verification: keep drafts while the cumulative confidence stays >= threshold."""
    if threshold is None or draft.confidence is None or len(draft) == 0:
        return len(draft)
    survival = np.cumprod(draft.confidence)
    return max(1, int(np.sum(survival >= threshold)))


def autoregressive_generate(
    target: ToyLM,
    prompt: list[int],
    max_new_tokens: int,
    temperature: float = 0.0,
    rng: np.random.Generator | None = None,
) -> GenerationResult:
    """Baseline: one target forward pass per token."""
    rng = rng or np.random.default_rng(0)
    sess = target.session(prompt)
    out = []
    for _ in range(max_new_tokens):
        p = apply_temperature(sess.next_probs(), temperature)
        tok = int(np.argmax(p)) if temperature <= 0 else sample(p, rng)
        out.append(tok)
        sess.extend([tok])
    return GenerationResult(out)


@dataclass
class StepEvent:
    """One engine step: the prefill (``drafted == 0``, ``prefill=True``) or one draft/verify round."""

    tokens: list[int]
    drafted: int = 0
    accepted: int = 0
    prefill: bool = False


def speculative_stream(
    target: ToyLM,
    drafter: Drafter,
    prompt: list[int],
    max_new_tokens: int,
    k: int = 5,
    temperature: float = 0.0,
    rng: np.random.Generator | None = None,
    confidence_threshold: float | None = None,
) -> Iterator[StepEvent]:
    """Speculative decoding as a stream of steps (what a server streams to clients).

    Every step emits ``accepted + 1`` tokens; the last step may be clipped to
    ``max_new_tokens``.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    rng = rng or np.random.default_rng(0)
    greedy = temperature <= 0
    drafter.reset()
    sess = target.session(prompt)

    # Prefill produces the first token; it becomes the "pending" token.
    p0 = apply_temperature(sess.next_probs(), temperature)
    first = int(np.argmax(p0)) if greedy else sample(p0, rng)
    sess.extend([first])
    produced = 1
    yield StepEvent([first], prefill=True)

    while produced < max_new_tokens:
        draft = drafter.propose(sess, k, temperature, rng).truncate(k)
        draft = draft.truncate(_adaptive_length(draft, confidence_threshold))
        base = len(sess)
        # One target forward over [pending, d_1..d_n] yields n + 1 distributions.
        sess.extend(draft.tokens)
        target_probs = np.array(
            [apply_temperature(sess.probs_at(base - 1 + i), temperature) for i in range(len(draft) + 1)]
        )
        result = verify_draft(draft.tokens, draft.probs, target_probs, rng, greedy=greedy)
        sess.truncate(base + result.num_accepted)  # roll back rejected drafts ("KV cache")
        sess.extend([result.next_token])
        new = (draft.tokens[: result.num_accepted] + [result.next_token])[: max_new_tokens - produced]
        produced += len(new)
        yield StepEvent(new, drafted=len(draft), accepted=result.num_accepted)


def speculative_generate(
    target: ToyLM,
    drafter: Drafter,
    prompt: list[int],
    max_new_tokens: int,
    k: int = 5,
    temperature: float = 0.0,
    rng: np.random.Generator | None = None,
    confidence_threshold: float | None = None,
) -> GenerationResult:
    """Generate with speculative decoding. Output distribution == target's (lossless).

    Args:
        k: ``num_speculative_tokens``, the maximum draft length per step.
        confidence_threshold: enables adaptive verification for drafters that
            report per-position confidence (DSpark); ``None`` verifies all drafts.
    """
    stats = SpecStats(k)
    out: list[int] = []
    for ev in speculative_stream(target, drafter, prompt, max_new_tokens, k, temperature, rng,
                                 confidence_threshold):
        if not ev.prefill:
            stats.record(ev.drafted, ev.accepted)
        out.extend(ev.tokens)
    return GenerationResult(out, stats)
