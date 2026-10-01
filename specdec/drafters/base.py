"""Drafter interface shared by every speculation method."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from specdec.models import Session


@dataclass
class Draft:
    """Proposed continuation for one verification step.

    Attributes:
        tokens: proposed tokens ``d_1..d_n`` (``n`` may be < k or 0).
        probs: ``(n, V)`` tempered distributions the tokens were sampled from,
            or ``None`` for deterministic drafters (treated as one-hot).
        confidence: optional ``(n,)`` predicted probability that each position
            is accepted given the previous ones were (DSpark confidence head).
    """

    tokens: list[int]
    probs: np.ndarray | None = None
    confidence: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.tokens)

    def truncate(self, n: int) -> "Draft":
        return Draft(
            self.tokens[:n],
            None if self.probs is None else self.probs[:n],
            None if self.confidence is None else self.confidence[:n],
        )


@dataclass(frozen=True)
class DraftCost:
    """Drafting work per verification step, consumed by :mod:`specdec.cost_model`.

    Attributes:
        passes: sequential forward passes of the draft network per step.
        tokens_per_pass: tokens processed in each pass (per sequence).
        rel_params: draft network size as a fraction of the target's parameters.
        cpu_ms: fixed host-side cost per step (e.g. n-gram matching).
    """

    passes: int
    tokens_per_pass: int
    rel_params: float
    cpu_ms: float = 0.0


class Drafter(ABC):
    """Proposes draft tokens; the target verifies them losslessly.

    ``session`` is the *target's* session (read-only). Its last token is the
    pending token emitted by the previous verification step; model-based
    drafters that reuse target hidden states may only read features up to the
    position before it (``session.states[-2]``) because the target has not yet
    run on the pending token in a real system.
    """

    name: str = "drafter"

    def reset(self) -> None:  # called at the start of every request
        pass

    @abstractmethod
    def propose(self, session: Session, k: int, temperature: float, rng: np.random.Generator) -> Draft:
        ...

    @abstractmethod
    def cost(self, k: int) -> DraftCost:
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
