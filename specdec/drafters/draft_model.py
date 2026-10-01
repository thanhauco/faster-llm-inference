"""Classic draft-model speculation: a smaller LM of the same family guesses first.

vLLM equivalent::

    {"method": "draft_model", "model": "Qwen/Qwen3-0.6B", "num_speculative_tokens": 5}
"""

from __future__ import annotations

import numpy as np

from specdec.drafters.base import Draft, DraftCost, Drafter
from specdec.models import Session, ToyLM
from specdec.sampling import apply_temperature, sample


class DraftModelDrafter(Drafter):
    def __init__(self, draft_model: ToyLM, rel_params: float = 0.08, name: str = "draft_model"):
        self.model = draft_model
        self.rel_params = rel_params
        self.name = name
        self._session: Session | None = None

    def reset(self) -> None:
        self._session = None

    def propose(self, session: Session, k: int, temperature: float, rng: np.random.Generator) -> Draft:
        # The draft model keeps its own cache, synced to the accepted context.
        if self._session is None:
            self._session = self.model.session(session.tokens)
        else:
            self._session.sync(session.tokens)
        tokens, probs = [], []
        for _ in range(k):
            q = apply_temperature(self._session.next_probs(), temperature)
            tok = int(np.argmax(q)) if temperature <= 0 else sample(q, rng)
            tokens.append(tok)
            probs.append(q)
            self._session.extend([tok])
        # Leave draft tokens cached; the next sync() rolls back whatever was rejected.
        return Draft(tokens=tokens, probs=np.array(probs) if probs else None)

    def cost(self, k: int) -> DraftCost:
        return DraftCost(passes=k, tokens_per_pass=1, rel_params=self.rel_params)
