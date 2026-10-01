"""N-gram / prompt-lookup drafting (vLLM ``method="ngram"``).

Resource #1 in the post: *start with n-gram, the simplest method*. Drafts come
from matches in your prompt (and the text generated so far): find the longest
suffix of the context, between ``prompt_lookup_min`` and ``prompt_lookup_max``
tokens long, that occurred earlier, and propose the ``num_speculative_tokens``
tokens that followed it. No draft model, no training, no GPU memory.

The post's suggested starting config, mirrored by :meth:`NgramDrafter.vllm_default`::

    {"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}
"""

from __future__ import annotations

import numpy as np

from specdec.drafters.base import Draft, DraftCost, Drafter
from specdec.models import Session
from specdec.ngram import propose_continuation


class NgramDrafter(Drafter):
    def __init__(self, prompt_lookup_min: int = 1, prompt_lookup_max: int = 4, cpu_ms: float = 0.05):
        if prompt_lookup_min < 1 or prompt_lookup_max < prompt_lookup_min:
            raise ValueError("need 1 <= prompt_lookup_min <= prompt_lookup_max")
        self.prompt_lookup_min = prompt_lookup_min
        self.prompt_lookup_max = prompt_lookup_max
        self.cpu_ms = cpu_ms
        self.name = f"ngram[{prompt_lookup_min}-{prompt_lookup_max}]"

    @classmethod
    def vllm_default(cls) -> "NgramDrafter":
        """The post's config: lookup of 4 (use with k = 5 speculative tokens)."""
        return cls(prompt_lookup_min=1, prompt_lookup_max=4)

    def propose(self, session: Session, k: int, temperature: float, rng: np.random.Generator) -> Draft:
        tokens = propose_continuation(session.tokens, k, self.prompt_lookup_min, self.prompt_lookup_max)
        # Deterministic proposal: q is one-hot, so probs=None (see verify_draft).
        return Draft(tokens=tokens, probs=None)

    def cost(self, k: int) -> DraftCost:
        return DraftCost(passes=0, tokens_per_pass=0, rel_params=0.0, cpu_ms=self.cpu_ms)
