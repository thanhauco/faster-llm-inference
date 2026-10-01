"""EAGLE-style drafting: a light head that extrapolates the *target's* hidden states.

Resource #2 (EAGLE 3.x on the vLLM blog): the draft head is built on the
target's hidden states rather than on raw tokens. Each step it takes the
current feature vector plus the embedding of the token just chosen, predicts
the next feature vector, and decodes it with the target's own (frozen) LM
head. EAGLE-3 fuses features from several target layers (low/mid/high)
instead of only the top layer, and trains with "training-time test" so the
head sees its own predictions as inputs, which is what makes acceptance hold
up deeper into the draft and on long contexts.

This module implements both flavours on :class:`~specdec.models.ToyLM`:

* ``layers="top"``  – EAGLE-1-like, top-layer features only
* ``layers="all"``  – EAGLE-3-like, fused multi-layer features (default)

and ``ttt_rounds`` emulates training-time test by refitting on inputs that
come from the head's own multi-step rollouts.

vLLM equivalent::

    {"method": "eagle3", "model": "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3",
     "num_speculative_tokens": 3}
"""

from __future__ import annotations

import numpy as np

from specdec.drafters.base import Draft, DraftCost, Drafter
from specdec.drafters.traces import Trace
from specdec.fit import add_bias, ridge
from specdec.models import Session, ToyLM
from specdec.sampling import apply_temperature, sample, softmax


class EagleDrafter(Drafter):
    def __init__(self, target: ToyLM, layers: str = "all", rel_params: float = 0.04, l2: float = 1e-2):
        if layers not in ("all", "top"):
            raise ValueError("layers must be 'all' (EAGLE-3 style) or 'top' (EAGLE-1 style)")
        self.target = target
        self.layers = layers
        self.rel_params = rel_params
        self.l2 = l2
        self.A: np.ndarray | None = None
        self.name = "eagle3" if layers == "all" else "eagle1"

    # ------------------------------------------------------------- features
    def _select(self, feats: np.ndarray) -> np.ndarray:
        H = self.target.hidden_size
        return feats[..., H:] if self.layers == "top" else feats

    def _inputs(self, f: np.ndarray, tokens: np.ndarray) -> np.ndarray:
        return add_bias(np.concatenate([f, self.target.E[tokens]], axis=1))

    def predict_features(self, f: np.ndarray, tokens: np.ndarray) -> np.ndarray:
        """Batched ``f_{t} = head(f_{t-1}, x_t)``."""
        return np.tanh(self._inputs(f, tokens) @ self.A)

    def _top(self, f: np.ndarray) -> np.ndarray:
        return f[..., -self.target.hidden_size:]

    # ------------------------------------------------------------- training
    def fit(self, traces: list[Trace], ttt_rounds: int = 1, ttt_depth: int = 3) -> "EagleDrafter":
        """Feature regression on on-policy target traces, plus training-time-test refits."""
        prev = np.concatenate([self._select(tr.feats[:-1]) for tr in traces])
        toks = np.concatenate([tr.tokens[1:] for tr in traces])
        nxt = np.concatenate([self._select(tr.feats[1:]) for tr in traces])
        X, Y = self._inputs(prev, toks), np.arctanh(np.clip(nxt, -0.999, 0.999))
        self.A = ridge(X, Y, self.l2)
        for _ in range(ttt_rounds):
            xs, ys = [X], [Y]
            for tr in traces:
                f_all = self._select(tr.feats)
                T = len(tr.tokens)
                # Roll the head forward from every anchor and record (own prediction -> true next).
                f_hat = f_all[:-1]
                for d in range(1, ttt_depth):
                    starts = np.arange(0, T - 1 - d)
                    f_hat = self.predict_features(f_hat[: len(starts)], tr.tokens[starts + d])
                    xs.append(self._inputs(f_hat, tr.tokens[starts + d + 1]))
                    ys.append(np.arctanh(np.clip(f_all[starts + d + 1], -0.999, 0.999)))
            self.A = ridge(np.concatenate(xs), np.concatenate(ys), self.l2)
        return self

    # ------------------------------------------------------------- drafting
    def propose(self, session: Session, k: int, temperature: float, rng: np.random.Generator) -> Draft:
        if self.A is None:
            raise RuntimeError("EagleDrafter.fit() must be called before drafting")
        dim = self.target.hidden_size * (1 if self.layers == "top" else 2)
        f = self._select(session.features(-2)) if len(session) >= 2 else np.zeros(dim)
        tok = session.tokens[-1]
        ctx = list(session.tokens)
        tokens, probs = [], []
        for _ in range(k):
            f = self.predict_features(f[None], np.array([tok]))[0]
            base = softmax(self.target.lm_head(self._top(f)))
            q = apply_temperature(self.target.mix(base, ctx), temperature)
            tok = int(np.argmax(q)) if temperature <= 0 else sample(q, rng)
            tokens.append(tok)
            probs.append(q)
            ctx.append(tok)
        return Draft(tokens=tokens, probs=np.array(probs) if probs else None)

    def cost(self, k: int) -> DraftCost:
        return DraftCost(passes=k, tokens_per_pass=1, rel_params=self.rel_params)
