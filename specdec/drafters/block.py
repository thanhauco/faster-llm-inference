"""Block drafting with sequential correction: DFlash and DSpark style drafters.

Resource #5 (DSpark, used for Kimi K3 in the post) and the DFlash/DSpark
algorithms in ``vllm-project/speculators``:

* **DFlash** drafts a *whole block* of ``block_size`` tokens in one forward pass,
  anchored on the last verified token and the target's hidden states. One pass
  instead of ``k`` makes drafting cheap, but positions inside the block cannot
  see each other, so acceptance decays toward the end of the block.
* **DSpark** keeps the block-parallel backbone and adds
  1. a low-rank **Markov head**: a logit bias ``A[prev] @ B`` that conditions
     each position on the token drafted just before it (the "sequential
     correction"), and
  2. a **confidence head** that predicts each position's acceptance
     probability, so the server can verify only as far as the block is worth
     verifying (vLLM's *adaptive verification*).

Here the backbone is a per-position softmax regression on
``[target features at t-1, embedding(anchor token), 1]``, the Markov head is a
residual softmax regression on the previous token factored to ``markov_rank``
via SVD, and the confidence head is a logistic regression on per-position
draft statistics. Like the real thing, the block also gets the context copy
signal (parallel span copy) so it can exploit repeated text.

vLLM equivalents::

    {"method": "dflash", "model": "RedHatAI/Qwen3-8B-speculator.dflash", "num_speculative_tokens": 8}
    {"method": "dspark", "model": "<dspark speculator>", "num_speculative_tokens": 8}
"""

from __future__ import annotations

import numpy as np

from specdec.drafters.base import Draft, DraftCost, Drafter
from specdec.drafters.traces import Trace
from specdec.fit import add_bias, fit_logistic_regression, fit_softmax_regression, low_rank
from specdec.models import Session, ToyLM
from specdec.ngram import find_suffix_match
from specdec.sampling import apply_temperature, entropy, sample, softmax


class BlockDrafter(Drafter):
    def __init__(
        self,
        target: ToyLM,
        block_size: int = 8,
        markov_rank: int = 32,
        confidence_head: bool = True,
        rel_params: float = 0.10,
    ):
        self.target = target
        self.block_size = block_size
        self.markov_rank = markov_rank
        self.use_confidence = confidence_head
        self.rel_params = rel_params
        self.heads: np.ndarray | None = None  # (block_size, D, V)
        self.markov_A: np.ndarray | None = None  # (V, r)
        self.markov_B: np.ndarray | None = None  # (r, V)
        self.conf_w: np.ndarray | None = None
        self.name = "dspark" if markov_rank > 0 else "dflash"

    @classmethod
    def dflash(cls, target: ToyLM, block_size: int = 8, **kw) -> "BlockDrafter":
        return cls(target, block_size=block_size, markov_rank=0, confidence_head=False, **kw)

    @classmethod
    def dspark(cls, target: ToyLM, block_size: int = 8, markov_rank: int = 32, **kw) -> "BlockDrafter":
        return cls(target, block_size=block_size, markov_rank=markov_rank, confidence_head=True, **kw)

    # ----------------------------------------------------------- components
    def _backbone_inputs(self, feats_prev: np.ndarray, anchor: np.ndarray) -> np.ndarray:
        return add_bias(np.concatenate([feats_prev, self.target.E[anchor]], axis=1))

    def _markov_bias(self, prev: np.ndarray) -> np.ndarray:
        if self.markov_A is None:
            return np.zeros((len(prev), self.target.vocab_size))
        return self.markov_A[prev] @ self.markov_B

    def _span_copy(self, tokens, position: int) -> tuple[float, int]:
        """Parallel copy candidate for draft offset ``position`` (1-based) from context alone."""
        max_n = self.target.config.copy_max_n
        n, end = find_suffix_match(tokens, 1, max_n)
        src = end + position - 1
        if n == 0 or src >= len(tokens):
            return 0.0, -1
        return self.target.config.copy_strength * n / max_n, int(tokens[src])

    def _conf_features(self, q: np.ndarray, pos: np.ndarray) -> np.ndarray:
        onehot = np.eye(self.block_size)[pos]
        return add_bias(np.column_stack([q.max(axis=-1), entropy(q), onehot]))

    # -------------------------------------------------------------- training
    def fit(self, traces: list[Trace], epochs: int = 6, max_rows: int = 20000, seed: int = 0) -> "BlockDrafter":
        rng = np.random.default_rng(seed)
        bs, V = self.block_size, self.target.vocab_size
        Xs, Ys, Prevs = [], [], []
        for tr in traces:
            T = len(tr.tokens)
            t = np.arange(1, T - bs)  # anchor positions (the pending token)
            Xs.append(self._backbone_inputs(tr.feats[t - 1], tr.tokens[t]))
            Ys.append(np.stack([tr.tokens[t + i] for i in range(1, bs + 1)], axis=1))
            Prevs.append(np.stack([tr.tokens[t + i - 1] for i in range(1, bs + 1)], axis=1))
        X, Y, P = np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Prevs)
        if len(X) > max_rows:
            keep = rng.choice(len(X), size=max_rows, replace=False)
            X, Y, P = X[keep], Y[keep], P[keep]

        # 1) Block-parallel backbone: one head per block position.
        self.heads = np.stack(
            [fit_softmax_regression(X, Y[:, i], V, epochs=epochs, seed=seed + i) for i in range(bs)]
        )

        # 2) DSpark Markov head: residual bias from the previous in-block token, then low-rank.
        if self.markov_rank > 0:
            rows = rng.choice(len(X) * bs, size=min(max_rows, len(X) * bs), replace=False)
            r_idx, r_pos = rows // bs, rows % bs
            offset = np.empty((len(rows), V))
            for i in range(bs):
                sel = r_pos == i
                offset[sel] = X[r_idx[sel]] @ self.heads[i]
            onehot_prev = np.eye(V)[P[r_idx, r_pos]]
            M = fit_softmax_regression(onehot_prev, Y[r_idx, r_pos], V, offset=offset, epochs=epochs, seed=seed)
            self.markov_A, self.markov_B = low_rank(M, self.markov_rank)

        # 3) Confidence head: P(position i accepted | earlier positions accepted).
        if self.use_confidence:
            feats, labels = [], []
            for tr in traces:
                T = len(tr.tokens)
                t = np.arange(1, T - bs)
                Xi = self._backbone_inputs(tr.feats[t - 1], tr.tokens[t])
                for i in range(bs):
                    prev = tr.tokens[t + i]  # teacher forcing: earlier positions were correct
                    q = softmax(Xi @ self.heads[i] + self._markov_bias(prev))
                    # Same parallel span-copy signal the drafter mixes in at inference time.
                    src = tr.copy_end[t] + i
                    has_copy = (tr.copy_end[t] >= 0) & (src <= t)
                    lam = np.where(has_copy, tr.copy_lam[t], 0.0)[:, None]
                    q = (1 - lam) * q
                    rows = np.nonzero(has_copy)[0]
                    q[rows, tr.tokens[src[rows]]] += lam[rows, 0]
                    d = q.argmax(axis=1)
                    # Soft label: target probability of the draft token = its expected acceptance.
                    labels.append(tr.probs[t + i, d])
                    feats.append(self._conf_features(q, np.full(len(t), i)))
            self.conf_w = fit_logistic_regression(np.concatenate(feats), np.concatenate(labels), seed=seed)
        return self

    # -------------------------------------------------------------- drafting
    def propose(self, session: Session, k: int, temperature: float, rng: np.random.Generator) -> Draft:
        if self.heads is None:
            raise RuntimeError("BlockDrafter.fit() must be called before drafting")
        k = min(k, self.block_size)
        f_prev = session.features(-2) if len(session) >= 2 else np.zeros(2 * self.target.hidden_size)
        anchor = session.tokens[-1]
        x = self._backbone_inputs(f_prev[None], np.array([anchor]))[0]
        backbone = np.einsum("d,kdv->kv", x, self.heads[:k])  # one parallel pass for the whole block
        tokens, probs, confs = [], [], []
        prev = anchor
        for i in range(k):
            logits = backbone[i] + self._markov_bias(np.array([prev]))[0]  # sequential correction
            base = softmax(logits)
            lam, tok_copy = self._span_copy(session.tokens, i + 1)
            if lam:
                base = (1 - lam) * base
                base[tok_copy] += lam
            if self.use_confidence:
                z = self._conf_features(base[None], np.array([i])) @ self.conf_w
                confs.append(float(1.0 / (1.0 + np.exp(-z[0]))))
            q = apply_temperature(base, temperature)
            tok = int(np.argmax(q)) if temperature <= 0 else sample(q, rng)
            tokens.append(tok)
            probs.append(q)
            prev = tok
        return Draft(
            tokens=tokens,
            probs=np.array(probs) if probs else None,
            confidence=np.array(confs) if confs else None,
        )

    def cost(self, k: int) -> DraftCost:
        return DraftCost(passes=1, tokens_per_pass=min(k, self.block_size) + 1, rel_params=self.rel_params)
