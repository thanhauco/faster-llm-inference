"""A small, deterministic target language model that runs anywhere numpy does.

Real speculative decoding needs a big target model on a GPU. To make every
algorithm in this repo runnable and testable on a laptop, we use ``ToyLM``:

* a 2-layer tanh RNN (so there are *multi-layer hidden states* for
  EAGLE-3-style feature fusion to use), plus
* an induction-style copy mechanism: when the current suffix occurred earlier
  in the context, the model puts extra mass on the token that followed it.
  The longer the match, the stronger the copy. This is what makes prompt
  lookup (n-gram) drafting work on code edits and RAG, exactly as on real LLMs.

The model is a proper conditional distribution ``p(x_{t+1} | x_{<=t})``, so the
lossless guarantees of speculative sampling can be tested exactly.
``Session`` plays the role of the KV cache: it stores per-position state and
supports cheap rollback when draft tokens are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from specdec.ngram import find_suffix_match
from specdec.sampling import softmax


@dataclass
class ToyLMConfig:
    vocab_size: int = 128
    hidden_size: int = 64
    copy_strength: float = 0.85
    copy_max_n: int = 4
    recurrent_scale: float = 0.95
    head_scale: float = 2.2
    seed: int = 0


class ToyLM:
    """Two-layer recurrent LM with a suffix-match copy head.

    Hidden state after consuming token ``x_t``::

        h1_t = tanh(W1 h1_{t-1} + E[x_t])
        h2_t = tanh(W2 h2_{t-1} + V h1_t)
        p(x_{t+1}) = (1 - lam) * softmax(U h2_t + b) + lam * onehot(copy_t)

    where ``copy_t``/``lam`` come from the longest earlier match of the context
    suffix (``lam = copy_strength * n / copy_max_n``).
    """

    def __init__(self, config: ToyLMConfig | None = None, **overrides):
        cfg = config or ToyLMConfig()
        for key, value in overrides.items():
            setattr(cfg, key, value)
        self.config = cfg
        rng = np.random.default_rng(cfg.seed)
        V, H = cfg.vocab_size, cfg.hidden_size
        self.E = rng.normal(0.0, 1.0, size=(V, H))
        self.W1 = rng.normal(0.0, cfg.recurrent_scale / np.sqrt(H), size=(H, H))
        self.W2 = rng.normal(0.0, cfg.recurrent_scale / np.sqrt(H), size=(H, H))
        self.Vm = rng.normal(0.0, 1.5 / np.sqrt(H), size=(H, H))
        self.U = rng.normal(0.0, cfg.head_scale, size=(H, V))
        self.b = rng.normal(0.0, 0.5, size=(V,))

    # ------------------------------------------------------------------ basics
    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    def initial_state(self) -> tuple[np.ndarray, np.ndarray]:
        H = self.hidden_size
        return np.zeros(H), np.zeros(H)

    def step(self, state: tuple[np.ndarray, np.ndarray], token: int) -> tuple[np.ndarray, np.ndarray]:
        h1, h2 = state
        h1 = np.tanh(self.W1 @ h1 + self.E[token])
        h2 = np.tanh(self.W2 @ h2 + self.Vm @ h1)
        return h1, h2

    def lm_head(self, h2: np.ndarray) -> np.ndarray:
        """Base (non-copy) logits. Shared with EAGLE-style heads, like a real LM head."""
        return h2 @ self.U + self.b

    def copy_component(self, tokens: Sequence[int]) -> tuple[float, int]:
        n, end = find_suffix_match(tokens, min_n=1, max_n=self.config.copy_max_n)
        if n == 0:
            return 0.0, -1
        lam = self.config.copy_strength * n / self.config.copy_max_n
        return lam, int(tokens[end])

    def mix(self, base_probs: np.ndarray, tokens: Sequence[int]) -> np.ndarray:
        lam, tok = self.copy_component(tokens)
        if lam == 0.0:
            return base_probs
        out = (1.0 - lam) * base_probs
        out[tok] += lam
        return out

    def next_token_probs(self, state, tokens: Sequence[int]) -> np.ndarray:
        """Untempered ``p(x_{t+1} | tokens)`` given the state after ``tokens``."""
        return self.mix(softmax(self.lm_head(state[1])), tokens)

    def session(self, prompt: Iterable[int]) -> "Session":
        return Session(self, list(prompt))


@dataclass
class Session:
    """Incremental decoding state, the toy analogue of a KV cache."""

    model: ToyLM
    tokens: list[int] = field(default_factory=list)
    states: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)

    def __post_init__(self):
        prompt, self.tokens, self.states = self.tokens, [], []
        if not prompt:
            raise ValueError("prompt must contain at least one token")
        self.extend(prompt)

    def __len__(self) -> int:
        return len(self.tokens)

    def extend(self, tokens: Iterable[int]) -> None:
        state = self.states[-1] if self.states else self.model.initial_state()
        for tok in tokens:
            state = self.model.step(state, int(tok))
            self.tokens.append(int(tok))
            self.states.append(state)

    def truncate(self, length: int) -> None:
        """Roll back to the first ``length`` tokens (rejected drafts are discarded)."""
        del self.tokens[length:]
        del self.states[length:]

    def sync(self, tokens: Sequence[int]) -> None:
        """Make the session match ``tokens``, reusing the longest common prefix."""
        common = 0
        limit = min(len(self.tokens), len(tokens))
        while common < limit and self.tokens[common] == tokens[common]:
            common += 1
        self.truncate(common)
        self.extend(tokens[common:])

    def probs_at(self, position: int) -> np.ndarray:
        """Distribution for the token after ``tokens[:position + 1]``."""
        return self.model.next_token_probs(self.states[position], self.tokens[: position + 1])

    def next_probs(self) -> np.ndarray:
        return self.probs_at(len(self.tokens) - 1)

    def features(self, position: int = -1, layers: str = "all") -> np.ndarray:
        h1, h2 = self.states[position]
        return h2.copy() if layers == "top" else np.concatenate([h1, h2])


def make_draft_model(target: ToyLM, noise: float = 0.15, copy_strength: float | None = None, seed: int = 1) -> ToyLM:
    """Build a smaller-quality draft model that is "from the same family" as the target.

    Real draft models (e.g. Qwen3-0.6B for Qwen3-8B) agree with the target on
    easy tokens and diverge on hard ones. We emulate that by perturbing every
    weight matrix with relative Gaussian noise.
    """
    cfg = ToyLMConfig(**vars(target.config))
    if copy_strength is not None:
        cfg.copy_strength = copy_strength
    draft = ToyLM(cfg)
    rng = np.random.default_rng(seed)
    for name in ("E", "W1", "W2", "Vm", "U", "b"):
        w = getattr(target, name)
        setattr(draft, name, w + noise * np.std(w) * rng.normal(size=w.shape))
    return draft


def generate_corpus(
    model: ToyLM,
    num_sequences: int,
    length: int,
    temperature: float = 1.0,
    prompt_len: int = 4,
    seed: int = 0,
) -> list[list[int]]:
    """Sample sequences from ``model``; used as on-policy training data for drafters."""
    from specdec.sampling import apply_temperature, sample

    rng = np.random.default_rng(seed)
    corpus = []
    for _ in range(num_sequences):
        prompt = rng.integers(0, model.vocab_size, size=prompt_len).tolist()
        sess = model.session(prompt)
        for _ in range(length):
            p = apply_temperature(sess.next_probs(), temperature)
            sess.extend([sample(p, rng)])
        corpus.append(list(sess.tokens))
    return corpus
