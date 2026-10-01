"""Numerically stable probability helpers shared by models, drafters and the verifier."""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = logits - np.max(logits, axis=axis, keepdims=True)
    return z - np.log(np.sum(np.exp(z), axis=axis, keepdims=True))


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Return p^(1/T) renormalised; T == 0 gives a one-hot argmax distribution.

    Working in probability space (instead of logits) lets models that are a
    mixture of distributions, like :class:`specdec.models.ToyLM`, expose a
    single well-defined sampling distribution for every temperature.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if temperature <= 0.0:
        out = np.zeros_like(probs)
        idx = np.argmax(probs, axis=-1)
        np.put_along_axis(out, np.expand_dims(idx, -1), 1.0, axis=-1)
        return out
    if temperature == 1.0:
        return probs / probs.sum(axis=-1, keepdims=True)
    logp = np.log(np.maximum(probs, _EPS)) / temperature
    return softmax(logp, axis=-1)


def sample(probs: np.ndarray, rng: np.random.Generator) -> int:
    """Sample one index from a 1-D probability vector."""
    p = probs / probs.sum()
    return int(rng.choice(p.shape[-1], p=p))


def entropy(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.maximum(probs, _EPS)
    return -np.sum(p * np.log(p), axis=axis)


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """TV distance. For a single position, P(accept) under rejection sampling = 1 - TV(p, q)."""
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())
