"""Tiny numpy training utilities used to fit drafters on target-model data.

Everything here is deliberately small: ridge regression for feature heads
(EAGLE), Adam-trained softmax regression for token heads (DFlash/DSpark
blocks, Markov bias) and logistic regression for confidence heads. The point
is to show *what* each drafter learns from the target, not to compete with
GPU training; for real models use ``vllm-project/speculators``.
"""

from __future__ import annotations

import numpy as np

from specdec.sampling import log_softmax, softmax


def add_bias(X: np.ndarray) -> np.ndarray:
    return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)


def ridge(X: np.ndarray, Y: np.ndarray, l2: float = 1e-2) -> np.ndarray:
    """Closed-form ridge regression ``argmin ||XA - Y||^2 + l2 ||A||^2``."""
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + l2 * np.eye(d), X.T @ Y)


class _Adam:
    def __init__(self, shape, lr):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0
        self.lr = lr

    def step(self, w, g, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        self.m = b1 * self.m + (1 - b1) * g
        self.v = b2 * self.v + (1 - b2) * g * g
        mhat = self.m / (1 - b1**self.t)
        vhat = self.v / (1 - b2**self.t)
        return w - self.lr * mhat / (np.sqrt(vhat) + eps)


def fit_softmax_regression(
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    offset: np.ndarray | None = None,
    l2: float = 1e-4,
    epochs: int = 8,
    lr: float = 0.05,
    batch_size: int = 2048,
    seed: int = 0,
) -> np.ndarray:
    """Multinomial logistic regression ``softmax(X W + offset)`` trained with Adam.

    ``offset`` holds fixed per-row logits (e.g. a frozen backbone), which lets
    us fit residual heads such as the DSpark Markov bias.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W = np.zeros((d, num_classes))
    opt = _Adam(W.shape, lr)
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            logits = X[idx] @ W
            if offset is not None:
                logits = logits + offset[idx]
            p = softmax(logits)
            p[np.arange(len(idx)), y[idx]] -= 1.0
            grad = X[idx].T @ p / len(idx) + l2 * W
            W = opt.step(W, grad)
    return W


def softmax_nll(X: np.ndarray, y: np.ndarray, W: np.ndarray, offset: np.ndarray | None = None) -> float:
    logits = X @ W if offset is None else X @ W + offset
    return float(-log_softmax(logits)[np.arange(len(y)), y].mean())


def fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 1e-4,
    epochs: int = 30,
    lr: float = 0.05,
    batch_size: int = 512,
    seed: int = 0,
) -> np.ndarray:
    """Binary logistic regression with (possibly soft) labels in [0, 1]."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = np.zeros(d)
    opt = _Adam(w.shape, lr)
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            pred = 1.0 / (1.0 + np.exp(-(X[idx] @ w)))
            grad = X[idx].T @ (pred - y[idx]) / len(idx) + l2 * w
            w = opt.step(w, grad)
    return w


def low_rank(M: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Factor ``M ~= A @ B`` with ``A: (m, rank)``, ``B: (rank, n)`` via truncated SVD."""
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    rank = min(rank, len(s))
    return U[:, :rank] * s[:rank], Vt[:rank]
