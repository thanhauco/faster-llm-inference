"""Benchmark workloads: synthetic inputs vs realistic, category-diverse prompts.

Resource #3 (SPEED-Bench) argues that drafters must be tested on diverse,
realistic prompts because acceptance depends heavily on the *kind* of text
being generated, and synthetic inputs (random tokens, repeated filler) give
misleading throughput numbers. These are toy analogues of the categories that
matter, built from text sampled from the target itself:

=================  ===========================================  ==========
workload           real-world analogue                          sampling
=================  ===========================================  ==========
code-edit          "apply this edit to the file" (output ~copies input) T=0
rag-qa             answer grounded in retrieved passages          T=0.7
multi-turn-chat    later turns that reuse earlier turns           T=0.8
open-chat          short prompt, open-ended answer                T=1.0
reasoning          long greedy chain-of-thought / math            T=0
synthetic-random   random token ids (``--dataset-name random``)   T=0
synthetic-repeat   one short phrase repeated to fill the prompt   T=0
=================  ===========================================  ==========
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from specdec.models import ToyLM
from specdec.sampling import apply_temperature, sample

SEP = 0  # separator token between prompt sections


def target_text(target: ToyLM, rng: np.random.Generator, length: int, temperature: float = 1.0) -> list[int]:
    """Natural-looking text for the toy world: a sample from the target itself."""
    seed = rng.integers(1, target.vocab_size, size=4).tolist()
    sess = target.session(seed)
    for _ in range(length):
        sess.extend([sample(apply_temperature(sess.next_probs(), temperature), rng)])
    return sess.tokens[4:]


@dataclass(frozen=True)
class Workload:
    name: str
    category: str  # "realistic" or "synthetic"
    temperature: float
    max_new_tokens: int
    make_prompt: Callable[[ToyLM, np.random.Generator], list[int]]
    description: str = ""

    def prompts(self, target: ToyLM, n: int, seed: int = 0) -> list[list[int]]:
        rng = np.random.default_rng(seed)
        return [self.make_prompt(target, rng) for _ in range(n)]


def _code_edit(target, rng):
    doc = target_text(target, rng, 160)
    return doc + [SEP] + doc[:6]


def _rag_qa(target, rng):
    passages = []
    for _ in range(3):
        passages += target_text(target, rng, 60) + [SEP]
    return passages + target_text(target, rng, 8)


def _multi_turn(target, rng):
    turns = []
    for _ in range(3):
        turns += target_text(target, rng, 40, temperature=0.8) + [SEP]
    return turns


def _open_chat(target, rng):
    return target_text(target, rng, 12)


def _reasoning(target, rng):
    return target_text(target, rng, 24)


def _synthetic_random(target, rng):
    return rng.integers(1, target.vocab_size, size=64).tolist()


def _synthetic_repeat(target, rng):
    phrase = rng.integers(1, target.vocab_size, size=8).tolist()
    return phrase * 8


WORKLOADS: dict[str, Workload] = {
    w.name: w
    for w in [
        Workload("code-edit", "realistic", 0.0, 128, _code_edit, "rewrite a document with edits"),
        Workload("rag-qa", "realistic", 0.7, 96, _rag_qa, "answer from retrieved passages"),
        Workload("multi-turn-chat", "realistic", 0.8, 96, _multi_turn, "conversation with history"),
        Workload("open-chat", "realistic", 1.0, 96, _open_chat, "short open-ended prompt"),
        Workload("reasoning", "realistic", 0.0, 160, _reasoning, "long greedy reasoning"),
        Workload("synthetic-random", "synthetic", 0.0, 128, _synthetic_random, "random token ids"),
        Workload("synthetic-repeat", "synthetic", 0.0, 128, _synthetic_repeat, "repeated filler phrase"),
    ]
}


def realistic() -> list[Workload]:
    return [w for w in WORKLOADS.values() if w.category == "realistic"]


def synthetic() -> list[Workload]:
    return [w for w in WORKLOADS.values() if w.category == "synthetic"]
