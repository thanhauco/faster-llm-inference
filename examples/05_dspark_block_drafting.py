"""Resource #5: block drafting plus sequential correction (DFlash -> DSpark).

* DFlash drafts a whole block in ONE pass. Cheap, but positions inside the
  block cannot see each other, so acceptance decays along the block.
* DSpark adds a low-rank Markov head (each position conditioned on the token
  drafted before it) and a confidence head that predicts how far the block is
  worth verifying (adaptive verification).

    python examples/05_dspark_block_drafting.py
"""

import numpy as np

from specdec import ToyLM, speculative_generate
from specdec.bench.workloads import WORKLOADS
from specdec.cost_model import CostModel
from specdec.drafters import BlockDrafter, EagleDrafter, collect_traces
from specdec.engine import SpecStats
from specdec.models import generate_corpus

target = ToyLM()
traces = collect_traces(target, generate_corpus(target, 120, 128, temperature=1.0, seed=1))
block = 8
dflash = BlockDrafter.dflash(target, block).fit(traces)
dspark = BlockDrafter.dspark(target, block, markov_rank=32).fit(traces)
eagle3 = EagleDrafter(target).fit(traces)

variants = [
    ("eagle3 (k passes)", eagle3, None),
    ("dflash (1 pass)", dflash, None),
    ("dspark (1 pass + Markov)", dspark, None),
    ("dspark + adaptive 0.5", dspark, 0.5),
    ("dspark + adaptive 0.3", dspark, 0.3),
]

cm = CostModel()
print(f"block/k = {block}; workloads: reasoning (math-like, greedy) and open-chat (T=1)\n")
for wname in ("reasoning", "open-chat"):
    wl = WORKLOADS[wname]
    print(f"[{wname}]")
    print(f"  {'variant':<26} {'tau':>5} {'verified':>8} {'B=1':>6} {'B=64':>6} {'B=256':>6}  per-position")
    for name, d, thr in variants:
        total = SpecStats(block)
        for i, p in enumerate(wl.prompts(target, 6, seed=0)):
            total = total.merge(speculative_generate(target, d, p, wl.max_new_tokens, k=block,
                                                     temperature=wl.temperature, rng=np.random.default_rng(i),
                                                     confidence_threshold=thr).stats)
        tau, ver = total.mean_acceptance_length, total.mean_verified_tokens
        sp = [cm.speedup(b, tau, d.cost(block), ver) for b in (1, 64, 256)]
        pos = " ".join(f"{x:.2f}" for x in total.per_position_acceptance())
        print(f"  {name:<26} {tau:5.2f} {ver:8.2f} {sp[0]:5.2f}x {sp[1]:5.2f}x {sp[2]:5.2f}x  [{pos}]")
    print()

print("Adaptive verification trades a little acceptance for far fewer verified tokens,")
print("which matters once the batch is large enough for verification to be compute-bound.")
