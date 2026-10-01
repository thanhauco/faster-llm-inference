"""Resource #2: EAGLE-style draft heads built on the target's hidden states.

Compares a classic separate draft model against two EAGLE-style heads:

* eagle1: extrapolates top-layer features only
* eagle3: fuses features from all layers (EAGLE-3 idea) and is trained with
  "training-time test" (it sees its own predictions as inputs)

Watch per-position acceptance: a better head keeps acceptance up deeper into
the draft, which is what longer acceptance lengths come from.

    python examples/02_eagle3_head.py
"""

import numpy as np

from specdec import ToyLM, make_draft_model, speculative_generate
from specdec.bench.workloads import WORKLOADS
from specdec.cost_model import CostModel
from specdec.drafters import DraftModelDrafter, EagleDrafter, collect_traces
from specdec.engine import SpecStats
from specdec.models import generate_corpus
from specdec.serving import eagle3_config, serve_command

target = ToyLM()
print("collecting on-policy target traces (what speculators extracts from vLLM) ...")
traces = collect_traces(target, generate_corpus(target, 120, 128, temperature=1.0, seed=1))

drafters = {
    "draft-model": DraftModelDrafter(make_draft_model(target)),
    "eagle1 (top layer)": EagleDrafter(target, layers="top").fit(traces, ttt_rounds=0),
    "eagle3 (fused, no TTT)": EagleDrafter(target, layers="all").fit(traces, ttt_rounds=0),
    "eagle3 (fused + TTT)": EagleDrafter(target, layers="all").fit(traces, ttt_rounds=1),
}

k = 6
cm = CostModel()
print(f"\nk={k}; realistic workloads; speedup from the roofline model (8B on H100)")
print(f"{'drafter':<24} {'tau':>5}  {'B=1':>6} {'B=16':>6}  per-position acceptance")
for name, d in drafters.items():
    total = SpecStats(k)
    for w in ("rag-qa", "multi-turn-chat", "open-chat", "reasoning"):
        wl = WORKLOADS[w]
        for i, p in enumerate(wl.prompts(target, 4, seed=0)):
            r = speculative_generate(target, d, p, wl.max_new_tokens, k=k, temperature=wl.temperature,
                                     rng=np.random.default_rng(i))
            total = total.merge(r.stats)
    tau = total.mean_acceptance_length
    sp = [cm.speedup(b, tau, d.cost(k), total.mean_verified_tokens) for b in (1, 16)]
    pos = " ".join(f"{x:.2f}" for x in total.per_position_acceptance())
    print(f"{name:<24} {tau:5.2f}  {sp[0]:5.2f}x {sp[1]:5.2f}x  [{pos}]")

print("\nServe a real EAGLE-3 head:")
print("  ", serve_command("meta-llama/Llama-3.1-8B-Instruct",
                          eagle3_config("RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3", 3)))
