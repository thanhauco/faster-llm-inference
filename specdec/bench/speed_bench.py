"""SPEED-Bench-style sweep: drafters x workloads x draft length x batch size.

Two stages, mirroring how you should evaluate speculation for real:

1. **Acceptance** (model-dependent, hardware-independent): run every drafter
   on every workload with lossless verification and record per-step accepted
   counts. One run at ``k_max`` covers all smaller ``k`` via
   :meth:`SpecStats.truncated`.
2. **Throughput** (hardware-dependent): turn acceptance into tokens/s and
   speedup at each batch size with the roofline :class:`CostModel`.

The report answers the three questions from the post: which drafter, what
draft length, and how much do synthetic prompts overestimate the gain.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

from specdec.bench.workloads import WORKLOADS, Workload
from specdec.cost_model import CostModel
from specdec.drafters import (
    BlockDrafter,
    DraftModelDrafter,
    Drafter,
    EagleDrafter,
    NgramDrafter,
    collect_traces,
)
from specdec.engine import SpecStats, speculative_generate
from specdec.models import ToyLM, generate_corpus, make_draft_model


@dataclass
class BenchDrafter:
    name: str
    drafter: Drafter
    confidence_threshold: float | None = None


@dataclass
class BenchConfig:
    k_max: int = 8
    ks: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8)
    batch_sizes: tuple[int, ...] = (1, 4, 16, 64, 256)
    prompts_per_workload: int = 6
    report_k: int = 5
    seed: int = 0
    cost_model: CostModel = field(default_factory=CostModel)


ALL_DRAFTERS = ("ngram", "draft-model", "eagle1", "eagle3", "dflash", "dspark", "dspark-adaptive")


def build_drafters(
    target: ToyLM,
    names: tuple[str, ...] = ALL_DRAFTERS,
    train_sequences: int = 120,
    train_length: int = 128,
    block_size: int = 8,
    seed: int = 0,
    log=print,
) -> list[BenchDrafter]:
    """Instantiate and (where needed) train drafters on on-policy target data."""
    out: list[BenchDrafter] = []
    traces = None
    if any(n in names for n in ("eagle1", "eagle3", "dflash", "dspark", "dspark-adaptive")):
        t0 = time.time()
        corpus = generate_corpus(target, train_sequences, train_length, temperature=1.0, seed=seed + 1000)
        traces = collect_traces(target, corpus)
        log(f"  collected {sum(len(c) for c in corpus)} on-policy training tokens in {time.time() - t0:.1f}s")
    dspark = None
    for name in names:
        t0 = time.time()
        if name == "ngram":
            out.append(BenchDrafter(name, NgramDrafter.vllm_default()))
        elif name == "draft-model":
            out.append(BenchDrafter(name, DraftModelDrafter(make_draft_model(target, seed=seed + 7))))
        elif name in ("eagle1", "eagle3"):
            layers = "top" if name == "eagle1" else "all"
            out.append(BenchDrafter(name, EagleDrafter(target, layers=layers).fit(traces)))
        elif name == "dflash":
            out.append(BenchDrafter(name, BlockDrafter.dflash(target, block_size).fit(traces, seed=seed)))
        elif name in ("dspark", "dspark-adaptive"):
            if dspark is None:
                dspark = BlockDrafter.dspark(target, block_size).fit(traces, seed=seed)
            threshold = 0.3 if name == "dspark-adaptive" else None
            out.append(BenchDrafter(name, dspark, confidence_threshold=threshold))
        else:
            raise ValueError(f"unknown drafter {name!r}; choose from {ALL_DRAFTERS}")
        log(f"  built {name:<16} in {time.time() - t0:.1f}s")
    return out


def run_acceptance(
    target: ToyLM,
    drafters: list[BenchDrafter],
    workloads: list[Workload],
    cfg: BenchConfig,
    log=print,
) -> dict[tuple[str, str], SpecStats]:
    results: dict[tuple[str, str], SpecStats] = {}
    for w in workloads:
        prompts = w.prompts(target, cfg.prompts_per_workload, seed=cfg.seed)
        for bd in drafters:
            t0 = time.time()
            total = SpecStats(cfg.k_max)
            for i, prompt in enumerate(prompts):
                r = speculative_generate(
                    target, bd.drafter, prompt, w.max_new_tokens, k=cfg.k_max,
                    temperature=w.temperature, rng=np.random.default_rng(cfg.seed + i),
                    confidence_threshold=bd.confidence_threshold,
                )
                total = total.merge(r.stats)
            results[(w.name, bd.name)] = total
            log(f"  {w.name:<17} {bd.name:<16} tau@k{cfg.k_max}={total.mean_acceptance_length:.2f} "
                f"({time.time() - t0:.1f}s)")
    return results


def analyse(
    results: dict[tuple[str, str], SpecStats],
    drafters: list[BenchDrafter],
    workloads: list[Workload],
    cfg: BenchConfig,
) -> dict:
    cm = cfg.cost_model
    by_name = {bd.name: bd for bd in drafters}
    rows = []
    for (wname, dname), stats in results.items():
        cost_of = by_name[dname].drafter.cost
        for k in cfg.ks:
            s = stats.truncated(k)
            for b in cfg.batch_sizes:
                rows.append({
                    "workload": wname,
                    "category": WORKLOADS[wname].category if wname in WORKLOADS else "custom",
                    "drafter": dname,
                    "k": k,
                    "batch": b,
                    "acceptance_length": s.mean_acceptance_length,
                    "acceptance_rate": s.draft_acceptance_rate,
                    "verified_tokens": s.mean_verified_tokens,
                    "speedup": cm.speedup(b, s.mean_acceptance_length, cost_of(k), s.mean_verified_tokens),
                    "tokens_per_s": cm.speculative_tokens_per_s(
                        b, s.mean_acceptance_length, cost_of(k), s.mean_verified_tokens),
                    "baseline_tokens_per_s": cm.baseline_tokens_per_s(b),
                })

    def mean(values):
        values = list(values)
        return float(np.mean(values)) if values else float("nan")

    def select(**kw):
        return [r for r in rows if all(r[key] == val for key, val in kw.items())]

    wnames = [w.name for w in workloads]
    dnames = [bd.name for bd in drafters]
    categories = sorted({WORKLOADS[w].category for w in wnames if w in WORKLOADS})
    k_ref, b_lo, b_hi = cfg.report_k, cfg.batch_sizes[0], cfg.batch_sizes[-1]

    acceptance = {
        w: {d: select(workload=w, drafter=d, k=k_ref, batch=b_lo)[0]["acceptance_length"] for d in dnames}
        for w in wnames
    }
    per_position = {
        w: {d: [round(x, 3) for x in results[(w, d)].per_position_acceptance()] for d in dnames} for w in wnames
    }
    by_category = {
        c: {
            d: {
                "acceptance_length": mean(r["acceptance_length"] for r in select(category=c, drafter=d, k=k_ref, batch=b_lo)),
                f"speedup_b{b_lo}": mean(r["speedup"] for r in select(category=c, drafter=d, k=k_ref, batch=b_lo)),
                f"speedup_b{b_hi}": mean(r["speedup"] for r in select(category=c, drafter=d, k=k_ref, batch=b_hi)),
            }
            for d in dnames
        }
        for c in categories
    }
    best_k = {}
    for d in dnames:
        best_k[d] = []
        for b in cfg.batch_sizes:
            scored = {k: mean(r["speedup"] for r in select(category="realistic", drafter=d, k=k, batch=b))
                      for k in cfg.ks}
            k_best = max(scored, key=scored.get)
            best_k[d].append({"batch": b, "k": k_best, "speedup": scored[k_best]})
    return {
        "config": {
            "ks": list(cfg.ks), "batch_sizes": list(cfg.batch_sizes), "k_max": cfg.k_max,
            "report_k": k_ref, "prompts_per_workload": cfg.prompts_per_workload,
            "cost_model": cm.describe(),
        },
        "acceptance_length": acceptance,
        "per_position_acceptance": per_position,
        "by_category": by_category,
        "best_k": best_k,
        "rows": rows,
    }


def to_markdown(report: dict) -> str:
    cfg = report["config"]
    k_ref = cfg["report_k"]
    bs = cfg["batch_sizes"]
    lines = ["# SPEED-Bench-style results (toy target)", ""]
    hw, model = cfg["cost_model"]["hardware"], cfg["cost_model"]["model"]
    lines += [
        f"Cost model: {model['name']} on {hw['name']}, context {cfg['cost_model']['context_len']} tokens. "
        f"{cfg['prompts_per_workload']} prompts per workload. Acceptance from a lossless run at k={cfg['k_max']}.",
        "",
        f"## Mean acceptance length (tokens per target pass) at k={k_ref}",
        "",
    ]
    acc = report["acceptance_length"]
    dnames = list(next(iter(acc.values())).keys())
    lines.append("| workload | " + " | ".join(dnames) + " |")
    lines.append("|---" * (len(dnames) + 1) + "|")
    for w, row in acc.items():
        lines.append(f"| {w} | " + " | ".join(f"{row[d]:.2f}" for d in dnames) + " |")

    lines += ["", f"## Synthetic vs realistic prompts (k={k_ref})", ""]
    cats = report["by_category"]
    lines.append("| drafter | " + " | ".join(
        f"{c} tau | {c} speedup@B={bs[0]} | {c} speedup@B={bs[-1]}" for c in cats) + " |")
    lines.append("|---" * (1 + 3 * len(cats)) + "|")
    for d in dnames:
        cells = []
        for c in cats:
            v = cats[c][d]
            cells += [f"{v['acceptance_length']:.2f}", f"{v[f'speedup_b{bs[0]}']:.2f}x", f"{v[f'speedup_b{bs[-1]}']:.2f}x"]
        lines.append(f"| {d} | " + " | ".join(cells) + " |")

    lines += ["", "## Best draft length per batch size (realistic workloads)", ""]
    lines.append("| drafter | " + " | ".join(f"B={b}" for b in bs) + " |")
    lines.append("|---" * (len(bs) + 1) + "|")
    for d, per_b in report["best_k"].items():
        lines.append(f"| {d} | " + " | ".join(f"k={e['k']} ({e['speedup']:.2f}x)" for e in per_b) + " |")
    lines += ["", "Toy-model numbers: use them to compare methods and see trends, not as production estimates.", ""]
    return "\n".join(lines)


def run(
    target: ToyLM | None = None,
    drafter_names: tuple[str, ...] = ALL_DRAFTERS,
    workload_names: tuple[str, ...] | None = None,
    cfg: BenchConfig | None = None,
    out_dir: str | None = None,
    log=print,
) -> dict:
    target = target or ToyLM()
    cfg = cfg or BenchConfig()
    workloads = [WORKLOADS[n] for n in (workload_names or tuple(WORKLOADS))]
    log("building drafters")
    drafters = build_drafters(target, drafter_names, seed=cfg.seed, log=log)
    log("measuring acceptance")
    results = run_acceptance(target, drafters, workloads, cfg, log=log)
    report = analyse(results, drafters, workloads, cfg)
    if out_dir:
        import os

        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "speed_bench.json"), "w") as f:
            json.dump(report, f, indent=1)
        with open(os.path.join(out_dir, "speed_bench.md"), "w") as f:
            f.write(to_markdown(report))
        log(f"wrote {out_dir}/speed_bench.json and speed_bench.md")
    return report
