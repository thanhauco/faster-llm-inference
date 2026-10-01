"""``specdec`` command line.

    specdec demo                         # every drafter on one prompt, lossless check
    specdec bench --out results/         # SPEED-Bench-style sweep on the toy target
    specdec whatif --alpha 0.7           # roofline speedup vs batch size and draft length
    specdec vllm-config ngram -k 5 --lookup-max 4 --target Qwen/Qwen3-8B
    specdec replay --traffic t.jsonl --endpoint base=http://localhost:8000/v1 --endpoint spec=...
    specdec metrics --endpoint http://localhost:8001/v1
    specdec mock-server --port 8001 --method ngram -k 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x.strip())


# --------------------------------------------------------------------- demo
def cmd_demo(args) -> int:
    from specdec.bench.speed_bench import build_drafters
    from specdec.bench.workloads import WORKLOADS
    from specdec.engine import autoregressive_generate, speculative_generate
    from specdec.models import ToyLM

    target = ToyLM()
    workload = WORKLOADS[args.workload]
    prompt = workload.prompts(target, 1, seed=args.seed)[0]
    print(f"workload={workload.name} ({workload.description}), prompt={len(prompt)} tokens, "
          f"T={workload.temperature}, k={args.k}\n")
    print("training drafters on on-policy target data ...")
    drafters = build_drafters(target, train_sequences=80, log=lambda *a: None)
    rng_seed = args.seed
    reference = autoregressive_generate(target, prompt, workload.max_new_tokens, workload.temperature,
                                        np.random.default_rng(rng_seed)).tokens
    print(f"\n{'drafter':<16} {'tau':>6} {'accept%':>8} {'target passes':>14} {'same as baseline':>17}")
    for bd in drafters:
        r = speculative_generate(target, bd.drafter, prompt, workload.max_new_tokens, k=args.k,
                                 temperature=workload.temperature, rng=np.random.default_rng(rng_seed),
                                 confidence_threshold=bd.confidence_threshold)
        same = "yes" if r.tokens == reference else ("n/a (sampled)" if workload.temperature > 0 else "NO")
        s = r.stats
        print(f"{bd.name:<16} {s.mean_acceptance_length:6.2f} {100 * s.draft_acceptance_rate:7.1f}% "
              f"{s.num_steps + 1:>8} / {len(r.tokens):<5} {same:>17}")
    print("\ntau = tokens per target forward pass. Baseline needs one pass per token.")
    return 0


# -------------------------------------------------------------------- bench
def _cost_model(args):
    from specdec.cost_model import HARDWARE, MODELS, CostModel

    return CostModel(HARDWARE[args.hardware], MODELS[args.model_size], context_len=args.context_len)


def cmd_bench(args) -> int:
    from specdec.bench.speed_bench import ALL_DRAFTERS, BenchConfig, run, to_markdown

    ks = _ints(args.ks)
    cfg = BenchConfig(
        k_max=max(ks), ks=ks, batch_sizes=_ints(args.batch_sizes), prompts_per_workload=args.prompts,
        report_k=args.report_k, seed=args.seed, cost_model=_cost_model(args),
    )
    drafters = tuple(args.drafters.split(",")) if args.drafters else ALL_DRAFTERS
    workloads = tuple(args.workloads.split(",")) if args.workloads else None
    t0 = time.time()
    report = run(drafter_names=drafters, workload_names=workloads, cfg=cfg, out_dir=args.out)
    print()
    print(to_markdown(report))
    print(f"done in {time.time() - t0:.1f}s")
    return 0


# ------------------------------------------------------------------- whatif
def cmd_whatif(args) -> int:
    from specdec.cost_model import expected_acceptance_length
    from specdec.drafters.base import DraftCost

    cm = _cost_model(args)
    ks = _ints(args.ks)
    batches = _ints(args.batch_sizes)
    print(f"{cm.model.name} on {cm.hardware.name}; per-token acceptance alpha={args.alpha}, "
          f"draft passes={'k' if args.draft_passes == 'k' else args.draft_passes}, "
          f"draft size={args.draft_rel_params:.0%} of target\n")
    print("speedup vs no speculation (rows: k, cols: batch)")
    print("k \\ B " + "".join(f"{b:>8}" for b in batches))
    best = {b: (None, 0.0) for b in batches}
    for k in ks:
        tau = expected_acceptance_length(args.alpha, k)
        passes = k if args.draft_passes == "k" else int(args.draft_passes)
        cost = DraftCost(passes=passes, tokens_per_pass=1 if args.draft_passes == "k" else k + 1,
                         rel_params=args.draft_rel_params, cpu_ms=0.05 if passes == 0 else 0.0)
        row = []
        for b in batches:
            sp = cm.speedup(b, tau, cost, k + 1)
            row.append(sp)
            if sp > best[b][1]:
                best[b] = (k, sp)
        print(f"{k:<6}" + "".join(f"{x:8.2f}" for x in row) + f"   (tau={tau:.2f})")
    print("best k" + "".join(f"{best[b][0]:>8}" for b in batches))
    return 0


# -------------------------------------------------------------- vllm-config
def cmd_vllm_config(args) -> int:
    from specdec.serving import configs

    if args.method == "preset":
        if args.preset not in configs.PRESETS:
            print(f"unknown preset; choose from: {', '.join(configs.PRESETS)}", file=sys.stderr)
            return 2
        target, spec = configs.PRESETS[args.preset]
        target = args.target or target
    elif args.method == "speculators":
        if not args.model:
            print("--model <speculator checkpoint> is required", file=sys.stderr)
            return 2
        print("# Speculators checkpoints are self-describing; vLLM reads speculators_config from config.json")
        print(configs.speculators_serve_command(args.model, args.port))
        return 0
    else:
        extra = {}
        if args.method == "ngram":
            spec = configs.ngram_config(args.k or 5, args.lookup_max, args.lookup_min)
        else:
            spec = configs.speculative_config(args.method, args.k, args.model,
                                              draft_tensor_parallel_size=args.draft_tp, **extra)
        target = args.target or "<target-model>"
    print("# --speculative-config")
    print(json.dumps(spec, indent=2))
    print("\n# online")
    print(configs.serve_command(target, spec, args.port))
    print("\n# offline")
    print(configs.offline_snippet(target, spec))
    return 0


# ------------------------------------------------------------------- replay
def cmd_replay(args) -> int:
    from specdec.serving import replay, traffic

    endpoints = {}
    for item in args.endpoint:
        label, _, url = item.partition("=")
        if not url:
            label, url = f"ep{len(endpoints)}", item
        endpoints[label] = url
    reqs = traffic.load_jsonl(args.traffic, limit=args.limit, shuffle=args.shuffle,
                              override_max_tokens=args.max_tokens, override_temperature=args.temperature)
    print(f"replaying {len(reqs)} requests against {len(endpoints)} endpoint(s) at concurrency {args.concurrency}")
    report = replay.sweep(endpoints, reqs, list(_ints(args.concurrency)), model=args.model,
                          api_key=args.api_key, warmup=args.warmup)
    print()
    print(replay.to_markdown(report))
    if args.out:
        replay.save(report, args.out)
        print(f"wrote {args.out}.json and {args.out}.md")
    return 0


# ------------------------------------------------------------------ metrics
def cmd_metrics(args) -> int:
    from specdec.serving import prom

    prev = None
    while True:
        snap = prom.scrape(args.endpoint)
        if not snap.found:
            print("no vllm:spec_decode_* metrics found (is speculative decoding enabled?)")
            return 1
        view = snap if prev is None else snap - prev
        d = view.to_dict()
        tau = d["mean_acceptance_length"]
        rate = d["draft_acceptance_rate"]
        pos = ", ".join(f"{x:.2f}" for x in d["per_position_acceptance"])
        scope = "since start" if prev is None else f"last {args.watch}s"
        print(f"[{scope}] drafts={d['num_drafts']:.0f} tau={'-' if tau is None else f'{tau:.3f}'} "
              f"acceptance={'-' if rate is None else f'{rate:.1%}'} per-position=[{pos}]")
        if not args.watch:
            return 0
        prev = snap
        time.sleep(args.watch)


# -------------------------------------------------------------- mock-server
def cmd_mock_server(args) -> int:
    from specdec.bench.speed_bench import build_drafters
    from specdec.models import ToyLM
    from specdec.serving.mock_server import MockEngine, serve

    target = ToyLM()
    drafter, threshold = None, None
    if args.method != "none":
        print(f"preparing drafter {args.method} ...")
        bd = build_drafters(target, (args.method,), train_sequences=80, log=lambda *a: None)[0]
        drafter, threshold = bd.drafter, bd.confidence_threshold
    engine = MockEngine(target, drafter, k=args.k, cost_model=_cost_model(args), time_scale=args.time_scale,
                        model_name=args.served_model_name, confidence_threshold=threshold)
    server = serve(engine, args.host, args.port)
    print(f"mock OpenAI-compatible server on http://{args.host}:{args.port}/v1 "
          f"(method={args.method}, k={args.k}); Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _add_cost_args(p):
    from specdec.cost_model import HARDWARE, MODELS

    p.add_argument("--hardware", default="h100", choices=sorted(HARDWARE))
    p.add_argument("--model-size", default="8b", choices=sorted(MODELS))
    p.add_argument("--context-len", type=int, default=2048)


def build_parser() -> argparse.ArgumentParser:
    from specdec.bench.workloads import WORKLOADS
    from specdec.serving.configs import METHODS

    parser = argparse.ArgumentParser(prog="specdec", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("demo", help="run every drafter on one prompt")
    p.add_argument("--workload", default="reasoning", choices=sorted(WORKLOADS))
    p.add_argument("-k", type=int, default=5)
    # Many greedy toy prompts fall into a repetition loop where every drafter is perfect;
    # seed 2 gives a prompt that separates the methods.
    p.add_argument("--seed", type=int, default=2)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("bench", help="SPEED-Bench-style sweep on the toy target")
    p.add_argument("--drafters", default="", help="comma list (default: all)")
    p.add_argument("--workloads", default="", help="comma list (default: all)")
    p.add_argument("--prompts", type=int, default=6, help="prompts per workload")
    p.add_argument("--ks", default="1,2,3,4,5,6,8")
    p.add_argument("--report-k", type=int, default=5)
    p.add_argument("--batch-sizes", default="1,4,16,64,256")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    _add_cost_args(p)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("whatif", help="roofline speedup table for a given acceptance rate")
    p.add_argument("--alpha", type=float, default=0.7, help="per-token acceptance probability")
    p.add_argument("--ks", default="1,2,3,4,5,6,8")
    p.add_argument("--batch-sizes", default="1,4,16,64,256")
    p.add_argument("--draft-passes", default="k", help="'k' (autoregressive drafter), '1' (block), '0' (n-gram)")
    p.add_argument("--draft-rel-params", type=float, default=0.04)
    _add_cost_args(p)
    p.set_defaults(func=cmd_whatif)

    p = sub.add_parser("vllm-config", help="print --speculative-config and vllm serve commands")
    p.add_argument("method", choices=[*METHODS, "speculators", "preset"])
    p.add_argument("-k", "--num-speculative-tokens", dest="k", type=int, default=None)
    p.add_argument("--model", help="draft / speculator model")
    p.add_argument("--target", help="target model to serve")
    p.add_argument("--lookup-max", type=int, default=4)
    p.add_argument("--lookup-min", type=int, default=None)
    p.add_argument("--draft-tp", type=int, default=None)
    p.add_argument("--preset", default="ngram-post")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_vllm_config)

    p = sub.add_parser("replay", help="replay your own traffic against one or more endpoints")
    p.add_argument("--traffic", required=True, help="JSONL with prompt/messages records")
    p.add_argument("--endpoint", action="append", required=True, help="label=http://host:port/v1 (repeatable)")
    p.add_argument("--concurrency", default="1,4,16")
    p.add_argument("--model", default=None, help="served model name (default: first from /v1/models)")
    p.add_argument("--api-key", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--max-tokens", type=int, default=None, help="override max_tokens")
    p.add_argument("--temperature", type=float, default=None, help="override temperature")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--out", default=None, help="path prefix for .json/.md reports")
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("metrics", help="acceptance stats from a vLLM /metrics endpoint")
    p.add_argument("--endpoint", required=True)
    p.add_argument("--watch", type=float, default=0, help="refresh every N seconds, showing deltas")
    p.set_defaults(func=cmd_metrics)

    p = sub.add_parser("mock-server", help="toy OpenAI-compatible server for trying the tooling")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--method", default="none",
                   choices=["none", "ngram", "draft-model", "eagle1", "eagle3", "dflash", "dspark",
                            "dspark-adaptive"])
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--time-scale", type=float, default=1.0, help="multiply simulated step latency")
    p.add_argument("--served-model-name", default="toy-target")
    _add_cost_args(p)
    p.set_defaults(func=cmd_mock_server)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
