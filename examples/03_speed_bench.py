"""Resource #3: SPEED-Bench-style evaluation.

Three lessons, reproduced on the toy target:

1. Test drafters on diverse, realistic prompts: acceptance swings a lot by category.
2. Synthetic inputs (random ids, repeated filler) give misleading throughput.
3. The best draft length changes with batch size.

    python examples/03_speed_bench.py            # ~30 s
    specdec bench --out results/                 # same thing, full CLI
"""

from specdec.bench import BenchConfig, run, to_markdown

report = run(cfg=BenchConfig(prompts_per_workload=4), log=lambda *a: None)
print(to_markdown(report))

real = report["by_category"]["realistic"]
synth = report["by_category"]["synthetic"]
print("How much synthetic prompts overstate acceptance length (k=5):")
for d in real:
    gap = synth[d]["acceptance_length"] / real[d]["acceptance_length"] - 1
    print(f"  {d:<16} {gap:+.0%}")
