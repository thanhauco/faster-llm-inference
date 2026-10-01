"""Benchmark harness: realistic vs synthetic workloads, draft length and batch sweeps."""

from specdec.bench.speed_bench import ALL_DRAFTERS, BenchConfig, analyse, build_drafters, run, run_acceptance, to_markdown
from specdec.bench.workloads import WORKLOADS, Workload, realistic, synthetic

__all__ = [
    "ALL_DRAFTERS",
    "BenchConfig",
    "WORKLOADS",
    "Workload",
    "analyse",
    "build_drafters",
    "realistic",
    "run",
    "run_acceptance",
    "synthetic",
    "to_markdown",
]
