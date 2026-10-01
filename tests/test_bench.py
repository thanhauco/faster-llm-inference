import json

from specdec.bench import BenchConfig, analyse, build_drafters, run_acceptance, to_markdown
from specdec.bench.workloads import WORKLOADS, realistic, synthetic
from specdec.cli import main


def test_workloads_cover_both_categories(target):
    assert realistic() and synthetic()
    for w in WORKLOADS.values():
        prompts = w.prompts(target, 2, seed=0)
        assert len(prompts) == 2 and all(len(p) > 0 for p in prompts)
        assert all(0 <= t < target.vocab_size for p in prompts for t in p)


def test_small_bench_report(target):
    cfg = BenchConfig(k_max=4, ks=(1, 2, 4), batch_sizes=(1, 64), prompts_per_workload=1, report_k=2)
    drafters = build_drafters(target, ("ngram", "draft-model"), log=lambda *a: None)
    workloads = [WORKLOADS["code-edit"], WORKLOADS["synthetic-random"]]
    results = run_acceptance(target, drafters, workloads, cfg, log=lambda *a: None)
    report = analyse(results, drafters, workloads, cfg)
    assert set(report["acceptance_length"]) == {"code-edit", "synthetic-random"}
    assert set(report["by_category"]) == {"realistic", "synthetic"}
    assert [e["batch"] for e in report["best_k"]["ngram"]] == [1, 64]
    assert len(report["rows"]) == 2 * 2 * 3 * 2
    json.dumps(report)  # serialisable
    md = to_markdown(report)
    assert "Best draft length" in md and "| ngram |" in md


def test_cli_vllm_config_and_whatif(capsys):
    assert main(["vllm-config", "ngram", "-k", "5", "--lookup-max", "4", "--target", "Qwen/Qwen3-8B"]) == 0
    out = capsys.readouterr().out
    assert '"prompt_lookup_max": 4' in out and "vllm serve Qwen/Qwen3-8B" in out
    assert main(["whatif", "--alpha", "0.6", "--ks", "1,3,5", "--batch-sizes", "1,256"]) == 0
    assert "best k" in capsys.readouterr().out
    assert main(["vllm-config", "preset", "--preset", "eagle3-qwen3-8b"]) == 0
    assert "eagle3" in capsys.readouterr().out
