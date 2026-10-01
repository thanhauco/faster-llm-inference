import json
import threading

import pytest

from specdec.drafters import NgramDrafter
from specdec.models import ToyLM
from specdec.serving import configs, prom, replay, traffic
from specdec.serving.client import Request, list_models, send
from specdec.serving.mock_server import MockEngine, serve


# ------------------------------------------------------------------ configs
def test_post_ngram_config_is_exact():
    assert configs.ngram_config(5, 4) == {"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}


def test_model_based_methods_need_a_model():
    with pytest.raises(ValueError):
        configs.speculative_config("eagle3", 3)
    with pytest.raises(ValueError):
        configs.speculative_config("bogus", 3)
    with pytest.raises(ValueError):
        configs.ngram_config(5, 2, prompt_lookup_min=3)
    with pytest.raises(ValueError):
        configs.speculative_config("ngram", 0)


def test_serve_command_is_shell_safe():
    cmd = configs.serve_command("Qwen/Qwen3-8B", configs.ngram_config())
    assert cmd.startswith("vllm serve Qwen/Qwen3-8B --port 8000 --speculative-config '")
    payload = cmd.split("--speculative-config ", 1)[1].strip("'")
    assert json.loads(payload)["method"] == "ngram"


def test_presets_are_valid():
    for name, (target, spec) in configs.PRESETS.items():
        assert target and spec["method"] in configs.METHODS, name


def test_speculators_checkpoint_needs_only_plain_serve():
    assert configs.speculators_serve_command("RedHatAI/Qwen3-8B-speculator.eagle3") == \
        "vllm serve RedHatAI/Qwen3-8B-speculator.eagle3 --port 8000"


# ------------------------------------------------------------------ metrics
METRICS = """
# HELP vllm:spec_decode_num_drafts_total Number of spec decoding drafts.
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} 100.0
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="m"} 500.0
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="m"} 250.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="0"} 80.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="1"} 60.0
vllm:num_requests_running{model_name="m"} 3
"""


def test_parse_and_diff_spec_metrics():
    snap = prom.SpecDecodeSnapshot.from_text(METRICS)
    assert snap.found
    assert snap.mean_acceptance_length == pytest.approx(3.5)
    assert snap.draft_acceptance_rate == pytest.approx(0.5)
    assert snap.per_position_acceptance() == [0.8, 0.6]
    later = prom.SpecDecodeSnapshot.from_text(METRICS.replace("100.0", "150.0").replace("250.0", "300.0"))
    delta = later - snap
    assert delta.num_drafts == 50 and delta.num_accepted_tokens == 50
    assert delta.mean_acceptance_length == pytest.approx(2.0)


def test_metrics_without_total_suffix_and_without_spec():
    text = "vllm:spec_decode_num_drafts 4\nvllm:spec_decode_num_accepted_tokens 6\n"
    assert prom.SpecDecodeSnapshot.from_text(text).mean_acceptance_length == pytest.approx(2.5)
    assert not prom.SpecDecodeSnapshot.from_text("vllm:num_requests_running 1\n").found


def test_metrics_url():
    assert prom.metrics_url("http://h:8000/v1") == "http://h:8000/metrics"
    assert prom.metrics_url("http://h:8000/") == "http://h:8000/metrics"


# ------------------------------------------------------------------ traffic
def test_traffic_loader_shapes(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"prompt": "hi", "max_tokens": 8},
        {"messages": [{"role": "user", "content": "yo"}], "temperature": 0.5},
        {"body": {"messages": [{"role": "user", "content": "x"}], "max_completion_tokens": 9, "top_p": 0.9}},
    ]) + "\n")
    reqs = traffic.load_jsonl(str(p), default_max_tokens=32)
    assert [r.is_chat for r in reqs] == [False, True, True]
    assert [r.max_tokens for r in reqs] == [8, 32, 9]
    assert reqs[1].temperature == 0.5 and reqs[2].extra == {"top_p": 0.9}
    with pytest.raises(ValueError):
        traffic.record_to_request({"foo": 1})


def test_sample_traffic_file_loads():
    reqs = traffic.load_jsonl("data/sample_traffic.jsonl")
    assert len(reqs) >= 5


# --------------------------------------------------------- end to end (mock)
@pytest.fixture(scope="module")
def servers():
    target = ToyLM()
    started = []
    urls = {}
    for label, drafter in (("baseline", None), ("ngram", NgramDrafter.vllm_default())):
        engine = MockEngine(target, drafter, k=5, time_scale=0.0)
        srv = serve(engine, "127.0.0.1", 0)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        started.append(srv)
        urls[label] = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    yield urls
    for srv in started:
        srv.shutdown()
        srv.server_close()


def test_client_streams_completion_and_chat(servers):
    url = servers["ngram"]
    assert list_models(url) == ["toy-target"]
    r = send(url, "toy-target", Request(prompt="abcabcabcabc", max_tokens=20), keep_text=True)
    assert r.ok and r.output_tokens == 20 and len(r.text) == 20
    assert r.num_chunks <= 20  # speculation streams several tokens per chunk
    r = send(url, "toy-target", Request(messages=[{"role": "user", "content": "hello"}], max_tokens=12))
    assert r.ok and r.output_tokens == 12


def test_speculation_is_lossless_through_the_server(servers):
    req = Request(prompt="the cat sat on the mat. the cat sat on", max_tokens=40)
    a = send(servers["baseline"], "toy-target", req, keep_text=True)
    b = send(servers["ngram"], "toy-target", req, keep_text=True)
    assert a.text == b.text


def test_replay_sweep_reports_acceptance(servers):
    reqs = [Request(prompt="abc " * 10, max_tokens=24), Request(prompt="xyz xyz xyz", max_tokens=16)] * 2
    report = replay.sweep(servers, reqs, [1, 2], warmup=1, log=lambda *a: None)
    rows = report["results"]
    assert len(rows) == 4 and all(r["errors"] == 0 for r in rows)
    spec_rows = [r for r in rows if r["endpoint"] == "ngram"]
    assert all(r["spec_decode"]["mean_acceptance_length"] >= 1.0 for r in spec_rows)
    assert "throughput_speedup" in spec_rows[0]
    assert "| ngram |" in replay.to_markdown(report)


def test_client_reports_http_errors(servers):
    bad = servers["baseline"].replace("/v1", "/nope")
    r = send(bad, "toy-target", Request(prompt="a", max_tokens=2))
    assert not r.ok and "404" in r.error


def test_percentile():
    assert replay.percentile([], 50) is None
    assert replay.percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
