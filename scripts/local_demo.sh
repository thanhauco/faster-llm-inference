#!/usr/bin/env bash
# GPU-free end-to-end demo of the measurement workflow using the toy mock server:
# baseline vs n-gram vs EAGLE-3-style, replayed on data/sample_traffic.jsonl.
set -euo pipefail
cd "$(dirname "$0")/.."
PIDS=()
cleanup() { kill "${PIDS[@]}" 2>/dev/null || true; }
trap cleanup EXIT
specdec mock-server --port 18000 --time-scale 0.5 & PIDS+=($!)
specdec mock-server --port 18001 --method ngram -k 5 --time-scale 0.5 & PIDS+=($!)
specdec mock-server --port 18002 --method eagle3 -k 4 --time-scale 0.5 & PIDS+=($!)
for port in 18000 18001 18002; do
  for _ in $(seq 1 60); do
    python3 -c "import urllib.request,sys; urllib.request.build_opener(urllib.request.ProxyHandler({})).open('http://127.0.0.1:$port/v1/models', timeout=1)" 2>/dev/null && break
    sleep 0.5
  done
done
mkdir -p results
specdec replay --traffic data/sample_traffic.jsonl \
  --endpoint baseline=http://127.0.0.1:18000/v1 \
  --endpoint ngram=http://127.0.0.1:18001/v1 \
  --endpoint eagle3=http://127.0.0.1:18002/v1 \
  --concurrency 1,4 --out results/local_demo
specdec metrics --endpoint http://127.0.0.1:18002/v1
