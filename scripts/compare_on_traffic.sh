#!/usr/bin/env bash
# "Measure on your own traffic before trusting any speedup number."
#
# Replays a JSONL sample of your real requests against a baseline server and one or
# more speculative servers at several concurrencies, and reports throughput, TPOT,
# TTFT and the acceptance length scraped from each server's /metrics.
#
#   scripts/compare_on_traffic.sh my_traffic.jsonl \
#       baseline=http://localhost:8000/v1 ngram=http://localhost:8001/v1 eagle3=http://localhost:8002/v1
#
# Env: CONCURRENCY (default 1,4,16,64), LIMIT (default all), OUT (default results/replay)
set -euo pipefail
TRAFFIC=${1:?traffic JSONL required}
shift
[[ $# -ge 1 ]] || { echo "need at least one label=url endpoint" >&2; exit 2; }
EP_ARGS=()
for ep in "$@"; do EP_ARGS+=(--endpoint "$ep"); done
mkdir -p "$(dirname "${OUT:-results/replay}")"
LIMIT_ARGS=()
[[ -n "${LIMIT:-}" ]] && LIMIT_ARGS=(--limit "$LIMIT")
exec specdec replay --traffic "$TRAFFIC" "${EP_ARGS[@]}" \
  --concurrency "${CONCURRENCY:-1,4,16,64}" "${LIMIT_ARGS[@]}" --out "${OUT:-results/replay}"
