#!/usr/bin/env bash
# "Measure on your own traffic before trusting any speedup number."
#
# 1. Export ~200-1000 real requests to JSONL (see data/sample_traffic.jsonl for the format).
# 2. Start a baseline server and a speculative one on the same GPU type.
# 3. Replay at the concurrencies you actually run, compare, and read acceptance from /metrics.
set -euo pipefail
TARGET=${TARGET:-Qwen/Qwen3-8B}
TRAFFIC=${TRAFFIC:-data/sample_traffic.jsonl}

cat <<MSG
Start these in separate terminals (or on separate GPUs):

  scripts/serve.sh $TARGET none 8000
  scripts/serve.sh $TARGET configs/ngram.json 8001

Then press enter to replay $TRAFFIC.
MSG
read -r _
CONCURRENCY=${CONCURRENCY:-1,4,16,64} scripts/compare_on_traffic.sh "$TRAFFIC" \
  baseline=http://localhost:8000/v1 ngram=http://localhost:8001/v1
specdec metrics --endpoint http://localhost:8001/v1
