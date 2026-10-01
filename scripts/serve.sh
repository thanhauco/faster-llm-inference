#!/usr/bin/env bash
# Serve a target model with vLLM, with or without speculative decoding.
#
#   scripts/serve.sh <target-model> [config.json|none] [port] [extra vllm args...]
#
# Examples:
#   scripts/serve.sh Qwen/Qwen3-8B none 8000                       # baseline
#   scripts/serve.sh Qwen/Qwen3-8B configs/ngram.json 8001         # n-gram, k=5, lookup 4
#   scripts/serve.sh meta-llama/Llama-3.1-8B-Instruct configs/eagle3.json 8002
#
# Add --per-request-spec-decode-metrics summary (newer vLLM) to get acceptance per request.
set -euo pipefail
MODEL=${1:?target model required}
CONFIG=${2:-none}
PORT=${3:-8000}
shift $(( $# < 3 ? $# : 3 ))

ARGS=(serve "$MODEL" --port "$PORT")
if [[ "$CONFIG" != "none" ]]; then
  ARGS+=(--speculative-config "$(tr -d '\n' < "$CONFIG")")
fi
echo "+ vllm ${ARGS[*]} $*"
exec vllm "${ARGS[@]}" "$@"
