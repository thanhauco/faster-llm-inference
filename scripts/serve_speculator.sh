#!/usr/bin/env bash
# Resource #4: a speculators checkpoint (EAGLE-3, DFlash, DSpark, P-EAGLE, MTP...) is
# self-describing. vLLM reads `speculators_config` from its config.json, loads the
# verifier named there and turns on speculative decoding - a plain `vllm serve` is enough.
#
#   scripts/serve_speculator.sh RedHatAI/Qwen3-8B-speculator.eagle3 [port]
#   scripts/serve_speculator.sh ./output/checkpoints/checkpoint_best 8001
set -euo pipefail
SPECULATOR=${1:?speculator checkpoint (HF id or local path) required}
PORT=${2:-8000}
shift $(( $# < 2 ? $# : 2 ))
echo "+ vllm serve $SPECULATOR --port $PORT $*"
exec vllm serve "$SPECULATOR" --port "$PORT" "$@"
