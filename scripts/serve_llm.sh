#!/usr/bin/env bash
# Serve the judge LLM on the compute node with vLLM, detached from any ssh session.
#
#   bash scripts/serve_llm.sh start     launch, then return
#   bash scripts/serve_llm.sh wait      block until the endpoint answers
#   bash scripts/serve_llm.sh env       print the OPENAI_* exports for the judge
#   bash scripts/serve_llm.sh stop
#
# Qwen3-30B-A3B-Instruct-2507 in FP8: a mixture-of-experts model with about 3B
# active parameters, so it is fast, and 30 GB of weights, so it fits one 48 GB RTX
# 6000 Ada with room left for batching. It is the non-thinking instruct release, so
# replies carry no reasoning block. The organizers re-run code submissions against
# their own model (gpt-oss-20b, Qwen2.5-7B), injected through OPENAI_MODEL, so
# nothing here is baked into the judge.
set -uo pipefail
cd "$(dirname "$0")/.."

VLLM="${VLLM_BIN:-$HOME/miniforge3/envs/llama-serve/bin/vllm}"
MODEL="${JUDGE_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}"
PORT="${JUDGE_PORT:-8000}"
export HF_HOME="${HF_HOME:-/media/20TB/shared-cache/huggingface}"

case "${1:-}" in
start)
    if curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        echo "already serving on port $PORT"; exit 0
    fi
    # HF_HUB_OFFLINE: the weights are already cached, and a serving process should
    # not wander onto the network. --served-model-name is the full id, so it lands
    # verbatim in each run's config.yml and in TIRA's Model column.
    HF_HUB_OFFLINE=1 setsid nohup "$VLLM" serve "$MODEL" \
        --served-model-name "$MODEL" \
        --host 127.0.0.1 --port "$PORT" \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.90 \
        --max-num-seqs 64 \
        > vllm.log 2>&1 < /dev/null &
    echo "vLLM starting, pid $!, model $MODEL, log vllm.log"
    ;;
wait)
    for i in $(seq 1 180); do
        if curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q '"id"'; then
            echo "ready after ~$((i * 5))s"; exit 0
        fi
        if ! pgrep -f "vllm serve" >/dev/null; then
            echo "vLLM exited, last log lines:"; tail -20 vllm.log; exit 1
        fi
        sleep 5
    done
    echo "timed out waiting for vLLM"; tail -20 vllm.log; exit 1
    ;;
env)
    echo "export OPENAI_BASE_URL=http://127.0.0.1:$PORT/v1"
    echo "export OPENAI_MODEL=$MODEL"
    echo "export OPENAI_API_KEY=EMPTY"
    ;;
stop)
    pkill -f "vllm serve" && echo "stopped" || echo "not running"
    ;;
*)
    sed -n '2,8p' "$0"; exit 1
    ;;
esac
