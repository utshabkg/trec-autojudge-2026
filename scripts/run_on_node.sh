#!/usr/bin/env bash
# Run a sequence of datasets on the compute node, detached from any ssh session.
#
#   bash scripts/run_on_node.sh launch rag25-gen ragtime25 dragun-repgen
#   bash scripts/run_on_node.sh launch --variant no-loto rag25-gen
#   bash scripts/run_on_node.sh launch --judge nuggetrecall rag25-gen
#   bash scripts/run_on_node.sh status
#
# --judge defaults to corroborate. A judge whose workflow.yml says uses_llm: true
# is pointed at the vLLM endpoint from scripts/serve_llm.sh, which must already be
# serving. The others get a deliberately dead endpoint, so an accidental LLM call
# fails loudly instead of quietly reaching some other service.
#
# `launch` starts a detached job and returns immediately: setsid gives it its own
# session and every file descriptor is redirected, so a dropped connection cannot
# take it down and ssh has nothing to wait on. Progress goes to node-run.log, one
# log per dataset, and a line per dataset with its exit code and wall time.
set -uo pipefail
cd "$(dirname "$0")/.."

ENV_BIN="${AUTOJUDGE_ENV_BIN:-$HOME/miniforge3/envs/autojudge/bin}"

case "${1:-}" in
launch)
    shift
    VARIANT="best"; JUDGE="corroborate"
    while [ "${1:-}" = "--variant" ] || [ "${1:-}" = "--judge" ]; do
        if [ "$1" = "--variant" ]; then VARIANT="$2"; else JUDGE="$2"; fi
        shift 2
    done
    [ $# -gt 0 ] || { echo "name at least one dataset"; exit 1; }
    [ -f "judges/$JUDGE/workflow.yml" ] || { echo "no judges/$JUDGE/workflow.yml"; exit 1; }
    setsid nohup bash "$0" _run "$JUDGE" "$VARIANT" "$@" > "node-run-$JUDGE.log" 2>&1 < /dev/null &
    echo "launched pid $! judge=$JUDGE variant=$VARIANT datasets: $*"
    ;;
_run)
    shift
    JUDGE="$1"; VARIANT="$2"; shift 2
    export PATH="$ENV_BIN:$PATH"
    export HF_HOME="${HF_HOME:-/media/20TB/shared-cache/huggingface}"
    # One cache directory per judge, so an LLM judge's prompt cache and Corroborate's
    # score cache never share a file.
    export CACHE_DIR="${CACHE_DIR:-./cache/$JUDGE}"
    if grep -qE "^uses_llm:[[:space:]]*true" "judges/$JUDGE/workflow.yml"; then
        eval "$(bash scripts/serve_llm.sh env)"
        # minima-llm allows 32 requests in flight by default, and vLLM here is
        # configured for 64 concurrent sequences. On the first rag25-gen run, vLLM
        # metrics showed exactly 31 running and 0 waiting: the client, not the GPU,
        # was the bottleneck. Match the server.
        export MAX_OUTSTANDING="${MAX_OUTSTANDING:-64}"
        export BATCH_NUM_WORKERS="${BATCH_NUM_WORKERS:-64}"
    else
        export OPENAI_MODEL=none OPENAI_BASE_URL=EMPTY OPENAI_API_KEY=EMPTY
    fi
    for ds in "$@"; do
        S=$SECONDS
        echo "=== $ds judge=$JUDGE variant=$VARIANT model=$OPENAI_MODEL start $(date -Is) ==="
        python run_all_datasets.py --workflow "judges/$JUDGE/workflow.yml" \
            --variant "$VARIANT" --dataset "$ds" --keep-going > "run-$JUDGE-$ds-$VARIANT.log" 2>&1
        echo "=== $ds exit $? after $((SECONDS - S))s ==="
    done
    echo "=== ALL DONE $(date -Is) ==="
    ;;
status)
    for f in node-run*.log; do [ -e "$f" ] && { echo "--- $f"; cat "$f"; }; done
    for f in run-*.log; do
        [ -e "$f" ] || continue
        last=$(tr '\r' '\n' < "$f" | grep -E "(Pooling|Judging) topics:" | tail -1)
        echo "  $f: ${last:-(starting)}"
    done
    pgrep -af "run_all_datasets" | grep -v pgrep | head -2 || echo "  nothing running"
    echo "  load: $(cat /proc/loadavg)"
    ;;
*)
    sed -n '2,12p' "$0"; exit 1
    ;;
esac
