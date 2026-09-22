#!/usr/bin/env bash
# Run a sequence of datasets on the compute node, detached from any ssh session.
#
#   bash scripts/run_on_node.sh launch rag25-gen ragtime25 dragun-repgen
#   bash scripts/run_on_node.sh launch --variant no-loto rag25-gen
#   bash scripts/run_on_node.sh status
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
    VARIANT="best"
    if [ "${1:-}" = "--variant" ]; then VARIANT="$2"; shift 2; fi
    [ $# -gt 0 ] || { echo "name at least one dataset"; exit 1; }
    setsid nohup bash "$0" _run "$VARIANT" "$@" > node-run.log 2>&1 < /dev/null &
    echo "launched pid $! variant=$VARIANT datasets: $*"
    ;;
_run)
    shift
    VARIANT="$1"; shift
    export PATH="$ENV_BIN:$PATH"
    export HF_HOME="${HF_HOME:-/media/20TB/shared-cache/huggingface}"
    export CACHE_DIR="${CACHE_DIR:-./cache/pilots}"
    export OPENAI_MODEL=none OPENAI_BASE_URL=EMPTY OPENAI_API_KEY=EMPTY
    for ds in "$@"; do
        S=$SECONDS
        echo "=== $ds variant=$VARIANT start $(date -Is) ==="
        python run_all_datasets.py --workflow judges/corroborate/workflow.yml \
            --variant "$VARIANT" --dataset "$ds" --keep-going > "run-$ds-$VARIANT.log" 2>&1
        echo "=== $ds exit $? after $((SECONDS - S))s ==="
    done
    echo "=== ALL DONE $(date -Is) ==="
    ;;
status)
    cat node-run.log 2>/dev/null || echo "(no node-run.log yet)"
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
