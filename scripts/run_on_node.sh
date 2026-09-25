#!/usr/bin/env bash
# Run a sequence of datasets on the compute node, detached from any ssh session.
#
#   bash scripts/run_on_node.sh launch rag25-gen ragtime25 dragun-repgen
#   bash scripts/run_on_node.sh launch --variant no-loto rag25-gen
#   bash scripts/run_on_node.sh launch --judge nuggetrecall rag25-gen
#   bash scripts/run_on_node.sh launch --judge nuggetrecall --out output-qwen2.5-7b rag25-gen
#   bash scripts/run_on_node.sh status
#   bash scripts/run_on_node.sh stop
#
# --out sets the output root. Use a separate one whenever the model changes: the
# default root holds the submitted results, and the judge REUSES an existing
# <filebase>.nuggets.jsonl rather than regenerating it, so running a second model
# into the same directory would silently score it with the first model's answer
# keys as well as overwriting the first model's results.
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

# Every process that belongs to a judge run, matched on exact argv fields rather
# than a substring, so a command line that merely mentions these names (an ssh
# check, a grep) can never match itself, which bit four times.
#
# The auto-judge line is the one that matters most. run_all_datasets.py runs the
# judge as a CHILD process, `.../bin/auto-judge run ...`, whose command line
# contains neither run_all_datasets.py nor run_on_node.sh. So killing those two
# left the actual judge running as an orphan, still writing into the same log and
# the same output directory. It happened on 2026-09-25: a "stopped" 4,096-token
# run kept going for eleven minutes beside its replacement and overwrote the
# replacement's leaderboard. It is also the likeliest real cause of the two-judges
# collision on 2026-09-22.
judge_procs() {
    ps -u "$(id -u)" -o pid=,sid=,args= | awk '
        ($3=="bash" && $4 ~ /run_on_node\.sh$/ && $5=="_run") ||
        ($3 ~ /(^|\/)python[0-9.]*$/ && $4=="run_all_datasets.py") ||
        ($4 ~ /(^|\/)auto-judge$/ && $5=="run") ||
        ($3 ~ /(^|\/)auto-judge$/ && $4=="run") { print }'
}

case "${1:-}" in
launch)
    shift
    VARIANT="best"; JUDGE="corroborate"; OUT="output"
    while [ "${1:-}" = "--variant" ] || [ "${1:-}" = "--judge" ] || [ "${1:-}" = "--out" ]; do
        case "$1" in
            --variant) VARIANT="$2" ;;
            --judge) JUDGE="$2" ;;
            --out) OUT="$2" ;;
        esac
        shift 2
    done
    [ $# -gt 0 ] || { echo "name at least one dataset"; exit 1; }
    [ -f "judges/$JUDGE/workflow.yml" ] || { echo "no judges/$JUDGE/workflow.yml"; exit 1; }
    # Refuse to start beside an existing run. Two judges writing the same
    # node-run-$JUDGE.log and the same --out directory produced a leaderboard that
    # was half one model and half another, on top of a vLLM restarted under both.
    # It cost an hour to notice, because the only symptom was bad numbers.
    #
    # `grep -v "bash -c"` matters: an ssh command whose text mentions the script
    # matches a naive pgrep, so a check can report a phantom job, or hide a real
    # one behind a count you misread. Ask for the list, never just the count.
    RUNNING=$(judge_procs)
    if [ -n "$RUNNING" ]; then
        echo "refusing to launch, a run is already going:" >&2
        echo "$RUNNING" >&2
        echo "stop it first: bash scripts/run_on_node.sh stop" >&2
        exit 1
    fi
    setsid nohup bash "$0" _run "$JUDGE" "$VARIANT" "$OUT" "$@" > "node-run-$JUDGE.log" 2>&1 < /dev/null &
    echo "launched pid $! judge=$JUDGE variant=$VARIANT out=$OUT datasets: $*"
    ;;
stop)
    # Kill every judge process, the auto-judge children included (see
    # judge_procs). Each run is started with setsid, so its session holds the
    # wrapper, run_all_datasets.py and the judge. Killing the session gets all
    # three at once, and the list catches any orphan whose session is gone.
    for sid in $(judge_procs | awk '{print $2}' | sort -u); do
        pkill -TERM -s "$sid" 2>/dev/null || true
    done
    judge_procs | awk '{print $1}' | xargs -r kill -TERM 2>/dev/null || true
    sleep 4
    judge_procs | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
    sleep 2
    LEFT=$(judge_procs)
    [ -z "$LEFT" ] && echo "CLEAN: no judge processes" || { echo "STILL RUNNING:"; echo "$LEFT"; exit 1; }
    ;;
_run)
    shift
    JUDGE="$1"; VARIANT="$2"; OUT="$3"; shift 3
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
            --variant "$VARIANT" --dataset "$ds" --out-dir "$OUT" --keep-going \
            > "run-$JUDGE-$ds-$VARIANT-$(basename "$OUT").log" 2>&1
        echo "=== $ds exit $? after $((SECONDS - S))s ==="
    done
    echo "=== ALL DONE $(date -Is) ==="
    ;;
procs)
    # What stop would kill, and what launch would refuse on. Changes nothing.
    judge_procs
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
