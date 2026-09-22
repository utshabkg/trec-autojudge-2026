#!/usr/bin/env bash
# Move work between this PC and the compute node, in both directions.
#
# This PC is the single source of truth for code and git: commits and pushes
# happen here, where the GitHub credentials live, so no credential is ever copied
# to the compute node. The node runs jobs. Code goes out, results come back.
#
#   bash scripts/sync_remote.sh push        code and data to the node
#   bash scripts/sync_remote.sh pull        outputs back to this PC
#   bash scripts/sync_remote.sh pull-cache  also the score caches (large, optional)
#   bash scripts/sync_remote.sh status      what differs, changes nothing
#
# `push` never deletes anything on the node except stale code files. `pull` never
# deletes anything here. Both are rsync, so a repeat run only moves what changed.
set -euo pipefail

HOST="${AUTOJUDGE_REMOTE_HOST:-54}"
REMOTE="${AUTOJUDGE_REMOTE_DIR:-/media/20TB/projects/trec-autojudge-2026}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

# Code only. Data, caches and outputs are handled separately and deliberately,
# so a code push can never overwrite results produced on the node.
CODE_EXCLUDES=(
    --exclude local-data --exclude local-data-dev
    --exclude 'output*/' --exclude cache/ --exclude .hf/
    --exclude metaeval/ --exclude models/ --exclude fusion/
    --exclude __pycache__/ --exclude '*.egg-info/' --exclude .pytest_cache/
    # Anything produced ON the node must be listed above, or --delete removes it
    # simply because it does not exist here. vllm.log went first, from under a
    # live server. fusion/ went next, three built leaderboards, minutes before
    # they were due to be uploaded. If you make the node write somewhere new,
    # add it here in the same commit.
    --exclude '*.log'
)

case "${1:-}" in
push)
    echo ">> code -> $HOST:$REMOTE"
    # --delete on code only, so a file removed here is removed there. .git goes
    # too, which keeps the node a real repository (tira and two of the tests read
    # git metadata) but carries no credentials: those live in ~/.git-credentials.
    rsync -az --delete "${CODE_EXCLUDES[@]}" "$HERE/" "$HOST:$REMOTE/"

    echo ">> data -> $HOST:$REMOTE/local-data"
    # -L follows the local-data symlink so the node gets real files. No --delete:
    # data only ever grows.
    rsync -azL --info=progress2 "$HERE/local-data/" "$HOST:$REMOTE/local-data/"
    echo ">> done"
    ;;
pull)
    echo ">> outputs <- $HOST:$REMOTE"
    for d in output metaeval; do
        if ssh -o BatchMode=yes "$HOST" "test -d '$REMOTE/$d'"; then
            rsync -az --info=progress2 "$HOST:$REMOTE/$d/" "$HERE/$d/"
        fi
    done
    # Log files written by runs on the node.
    rsync -az --include='*.log' --exclude='*' "$HOST:$REMOTE/" "$HERE/" 2>/dev/null || true
    echo ">> done. Code changes made on the node are NOT pulled: make them here."
    ;;
pull-cache)
    "$0" pull
    echo ">> caches <- $HOST:$REMOTE/cache"
    rsync -az --info=progress2 "$HOST:$REMOTE/cache/" "$HERE/cache/"
    ;;
status)
    echo ">> code that differs (dry run, nothing changes)"
    rsync -azn --delete --itemize-changes "${CODE_EXCLUDES[@]}" "$HERE/" "$HOST:$REMOTE/" | head -40
    echo ">> outputs on the node"
    ssh -o BatchMode=yes "$HOST" "cd '$REMOTE' 2>/dev/null && find output -name '*.eval.txt' 2>/dev/null | sort" || true
    ;;
*)
    sed -n '2,16p' "$0"
    exit 1
    ;;
esac
