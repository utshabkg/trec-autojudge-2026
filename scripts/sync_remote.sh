#!/usr/bin/env bash
# Move work between this PC and the compute node, in both directions.
#
# This PC is the single source of truth for code and git: commits and pushes
# happen here, where the GitHub credentials live, so no credential is ever copied
# to the compute node. The node runs jobs. Code goes out, results come back.
#
#   bash scripts/sync_remote.sh push        code and data to the node
#   bash scripts/sync_remote.sh push --preview   show what push would change
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
    --exclude local-data --exclude local-data-dev --exclude results/
    --exclude 'output*/' --exclude cache/ --exclude .hf/
    --exclude metaeval/ --exclude models/ --exclude fusion/
    --exclude __pycache__/ --exclude '*.egg-info/' --exclude .pytest_cache/
    --exclude '*.log'
    # Claude Code's own machine-local files: permission grants and a runtime lock.
    --exclude .claude/settings.json --exclude .claude/settings.local.json
    --exclude .claude/scheduled_tasks.lock
)

# Why push no longer runs --delete over the whole tree.
#
# It used to, protected only by the exclude list above, so anything the node
# wrote that nobody had remembered to list was deleted by the next code push,
# simply because it did not exist here. That happened three times: vllm.log from
# under a live server, then fusion/ minutes before its leaderboards were due to
# be uploaded, then upload-rank/. A list someone must remember to extend fails
# again the next time someone forgets. So --delete now runs only inside
# directories that git tracks. A code file removed here is still removed there,
# and anything the node creates anywhere else cannot be touched by a push.
tracked_dirs() {
    # `if`, not `[ -d ] && echo`: the && form leaves the loop's status at 1 when the
    # last entry is a file, and `tracked=$(tracked_dirs)` under set -e then exits
    # the script silently. That is exactly how the first new pull did nothing while
    # push, which calls this inside a for list where set -e does not apply, worked.
    git -C "$HERE" ls-files | cut -d/ -f1 | sort -u | while read -r p; do
        if [ -d "$HERE/$p" ]; then echo "$p"; fi
    done
}

case "${1:-}" in
push)
    DRY=()
    [ "${2:-}" = "--preview" ] && { DRY=(-n -i); echo ">> PREVIEW, nothing is changed"; }
    echo ">> code -> $HOST:$REMOTE"
    # Everything, with no --delete, so nothing on the node is ever removed here.
    rsync -az "${DRY[@]}" "${CODE_EXCLUDES[@]}" "$HERE/" "$HOST:$REMOTE/"
    # Then --delete, but only inside directories git tracks. See tracked_dirs.
    for d in $(tracked_dirs); do
        rsync -az --delete "${DRY[@]}" "${CODE_EXCLUDES[@]}" "$HERE/$d/" "$HOST:$REMOTE/$d/"
    done
    # .git mirrors exactly, which keeps the node a real repository (tira and two
    # of the tests read git metadata). It carries no credentials: those live in
    # ~/.git-credentials here, and checkout never writes them into .git/config.
    rsync -az --delete "${DRY[@]}" "$HERE/.git/" "$HOST:$REMOTE/.git/"
    [ ${#DRY[@]} -gt 0 ] && { echo ">> preview done"; exit 0; }

    echo ">> data -> $HOST:$REMOTE/local-data"
    # -L follows the local-data symlink so the node gets real files. No --delete:
    # data only ever grows.
    rsync -azL --info=progress2 "$HERE/local-data/" "$HOST:$REMOTE/local-data/"
    echo ">> done"
    ;;
pull)
    echo ">> outputs <- $HOST:$REMOTE"
    # Every top-level directory the node created, meaning anything git does not
    # track, minus inputs and caches. Not a list of names: the first version
    # pulled output and metaeval only, and the second a list of prefixes, and
    # both stranded the next new folder (output-gate, fusion, then ensemble).
    # This mirrors push, which by the same rule never deletes such a folder.
    tracked=" $(tracked_dirs | tr '\n' ' ') "
    for d in $(ssh -o BatchMode=yes "$HOST" \
            "cd '$REMOTE' && find . -mindepth 1 -maxdepth 1 -type d -printf '%f\n'" </dev/null); do
        case "$d" in
            .git|local-data*|cache|.hf|models|__pycache__|.pytest_cache|*.egg-info) continue ;;
        esac
        [[ "$tracked" == *" $d "* ]] && continue
        echo "   $d"
        # Caches live in results/cache since the 2026-09-26 cleanup. They are large
        # (6.5 GB), so a plain pull skips them and pull-cache fetches them.
        rsync -az --exclude 'cache/' "$HOST:$REMOTE/$d/" "$HERE/$d/"
    done
    echo ">> done. Code changes made on the node are NOT pulled: make them here."
    ;;
pull-cache)
    "$0" pull
    echo ">> caches <- $HOST:$REMOTE/results/cache"
    rsync -az --info=progress2 "$HOST:$REMOTE/results/cache/" "$HERE/results/cache/"
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
