#!/usr/bin/env bash
# Unattended finish of the rag26 ensemble submission, run on the compute node so it
# survives the assistant's session ending. It uploads ONLY if every check passes,
# the same checks that would be run by hand, and it never uploads twice.
#
#   1. wait for the gpt-oss rag26 run to finish
#   2. gate on its unreadable-reply and fallback-key rates
#   3. check the gpt-oss leaderboard is complete and varies
#   4. build the equal-weight Qwen3 + gpt-oss ensemble, exactly as on the pilots
#   5. check the ensemble is complete and varies
#   6. upload once, then free the GPU
set -uo pipefail
cd /media/20TB/projects/trec-autojudge-2026 || exit 1
export PATH=/home/ugdf8/miniforge3/envs/autojudge/bin:$PATH
LOG=finish-rag26-ensemble.log
exec >>"$LOG" 2>&1
MARK=ensemble/.rag26-uploaded
abort() { echo "ABORT: $*"; echo "=== end $(date -Is), NOTHING uploaded"; exit 1; }

echo "=== start $(date -Is)"
[ -f "$MARK" ] && { echo "already uploaded earlier, nothing to do"; exit 0; }

until grep -q "ALL DONE" node-run-nuggetrecall.log && grep -q "rag26-generation" node-run-nuggetrecall.log; do
    sleep 60
done
echo "gpt-oss run finished $(date -Is)"
tr -d '\000' < node-run-nuggetrecall.log | tr '\r' '\n' | grep -a '^==='
grep -q "rag26-generation exit 0" node-run-nuggetrecall.log || abort "the gpt-oss run did not exit 0"

L=run-nuggetrecall-rag26-generation-best-output-gptoss.log
flat() { tr -d '\000' < "$L" | tr '\r' '\n'; }
# FIRST match, not last. On 2026-09-25 an orphaned judge from a stopped run kept
# writing into this same log at its old offset, so the file holds the new run's
# output from the start and the orphan's leftover text after it, including its own
# summary lines. The new run writes from offset 0, so its line always comes first.
UNP=$(flat | grep -aoE '[0-9]+ of [0-9]+ judgments stayed unparsable' | head -1)
KEY=$(flat | grep -aoE '[0-9]+ of [0-9]+ topics produced no usable answer key' | head -1)
FAIL=$(flat | grep -aoE '[0-9]+ of [0-9]+ LLM calls failed' | head -1)
echo "unparsable:    ${UNP:-none reported}"
echo "fallback keys: ${KEY:-none reported}"
echo "failed calls:  ${FAIL:-none reported}"
# Pilot rates were 0.75% unreadable on rag25-gen and about 3% on ragtime. 10% on
# either gate means something went wrong, not that the model had a bad day.
rate_ok() { [ -z "$1" ] && return 0; set -- $1; awk -v a="$1" -v b="$3" 'BEGIN{exit !(b>0 && a/b<0.10)}'; }
rate_ok "$UNP"  || abort "unreadable rate is 10% or more: $UNP"
rate_ok "$KEY"  || abort "fallback-key rate is 10% or more: $KEY"
rate_ok "$FAIL" || abort "failed-call rate is 10% or more: $FAIL"

python scripts/check_upload_dir.py output-gptoss/rag26-generation/nuggetrecall/best-all-all \
    --expect-runs 83 --expect-topics 119 || abort "the gpt-oss leaderboard failed its check"

python scripts/fuse_leaderboards.py \
    --left  output/rag26-generation/nuggetrecall/best-all-all/best.eval.txt --left-measure NR_VITAL \
    --right output-gptoss/rag26-generation/nuggetrecall/best-all-all/best.eval.txt --right-measure NR_VITAL \
    --weight 0.5 --name NR_ENSEMBLE --out ensemble/rag26-generation/fusion.eval.txt \
    || abort "fusion failed"
python scripts/check_upload_dir.py ensemble/rag26-generation \
    --expect-runs 83 --expect-topics 119 || abort "the ensemble failed its check"

OUT=$(tira-cli upload --dataset rag26-20260827_1-test --directory ensemble/rag26-generation \
      --system ensemble-qwen-gptoss 2>&1)
echo "$OUT" | tail -5
if echo "$OUT" | grep -q "Done. Your run is available"; then
    date -Is > "$MARK"
    echo "UPLOADED ensemble-qwen-gptoss to rag26-20260827_1-test"
else
    abort "tira-cli upload did not confirm"
fi

# Nothing else needs the model server, and it held the only GPU idle for two days
# once before.
bash scripts/serve_llm.sh stop
echo "=== end $(date -Is)"
