#!/usr/bin/env bash
# Retry the rag26 ensemble upload while TIRA's API is returning HTTP 500.
#
# The ensemble is already built and verified (byte-identical to a fresh rebuild,
# 83 runs over 119 topics, check_upload_dir green). Only the upload failed, twice,
# on TIRA's side: tira-cli dies in metadata_for_task, its first API call, because
# the server answers with an error page instead of JSON. So: try every 10 minutes
# for up to 12 hours, upload at most once, and write the marker only when tira-cli
# itself confirms "Done. Your run is available".
set -uo pipefail
cd /media/20TB/projects/trec-autojudge-2026 || exit 1
export PATH=/home/ugdf8/miniforge3/envs/autojudge/bin:$PATH
LOG=retry-rag26-upload.log
exec >>"$LOG" 2>&1
MARK=ensemble/.rag26-uploaded
echo "=== start $(date -Is)"
for i in $(seq 1 72); do
    [ -f "$MARK" ] && { echo "marker present, already uploaded, stopping"; exit 0; }
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 60 \
        "https://www.tira.io/api/evaluations/trec-auto-judge/rag26-20260827_1-test")
    echo "try $i at $(date -Is): TIRA API HTTP $code"
    if [ "$code" = "200" ]; then
        OUT=$(tira-cli upload --dataset rag26-20260827_1-test \
              --directory ensemble/rag26-generation --system ensemble-qwen-gptoss 2>&1)
        echo "$OUT" | grep -E "Done|✖|Error" | tail -2
        if echo "$OUT" | grep -q "Done. Your run is available"; then
            date -Is > "$MARK"
            echo "UPLOADED ensemble-qwen-gptoss to rag26-20260827_1-test"
            echo "UPLOADED by retry-rag26-upload.sh at $(date -Is)" >> finish-rag26-ensemble.log
            exit 0
        fi
    fi
    sleep 600
done
echo "GAVE UP after 12 hours, nothing uploaded"
