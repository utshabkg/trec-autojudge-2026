# One-off scripts

Kept as a record of how two submissions were actually made. They are not part of
the normal workflow, and they use the file layout from before 2026-09-26, when run
outputs sat at the repository root rather than under `results/`.

- `finish_rag26_ensemble.sh` waited for the gpt-oss run on rag26, gated it on its
  unreadable-reply rate, built the Qwen3 plus gpt-oss ensemble and uploaded it once.
- `retry_rag26_upload.sh` retried that upload every 10 minutes while TIRA's API was
  returning HTTP 500 on 2026-09-25.

Both ran from the compute node's project root. See docs/WORK_LOG.md, 2026-09-25.
