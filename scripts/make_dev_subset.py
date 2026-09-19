#!/usr/bin/env python3
"""Build a small dev subset so iteration does not cost a full parse every time.

`load_runs_failsave` reads and pydantic-parses every line of every run file
before anything is filtered, so `--limit-topics` and `--topic` do not save load
time: on rag26 that is 15 to 25 minutes per iteration. A physical subset on disk
is what makes the development loop usable.

The selected topics are the FIRST N in the dataset's topics file, in file order,
which is exactly the window the data policy makes inspectable. Selection reads
only metadata (topic ids), never report text. Sorting, sampling or reordering
would change which topics are permitted, so this does none of those.

    python scripts/make_dev_subset.py \\
        --topics local-data/rag26/topics/topics.jsonl \\
        --runs   local-data/rag26/runs/generation \\
        --out    local-data-dev/rag26 --n 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topics", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=10, help="how many leading topics to keep")
    args = ap.parse_args()

    keep: list[str] = []
    with open(args.topics, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tid = str(obj.get("request_id") or obj.get("topic_id") or obj.get("narrative_id"))
            keep.append(tid)
            if len(keep) >= args.n:
                break
    if not keep:
        print(f"FAIL: no topics parsed from {args.topics}")
        return 2
    keep_set = set(keep)

    task = args.runs.name
    out_runs = args.out / "runs" / task
    out_topics = args.out / "topics"
    out_runs.mkdir(parents=True, exist_ok=True)
    out_topics.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(args.topics, encoding="utf-8") as src, \
         open(out_topics / args.topics.name, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            obj = json.loads(line)
            tid = str(obj.get("request_id") or obj.get("topic_id") or obj.get("narrative_id"))
            if tid in keep_set:
                dst.write(line if line.endswith("\n") else line + "\n")
                written += 1

    files = sorted(p for p in args.runs.glob("*") if p.is_file())
    kept_records = 0
    for f in files:
        with open(f, encoding="utf-8") as src, open(out_runs / f.name, "w", encoding="utf-8") as dst:
            for line in src:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                md = d.get("metadata") or {}
                tid = str(md.get("topic_id") or md.get("narrative_id") or md.get("request_id"))
                if tid in keep_set:
                    dst.write(line if line.endswith("\n") else line + "\n")
                    kept_records += 1

    print(f"topics kept   : {written} ({', '.join(keep)})")
    print(f"run files     : {len(files)}")
    print(f"records kept  : {kept_records}")
    print(f"written to    : {args.out}")
    print(f"\nauto-judge run --workflow judges/corroborate/workflow.yml --variant best \\")
    print(f"    --rag-responses {out_runs}/ \\")
    print(f"    --rag-topics {out_topics / args.topics.name} \\")
    print(f"    --out-dir ./output-dev/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
