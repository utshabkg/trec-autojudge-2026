#!/usr/bin/env python3
"""Pre-upload guard: would tira accept this output directory?

The TIRA evaluator locates your leaderboard by scanning the run directory and
accepting any file whose lines split into exactly four whitespace-separated
tokens. A TREC qrels file (`topic 0 doc grade`) matches that shape, so shipping
qrels beside the leaderboard makes the upload fail with "No unique file in
trec-eval -q format was found" instead of scoring anything. Verified locally.

This runs tira's own checker rather than reimplementing it, so it cannot drift
from what the server does, and adds the checks tira does not make: that every
run and topic we expected is actually present, and that no measure is constant
across runs.

    python scripts/check_upload_dir.py output/rag26-generation/corroborate/best-all-all \
        --expect-runs 83 --expect-topics 119

Exit status is 0 only if the directory is safe to upload.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", type=Path)
    ap.add_argument("--expect-runs", type=int, default=None, help="required number of distinct run ids")
    ap.add_argument("--expect-topics", type=int, default=None,
                    help="required number of distinct topic ids, not counting the 'all' aggregate")
    ap.add_argument("--all-topic-id", default="all")
    args = ap.parse_args()

    problems: list[str] = []
    notes: list[str] = []

    if not args.directory.is_dir():
        print(f"FAIL: {args.directory} is not a directory")
        return 2

    # 1. tira's own verdict.
    try:
        from tira.check_format import FormatMsgType, TrecEvalLeaderboard

        checker = TrecEvalLeaderboard()
        checker.apply_configuration_and_throw_if_invalid(None)
        matches = checker.matching_files(args.directory)
        status, message = checker.check_format(args.directory)
        notes.append(f"tira sees {len(matches)} leaderboard-shaped file(s): "
                     f"{sorted(p.name for p in matches)}")
        # FormatMsgType is a plain Enum, not an IntEnum, so compare members.
        if status is not FormatMsgType.OK:
            problems.append(f"tira check_format ({status.name}): {message}")
    except ImportError:
        notes.append("tira not installed, skipped the authoritative check")
        matches = sorted(args.directory.glob("*eval.txt"))
        if len(matches) != 1:
            problems.append(f"expected exactly one *eval.txt, found {len(matches)}")

    # 2. Coverage and separation, which tira does not check.
    leaderboards = sorted(args.directory.glob("*eval.txt"))
    if len(leaderboards) != 1:
        problems.append(f"expected exactly one *eval.txt in the directory, found {len(leaderboards)}")
    if leaderboards:
        runs: set[str] = set()
        topics: set[str] = set()
        values: dict[str, set[str]] = defaultdict(set)
        bad = 0
        with open(leaderboards[0], encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) != 4:
                    if line.strip():
                        bad += 1
                    continue
                run, topic, measure, value = parts
                runs.add(run)
                if topic == args.all_topic_id:
                    values[measure].add(value)
                else:
                    topics.add(topic)
        if bad:
            problems.append(f"{bad} line(s) are not four whitespace-separated columns")
        notes.append(f"{len(runs)} runs, {len(topics)} topics, "
                     f"{len(values)} measures with aggregate rows")
        if not values:
            problems.append(f"no rows with topic_id == '{args.all_topic_id}': "
                            "correlations are computed only from the aggregate rows")
        if args.expect_runs is not None and len(runs) != args.expect_runs:
            problems.append(f"expected {args.expect_runs} runs, found {len(runs)}. "
                            "A partial submission cannot be accepted")
        if args.expect_topics is not None and len(topics) != args.expect_topics:
            problems.append(f"expected {args.expect_topics} topics, found {len(topics)}")
        for measure, vals in sorted(values.items()):
            if len(vals) <= 1:
                problems.append(f"measure {measure} is constant across all runs, "
                                "so it carries no ranking")
            elif len(vals) < max(3, 0.5 * len(runs)):
                notes.append(f"WARN measure {measure} has only {len(vals)} distinct "
                             f"aggregate values over {len(runs)} runs, ties hurt tauap_b")

    for n in notes:
        print(f"  {n}")
    if problems:
        print("\nFAIL, do not upload:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nOK, safe to upload.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
