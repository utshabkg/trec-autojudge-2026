#!/usr/bin/env python3
"""Fuse two judges' leaderboards into one, by within-topic rank.

Corroborate and NuggetRecall disagree more than they agree (Kendall 0.50 on
ragtime25, 0.23 on rag25-gen), and each fails where its own assumption breaks.
That is the textbook case for fusion: two noisy rankers whose errors are not the
same errors. This script tests whether combining them beats either alone.

It works on the judges' `.eval.txt` outputs alone, so it needs no re-run and no
LLM. Those files hold run ids, topic ids and scores, all of which are readable
under the data policy on any topic.

Fusion is on within-topic percentile rank rather than on the raw scores. The two
judges' scores are on different scales with different spreads, so averaging them
directly would just hand the ranking to whichever has the larger variance. Rank
also removes topic difficulty, which is the dominant noise term when ordering
systems. Percentile rank keeps the result continuous, which matters because
tauap_b breaks ties by input order.

Usage:
    python scripts/fuse_leaderboards.py \
        --left  output/ragtime25/corroborate/best-all-all/best.eval.txt \
        --left-measure NF1 \
        --right output/ragtime25/nuggetrecall/best-all-all/best.eval.txt \
        --right-measure NR_VITAL \
        --weight 0.5 --name FUSE --out fusion/ragtime25-w50/fusion.eval.txt
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

AGG = "all"


def read_eval(path: Path, measure: str) -> Dict[str, Dict[str, float]]:
    """topic_id -> run_id -> value, aggregate rows dropped."""
    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    with path.open() as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 4:
                continue
            run, topic, name, value = parts
            if name != measure or topic == AGG:
                continue
            try:
                out[topic][run] = float(value)
            except ValueError:
                continue
    if not out:
        raise SystemExit(f"{path}: no per-topic rows for measure {measure}")
    return out


def percentiles(scores: Dict[str, float]) -> Dict[str, float]:
    """|{r' : v(r') < v(r)}| / (n-1), so worst is 0.0 and best is 1.0.

    Ties share the value below them, which is what keeps a judge that cannot
    separate two systems from inventing an order between them.
    """
    n = len(scores)
    if n < 2:
        return {r: 0.0 for r in scores}
    values = sorted(scores.values())
    # index of the first element not less than v, which is exactly the count below
    below = {}
    for run, v in scores.items():
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if values[mid] < v:
                lo = mid + 1
            else:
                hi = mid
        below[run] = lo / (n - 1)
    return below


def fuse(left: Dict[str, Dict[str, float]], right: Dict[str, Dict[str, float]],
         weight: float) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    topics = sorted(set(left) & set(right))
    if not topics:
        raise SystemExit("the two leaderboards share no topics")
    # Only judges that actually carry weight get a say in which runs survive.
    # At weight 1.0 this is the pure right-hand judge re-aggregated by rank, which
    # is the control that separates "fusion helped" from "rank aggregation
    # helped", and it must not lose runs to a judge contributing nothing.
    sides = []
    if weight < 1.0:
        sides.append(left)
    if weight > 0.0:
        sides.append(right)

    # Every run a contributing judge scored on any topic. Not the intersection:
    # dropping a run makes the leaderboard incomplete, which TIRA rejects outright
    # ("The all entry for run X is missing") and which check_upload_dir cannot
    # see, because it only inspects what is present. A run absent on a topic
    # scores 0.0 there, which is the framework's own fix_aggregate convention and
    # is why this fuses percentiles rather than z-scores: for a percentile 0.0
    # correctly means "worst", where for a z-score it would wrongly mean "average".
    runs = sorted(set().union(*[set(s[t]) for s in sides for t in topics]))

    per_topic: Dict[str, Dict[str, float]] = {}
    filled = 0
    for t in topics:
        lp, rp = percentiles(left[t]), percentiles(right[t])
        per_topic[t] = {}
        for r in runs:
            if not all(r in s[t] for s in sides):
                filled += 1
            per_topic[t][r] = (1.0 - weight) * lp.get(r, 0.0) + weight * rp.get(r, 0.0)
    if filled:
        print(f"  note: {filled} of {len(runs) * len(topics)} cells were absent from "
              f"a contributing judge and scored 0.0, as fix_aggregate would")

    agg = {r: sum(per_topic[t][r] for t in topics) / len(topics) for r in runs}
    return per_topic, agg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--left", required=True, type=Path)
    p.add_argument("--left-measure", required=True)
    p.add_argument("--right", required=True, type=Path)
    p.add_argument("--right-measure", required=True)
    p.add_argument("--weight", type=float, default=0.5,
                   help="weight on the right judge, 0.0 to 1.0 (default 0.5)")
    p.add_argument("--name", default="FUSE", help="measure name to emit")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    if not 0.0 <= a.weight <= 1.0:
        raise SystemExit("--weight must be between 0.0 and 1.0")

    left = read_eval(a.left, a.left_measure)
    right = read_eval(a.right, a.right_measure)
    per_topic, agg = fuse(left, right, a.weight)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as fh:
        for t in sorted(per_topic):
            for r in sorted(per_topic[t]):
                fh.write(f"{r}\t{t}\t{a.name}\t{per_topic[t][r]!r}\n")
        for r in sorted(agg):
            fh.write(f"{r}\t{AGG}\t{a.name}\t{agg[r]!r}\n")

    distinct = len(set(round(v, 12) for v in agg.values()))
    print(f"{a.out}: {len(agg)} runs, {len(per_topic)} topics, "
          f"{distinct} distinct aggregate values")
    if distinct < len(agg):
        print(f"  warning: {len(agg) - distinct} ties at the top level, which "
              f"tauap_b breaks by input order")


if __name__ == "__main__":
    main()
