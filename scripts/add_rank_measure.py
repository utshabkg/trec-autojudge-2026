#!/usr/bin/env python3
"""Add NR_RANK to a leaderboard the judge already produced, without re-judging.

The judge emits NR_RANK natively now, but our lodged test-set submissions predate
it. Re-running the judge would add the measure, and would also re-ask the LLM for
anything not in the prompt cache, and vLLM is not bitwise deterministic across
batch compositions. Measured: re-running NuggetRecall on ragtime25 with only an
added measure moved max Kendall by -0.010, which adding a measure cannot do, since
the reported score is a maximum. So a re-run carries about 0.01 of noise in both
directions.

Deriving the measure from the existing .eval.txt avoids that entirely. The
judgments are exactly the ones already submitted, the four original measures come
through byte for byte, and NR_RANK is computed by the same helper the judge uses,
so the result is what the judge would have emitted for that run.

This reads only .eval.txt, which holds run ids, topic ids and scores. All three
are readable under the data policy on any topic.

    python scripts/add_rank_measure.py output/rag26-generation/.../best.eval.txt \
        --out upload/rag26-rank/best.eval.txt
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from judges.nuggetrecall.measures import (  # noqa: E402
    MEASURE_NAMES, RANK_BASE, within_topic_percentile,
)

AGG = "all"
NEW = "NR_RANK"


def read(path: Path) -> Tuple[List[Tuple[str, str, str, str]], Dict[str, Dict[str, float]]]:
    """Every row verbatim, plus topic -> run -> RANK_BASE value (aggregates excluded)."""
    rows: List[Tuple[str, str, str, str]] = []
    base: Dict[str, Dict[str, float]] = defaultdict(dict)
    with path.open() as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 4:
                continue
            run, topic, measure, value = parts
            rows.append((run, topic, measure, value))
            if measure == RANK_BASE and topic != AGG:
                base[topic][run] = float(value)
    if not rows:
        raise SystemExit(f"{path}: no 4-column rows")
    if not base:
        raise SystemExit(f"{path}: no per-topic {RANK_BASE} rows to rank")
    return rows, base


def compute_rank(base: Dict[str, Dict[str, float]]):
    """(topic -> run -> percentile, run -> aggregate), matching the judge exactly.

    Verified bit for bit against the judge's own NR_RANK on rag25-gen: strip the
    measure from its output, recompute here, and every one of the 51 aggregates
    agrees to twelve decimal places. Getting there needed both halves right, and
    they do NOT use the same convention:

    Per topic, rank over exactly the runs that have a row on that topic. The
    leaderboards are ragged, 51 runs on most rag25-gen topics but 50 on twenty of
    them and 48 on one. Padding to a full grid with 0.0 looks reasonable and is
    wrong: it changes the denominator from (present - 1) to (all - 1) and inserts
    an extra bottom entry, shifting every percentile on that topic.

    The aggregate is not the judge's own work. LeaderboardBuilder.build computes
    it under on_missing="fix_aggregate", which counts a topic the run is absent
    from as 0.0 and still divides by every topic. So here padding IS correct.
    """
    topics = sorted(base)
    per_topic = {t: within_topic_percentile(base[t]) for t in topics}
    runs = sorted({r for t in topics for r in base[t]})
    agg = {r: sum(per_topic[t].get(r, 0.0) for t in topics) / len(topics) for r in runs}
    return per_topic, agg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("eval_txt", type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()

    rows, base = read(a.eval_txt)
    present = {m for _, _, m, _ in rows}
    if NEW in present:
        raise SystemExit(f"{a.eval_txt}: already has {NEW}, nothing to do")
    missing = set(MEASURE_NAMES) - present - {NEW}
    if missing:
        raise SystemExit(f"{a.eval_txt}: missing {sorted(missing)}, not a NuggetRecall leaderboard")

    per_topic, agg = compute_rank(base)

    topics = sorted(base)
    runs = sorted({r for t in topics for r in base[t]})

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as fh:
        for run, topic, measure, value in rows:      # everything already there, unchanged
            fh.write(f"{run}\t{topic}\t{measure}\t{value}\n")
        for t in topics:
            for r in sorted(per_topic[t]):
                fh.write(f"{r}\t{t}\t{NEW}\t{per_topic[t][r]!r}\n")
        for r in runs:
            fh.write(f"{r}\t{AGG}\t{NEW}\t{agg[r]!r}\n")

    distinct = len(set(round(v, 12) for v in agg.values()))
    print(f"{a.out}: {len(runs)} runs, {len(topics)} topics, {NEW} added "
          f"({distinct} distinct aggregate values)")
    if distinct < len(agg):
        print(f"  warning: {len(agg) - distinct} ties, which tauap_b breaks by input order")


if __name__ == "__main__":
    main()
