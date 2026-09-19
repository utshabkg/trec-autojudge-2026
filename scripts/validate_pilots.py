#!/usr/bin/env python3
"""Measure a judge against a pilot leaderboard, and against the length confound.

Raw Kendall is not evidence that a judge works. The starter kit's naive judge
scores by answer length and already reaches Kendall 0.558 on ragtime25, so a
measure that is a restatement of length will read as "working" at around 0.55
while telling us nothing a word count would not.

So for every measure this reports four things beyond the correlation:

  conf(len)   Kendall of the measure against LENGTH. Above about 0.85 the
              measure is length wearing a hat.
  disc-acc    Discordant-pair accuracy. Of the system pairs that LENGTH orders
              the WRONG way against truth, the share this measure orders right.
              Length scores 0.0 here by construction, so this is the decisive
              test: it is the part of the signal length cannot explain.
  partial     Rank-partial correlation of measure with truth, controlling for
              length.
  n_disc      How many discordant pairs there were, so a good disc-acc on three
              pairs is not mistaken for evidence.

Correlations come from the official evaluation package, so the numbers here are
the same ones the organizers' meta-evaluation would produce.

    python scripts/validate_pilots.py \\
        --truth local-data/ragtime25/eval/ragtime.repgen.official.eval.jsonl \\
        --eval  output/ragtime25/corroborate/best-all-all/best.eval.txt \\
        --runs  local-data/ragtime25/runs/repgen/ --label ragtime25
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

from autojudge_evaluate.evaluation import correlation
from autojudge_evaluate.pyircore import tauap_b

ALL = "all"


# ----------------------------------------------------------------- loading


def load_leaderboard(path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """-> {measure: {run: {topic: value}}}, accepting ir_measures or jsonl."""
    out: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                    run, topic = str(obj["run_id"]), str(obj["topic_id"])
                    measure, value = str(obj["measure"]), obj["value"]
                except Exception:
                    continue
            else:
                parts = line.split()
                if len(parts) != 4:
                    continue
                run, topic, measure, value = parts
                if lineno == 1 and not _is_float(value):
                    continue  # header row
            if not _is_float(value):
                continue
            out[measure][run][topic] = float(value)
    return {m: dict(r) for m, r in out.items()}


def _is_float(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def aggregate(per_topic: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Use the 'all' row when present, otherwise macro-average the topics."""
    out: Dict[str, float] = {}
    for run, topics in per_topic.items():
        if ALL in topics:
            out[run] = topics[ALL]
        else:
            vals = [v for t, v in topics.items() if t != ALL]
            if vals:
                out[run] = mean(vals)
    return out


def length_per_run(runs_dir: Path) -> Dict[str, float]:
    """Macro-averaged answer length in words per run, the confound to beat.

    Parsed straight from the JSONL rather than through pydantic: this only needs
    word counts, and the run files are large.
    """
    per: Dict[str, Dict[str, int]] = defaultdict(dict)
    files = sorted(
        p for p in list(runs_dir.glob("*")) + list(runs_dir.glob("*/*")) if p.is_file()
    )
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    md = d.get("metadata") or {}
                    run = str(md.get("run_id") or f.stem)
                    topic = str(
                        md.get("topic_id")
                        or md.get("narrative_id")
                        or md.get("request_id")
                        or ""
                    )
                    sents = d.get("responses") or d.get("answer") or []
                    words = sum(len((s.get("text") or "").split()) for s in sents)
                    per[run][topic] = words
        except Exception as exc:
            print(f"  warn: could not read {f.name}: {exc}", file=sys.stderr)
    return {run: mean(t.values()) for run, t in per.items() if t}


# --------------------------------------------------------------- statistics


def _ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def discordant_pair_accuracy(
    truth: Sequence[float], measure: Sequence[float], length: Sequence[float]
) -> Tuple[Optional[float], int]:
    """Accuracy on exactly the pairs LENGTH gets wrong.

    Ties in the measure count as half right, which is what they are worth: the
    measure expressed no preference.
    """
    right = 0.0
    total = 0
    n = len(truth)
    for i in range(n):
        for j in range(i + 1, n):
            if truth[i] == truth[j] or length[i] == length[j]:
                continue
            t = truth[i] > truth[j]
            if (length[i] > length[j]) == t:
                continue  # length already gets this pair right
            total += 1
            if measure[i] == measure[j]:
                right += 0.5
            elif (measure[i] > measure[j]) == t:
                right += 1.0
    return (right / total if total else None), total


def partial_correlation(
    x: Sequence[float], y: Sequence[float], z: Sequence[float]
) -> Optional[float]:
    """Rank-partial correlation of x and y controlling for z."""
    try:
        rx, ry, rz = _ranks(x), _ranks(y), _ranks(z)
        rxz = correlation(rx, rz, "pearson")
        ryz = correlation(ry, rz, "pearson")
        rxy = correlation(rx, ry, "pearson")
        denom = ((1 - rxz**2) * (1 - ryz**2)) ** 0.5
        if denom == 0:
            return None
        return (rxy - rxz * ryz) / denom
    except Exception:
        return None


def _safe(fn, *a) -> Optional[float]:
    try:
        return fn(*a)
    except Exception:
        return None


def _fmt(v: Optional[float], width: int = 7) -> str:
    return " " * (width - 1) + "-" if v is None else f"{v:>{width}.3f}"


# ---------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--eval", type=Path, required=True, dest="eval_path")
    ap.add_argument("--runs", type=Path, default=None,
                    help="run directory, for the LENGTH confound. Omit to skip those columns")
    ap.add_argument("--label", default=None)
    ap.add_argument("--truth-measure", action="append", default=None)
    ap.add_argument("--min-runs", type=int, default=3,
                    help="the evaluation package refuses fewer than 3 runs")
    args = ap.parse_args()

    label = args.label or args.eval_path.parent.name
    truth_lb = load_leaderboard(args.truth)
    eval_lb = load_leaderboard(args.eval_path)
    if not truth_lb:
        print(f"FAIL: no usable rows in {args.truth}")
        return 2
    if not eval_lb:
        print(f"FAIL: no usable rows in {args.eval_path}")
        return 2

    lengths = length_per_run(args.runs) if args.runs else {}

    truth_measures = args.truth_measure or sorted(truth_lb)
    print(f"\n=== {label} ===")
    print(f"truth measures: {', '.join(truth_measures)}")
    print(f"eval measures : {', '.join(sorted(eval_lb))}")
    if lengths:
        print(f"LENGTH computed for {len(lengths)} runs from {args.runs}")

    header = (
        f"{'truth':<26} {'measure':<10} {'kendall':>8} {'tauap_b':>8} "
        f"{'k@10':>8} {'conf(len)':>10} {'disc-acc':>9} {'n_disc':>7} {'partial':>8}"
    )
    print("\n" + header)
    print("-" * len(header))

    worst_kendall = None
    best_kendall = None
    rows_emitted = 0

    for tm in truth_measures:
        t_agg = aggregate(truth_lb.get(tm, {}))
        if len(t_agg) < args.min_runs:
            print(f"{tm:<26} (only {len(t_agg)} runs in truth, skipped)")
            continue
        # LENGTH's own correlation, the bar every measure has to clear.
        if lengths:
            shared_len = sorted(set(t_agg) & set(lengths))
            if len(shared_len) >= args.min_runs:
                tv = [t_agg[r] for r in shared_len]
                lv = [lengths[r] for r in shared_len]
                print(
                    f"{tm:<26} {'LENGTH':<10} {_fmt(_safe(correlation, tv, lv, 'kendall'),8)} "
                    f"{_fmt(_safe(tauap_b, tv, lv),8)} {'':>8} {'':>10} "
                    f"{_fmt(0.0,9)} {len(shared_len):>7} {'':>8}"
                )

        for em in sorted(eval_lb):
            e_agg = aggregate(eval_lb[em])
            shared = sorted(set(t_agg) & set(e_agg))
            if len(shared) < args.min_runs:
                continue
            tv = [t_agg[r] for r in shared]
            ev = [e_agg[r] for r in shared]
            k = _safe(correlation, tv, ev, "kendall")
            tb = _safe(tauap_b, tv, ev)
            k10 = None
            if len(shared) > 10:
                top = sorted(shared, key=lambda r: t_agg[r], reverse=True)[:10]
                k10 = _safe(
                    correlation, [t_agg[r] for r in top], [e_agg[r] for r in top], "kendall"
                )
            conf = disc = part = None
            ndisc = 0
            if lengths:
                sl = [r for r in shared if r in lengths]
                if len(sl) >= args.min_runs:
                    tvl = [t_agg[r] for r in sl]
                    evl = [e_agg[r] for r in sl]
                    lvl = [lengths[r] for r in sl]
                    conf = _safe(correlation, evl, lvl, "kendall")
                    disc, ndisc = discordant_pair_accuracy(tvl, evl, lvl)
                    part = partial_correlation(evl, tvl, lvl)
            print(
                f"{tm:<26} {em:<10} {_fmt(k,8)} {_fmt(tb,8)} {_fmt(k10,8)} "
                f"{_fmt(conf,10)} {_fmt(disc,9)} {ndisc:>7} {_fmt(part,8)}"
            )
            rows_emitted += 1
            if k is not None:
                best_kendall = k if best_kendall is None else max(best_kendall, k)
                worst_kendall = k if worst_kendall is None else min(worst_kendall, k)

    if not rows_emitted:
        print("\nFAIL: no (truth, eval) measure pair had enough shared runs")
        return 2

    print(
        f"\nMax(kendall) = {_fmt(best_kendall).strip()}   "
        f"Min(kendall) = {_fmt(worst_kendall).strip()}   "
        "(TIRA reports both, so a negative Min costs us for nothing)"
    )
    print(
        "Gate: a measure only counts as beating length if disc-acc >= 0.60 on a "
        "meaningful n_disc and conf(len) stays below ~0.85."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
