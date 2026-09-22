"""Leaderboard schema for query-derived nugget recall.

Four measures, all higher-is-better and continuous (they are averages over topics
of fractions over up to twenty facts, so ties between systems are rare). Each is a
variant of nugget recall, which is what TREC RAG's human assessment measures, so
each is a plausible match for a truth measure and Max() over them is legitimate.
"""

from autojudge_base import LeaderboardSpec, MeasureSpec

MEASURE_NAMES = ("NR_VITAL", "NR_ALL", "NR_VITAL_STRICT", "NR_WEIGHTED", "NR_RANK")

# The measure NR_RANK is built from. Not a separate judgment, just a different
# way of aggregating the one we already make.
RANK_BASE = "NR_VITAL"

NUGGETRECALL_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("NR_VITAL", float, description=(
        "Recall of the VITAL facts in a query-derived answer key, support counting 1 "
        "and partial 0.5. The key is generated from the request alone, so it does not "
        "depend on what any system happened to say. 0 to 1, higher is better.")),
    MeasureSpec("NR_ALL", float, description=(
        "Recall of all facts in the query-derived answer key, support 1 and partial "
        "0.5. 0 to 1, higher is better.")),
    MeasureSpec("NR_VITAL_STRICT", float, description=(
        "Recall of the VITAL facts counting only full support, so hedged or partial "
        "statements earn nothing. 0 to 1, higher is better.")),
    MeasureSpec("NR_WEIGHTED", float, description=(
        "Importance-weighted recall over all facts, vital weight 1 and okay weight "
        "0.5, support 1 and partial 0.5. 0 to 1, higher is better.")),
    MeasureSpec("NR_RANK", float, description=(
        "NR_VITAL re-expressed as the fraction of other runs this run beats on the "
        "same topic, averaged over topics. Same judgments, different aggregation: "
        "it removes topic difficulty, which is the largest source of variance when "
        "ordering systems. 0 to 1, higher is better.")),
))


def within_topic_percentile(scores: dict) -> dict:
    """Fraction of other runs this run beats on this topic, ties split evenly.

    With R runs the result is in [0, 1] where 0 is worst and 1 is best. The
    orientation matters: the framework fills a missing cell with 0.0, which is the
    right "worst" for a percentile but would wrongly mean "average" for a z-score.
    That is why this is not a z-score.

    A single-run topic carries no ranking information, so it returns 0.5 rather
    than inventing a win or a loss.

    Deliberately duplicated from judges/corroborate/scoring.py rather than
    imported: that module pulls in numpy, torch and sentence-transformers, and
    this judge needs none of them. A test asserts the two stay identical.
    """
    runs = sorted(scores)
    n = len(runs)
    if n <= 1:
        return {r: 0.5 for r in runs}
    out = {}
    for r in runs:
        s = scores[r]
        less = sum(1 for o in runs if o != r and scores[o] < s)
        equal = sum(1 for o in runs if o != r and scores[o] == s)
        out[r] = (less + 0.5 * equal) / (n - 1)
    return out


def zero_row() -> dict:
    return {m: 0.0 for m in MEASURE_NAMES}
