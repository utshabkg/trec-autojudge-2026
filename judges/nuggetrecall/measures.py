"""Leaderboard schema for query-derived nugget recall.

Four measures, all higher-is-better and continuous (they are averages over topics
of fractions over up to twenty facts, so ties between systems are rare). Each is a
variant of nugget recall, which is what TREC RAG's human assessment measures, so
each is a plausible match for a truth measure and Max() over them is legitimate.
"""

from autojudge_base import LeaderboardSpec, MeasureSpec

MEASURE_NAMES = ("NR_VITAL", "NR_ALL", "NR_VITAL_STRICT", "NR_WEIGHTED")

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
))


def zero_row() -> dict:
    return {m: 0.0 for m in MEASURE_NAMES}
