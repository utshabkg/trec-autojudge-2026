"""The leaderboard schema.

Five measures, all oriented so that higher is better, all continuous.

Both orientation and continuity are deliberate. The TIRA evaluator reports
`Min (kendall)` as well as `Max (kendall)`, so a measure that anti-correlates
with the human truth is pure downside, which is why there is no length penalty
or other deliberately inverted diagnostic here. And `tauap_b` breaks ties by
input order, so a coarse score costs us precisely at the top of the leaderboard
where that measure is most sensitive.

Raw answer length is deliberately absent. The shipped naive baseline scores by
length and already reaches Kendall 0.558 on ragtime25, so emitting length would
let `Max` credit this judge with the baseline's performance. That would be
gaming the aggregation rather than judging.
"""

from __future__ import annotations

from autojudge_base import LeaderboardSpec, MeasureSpec

MEASURE_NAMES = ("NCOV", "NPREC", "NF1", "NF1_RANK", "CITE")

CORROBORATE_SPEC = LeaderboardSpec(
    measures=(
        MeasureSpec(
            "NCOV",
            float,
            description=(
                "Pooled-consensus nugget coverage (recall). Sentences from every run for a "
                "topic are clustered; each cluster is weighted by the number of distinct teams "
                "asserting it, leave-one-team-out, so a run earns nothing for a cluster only "
                "its own team created. Value is the team-weighted share of the topic's bank "
                "this report covers, with partial credit from cross-encoder entailment for "
                "near-matches. 0 to 1, higher is better."
            ),
        ),
        MeasureSpec(
            "NPREC",
            float,
            description=(
                "Corroboration precision: the share of this report's own substantive sentences "
                "that are corroborated by at least two other teams, gated by topical "
                "specificity so boilerplate does not count as universally corroborated. "
                "Length-normalised by construction, so it penalises padding and "
                "uncorroborated assertions. 0 to 1, higher is better."
            ),
        ),
        MeasureSpec(
            "NF1",
            float,
            description=(
                "Harmonic mean of NCOV and NPREC: covering what the pool agrees matters, "
                "without padding. 0 to 1, higher is better."
            ),
        ),
        MeasureSpec(
            "NF1_RANK",
            float,
            description=(
                "Within-topic percentile rank of NF1, that is the share of the other runs this "
                "run beats on this topic, macro-averaged over topics. Removes topic-difficulty "
                "variance from the system ordering. 0 means worst on every topic, 1 means best "
                "on every topic."
            ),
        ),
        MeasureSpec(
            "CITE",
            float,
            description=(
                "Citation support. When runs carry their cited documents inline, this is the "
                "mean over cited sentences of the entailment of the sentence from the "
                "best-matching chunk of a document it cites. When they do not, it is pooled "
                "citation agreement, the leave-one-team-out share of other teams citing the "
                "same document for this topic, times citation density. 0 to 1, higher is better."
            ),
        ),
    )
)


def zero_row() -> dict:
    """The bottom grade for every measure.

    Used for a report with no scoreable content. Such a report must be scored
    rather than dropped, because dropping a topic would silently shrink the
    leaderboard and a partial submission cannot be accepted.
    """
    return {name: 0.0 for name in MEASURE_NAMES}
