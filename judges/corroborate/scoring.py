"""Per-topic scoring: NCOV, NPREC, NF1, and within-topic percentile rank.

Scoring is organised per topic rather than per report so that every ambiguous
sentence-to-cluster comparison in the topic can go to the cross-encoder in one
batch. Doing it per report would multiply the number of GPU round trips by the
number of runs for no benefit.

All scores are continuous. `tauap_b` breaks ties by input order, which is
effectively random, so a coarse or thresholded score actively costs us at the
top of the leaderboard where that measure is most sensitive.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .pool import Cluster, TopicPool
from .textutil import scoreable_sentences

log = logging.getLogger(__name__)


def harmonic(a: float, b: float) -> float:
    """F1-style combination. Zero if either side is zero."""
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 * a * b / (a + b)


def within_topic_percentile(scores: Dict[str, float]) -> Dict[str, float]:
    """Fraction of other runs this run beats on this topic, ties split evenly.

    With R runs the result is in [0, 1] where 0 means worst and 1 means best.
    That orientation matters: the framework fills a missing cell with 0.0, which
    is the correct "worst" for a percentile but would wrongly mean "average" for
    a z-score, which is why this is not a z-score.

    A single-run topic carries no ranking information, so it returns 0.5 rather
    than inventing a win or a loss.
    """
    runs = sorted(scores)
    n = len(runs)
    if n <= 1:
        return {r: 0.5 for r in runs}
    out: Dict[str, float] = {}
    for r in runs:
        s = scores[r]
        less = sum(1 for o in runs if o != r and scores[o] < s)
        equal = sum(1 for o in runs if o != r and scores[o] == s)
        out[r] = (less + 0.5 * equal) / (n - 1)
    return out


def score_topic(
    reports: Sequence,
    pool: TopicPool,
    bank: Sequence[Cluster],
    encoders,
    *,
    tau: float,
    band_low: float,
    min_other_teams: int,
    min_sentence_words: int,
    specificity_floor: float,
    use_cross_encoder: bool,
    vote_unit: str = "team",
) -> Dict[str, Dict[str, float]]:
    """Score every report for one topic.

    Returns {run_id: {"NCOV": .., "NPREC": ..}}. NF1 and the rank measure are
    derived by the caller, which needs all runs to compute a percentile.

    Coverage is soft. A cluster the report has a sentence assigned to counts
    fully. Otherwise the report's best sentence for that cluster is looked at:
    at or above `tau` it counts fully, inside [band_low, tau) the cross-encoder
    decides and its entailment probability is the credit, and below `band_low`
    it counts as zero.
    """
    results: Dict[str, Dict[str, float]] = {}
    if not reports:
        return results

    ordered = sorted(reports, key=lambda r: str(r.metadata.run_id))
    bank_ids = [c.cluster_id for c in bank]
    bank_rows = [pool.index_of[cid] for cid in bank_ids if cid in pool.index_of]
    leader_vecs = pool.leader_vecs[bank_rows] if bank_rows else np.zeros((0, 0), np.float32)
    bank = [c for c in bank if c.cluster_id in pool.index_of]

    # Pass 1: per report, gather sentences, direct hits, and best-similarity
    # candidates. Collect the ambiguous band for one batched cross-encoder call.
    per_report: List[dict] = []
    ce_pairs: List[Tuple[str, str]] = []
    ce_slots: List[Tuple[int, int]] = []  # (report position, bank position)

    for pos, rep in enumerate(ordered):
        run_id = str(rep.metadata.run_id)
        team_id = str(rep.metadata.team_id or run_id)
        if vote_unit == "run":
            team_id = run_id
        sents = scoreable_sentences(rep.get_sentences() or [], min_sentence_words)
        info = {
            "run_id": run_id,
            "team_id": team_id,
            "n_sents": len(sents),
            "cov": np.zeros((len(bank),), dtype=np.float32),
            "best_text": [None] * len(bank),
        }

        if sents and len(bank):
            vecs = encoders.embed([t for _, t in sents])
            sims = vecs @ leader_vecs.T  # (n_sents, n_bank)
            best_idx = np.argmax(sims, axis=0)
            best_sim = sims[best_idx, np.arange(sims.shape[1])]

            # A sentence assigned to this cluster during pooling covers it
            # outright, even if some other cluster was a marginally better fit.
            assigned = {
                pool.assignment.get((run_id, i))
                for i, _ in sents
            }
            for bi, c in enumerate(bank):
                if c.cluster_id in assigned or best_sim[bi] >= tau:
                    info["cov"][bi] = 1.0
                elif use_cross_encoder and best_sim[bi] >= band_low:
                    info["best_text"][bi] = sents[int(best_idx[bi])][1]
                    ce_pairs.append((sents[int(best_idx[bi])][1], c.leader))
                    ce_slots.append((pos, bi))
                # else stays 0.0

        # NPREC: this report's own sentences, corroborated by other teams.
        supported = 0
        for i, _t in sents:
            cid = pool.assignment.get((run_id, i))
            if cid is None:
                continue
            row = pool.index_of.get(cid)
            if row is None:
                continue
            c = pool.clusters[row]
            if c.specificity < specificity_floor:
                continue
            if c.other_teams(team_id) >= min_other_teams:
                supported += 1
        info["nprec"] = (supported / len(sents)) if sents else 0.0
        per_report.append(info)

    # Pass 2: one cross-encoder batch for the whole topic.
    if ce_pairs:
        scores = encoders.cross(ce_pairs)
        for (pos, bi), s in zip(ce_slots, scores):
            per_report[pos]["cov"][bi] = float(s)

    # Pass 3: LOTO-weighted recall. Weights depend on who is being scored, so
    # this is the step that cannot be shared across reports.
    for info in per_report:
        team_id = info["team_id"]
        w = np.asarray([c.loto_weight(team_id) for c in bank], dtype=np.float32)
        denom = float(w.sum())
        ncov = float((w * info["cov"]).sum() / denom) if denom > 0 else 0.0
        results[info["run_id"]] = {"NCOV": ncov, "NPREC": float(info["nprec"])}

    return results
