"""Per-topic answer pooling: cluster the pool, count team votes, apply LOTO.

The pooling assumption from retrieval evaluation, moved from documents to
answers: for a given topic we hold every system's answer, and a fact that many
independent teams state is probably a real and important fact about that topic.

Two corrections make that usable rather than circular.

Votes are counted per team, not per run. rag26 is 83 runs from 25 teams, so
about three runs per team, and teams submit close variants of one pipeline.
Counting runs would measure herd size and let one team vote several times for
its own phrasing.

No team votes for its own claims. A report's own sentences are in the pool its
bank is built from, so two rules apply. A cluster needs at least two teams to
enter the bank, which keeps out a claim only one team makes. Cluster weights are
then recomputed per scored team as |teams(c) - {t}|, a team-level version of the
leave-one-run-out test of pool bias from retrieval evaluation.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from .textutil import normalize, scoreable_sentences, topic_text

log = logging.getLogger(__name__)


# The specificity gate silently did nothing for the whole of development, and the
# sweep that was supposed to validate it could not see that, because all four swept
# values produced byte-identical banks. The cause: `specificity` is a raw cosine
# against the topic under BAAI/bge-base-en-v1.5, and BGE is anisotropic, so nothing
# ever scores near zero. Measured against a real topic with the shipped encoder:
#
#   specific facts     0.587 to 0.753
#   boilerplate hedges 0.361 to 0.472
#   literal gibberish  0.266 to 0.319
#
# The shipped floor was 0.15 and the sweep was [0.0, 0.10, 0.15, 0.20], all of
# which sit below the floor of the distribution. So every cluster passed, hedges
# took the top of the consensus bank (it is sorted by team count, and a hedge is
# exactly what every team writes), and NPREC counted them as corroborated, which
# inverts the measure that is supposed to penalise padding.
#
# A raw threshold on a cosine is meaningless without knowing the encoder's scale,
# so the number alone is not the fix. This is: say so when the gate is not gating.
_GATE_REPORTED = set()


def _warn_if_gate_is_inert(clusters, floor: float) -> None:
    """Say so when the specificity gate rejects everything or nothing.

    Either extreme means the floor is off the scale of whatever encoder is in use,
    and the parameter is not doing its job. Reported once per floor per process, so
    a per-topic loop does not flood the log. A negative floor is how the tests
    disable the gate on purpose, so that is left alone.
    """
    if floor < 0 or not clusters or floor in _GATE_REPORTED:
        return
    _GATE_REPORTED.add(floor)
    kept = sum(1 for c in clusters if c.specificity >= floor)
    share = kept / len(clusters)
    if share > 0.995 or share < 0.005:
        lo = min(c.specificity for c in clusters)
        hi = max(c.specificity for c in clusters)
        log.warning(
            "the specificity gate keeps %.1f%% of clusters at floor %.2f, so it is "
            "not gating. Observed specificity runs %.3f to %.3f, so pick a floor "
            "inside that range", share * 100, floor, lo, hi)


@dataclass(frozen=True)
class Cluster:
    """One pooled claim.

    `leader` is the first sentence assigned to the cluster in
    (run_id, sentence_index) order. Using the first member rather than a
    recomputed medoid keeps clustering deterministic and single-pass. Which run
    supplies it is arbitrary but stable, and it cannot advantage that run
    because every member is within `tau` of it anyway and LOTO removes
    self-credit regardless.
    """

    cluster_id: str
    leader: str
    teams: FrozenSet[str]
    runs: FrozenSet[str]
    specificity: float
    size: int

    def loto_weight(self, team_id: Optional[str]) -> int:
        """Team votes excluding the team being scored."""
        if team_id is None:
            return len(self.teams)
        return len(self.teams - {team_id})

    def other_teams(self, team_id: Optional[str]) -> int:
        return self.loto_weight(team_id)


@dataclass
class TopicPool:
    """Everything the scorer needs for one topic."""

    topic_id: str
    clusters: List[Cluster]
    # (run_id, sentence_index) -> cluster_id, for NPREC and for exact coverage
    assignment: Dict[Tuple[str, int], str]
    # cluster_id -> row index into `leader_vecs`
    index_of: Dict[str, int]
    leader_vecs: np.ndarray
    n_teams: int

    def bank(
        self, *, specificity_floor: float, min_teams: int, max_clusters: int
    ) -> List[Cluster]:
        """The scoring bank: specific enough, corroborated enough, capped.

        `min_teams >= 2` is what guarantees no bank cluster can be the private
        invention of a single team, so a LOTO weight can never fall to zero and
        the recall denominator stays well defined for every report.

        Selection happens once, globally, and LOTO then reweights within that
        fixed bank. Re-selecting per team would make the denominator depend on
        who is being scored, which is harder to interpret and no fairer.
        """
        elig = [
            c
            for c in self.clusters
            if c.specificity >= specificity_floor and len(c.teams) >= min_teams
        ]
        _warn_if_gate_is_inert(self.clusters, specificity_floor)
        # Sort by votes, then id, so the cap is deterministic under ties.
        elig.sort(key=lambda c: (-len(c.teams), c.cluster_id))
        if max_clusters and max_clusters > 0:
            elig = elig[:max_clusters]
        return elig


def _cluster_id(text: str) -> str:
    return hashlib.md5(normalize(text).encode("utf-8")).hexdigest()


def _assign(best, run_id, team_id, idx, text, v, leaders, leader_text,
            teams, runs, sizes, cluster_ids, assignment) -> None:
    """Place one sentence: join cluster `best`, or start a new one if best < 0.

    Factored out so the batched loop and the reference implementation in the
    tests share exactly one definition of what assignment means.
    """
    if best < 0:
        n = len(cluster_ids)
        leaders[n] = v
        leader_text.append(text)
        cid = _cluster_id(text)
        # Distinct sentences can normalise to the same id only if identical,
        # which dedupe already prevents within a report but not across them.
        while cid in cluster_ids:
            cid = hashlib.md5((cid + "+").encode("utf-8")).hexdigest()
        cluster_ids.append(cid)
        teams.append({team_id})
        runs.append({run_id})
        sizes.append(1)
        best = n
    else:
        teams[best].add(team_id)
        runs[best].add(run_id)
        sizes[best] += 1
    assignment[(run_id, idx)] = cluster_ids[best]


def build_topic_pool(
    reports: Sequence,
    topic,
    encoders,
    *,
    tau: float = 0.82,
    min_sentence_words: int = 5,
    vote_unit: str = "team",
    block_size: int = 512,
) -> TopicPool:
    """Cluster every scoreable sentence of every report for one topic.

    `reports` must already be filtered to this topic. Processing order is fixed
    to (run_id, sentence_index) so the pool is reproducible.

    `vote_unit="run"` is the ablation that counts runs instead of teams. It is
    implemented by treating each run as its own team, so the rest of the
    pipeline is untouched.
    """
    ordered = sorted(reports, key=lambda r: str(r.metadata.run_id))

    items: List[Tuple[str, str, int, str]] = []  # run_id, team_id, sent_idx, text
    for rep in ordered:
        run_id = str(rep.metadata.run_id)
        team_id = str(rep.metadata.team_id or run_id)
        if vote_unit == "run":
            team_id = run_id
        sents = rep.get_sentences() or []
        for idx, text in scoreable_sentences(sents, min_sentence_words):
            items.append((run_id, team_id, idx, text))

    topic_str = topic_text(
        getattr(topic, "title", None),
        getattr(topic, "background", None),
        getattr(topic, "problem_statement", None),
    )

    if not items:
        log.warning("topic %s: no scoreable sentences in the pool", topic.request_id)
        return TopicPool(
            topic_id=str(topic.request_id),
            clusters=[],
            assignment={},
            index_of={},
            leader_vecs=np.zeros((0, 0), dtype=np.float32),
            n_teams=0,
        )

    vecs = encoders.embed([t for _, _, _, t in items])
    topic_vec = encoders.embed([topic_str])[0] if topic_str else None

    dim = vecs.shape[1]
    leaders = np.zeros((len(items), dim), dtype=np.float32)
    n_leaders = 0
    leader_text: List[str] = []
    teams: List[set] = []
    runs: List[set] = []
    sizes: List[int] = []
    assignment: Dict[Tuple[str, int], str] = {}
    cluster_ids: List[str] = []

    # Similarities are computed one block at a time rather than one sentence at a
    # time. The naive version issued a separate small matrix-vector product per
    # sentence, roughly 300k of them per dataset, and each one launched a
    # 60-thread BLAS parallel region for a few milliseconds of arithmetic. That
    # cost 15 seconds per topic and drove load average to 123 on a shared
    # 120-core machine, almost entirely in thread launch rather than computation.
    #
    # The result is identical to the sequential version, not merely similar.
    # Within a block, sentences are still assigned strictly in order, and a
    # sentence is compared both against the leaders that existed when the block
    # started (the precomputed rows) and against any leader created earlier in
    # the same block (the small incremental product). `test_batched_clustering_
    # matches_sequential` pins that equivalence.
    block = max(1, int(block_size))
    for start in range(0, len(items), block):
        stop = min(start + block, len(items))
        bvecs = vecs[start:stop]
        base_n = n_leaders
        pre = bvecs @ leaders[:base_n].T if base_n else None

        for bi in range(stop - start):
            run_id, team_id, idx, text = items[start + bi]
            v = bvecs[bi]
            best, best_sim = -1, -1.0
            if base_n:
                row = pre[bi]
                j = int(np.argmax(row))
                if float(row[j]) >= tau:
                    best, best_sim = j, float(row[j])
            if n_leaders > base_n:
                # Leaders created earlier in this block are not in `pre`.
                extra = leaders[base_n:n_leaders] @ v
                j = int(np.argmax(extra))
                if float(extra[j]) >= tau and float(extra[j]) > best_sim:
                    best, best_sim = base_n + j, float(extra[j])
            _assign(best, run_id, team_id, idx, text, v, leaders, leader_text,
                    teams, runs, sizes, cluster_ids, assignment)
            n_leaders = len(cluster_ids)


    leader_vecs = leaders[:n_leaders].copy()
    if topic_vec is not None and n_leaders:
        spec = leader_vecs @ topic_vec
    else:
        spec = np.ones((n_leaders,), dtype=np.float32)

    clusters = [
        Cluster(
            cluster_id=cluster_ids[i],
            leader=leader_text[i],
            teams=frozenset(teams[i]),
            runs=frozenset(runs[i]),
            specificity=float(spec[i]),
            size=sizes[i],
        )
        for i in range(n_leaders)
    ]
    index_of = {cluster_ids[i]: i for i in range(n_leaders)}
    all_teams = {t for s in teams for t in s}

    log.info(
        "topic %s: %d sentences -> %d clusters, %d teams, %d clusters with >=2 teams",
        topic.request_id, len(items), n_leaders, len(all_teams),
        sum(1 for c in clusters if len(c.teams) >= 2),
    )
    return TopicPool(
        topic_id=str(topic.request_id),
        clusters=clusters,
        assignment=assignment,
        index_of=index_of,
        leader_vecs=leader_vecs,
        n_teams=len(all_teams),
    )
