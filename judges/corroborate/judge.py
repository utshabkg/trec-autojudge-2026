#!/usr/bin/env python3
"""Corroborate: the AutoJudge protocol implementation.

One class serves both the nugget phase and the judge phase. The runner caches
instances by class path, so declaring only `judge_class` in workflow.yml gives
both phases the same object and the pooled clustering is computed once and
reused rather than repeated.

The judge makes no LLM calls. `workflow.yml` declares `uses_llm: false`, and
`llm_config` is used only for its `cache_dir`, which is where the embedding and
cross-encoder score caches live. That keeps CACHE_DIR meaningful for resume and
for a warm re-run even though no prompt ever goes to a model.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Type

from autojudge_base import (
    Leaderboard,
    LeaderboardBuilder,
    LeaderboardVerification,
    LlmConfigProtocol,
    NuggetBanks,
    NuggetBanksProtocol,
    Qrels,
    Report,
    Request,
    auto_judge_to_click_command,
)
from autojudge_base.nugget_data import Creator, NuggetBank, NuggetClaim

try:
    from tqdm import tqdm
except ImportError:  # tqdm ships in the [all] extra
    def tqdm(iterable, *args, **kwargs):
        return iterable

from .attribution import (
    build_citation_index,
    detect_mode,
    score_inline,
    score_pooled,
)
from .encoders import Encoders, limit_threads
from .measures import CORROBORATE_SPEC, zero_row
from .pool import TopicPool, build_topic_pool
from .textutil import normalize
from .scoring import harmonic, score_topic, within_topic_percentile

log = logging.getLogger(__name__)


class CorroborateJudge:
    """Pooled corroboration with leave-one-team-out weighting."""

    # Required so the runner can deserialise a bank handed in with --nugget-banks,
    # even though this judge builds its own.
    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks

    def __init__(self) -> None:
        self._encoders: Optional[Encoders] = None
        self._enc_sig: Optional[tuple] = None
        self._pools: Dict[str, TopicPool] = {}
        self._pool_sig: Optional[tuple] = None
        self._scratch_cache: Optional[str] = None

    # ------------------------------------------------------------------ helpers

    def _get_encoders(
        self,
        llm_config: Optional[LlmConfigProtocol],
        *,
        bi_encoder: str,
        cross_encoder: Optional[str],
        embed_batch_size: int,
        cross_batch_size: int,
        cross_max_length: int,
        hf_home: Optional[str],
        device: Optional[str],
    ) -> Encoders:
        if hf_home and Path(hf_home).is_dir():
            # Local convenience so a shared model cache can be named in
            # workflow.yml without exporting anything.
            #
            # Guarded on the directory existing, because this value is a path on
            # a specific machine and workflow.yml travels into the container.
            # Setting it blindly is how the fourth CI dry run failed: HF_HOME was
            # harmless because the image sets it and setdefault is then a no-op,
            # but HF_HUB_CACHE is NOT set in the image, so setdefault planted a
            # host path inside the container. HF_HUB_CACHE wins over HF_HOME, so
            # the baked weights at /models/hub were never looked at and the judge
            # tried to reach huggingface.co from a sandbox with no network.
            #
            # HF_HUB_CACHE is deliberately not set here at all: HF derives it
            # from HF_HOME, and one variable is one thing that can go wrong.
            os.environ.setdefault("HF_HOME", hf_home)
        elif hf_home:
            log.info("hf_home %s does not exist here, leaving HF_HOME alone", hf_home)
        cache_dir = getattr(llm_config, "cache_dir", None) if llm_config else None
        if cache_dir is None:
            # No CACHE_DIR, which is exactly the organizers' graded run: the code
            # submission mounts one only for LLM judges. Without a cache the score
            # cache switched itself off, so every cited document chunk was embedded
            # again for every (sentence, document) pair that cited it, and every
            # sentence pooling had already embedded was embedded again in scoring.
            # A document cited by many of rag26's 83 runs paid that each time. Our
            # 8.5 h CPU timing on rag25-gen was WITH a cache, so the graded run
            # would have been far slower than anything we measured, estimated at
            # days on rag26. One scratch directory per process fixes it, shared by
            # both phases so the signature below stays stable between them. The
            # cache is content addressed, so results are unchanged, and every
            # read and write is guarded, so a full disk only costs speed.
            if self._scratch_cache is None:
                import tempfile
                self._scratch_cache = tempfile.mkdtemp(prefix="corroborate-cache-")
                log.info("no CACHE_DIR set, using scratch score cache %s", self._scratch_cache)
            cache_dir = self._scratch_cache
        sig = (bi_encoder, cross_encoder, str(cache_dir), device)
        if self._encoders is None or self._enc_sig != sig:
            if self._encoders is not None:
                self._encoders.close()
            self._encoders = Encoders(
                bi_encoder,
                cross_encoder,
                cache_dir=cache_dir,
                device=device,
                embed_batch_size=embed_batch_size,
                cross_batch_size=cross_batch_size,
                cross_max_length=cross_max_length,
                local_files_only=False,
            )
            self._enc_sig = sig
        return self._encoders

    @staticmethod
    def _by_topic(reports: Sequence[Report], expected: Sequence[str]) -> Dict[str, List[Report]]:
        """Group reports by topic, dropping topics not in the topics file.

        The nugget-bank verification raises on an unexpected topic, and rag26
        tolerates extra topics as a smell, so runs may genuinely contain them.
        Dropping is counted and logged rather than done silently.
        """
        allowed = set(expected)
        out: Dict[str, List[Report]] = {t: [] for t in expected}
        dropped = 0
        for r in reports:
            tid = str(r.metadata.topic_id)
            if tid in allowed:
                out[tid].append(r)
            else:
                dropped += 1
        if dropped:
            log.warning("dropped %d report(s) whose topic is not in the topics file", dropped)
        return out

    def _build_pools(
        self,
        reports: Sequence[Report],
        topics: Sequence[Request],
        encoders: Encoders,
        *,
        tau: float,
        min_sentence_words: int,
        vote_unit: str,
        num_threads: int = 8,
        block_size: int = 512,
    ) -> Dict[str, TopicPool]:
        sig = (tau, min_sentence_words, vote_unit, len(reports), len(topics))
        if self._pools and self._pool_sig == sig:
            log.info("reusing pooled clustering for %d topics", len(self._pools))
            return self._pools
        expected = [str(t.request_id) for t in topics]
        grouped = self._by_topic(reports, expected)
        pools: Dict[str, TopicPool] = {}
        # Capped inside the loop, not globally, because this is where the many
        # matrix products are and where an uncapped BLAS pool made a shared
        # machine unusable.
        with limit_threads(num_threads):
            for topic in tqdm(topics, "Pooling topics"):
                tid = str(topic.request_id)
                pools[tid] = build_topic_pool(
                    grouped.get(tid, []),
                    topic,
                    encoders,
                    tau=tau,
                    min_sentence_words=min_sentence_words,
                    vote_unit=vote_unit,
                    block_size=block_size,
                )
        self._pools, self._pool_sig = pools, sig
        return pools

    # ------------------------------------------------------------ nugget phase

    def create_nuggets(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        corpus: Optional[str] = None,
        filebase: str = "default",
        outdir: Path = Path("."),
        *,
        bi_encoder: str = "BAAI/bge-base-en-v1.5",
        cross_encoder: Optional[str] = None,
        embed_batch_size: int = 256,
        cross_batch_size: int = 128,
        cross_max_length: int = 512,
        hf_home: Optional[str] = None,
        device: Optional[str] = None,
        tau: float = 0.80,
        min_sentence_words: int = 5,
        specificity_floor: float = 0.15,
        min_teams: int = 2,
        max_clusters: int = 40,
        vote_unit: str = "team",
        num_threads: int = 8,
        block_size: int = 512,
        **kwargs: Any,
    ) -> Optional[NuggetBanksProtocol]:
        """Mine a nugget bank from the pool of answers.

        This is an output artifact in its own right. The judge phase rebuilds the
        same pools from the same deterministic function, so the published bank
        and the bank used for scoring agree.
        """
        encoders = self._get_encoders(
            llm_config,
            bi_encoder=bi_encoder,
            cross_encoder=cross_encoder,
            embed_batch_size=embed_batch_size,
            cross_batch_size=cross_batch_size,
            cross_max_length=cross_max_length,
            hf_home=hf_home,
            device=device,
        )
        pools = self._build_pools(
            rag_responses, rag_topics, encoders,
            tau=tau, min_sentence_words=min_sentence_words, vote_unit=vote_unit,
            num_threads=num_threads, block_size=block_size,
        )

        creator = Creator(
            is_human=False,
            llm_prompt_strategy="pool-corroboration-loto",
            format=f"bi_encoder={bi_encoder};tau={tau};vote_unit={vote_unit}",
        )
        banks: List[NuggetBank] = []
        empty = 0
        for topic in rag_topics:
            tid = str(topic.request_id)
            pool = pools.get(tid)
            selected = (
                pool.bank(
                    specificity_floor=specificity_floor,
                    min_teams=min_teams,
                    max_clusters=max_clusters,
                )
                if pool
                else []
            )
            bank = NuggetBank(query_id=tid, title_query=topic.title)
            if not selected:
                # Nothing in this topic's pool was corroborated by min_teams
                # teams. Fall back to the single best-supported cluster so the
                # bank still says something about the pool.
                empty += 1
                selected = sorted(
                    pool.clusters if pool else [],
                    key=lambda c: (-len(c.teams), c.cluster_id),
                )[:1]
            claims = [
                NuggetClaim.from_lazy(
                    query_id=tid,
                    claim=c.leader,
                    importance=len(c.teams),
                    metadata={
                        "teams": len(c.teams),
                        "runs": len(c.runs),
                        "sentences": c.size,
                        "specificity": round(c.specificity, 4),
                    },
                )
                for c in selected
            ]
            if not claims:
                # The pool holds no scoreable sentence at all for this topic, so
                # there is no cluster to fall back to either. This happens when
                # every run skipped the topic (responses: []), which is a case
                # the framework explicitly requires us to survive rather than
                # crash on. An empty bank is rejected by the bank verification,
                # so emit one claim derived from the topic itself. The Request is
                # unrestricted content, so this leaks nothing, and judge() scores
                # such a topic as zero for every run independently, which is
                # rank-preserving.
                placeholder = normalize(topic.title or "") or (
                    f"No corroborated content was found in the pool for topic {tid}."
                )
                log.warning(
                    "topic %s: pool has no scoreable sentences at all; emitting a "
                    "topic-derived placeholder nugget so the bank is not empty",
                    tid,
                )
                claims = [
                    NuggetClaim.from_lazy(
                        query_id=tid,
                        claim=placeholder,
                        importance=0,
                        metadata={"fallback": True, "reason": "empty pool for this topic"},
                    )
                ]
            bank.add_nuggets(claims)
            bank.add_creator(creator)
            banks.append(bank)

        if empty:
            log.warning(
                "%d/%d topics had no cluster corroborated by >= %d teams; "
                "NCOV is uninformative for those topics",
                empty, len(rag_topics), min_teams,
            )
        return NuggetBanks.from_banks_list(banks)

    def create_qrels(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        corpus: Optional[str] = None,
        filebase: str = "default",
        outdir: Path = Path("."),
        **kwargs: Any,
    ) -> Optional[Qrels]:
        """Intentionally not implemented.

        A qrels file alongside the leaderboard in an upload directory breaks
        tira-cli's format check, because TREC qrels also parse as four
        whitespace-separated columns and the evaluator then cannot tell which
        file is the leaderboard. workflow.yml sets `create_qrels: false`; this
        returns None so that flipping the flag cannot silently produce an
        unuploadable output directory either.
        """
        log.info("create_qrels is disabled by design (see docstring)")
        return None

    # ------------------------------------------------------------- judge phase

    def judge(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        qrels: Optional[Qrels] = None,
        corpus: Optional[str] = None,
        filebase: str = "default",
        outdir: Path = Path("."),
        *,
        bi_encoder: str = "BAAI/bge-base-en-v1.5",
        cross_encoder: Optional[str] = "cross-encoder/nli-deberta-v3-base",
        multilingual_cross_encoder: Optional[str] = None,
        use_cross_encoder: bool = True,
        embed_batch_size: int = 256,
        cross_batch_size: int = 128,
        cross_max_length: int = 512,
        hf_home: Optional[str] = None,
        device: Optional[str] = None,
        tau: float = 0.80,
        band_low: float = 0.62,
        min_sentence_words: int = 5,
        specificity_floor: float = 0.15,
        min_teams: int = 2,
        max_clusters: int = 40,
        min_other_teams: int = 1,
        vote_unit: str = "team",
        attribution_mode: str = "auto",
        doc_chunk_chars: int = 1200,
        max_doc_chunks: int = 6,
        on_missing_evals: str = "fix_aggregate",
        num_threads: int = 8,
        block_size: int = 512,
        **kwargs: Any,
    ) -> Leaderboard:
        expected = [str(t.request_id) for t in rag_topics]
        grouped = self._by_topic(rag_responses, expected)

        active_cross = multilingual_cross_encoder or cross_encoder
        encoders = self._get_encoders(
            llm_config,
            bi_encoder=bi_encoder,
            cross_encoder=active_cross,
            embed_batch_size=embed_batch_size,
            cross_batch_size=cross_batch_size,
            cross_max_length=cross_max_length,
            hf_home=hf_home,
            device=device,
        )
        pools = self._build_pools(
            rag_responses, rag_topics, encoders,
            tau=tau, min_sentence_words=min_sentence_words, vote_unit=vote_unit,
            num_threads=num_threads, block_size=block_size,
        )
        mode = detect_mode(list(rag_responses), attribution_mode)

        builder = LeaderboardBuilder(CORROBORATE_SPEC)
        n_rows = 0
        thread_cap = limit_threads(num_threads)
        thread_cap.__enter__()
        for topic in tqdm(rag_topics, "Judging topics"):
            tid = str(topic.request_id)
            reports = grouped.get(tid, [])
            if not reports:
                continue
            pool = pools[tid]
            bank = pool.bank(
                specificity_floor=specificity_floor,
                min_teams=min_teams,
                max_clusters=max_clusters,
            )
            scores = score_topic(
                reports, pool, bank, encoders,
                tau=tau,
                band_low=band_low,
                min_other_teams=min_other_teams,
                min_sentence_words=min_sentence_words,
                specificity_floor=specificity_floor,
                use_cross_encoder=use_cross_encoder and bool(active_cross),
                vote_unit=vote_unit,
            )

            cite_index = (
                build_citation_index(reports, vote_unit=vote_unit) if mode == "pool" else {}
            )
            cite: Dict[str, float] = {}
            for rep in reports:
                run_id = str(rep.metadata.run_id)
                if mode == "inline":
                    value, _judged, _cited = score_inline(
                        rep, encoders,
                        doc_chunk_chars=doc_chunk_chars,
                        max_doc_chunks=max_doc_chunks,
                        min_sentence_words=min_sentence_words,
                    )
                else:
                    value = score_pooled(
                        rep, cite_index, pool.n_teams,
                        min_sentence_words=min_sentence_words,
                        vote_unit=vote_unit,
                    )
                cite[run_id] = value

            nf1 = {
                run: harmonic(v["NCOV"], v["NPREC"]) for run, v in scores.items()
            }
            rank = within_topic_percentile(nf1)

            for rep in reports:
                run_id = str(rep.metadata.run_id)
                base = scores.get(run_id)
                if base is None:
                    # No scoreable content at all. Score it at the bottom rather
                    # than dropping the topic.
                    vals = zero_row()
                else:
                    vals = {
                        "NCOV": base["NCOV"],
                        "NPREC": base["NPREC"],
                        "NF1": nf1[run_id],
                        "NF1_RANK": rank[run_id],
                        "CITE": cite.get(run_id, 0.0),
                    }
                builder.add(run_id=run_id, topic_id=tid, values=vals)
                n_rows += 1

        thread_cap.__exit__(None, None, None)
        leaderboard = builder.build(
            expected_topic_ids=expected, on_missing=on_missing_evals
        )
        LeaderboardVerification(
            leaderboard, on_missing=on_missing_evals, warn=True
        ).all()
        runs = {e.run_id for e in leaderboard.entries}
        log.info(
            "leaderboard: %d rows, %d runs, %d topics, attribution=%s",
            n_rows, len(runs), len(expected), mode,
        )
        _warn_on_degenerate_measures(leaderboard, len(runs))
        return leaderboard


def _warn_on_degenerate_measures(leaderboard: Leaderboard, n_runs: int) -> None:
    """Shout if a measure cannot separate the systems.

    Two ways a measure silently becomes worthless, both of which we hit on the
    kiddie pool during development:

    - every run gets the same value, so the measure carries no ranking at all
      and any correlation computed from it is meaningless (tauap_b returns 0.0
      for an all-tied input);
    - many runs share a value, so the ordering among them is decided by input
      order, which is what tauap_b does with ties and it is effectively random.

    This is a warning rather than an error because a degenerate measure on a
    tiny pool is expected, whereas on the real datasets it means a parameter is
    wrong and the run should not be submitted.
    """
    all_id = leaderboard.all_topic_id
    for measure in leaderboard.measures:
        vals = [
            e.values[measure]
            for e in leaderboard.entries
            if e.topic_id == all_id and measure in e.values
        ]
        if not vals:
            log.warning("measure %s has no aggregate rows", measure)
            continue
        distinct = len({round(float(v), 12) for v in vals})
        if distinct <= 1:
            log.warning(
                "measure %s is CONSTANT across all %d runs: it carries no ranking "
                "and its correlations will be meaningless. Check the settings that "
                "feed it (min_other_teams, min_teams, specificity_floor).",
                measure, len(vals),
            )
        elif distinct < max(3, 0.5 * n_runs):
            log.warning(
                "measure %s has only %d distinct values over %d runs: ties will be "
                "broken by input order, which costs us most under tauap_b.",
                measure, distinct, len(vals),
            )
        else:
            log.info("measure %s: %d distinct values over %d runs", measure, distinct, len(vals))


if __name__ == "__main__":
    auto_judge_to_click_command(CorroborateJudge(), "corroborate")()
