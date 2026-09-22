#!/usr/bin/env python3
"""NuggetRecall: query-derived nugget recall, judged by an LLM.

The complement to Corroborate. On the pilots Corroborate is competitive on
long-form report generation (dragun-repgen rank 7 of 60) but near the bottom on
short RAG answers (rag25-gen rank 86 of 91), where the strongest public judges
build their nuggets from the question rather than from the pool of answers. This
judge does that.

One class serves both phases, as in Corroborate, so the runner reuses a single
instance. The nugget bank is written by the framework to <filebase>.nuggets.jsonl
straight after create_nuggets(), which means an interrupted judge phase resumes
without regenerating it.

Everything goes through minima-llm, whose SQLite prompt cache under $CACHE_DIR is
keyed on the model and the exact messages and deliberately excludes the endpoint
URL. That is what lets TIRA re-execute this judge with OPENAI_BASE_URL=EMPTY and get
identical output purely from the cache, which is the reproducibility proof a code
submission is checked against.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

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
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from .measures import MEASURE_NAMES, NUGGETRECALL_SPEC, zero_row
from .prompts import (
    Nugget,
    generation_messages,
    judge_messages,
    parse_judgments,
    parse_nuggets,
    request_text,
    score,
)

log = logging.getLogger(__name__)

# Appended on retries. Each one changes the prompt, and so the cache key, which is
# essential: minima-llm caches on the exact messages, so re-sending an identical
# prompt would replay the same unusable reply from the cache every time. The
# suffixes are fixed strings, so a replay from cache is still fully deterministic.
_RETRY_SUFFIX = (
    "",
    "\n\nReminder: follow the output format exactly, with no other text.",
    "\n\nImportant: output only the numbered lines, exactly as specified, nothing else.",
)


def _llm_backend(llm_config):
    """Build the minima-llm backend from the environment contract.

    llm_config.raw is always empty under the framework's environment-only
    configuration, so from_env() is the path that actually runs, exactly as in the
    starter kit's reference LLM judge.
    """
    from minima_llm import MinimaLlmConfig, OpenAIMinimaLlm

    raw = getattr(llm_config, "raw", None) if llm_config else None
    cfg = MinimaLlmConfig.from_dict(raw) if raw else MinimaLlmConfig.from_env()
    return OpenAIMinimaLlm(cfg)


def _run(backend, jobs: List[Tuple[str, List[Dict[str, str]]]], *, max_tokens: int,
         temperature: float) -> Dict[str, Optional[str]]:
    """Send a batch. Returns request_id -> reply text, or None on failure.

    Failures are counted and logged by type only. A failure message from the
    endpoint can echo the prompt, which holds restricted report text.
    """
    from minima_llm import MinimaLlmRequest, MinimaLlmResponse

    if not jobs:
        return {}
    reqs = [MinimaLlmRequest(request_id=rid, messages=msgs, temperature=temperature,
                             max_tokens=max_tokens) for rid, msgs in jobs]
    try:
        results = asyncio.run(backend.run_batched(reqs))
    except Exception as exc:
        log.warning("LLM batch failed entirely (%s), scoring affected items as unjudged",
                    type(exc).__name__)
        return {rid: None for rid, _ in jobs}
    out: Dict[str, Optional[str]] = {}
    failed = 0
    for (rid, _), res in zip(jobs, results):
        if isinstance(res, MinimaLlmResponse):
            out[rid] = res.text
        else:
            out[rid] = None
            failed += 1
    if failed:
        log.warning("%d of %d LLM calls failed", failed, len(jobs))
    return out


def _with_suffix(msgs: List[Dict[str, str]], attempt: int) -> List[Dict[str, str]]:
    if attempt == 0:
        return msgs
    head, last = msgs[:-1], dict(msgs[-1])
    last["content"] = last["content"] + _RETRY_SUFFIX[attempt]
    return head + [last]


class NuggetRecallJudge:
    """Query-derived nugget recall."""

    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks

    # ------------------------------------------------------------ nugget phase

    def create_nuggets(
        self,
        rag_responses: Optional[Sequence[Report]],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        corpus: Optional[str] = None,
        filebase: str = "default",
        outdir: Path = Path("."),
        *,
        min_nuggets: int = 8,
        max_nuggets: int = 20,
        max_tokens_generate: int = 1536,
        temperature: float = 0.0,
        retries: int = 2,
        **kwargs: Any,
    ) -> Optional[NuggetBanksProtocol]:
        """Generate an answer key per topic from the request alone.

        Uses no responses at all, so the bank is readable under the data policy for
        every topic, and `nugget_depends_on_responses: false` passes None here.
        """
        topics = sorted(rag_topics, key=lambda t: str(t.request_id))
        backend = _llm_backend(llm_config)
        requests = {
            str(t.request_id): request_text(t.title, t.problem_statement, t.background)
            for t in topics
        }
        parsed: Dict[str, List[Nugget]] = {}
        pending = list(requests)
        for attempt in range(retries + 1):
            if not pending:
                break
            jobs = [(f"gen:{tid}:{attempt}",
                     _with_suffix(generation_messages(requests[tid], min_nuggets, max_nuggets),
                                  attempt))
                    for tid in pending]
            replies = _run(backend, jobs, max_tokens=max_tokens_generate, temperature=temperature)
            still = []
            for tid in pending:
                ns = parse_nuggets(replies.get(f"gen:{tid}:{attempt}") or "", max_nuggets)
                if ns:
                    parsed[tid] = ns
                else:
                    still.append(tid)
            pending = still

        if pending:
            log.warning("%d of %d topics produced no usable answer key after %d attempts; "
                        "those use the request title as a single vital fact",
                        len(pending), len(topics), retries + 1)

        creator = Creator(is_human=False, llm_prompt_strategy="query-only-answer-key",
                          llm_model=getattr(llm_config, "model", None))
        banks: List[NuggetBank] = []
        for t in topics:
            tid = str(t.request_id)
            ns = parsed.get(tid) or [Nugget(text=(t.title or tid).strip(), vital=True)]
            bank = NuggetBank(query_id=tid, title_query=t.title)
            bank.add_nuggets([
                NuggetClaim.from_lazy(
                    query_id=tid, claim=n.text,
                    importance="vital" if n.vital else "okay",
                    metadata={"index": i, "fallback": tid not in parsed},
                )
                for i, n in enumerate(ns)
            ])
            bank.add_creator(creator)
            banks.append(bank)
        n_vital = sum(sum(1 for n in v if n.vital) for v in parsed.values())
        n_all = sum(len(v) for v in parsed.values())
        log.info("answer keys: %d topics, %d facts, %d vital", len(parsed), n_all, n_vital)
        return NuggetBanks.from_banks_list(banks)

    def create_qrels(self, *args: Any, **kwargs: Any) -> Optional[Qrels]:
        """Not implemented: a qrels file beside the leaderboard breaks tira-cli's
        upload format check. See judges/corroborate for the full story."""
        return None

    # ------------------------------------------------------------- judge phase

    @staticmethod
    def _nuggets_from_bank(bank) -> List[Nugget]:
        claims = list((getattr(bank, "claim_bank", None) or {}).values()) \
            if isinstance(getattr(bank, "claim_bank", None), dict) \
            else list(getattr(bank, "claim_bank", None) or [])
        claims.sort(key=lambda c: (c.metadata or {}).get("index", 0))
        return [Nugget(text=c.claim, vital=(str(c.importance).lower() == "vital"))
                for c in claims]

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
        max_tokens_judge: int = 1024,
        max_answer_chars: int = 40000,
        temperature: float = 0.0,
        retries: int = 2,
        on_missing_evals: str = "fix_aggregate",
        **kwargs: Any,
    ) -> Leaderboard:
        expected = [str(t.request_id) for t in rag_topics]
        allowed = set(expected)
        topic_by_id = {str(t.request_id): t for t in rag_topics}
        if nugget_banks is None:
            raise ValueError("NuggetRecall needs its answer keys: set judge_uses_nuggets: true")
        keys: Dict[str, List[Nugget]] = {
            tid: self._nuggets_from_bank(nugget_banks.banks[tid])
            for tid in expected if tid in nugget_banks.banks
        }

        # Deterministic order, so prompts and cache keys are identical across runs.
        reports = sorted(
            (r for r in rag_responses if str(r.metadata.topic_id) in allowed),
            key=lambda r: (str(r.metadata.run_id), str(r.metadata.topic_id)),
        )
        backend = _llm_backend(llm_config)
        results: Dict[Tuple[str, str], Dict[str, float]] = {}
        todo: List[Tuple[str, str, List[Dict[str, str]], int]] = []
        truncated = 0
        for r in reports:
            run_id, tid = str(r.metadata.run_id), str(r.metadata.topic_id)
            answer = " ".join(s for s in (r.get_sentences() or []) if s and s.strip())
            nuggets = keys.get(tid) or []
            if not answer.strip() or not nuggets:
                results[(run_id, tid)] = zero_row()
                continue
            if len(answer) > max_answer_chars:
                answer = answer[:max_answer_chars]
                truncated += 1
            t = topic_by_id[tid]
            req = request_text(t.title, t.problem_statement, t.background)
            todo.append((run_id, tid, judge_messages(req, answer, nuggets), len(nuggets)))
        if truncated:
            log.warning("%d answers exceeded %d characters and were truncated",
                        truncated, max_answer_chars)

        pending = todo
        for attempt in range(retries + 1):
            if not pending:
                break
            jobs = [(f"judge:{run}:{tid}:{attempt}", _with_suffix(msgs, attempt))
                    for run, tid, msgs, _ in pending]
            replies = _run(backend, jobs, max_tokens=max_tokens_judge, temperature=temperature)
            still = []
            for (run, tid, msgs, count) in pending:
                lab = parse_judgments(replies.get(f"judge:{run}:{tid}:{attempt}") or "", count)
                if lab is None:
                    still.append((run, tid, msgs, count))
                else:
                    results[(run, tid)] = score(lab, keys[tid])
            pending = still

        if pending:
            # Scored at zero rather than dropped: every (run, topic) must appear, and
            # a partial submission cannot be accepted. Counted so a high rate is seen.
            log.warning("%d of %d judgments stayed unparsable after %d attempts and "
                        "score zero", len(pending), len(todo), retries + 1)
            for run, tid, _, _ in pending:
                results[(run, tid)] = zero_row()

        builder = LeaderboardBuilder(NUGGETRECALL_SPEC)
        for r in reports:
            key = (str(r.metadata.run_id), str(r.metadata.topic_id))
            builder.add(run_id=key[0], topic_id=key[1], values=results.get(key, zero_row()))
        leaderboard = builder.build(expected_topic_ids=expected, on_missing=on_missing_evals)
        LeaderboardVerification(leaderboard, on_missing=on_missing_evals, warn=True).all()
        log.info("leaderboard: %d reports judged, %d unparsable", len(todo), len(pending))
        return leaderboard


if __name__ == "__main__":
    auto_judge_to_click_command(NuggetRecallJudge(), "nuggetrecall")()
