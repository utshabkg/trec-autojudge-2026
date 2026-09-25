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
import time
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

from .measures import (
    MEASURE_NAMES,
    NUGGETRECALL_SPEC,
    RANK_BASE,
    within_topic_percentile,
    zero_row,
)
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

# Batch-level retries, which are not the same thing as the prompt retries above.
# These re-send the identical batch after a transport-level throw, so the cache
# replays everything that already succeeded.
_BATCH_RETRIES = 2
_BATCH_BACKOFF_S = 5.0

# minima-llm settings this judge decides for itself, per phase.
#
# These must NOT be applied with setdefault. _llm_backend runs once per phase, so
# the nugget phase's decision would stick to the judge phase: one preflight miss
# while vLLM is still loading pins TIMEOUT_S=5 and MAX_ATTEMPTS=1 onto a judge
# phase that then runs against a perfectly healthy server. Every call needing more
# than 5 s scores zero, and because the zeros land on the longest answers the
# leaderboard still varies, so the constancy guard never fires and the file
# uploads. Instead we record once what the operator set before we touched
# anything, never overwrite those, and assign the rest per phase.
_TUNABLES = ("MAX_ATTEMPTS", "TIMEOUT_S", "COOLDOWN_CAP_S", "MAX_BACKOFF_S",
             "BATCH_MAX_FAILURES", "RPM")
_OPERATOR_TUNING = None


def _apply_tuning(policy: Dict[str, str]) -> None:
    """Set this phase's minima-llm policy, leaving operator overrides alone."""
    import os

    global _OPERATOR_TUNING
    if _OPERATOR_TUNING is None:
        _OPERATOR_TUNING = {k: os.environ[k] for k in _TUNABLES if k in os.environ}
    for key, value in policy.items():
        if key not in _OPERATOR_TUNING:
            os.environ[key] = value


# Where an unconfigured judge points. Port 9 is the discard service and is closed
# on any normal host, so a call fails in milliseconds instead of hanging.
_NO_ENDPOINT = "http://127.0.0.1:9/v1"


# How long to wait for an operator-supplied http(s) endpoint that is not answering
# yet. It used to be about four seconds, three quick tries, and the whole phase was
# then run in fast-fail mode. A server still loading its model when our container
# starts, or one of the "short hiccups" the organizers told us their cluster has,
# would therefore zero an entire phase with nothing more than a warning line. Five
# minutes covers a model load. It costs nothing when the endpoint is up, since the
# first try returns, and nothing in TIRA's replay, where the URL is EMPTY and is
# rejected before any request is made.
_ENDPOINT_PATIENCE_S = 300.0

# Statuses that mean "a gateway is here, the model behind it is not ready yet".
# A proxy or ingress in front of a loading server answers these at once, and
# treating that answer as "up" would skip the whole wait on exactly the setup a
# university cluster is likely to have.
_NOT_READY_STATUSES = frozenset({502, 503, 504})


def _endpoint_is_usable_url(base_url: str) -> bool:
    """Could this URL ever answer? EMPTY, blanks and port 9 cannot.

    Port 9 is matched by host and port rather than by exact string, because the
    placeholder turns up with and without a /v1 suffix, and treating a variant as a
    real endpoint would mean waiting minutes on a port that is closed by design.
    """
    from urllib.parse import urlsplit

    b = (base_url or "").strip()
    if not b.lower().startswith(("http://", "https://")):
        return False
    try:
        parts = urlsplit(b)
        return not (parts.port == 9 and parts.hostname in ("127.0.0.1", "localhost"))
    except ValueError:
        return False


def _endpoint_reachable(base_url: str, api_key: Optional[str], *, attempts: int = 3,
                        timeout: float = 5.0, patience_s: Optional[float] = None) -> bool:
    """Preflight: does anything answer at this endpoint?

    A URL that can never answer (EMPTY, which is TIRA's replay value, a blank, or
    our port-9 placeholder) is unreachable at once, with no request and no sleep.

    Any other HTTP response counts as reachable, including 401 or 404, because it
    proves a server is there. The exception is 502, 503 and 504 while we are being
    patient: that is a gateway whose model is still loading, so we keep waiting,
    and if patience runs out we still report reachable, since something answers.

    A real http(s) URL that does not answer at all is polled for up to
    `patience_s` with growing sleeps. With patience off it gets `attempts` quick
    tries, as before.
    """
    import os
    import time
    import urllib.error
    import urllib.request

    if not _endpoint_is_usable_url(base_url):
        return False
    if patience_s is None:
        patience_s = float(os.environ.get("NUGGETRECALL_ENDPOINT_PATIENCE_S",
                                          _ENDPOINT_PATIENCE_S))
    url = base_url.strip().rstrip("/") + "/models"
    patient = patience_s > 0
    start = time.monotonic()
    i = 0
    while True:
        try:
            req = urllib.request.Request(
                url, headers={"Authorization": f"Bearer {api_key or 'EMPTY'}"})
            urllib.request.urlopen(req, timeout=timeout).close()
            return True
        except urllib.error.HTTPError as exc:
            if not (patient and exc.code in _NOT_READY_STATUSES):
                return True
            if time.monotonic() - start >= patience_s:
                return True     # still a gateway answering, so call it reachable
            answered = True
        except Exception:
            answered = False
        i += 1
        waited = time.monotonic() - start
        if not patient:
            if i >= attempts:
                return False
            time.sleep(2)
            continue
        if waited >= patience_s:
            return answered
        if i == attempts:
            log.warning("LLM endpoint not ready yet, waiting up to %.0f s for it", patience_s)
        time.sleep(min(2.0 * i, 30.0, max(0.0, patience_s - waited)))


def _served_models(base_url: str, api_key: Optional[str], *, timeout: float = 5.0):
    """Model ids the endpoint lists, or None if it will not say.

    Model ids are not restricted content, so this is safe to log. None means the
    endpoint does not answer /models usefully, which is normal for a gateway, and
    is treated as "cannot tell" rather than as a mismatch.
    """
    import json
    import urllib.request

    try:
        req = urllib.request.Request(
            (base_url or "").rstrip("/") + "/models",
            headers={"Authorization": f"Bearer {api_key or 'EMPTY'}"})
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            data = json.load(fh)
        ids = [m["id"] for m in data.get("data", []) if isinstance(m, dict) and "id" in m]
        return ids or None
    except Exception:
        return None


def _llm_backend(llm_config):
    """Build the minima-llm backend from the environment contract, never raising.

    llm_config.raw is always empty under the framework's environment-only
    configuration, so from_env() is the path that actually runs, exactly as in the
    starter kit's reference LLM judge.

    from_env() raises RuntimeError when OPENAI_BASE_URL or OPENAI_MODEL is unset.
    That happens in the code-submission dry run, where no endpoint is forwarded,
    and a crash there means no leaderboard and a failed submission. The
    organizers' guidance is explicit that a judge must warn rather than raise,
    because TIRA's deterministic replay runs it with no working endpoint and must
    still complete from the prompt cache.

    So a missing endpoint is replaced with a closed local port. The backend is still
    built, which matters: prompt-cache hits are served without touching the
    endpoint, so a judge with a warm cache completes normally, and only cache misses
    fail, fast, and fall back.
    """
    import os

    from minima_llm import MinimaLlmConfig, OpenAIMinimaLlm

    raw = getattr(llm_config, "raw", None) if llm_config else None
    if raw:
        return OpenAIMinimaLlm(MinimaLlmConfig.from_dict(raw))
    if not os.environ.get("OPENAI_BASE_URL") and not os.environ.get("OPENAI_API_BASE"):
        log.warning("no LLM endpoint configured (OPENAI_BASE_URL unset)")
        os.environ["OPENAI_BASE_URL"] = _NO_ENDPOINT

    # Preflight, and fail fast only when nothing answers.
    #
    # minima-llm retries a failed call up to 50 times by default, with backoff up
    # to 20 s and a cooldown up to 60 s. That is the right behaviour against a live
    # server that hiccups, and a disaster against one that is not there: the first
    # NuggetRecall dry run sat in the submission step for over 50 minutes, because
    # CI set OPENAI_BASE_URL to a closed port and the fast-fail below used to apply
    # only when the variable was unset. It is now decided by whether anything
    # answers, so it covers an unset endpoint, a placeholder, TIRA's replay value
    # EMPTY, and a real endpoint that is down, and it never throttles a live one.
    # Cache hits are still served in every case, since they need no network.
    base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or ""
    reachable = _endpoint_reachable(base, os.environ.get("OPENAI_API_KEY"))
    if not reachable:
        log.warning("LLM endpoint is not reachable. Cached judgments are still used, "
                    "and anything not in the cache fails fast, falls back and scores zero")
        # The batch breaker is OFF here too. It used to be left at 25 on the theory
        # that with nothing answering every miss fails anyway, so aborting early
        # reaches the same all-zero result faster. That was wrong, because a batch
        # holds cache HITS as well as misses, and minima-llm's abort throws away
        # every result in the batch. This branch is exactly TIRA's replay
        # (OPENAI_BASE_URL=EMPTY), which must reproduce the first run from cache.
        # Any call that failed in the first run left no cache entry, so a first run
        # with 26 transport failures in one batch, one server restart drops more
        # than that, would replay as a constant, all-zero leaderboard. With the
        # breaker off the misses simply fail one by one, which with EMPTY happens
        # before any I/O, and every hit is kept.
        #
        # RPM 0 is minima-llm's "unlimited" (wait_turn returns at once when the
        # effective rate is <= 0). With the breaker off, every miss would
        # otherwise queue at the default 600 per minute before failing, so a
        # replay of a first run with thousands of failed calls would crawl.
        _apply_tuning({"MAX_ATTEMPTS": "1", "TIMEOUT_S": "5",
                       "COOLDOWN_CAP_S": "0", "MAX_BACKOFF_S": "0",
                       "BATCH_MAX_FAILURES": "none", "RPM": "0"})
    else:
        # Defaults tuned for a reasoning model behind a local server, which is
        # what the organizers run submitted code against. Measured, not guessed:
        # gpt-oss-20b on rag25-gen scored 0.000 and ranked last under the stock
        # defaults, with all 5302 judgments lost.
        #
        # TIMEOUT_S is 60 s by default. A reasoning model with a 4096-token reply
        # budget, behind 32 outstanding requests, does not answer a long judge
        # prompt in 60 s, so calls time out that would have succeeded.
        #
        # BATCH_MAX_FAILURES is 25 by default, and it aborts the *whole* batch,
        # not the failing calls. On 5302 judgments that is a 0.5% failure rate
        # taking down 100% of the work. We already handle a failed call properly:
        # it is retried, then scored zero, and both are counted in the log. A
        # whole-batch abort is strictly worse than that, so the breaker is turned
        # off and our own accounting is what reports trouble.
        #
        # These change no cache key, which is (model, messages, temperature,
        # max_tokens), so TIRA's replay-from-cache check is unaffected. An
        # operator's own value for any of them still wins.
        # RPM is set back to minima-llm's own default of 600 explicitly, because
        # the unreachable branch may have set it to 0 in an earlier phase of this
        # same process, and against a live server the pacing is wanted.
        _apply_tuning({"TIMEOUT_S": "900", "MAX_ATTEMPTS": "4",
                       "COOLDOWN_CAP_S": "60", "MAX_BACKOFF_S": "20",
                       "BATCH_MAX_FAILURES": "none", "RPM": "600"})
    if not os.environ.get("OPENAI_MODEL") and not os.environ.get("LLM_MODEL"):
        os.environ["OPENAI_MODEL"] = getattr(llm_config, "model", None) or "unconfigured"

    # Say so, loudly, when we are about to ask for a model the endpoint does not
    # have. Every such call 404s, every judgment then scores zero, and the run
    # looks exactly like a judge that cannot judge. It has already happened once:
    # the launcher exported Qwen3-30B while the server was on gpt-oss-20b, and the
    # whole rag25-gen run came back 0.000. An endpoint that will not list its
    # models tells us nothing, so that case stays silent rather than crying wolf.
    want = os.environ.get("OPENAI_MODEL") or os.environ.get("LLM_MODEL")
    served = _served_models(base, os.environ.get("OPENAI_API_KEY")) if reachable else None
    if served and want and want not in served:
        log.error("the endpoint does not serve %r, it serves %s. Every call will "
                  "fail and every judgment will score zero. Fix OPENAI_MODEL.",
                  want, ", ".join(repr(s) for s in served[:5]))
    try:
        return OpenAIMinimaLlm(MinimaLlmConfig.from_env())
    except Exception as exc:
        log.warning("could not build the LLM backend (%s); every call falls back",
                    type(exc).__name__)
        return None


def _run(backend, jobs: List[Tuple[str, List[Dict[str, str]]]], *, max_tokens: int,
         temperature: float, batch_retries: int = _BATCH_RETRIES) -> Dict[str, Optional[str]]:
    """Send a batch. Returns request_id -> reply text, or None on failure.

    Failures are counted and logged by type only. A failure message from the
    endpoint can echo the prompt, which holds restricted report text.

    run_batched can raise for the whole batch rather than per item, which we have
    seen once on gpt-oss-20b at the start of the judge phase. That must not lose
    thousands of answers at once, so a batch-level exception is retried with the
    *same* requests: identical prompts hit the cache, so whatever already
    succeeded replays for free and only the rest is re-sent. This is why the
    retry lives here and not in the caller, whose retries deliberately change the
    prompt to defeat the cache.
    """
    from minima_llm import MinimaLlmRequest, MinimaLlmResponse

    if not jobs:
        return {}
    if backend is None:
        return {rid: None for rid, _ in jobs}
    reqs = [MinimaLlmRequest(request_id=rid, messages=msgs, temperature=temperature,
                             max_tokens=max_tokens) for rid, msgs in jobs]
    results = None
    for attempt in range(batch_retries + 1):
        try:
            results = asyncio.run(backend.run_batched(reqs))
            break
        except Exception as exc:
            log.warning("LLM batch raised %s (attempt %d of %d)",
                        type(exc).__name__, attempt + 1, batch_retries + 1)
            if attempt < batch_retries:
                time.sleep(_BATCH_BACKOFF_S * (attempt + 1))
    if results is None:
        log.warning("LLM batch failed entirely, scoring %d items as unjudged", len(jobs))
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


# The reply budget grows on retries, because the commonest reason a reply cannot be
# read is that a reasoning model ran out of budget while still thinking and never
# wrote the answer. Measured on rag26 with gpt-oss-20b: 5,020 of 13,275 requests
# ended with finish_reason "length", 2,692 of 9,838 first-pass judgments (27%) were
# unreadable, and because every retry used the same 4,096 tokens the hard cases just
# ran out again. Rewording the prompt, which is all a retry used to do, cannot help a
# model that needed more room.
#
# The first attempt keeps the configured budget exactly. That is what keeps every
# existing prompt cache valid, since max_tokens is part of the cache key: the lodged
# submissions, the CI rehearsal's warmed cache, and any replay all hit as before.
# The schedule is fixed, so a replay rebuilds the same retry keys and still hits.
_RETRY_BUDGET_FACTORS = (1, 2, 4)
_MAX_REPLY_BUDGET = 16384


def _budget_for(attempt: int, base: int) -> int:
    """Reply budget for this attempt: the configured value, then doubled, then 4x.

    Never below the configured value, so an operator who set a large budget is not
    cut back, and capped so a long prompt plus the reply still fits a 32k context.
    """
    factor = _RETRY_BUDGET_FACTORS[min(attempt, len(_RETRY_BUDGET_FACTORS) - 1)]
    return max(base, min(base * factor, _MAX_REPLY_BUDGET))


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
        max_tokens_generate: int = 4096,
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
            replies = _run(backend, jobs, max_tokens=_budget_for(attempt, max_tokens_generate),
                           temperature=temperature)
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
        max_tokens_judge: int = 4096,
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
            replies = _run(backend, jobs, max_tokens=_budget_for(attempt, max_tokens_judge),
                           temperature=temperature)
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

        # NR_RANK, added last because it is the only measure that needs every
        # other run's score for the same topic.
        #
        # Measured on the pilots and the reason this exists: on rag25-gen the same
        # judgments aggregated this way score 0.809 max Kendall against 0.788 for
        # the plain mean, and better on tauap_b and Min(Kendall) too. Averaging a
        # recall fraction over topics lets an easy topic, where everyone scores
        # high, count as much as a discriminating one. A within-topic rank removes
        # that. On ragtime25 it is slightly worse than the mean (0.726 against
        # 0.739), which costs nothing: the reported score is the max over our
        # measures, so the mean simply wins there instead.
        rows = {}
        for r in reports:
            key = (str(r.metadata.run_id), str(r.metadata.topic_id))
            rows[key] = dict(results.get(key, zero_row()))
        by_topic = {}
        for (run_id, topic_id), values in rows.items():
            by_topic.setdefault(topic_id, {})[run_id] = values[RANK_BASE]
        for topic_id, scores in by_topic.items():
            for run_id, pct in within_topic_percentile(scores).items():
                rows[(run_id, topic_id)]["NR_RANK"] = pct

        builder = LeaderboardBuilder(NUGGETRECALL_SPEC)
        for r in reports:
            key = (str(r.metadata.run_id), str(r.metadata.topic_id))
            builder.add(run_id=key[0], topic_id=key[1], values=rows[key])
        leaderboard = builder.build(expected_topic_ids=expected, on_missing=on_missing_evals)
        LeaderboardVerification(leaderboard, on_missing=on_missing_evals, warn=True).all()
        log.info("leaderboard: %d reports judged, %d unparsable", len(todo), len(pending))
        return leaderboard


if __name__ == "__main__":
    auto_judge_to_click_command(NuggetRecallJudge(), "nuggetrecall")()
