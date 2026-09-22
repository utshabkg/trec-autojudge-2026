"""Unit tests for the NuggetRecall judge.

The LLM is stubbed throughout, so these need no model, no GPU and no network,
which matters because tira-cli runs pytest on the machine making the submission.
"""

from __future__ import annotations

import pytest

from autojudge_base import Request
from autojudge_base.report import Report, ReportMetaData
from autojudge_base.report_sentences import NeuclirReportSentence

import judges.nuggetrecall.judge as J
import judges.nuggetrecall.prompts as J_PROMPTS
from judges.nuggetrecall.measures import MEASURE_NAMES
from judges.nuggetrecall.prompts import (
    Nugget, parse_judgments, parse_nuggets, score, strip_reasoning,
)


# _run imports minima_llm lazily. Use the real package when it is installed, so
# the request objects keep the real signature, and a minimal stand-in otherwise,
# so the suite still runs on a machine without an LLM backend.
try:  # pragma: no cover - depends on the environment
    import minima_llm as _MOD
except ImportError:  # pragma: no cover
    import sys
    import types
    from dataclasses import dataclass, field

    _MOD = types.ModuleType("minima_llm")

    @dataclass
    class _Req:
        request_id: str
        messages: list = field(default_factory=list)
        temperature: float = 0.0
        max_tokens: int = 0

    @dataclass
    class _Resp:
        request_id: str
        text: str

    _MOD.MinimaLlmRequest, _MOD.MinimaLlmResponse = _Req, _Resp
    sys.modules["minima_llm"] = _MOD


class _NoSleep:
    """Stands in for the time module so batch backoff does not slow the suite."""

    @staticmethod
    def sleep(_seconds):
        return None


TOPIC = Request(request_id="t1", title="how do bees make honey")
KEY = ("1. [VITAL] Bees collect nectar from flowers.\n"
       "2. [VITAL] Bees add enzymes to nectar.\n"
       "3. [OKAY] Bees fan their wings to dry nectar.\n")


def _report(run, team, sents, topic="t1"):
    return Report(
        metadata=ReportMetaData(team_id=team, run_id=run, topic_id=topic),
        responses=[NeuclirReportSentence(text=t, citations=[]) for t in sents],
    )


# ------------------------------------------------------------------ parsing


def test_parse_nuggets_tolerates_bold_and_drops_duplicates():
    raw = ("1. [VITAL] Fact one is here.\n2. **[OKAY]** Fact two is here.\n"
           "3. [VITAL] Fact one is here.\n4. **[VITAL] Fact three is here.**")
    ns = parse_nuggets(raw)
    assert [n.text for n in ns] == ["Fact one is here.", "Fact two is here.", "Fact three is here."]
    assert [n.vital for n in ns] == [True, False, True]


def test_parse_nuggets_caps_the_key_so_it_cannot_inflate():
    raw = "\n".join(f"{i}. [VITAL] Distinct fact number {i}." for i in range(1, 40))
    assert len(parse_nuggets(raw, max_n=20)) == 20


def test_parse_nuggets_rejects_a_key_too_small_to_use():
    assert parse_nuggets("1. [VITAL] Only one fact here.") == []
    assert parse_nuggets("no list at all") == []


def test_reasoning_blocks_are_stripped():
    """The organizers re-run against gpt-oss-20b, a reasoning model."""
    assert strip_reasoning("<think>1. [VITAL] leaked</think>real") == "real"
    ns = parse_nuggets("<think>1. [VITAL] leaked fact</think>\n1. [VITAL] Real fact one.\n2. [OKAY] Real fact two.")
    assert [n.text for n in ns] == ["Real fact one.", "Real fact two."]


def test_parse_judgments_accepts_label_variants():
    lab = parse_judgments("1: support\n2) partial\n3. not support\n4: **no**", 4)
    assert lab == {1: "support", 2: "partial", 3: "no", 4: "no"}


def test_parse_judgments_rejects_a_truncated_reply():
    """Back-filling unlabelled facts as 'no' would silently invent misses."""
    assert parse_judgments("1: support", 10) is None


def test_score_formulas():
    ns = [Nugget("a", True), Nugget("b", True), Nugget("c", False)]
    s = score({1: "support", 2: "partial", 3: "no"}, ns)
    assert s["NR_VITAL"] == pytest.approx(0.75)          # (1 + 0.5) / 2
    assert s["NR_ALL"] == pytest.approx(0.5)             # (1 + 0.5 + 0) / 3
    assert s["NR_VITAL_STRICT"] == pytest.approx(0.5)    # only full support
    assert s["NR_WEIGHTED"] == pytest.approx(1.5 / 2.5)  # weights 1, 1, 0.5


def test_score_without_vital_items_falls_back_to_all():
    ns = [Nugget("a", False), Nugget("b", False)]
    assert score({1: "support", 2: "no"}, ns)["NR_VITAL"] == pytest.approx(0.5)


# ------------------------------------------------------- the judge, stubbed


class FakeLLM:
    """Records every prompt, answers from a script keyed by request id prefix."""

    def __init__(self, gen=KEY, judge="1: support\n2: partial\n3: no", fail_first=0):
        self.gen, self.judge_reply, self.fail_first = gen, judge, fail_first
        self.prompts = []
        self.calls = 0

    def run(self, backend, jobs, *, max_tokens, temperature):
        out = {}
        for rid, msgs in jobs:
            self.prompts.append((rid, msgs[-1]["content"]))
            self.calls += 1
            if rid.startswith("gen:"):
                out[rid] = self.gen
            elif self.fail_first and rid.endswith(":0"):
                out[rid] = "I cannot comply"
            else:
                out[rid] = self.judge_reply
        return out


@pytest.fixture
def fake(monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr(J, "_run", llm.run)
    monkeypatch.setattr(J, "_llm_backend", lambda cfg: object())
    return llm


def _run_judge(reports, topics):
    j = J.NuggetRecallJudge()
    banks = j.create_nuggets(None, topics, llm_config=None)
    return banks, j.judge(reports, topics, llm_config=None, nugget_banks=banks)


def test_answer_key_comes_from_the_request_alone(fake):
    """create_nuggets must not need responses: the workflow passes None."""
    banks, _ = _run_judge([_report("r1", "a", ["Bees collect nectar."])], [TOPIC])
    assert len(banks.banks["t1"].claim_bank) == 3
    gen_prompts = [p for rid, p in fake.prompts if rid.startswith("gen:")]
    assert gen_prompts and "Bees collect nectar." not in gen_prompts[0]


def test_every_run_and_topic_is_scored_and_empty_reports_score_zero(fake):
    reports = [_report("r1", "a", ["Bees collect nectar."]), _report("r2", "b", [])]
    _, lb = _run_judge(reports, [TOPIC])
    rows = {e.run_id: e.values for e in lb.entries if e.topic_id == "t1"}
    assert set(rows) == {"r1", "r2"}
    assert all(rows["r2"][m] == 0.0 for m in MEASURE_NAMES)
    assert rows["r1"]["NR_VITAL"] == pytest.approx(0.75)
    assert all(set(e.values) == set(MEASURE_NAMES) for e in lb.entries)


def test_a_retry_sends_a_different_prompt(monkeypatch):
    """Regression guard for the cache-replay trap.

    minima-llm caches on the exact messages. Re-sending an identical prompt after
    an unparsable reply would replay the same bad reply from the cache forever. A
    retry must therefore change the prompt, deterministically.
    """
    llm = FakeLLM(fail_first=1)
    monkeypatch.setattr(J, "_run", llm.run)
    monkeypatch.setattr(J, "_llm_backend", lambda cfg: object())
    _, lb = _run_judge([_report("r1", "a", ["Bees collect nectar."])], [TOPIC])
    judge_prompts = [p for rid, p in llm.prompts if rid.startswith("judge:")]
    assert len(judge_prompts) == 2
    assert judge_prompts[0] != judge_prompts[1]
    row = next(e.values for e in lb.entries if e.run_id == "r1" and e.topic_id == "t1")
    assert row["NR_VITAL"] == pytest.approx(0.75), "the retry's good reply must be used"


class _ThrowingBackend:
    """Raises for the whole batch the first n times, then answers everything."""

    def __init__(self, throws):
        self.throws, self.seen = throws, []

    async def run_batched(self, reqs):
        self.seen.append([tuple(m["content"] for m in r.messages) for r in reqs])
        if self.throws > 0:
            self.throws -= 1
            raise RuntimeError("Event loop is closed")
        return [_MOD.MinimaLlmResponse(request_id=r.request_id, text="ok") for r in reqs]


def _jobs(n=3):
    return [(f"judge:r{i}:t1:0", [{"role": "user", "content": f"q{i}"}]) for i in range(n)]


def test_a_thrown_batch_is_resent_unchanged(monkeypatch):
    """Regression guard for the whole-batch throw seen on gpt-oss-20b.

    run_batched can raise for the entire batch rather than per item. Losing
    thousands of judgments to one transport error would score them all zero, so
    _run resends. The resend must use the *identical* prompts, because that is
    what lets the cache replay whatever already succeeded. This is the opposite
    of the prompt retries above, which must change the prompt.
    """
    monkeypatch.setattr(J, "time", _NoSleep)
    b = _ThrowingBackend(throws=1)
    out = J._run(b, _jobs(), max_tokens=64, temperature=0.0)
    assert len(b.seen) == 2, "the batch must be resent once"
    assert b.seen[0] == b.seen[1], "the resend must not change the prompts"
    assert all(v == "ok" for v in out.values())


def test_a_batch_that_keeps_throwing_scores_zero_rather_than_crashing(monkeypatch):
    monkeypatch.setattr(J, "time", _NoSleep)
    b = _ThrowingBackend(throws=99)
    out = J._run(b, _jobs(), max_tokens=64, temperature=0.0)
    assert len(b.seen) == J._BATCH_RETRIES + 1
    assert set(out) == {rid for rid, _ in _jobs()} and all(v is None for v in out.values())


def test_unparsable_after_all_retries_scores_zero_not_dropped(monkeypatch):
    llm = FakeLLM(judge="garbage with no labels")
    monkeypatch.setattr(J, "_run", llm.run)
    monkeypatch.setattr(J, "_llm_backend", lambda cfg: object())
    _, lb = _run_judge([_report("r1", "a", ["Bees collect nectar."])], [TOPIC])
    row = next(e.values for e in lb.entries if e.run_id == "r1" and e.topic_id == "t1")
    recall = [m for m in MEASURE_NAMES if m != "NR_RANK"]
    assert all(row[m] == 0.0 for m in recall)
    # NR_RANK is a rank, not a recall. One run on a topic carries no ranking
    # information, so 0.5 is right and 0.0 would be a claim we cannot make.
    assert row["NR_RANK"] == 0.5


def test_prompts_are_byte_identical_across_runs(fake):
    """Deterministic prompts are what make the TIRA cache replay work."""
    reports = [_report("r2", "b", ["Bees add enzymes."]), _report("r1", "a", ["Bees collect nectar."])]
    _run_judge(reports, [TOPIC])
    first = list(fake.prompts)
    fake.prompts.clear()
    _run_judge(list(reversed(reports)), [TOPIC])
    assert fake.prompts == first, "input order must not change the prompts or their order"


def test_a_topic_with_no_usable_key_falls_back_rather_than_crashing(monkeypatch):
    llm = FakeLLM(gen="no list here")
    monkeypatch.setattr(J, "_run", llm.run)
    monkeypatch.setattr(J, "_llm_backend", lambda cfg: object())
    banks, lb = _run_judge([_report("r1", "a", ["Bees collect nectar."])], [TOPIC])
    assert len(banks.banks["t1"].claim_bank) == 1, "bank must never be empty"
    assert any(e.run_id == "r1" for e in lb.entries)


def test_qrels_creation_is_disabled():
    assert J.NuggetRecallJudge().create_qrels() is None


# ------------------------------------------------- no endpoint configured


def test_completes_with_no_endpoint_configured(monkeypatch, tmp_path):
    """The code-submission dry run forwards no endpoint at all.

    MinimaLlmConfig.from_env() raises when OPENAI_BASE_URL is unset, so without
    handling the judge would crash, write no leaderboard, and fail the submission.
    It must instead complete, with every (run, topic) present.
    """
    for var in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_MODEL", "LLM_MODEL",
                "MAX_ATTEMPTS", "TIMEOUT_S"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    reports = [_report("r1", "a", ["Bees collect nectar."]), _report("r2", "b", ["Bees make honey."])]
    j = J.NuggetRecallJudge()
    banks = j.create_nuggets(None, [TOPIC], llm_config=None)
    lb = j.judge(reports, [TOPIC], llm_config=None, nugget_banks=banks)
    runs = {e.run_id for e in lb.entries if e.topic_id == "t1"}
    assert runs == {"r1", "r2"}
    assert len(banks.banks["t1"].claim_bank) >= 1, "bank must not be empty"


def test_a_set_but_dead_endpoint_fails_fast(monkeypatch, tmp_path):
    """Regression: a placeholder endpoint used to stall the judge for over an hour.

    The CI dry run sets OPENAI_BASE_URL to a closed port. minima-llm retries each
    failed call up to 50 times with backoff, and the fast-fail path used to apply
    only when the variable was unset, so the dry run sat in the submission step for
    more than 50 minutes. With the preflight, an endpoint that does not answer is
    detected up front and the run completes in seconds.
    """
    import time
    monkeypatch.setattr(J, "_OPERATOR_TUNING", None, raising=False)
    for var in J._TUNABLES:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("OPENAI_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    monkeypatch.setenv("OPENAI_API_KEY", "EMPTY")
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    reports = [_report("r1", "a", ["Bees collect nectar."]), _report("r2", "b", ["Bees make honey."])]
    j = J.NuggetRecallJudge()
    start = time.monotonic()
    banks = j.create_nuggets(None, [TOPIC], llm_config=None)
    lb = j.judge(reports, [TOPIC], llm_config=None, nugget_banks=banks)
    elapsed = time.monotonic() - start
    assert {e.run_id for e in lb.entries if e.topic_id == "t1"} == {"r1", "r2"}
    assert elapsed < 60, f"took {elapsed:.0f}s: the dead endpoint was retried, not failed fast"


def _serve_models(ids):
    """A tiny /v1/models endpoint. Returns (base_url, shutdown)."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Models(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"data": [{"id": i} for i in ids]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Models)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_port}/v1", srv.shutdown


def test_a_dead_first_phase_does_not_cripple_a_live_second_phase(monkeypatch):
    """Regression for the stickiest bug in this judge.

    _llm_backend runs once per phase in one process. It used to apply its tuning
    with setdefault, so if the preflight missed while vLLM was still loading, the
    nugget phase pinned TIMEOUT_S=5 and MAX_ATTEMPTS=1 and the judge phase
    inherited them against a healthy server. Every call needing more than five
    seconds scored zero, and since those are the longest answers the leaderboard
    still varied, so the constancy guard could not catch it and the file uploaded.
    """
    import os
    monkeypatch.setattr(J, "_OPERATOR_TUNING", None, raising=False)
    for var in J._TUNABLES:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("OPENAI_API_KEY", "EMPTY")

    # Phase one: nothing is listening yet.
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    J._llm_backend(None)
    assert os.environ["TIMEOUT_S"] == "5", "a dead endpoint should still fail fast"

    # Phase two: the server came up.
    base, shutdown = _serve_models(["openai/gpt-oss-20b"])
    try:
        monkeypatch.setenv("OPENAI_BASE_URL", base)
        J._llm_backend(None)
    finally:
        shutdown()
    assert os.environ["TIMEOUT_S"] == "900", "the judge phase inherited the dead phase's timeout"
    assert os.environ["MAX_ATTEMPTS"] == "4"
    assert os.environ["BATCH_MAX_FAILURES"] == "none"


def test_an_operator_override_survives_both_phases(monkeypatch):
    """The documented escape hatch must still win over either phase's policy."""
    import os
    monkeypatch.setattr(J, "_OPERATOR_TUNING", None, raising=False)
    for var in J._TUNABLES:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TIMEOUT_S", "123")
    monkeypatch.setenv("OPENAI_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    J._llm_backend(None)
    assert os.environ["TIMEOUT_S"] == "123"
    base, shutdown = _serve_models(["openai/gpt-oss-20b"])
    try:
        monkeypatch.setenv("OPENAI_BASE_URL", base)
        J._llm_backend(None)
    finally:
        shutdown()
    assert os.environ["TIMEOUT_S"] == "123", "the operator's value was overwritten"


def test_a_live_endpoint_gets_the_reasoning_model_defaults(monkeypatch):
    """Regression: the stock batch defaults cost us a whole run.

    gpt-oss-20b on rag25-gen scored 0.000 and ranked last, because minima-llm
    times a call out after 60 s and aborts the entire batch after 25 failures.
    On 5302 judgments that is half a percent of the calls destroying all of them.
    A reasoning model needs the longer timeout, and a judge that already scores a
    failed call as zero and counts it must not have the batch pulled from under
    it. None of these touches a cache key, so TIRA's replay check is unaffected.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Ok(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.end_headers()
        def log_message(self, *a):
            pass

    monkeypatch.setattr(J, "_OPERATOR_TUNING", None, raising=False)
    for var in J._TUNABLES:
        monkeypatch.delenv(var, raising=False)
    srv = HTTPServer(("127.0.0.1", 0), Ok)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{srv.server_port}/v1")
        monkeypatch.setenv("OPENAI_MODEL", "openai/gpt-oss-20b")
        monkeypatch.setenv("OPENAI_API_KEY", "EMPTY")
        J._llm_backend(None)
    finally:
        srv.shutdown()
    import os
    assert float(os.environ["TIMEOUT_S"]) >= 600, "a reasoning model needs more than 60 s"
    assert os.environ["BATCH_MAX_FAILURES"] == "none", "a few failures must not abort the batch"
    assert int(os.environ["MAX_ATTEMPTS"]) <= 6


def test_a_model_the_endpoint_does_not_serve_is_reported(monkeypatch, caplog):
    """Regression: asking for the wrong model looks exactly like a broken judge.

    The launcher exported Qwen3-30B while vLLM was serving gpt-oss-20b. Every call
    404'd, every judgment scored zero, and nothing in the output said why. Model
    ids are not restricted content, so naming both is safe and saves hours.
    """
    import json
    import logging
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Models(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"data": [{"id": "openai/gpt-oss-20b"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Models)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}/v1"
    try:
        assert J._served_models(base, None) == ["openai/gpt-oss-20b"]
        monkeypatch.setenv("OPENAI_BASE_URL", base)
        monkeypatch.setenv("OPENAI_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
        monkeypatch.setenv("OPENAI_API_KEY", "EMPTY")
        with caplog.at_level(logging.ERROR, logger=J.log.name):
            J._llm_backend(None)
    finally:
        srv.shutdown()
    said = " ".join(r.getMessage() for r in caplog.records)
    assert "does not serve" in said and "gpt-oss-20b" in said

    # An endpoint that will not list its models must not trigger a false alarm.
    assert J._served_models("http://127.0.0.1:9/v1", None, timeout=1) is None


def test_preflight_treats_any_http_reply_as_reachable():
    """A 401 or 404 still proves a server is there, so it must not trigger fast-fail."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class NotFound(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(404); self.end_headers()
        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), NotFound)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert J._endpoint_reachable(f"http://127.0.0.1:{srv.server_port}/v1", None)
    finally:
        srv.shutdown()
    assert not J._endpoint_reachable("http://127.0.0.1:9/v1", None, attempts=1, timeout=1)
    assert not J._endpoint_reachable("EMPTY", None, attempts=1, timeout=1)


# ------------------------------------------------- the CI stub endpoint


def test_the_ci_stub_endpoint_speaks_what_our_parsers_read():
    """The stub is only useful if its replies survive our own parsers.

    It exists so the code submission's container test really calls an LLM. If a
    prompt change made the stub's replies unparsable, the container would fall
    back to zeros and the rehearsal would go back to proving nothing, quietly.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "stub_openai.py"
    spec = importlib.util.spec_from_file_location("stub_openai", path)
    stub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stub)

    gen = stub._reply_for(J_PROMPTS.generation_messages("how do bees make honey")[0]["content"])
    nuggets = parse_nuggets(gen)
    assert len(nuggets) >= 5 and any(n.vital for n in nuggets), gen

    def ask(answer):
        return stub._reply_for(
            J_PROMPTS.judge_messages("how do bees make honey", answer, nuggets)[0]["content"])

    labels = parse_judgments(ask("Bees collect nectar."), len(nuggets))
    assert labels is not None and len(labels) == len(nuggets)

    # Different answers must score differently, or the leaderboard is constant
    # and check_upload_dir would refuse it for the stub's sake, not the judge's.
    assert ask("Bees fan their wings to evaporate water from nectar.") != ask("Bees collect nectar.")


# ------------------------------------------------------------- NR_RANK


def test_the_two_percentile_helpers_stay_identical():
    """measures.py duplicates Corroborate's helper to avoid importing torch.

    Duplication is only safe if it cannot drift, so pin the two together.
    """
    from judges.corroborate.scoring import within_topic_percentile as theirs
    from judges.nuggetrecall.measures import within_topic_percentile as ours

    for scores in ({"a": 0.9, "b": 0.5, "c": 0.1},
                   {"a": 0.5, "b": 0.5, "c": 0.5},
                   {"a": 1.0},
                   {"a": 0.0, "b": 1.0},
                   {"a": 0.3, "b": 0.3, "c": 0.9, "d": 0.0}):
        assert ours(scores) == theirs(scores), scores


def test_nr_rank_removes_topic_difficulty(monkeypatch):
    """The reason NR_RANK exists, as a worked example.

    Two topics, one easy (everyone scores high) and one hard. Run A wins the hard
    topic, run B wins the easy one by a hair. Averaging the raw recall lets the
    easy topic dominate; ranking within a topic does not.
    """
    from judges.nuggetrecall.measures import within_topic_percentile

    topics = {
        "easy": {"A": 0.30, "B": 0.95},   # B wins by a mile
        "hard1": {"A": 0.25, "B": 0.20},  # A wins narrowly
        "hard2": {"A": 0.22, "B": 0.18},  # A wins narrowly
    }
    mean = {r: sum(t[r] for t in topics.values()) / len(topics) for r in ("A", "B")}
    pct = {k: within_topic_percentile(v) for k, v in topics.items()}
    rank = {r: sum(p[r] for p in pct.values()) / len(pct) for r in ("A", "B")}
    assert mean["B"] > mean["A"], "the easy topic must dominate the raw mean"
    assert rank["A"] > rank["B"], "A wins two topics of three, so it must win on rank"


def test_nr_rank_is_emitted_for_every_row(fake):
    reports = [_report("r1", "a", ["Bees collect nectar."]),
               _report("r2", "b", ["Bees add enzymes to nectar."]),
               _report("r3", "c", [])]
    _, lb = _run_judge(reports, [TOPIC])
    rows = {e.run_id: e.values for e in lb.entries if e.topic_id == "t1"}
    assert set(rows) == {"r1", "r2", "r3"}
    assert all("NR_RANK" in v for v in rows.values())
    # The empty report scored zero, so it must rank last.
    assert rows["r3"]["NR_RANK"] == min(v["NR_RANK"] for v in rows.values())
    assert all(0.0 <= v["NR_RANK"] <= 1.0 for v in rows.values())
