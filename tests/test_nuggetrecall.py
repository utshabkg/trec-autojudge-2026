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
from judges.nuggetrecall.measures import MEASURE_NAMES
from judges.nuggetrecall.prompts import (
    Nugget, parse_judgments, parse_nuggets, score, strip_reasoning,
)

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


def test_unparsable_after_all_retries_scores_zero_not_dropped(monkeypatch):
    llm = FakeLLM(judge="garbage with no labels")
    monkeypatch.setattr(J, "_run", llm.run)
    monkeypatch.setattr(J, "_llm_backend", lambda cfg: object())
    _, lb = _run_judge([_report("r1", "a", ["Bees collect nectar."])], [TOPIC])
    row = next(e.values for e in lb.entries if e.run_id == "r1" and e.topic_id == "t1")
    assert all(row[m] == 0.0 for m in MEASURE_NAMES)


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
