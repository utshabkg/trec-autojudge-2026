"""Unit tests for the Corroborate judge.

These run without a GPU, without network, and without any model weights. That
matters because `tira-cli code-submission` runs pytest on the machine doing the
submission, which for us is a GitHub Actions runner where the encoder weights
are not present: they are baked into the container image, not the repo. So the
logic is tested against a deterministic stub encoder, and the real-model path is
a separate opt-in test at the bottom.

The stub is a hashing bag-of-words encoder. Sentences that share words get a
high cosine, which is enough to exercise clustering, team voting, LOTO, and the
scoring maths with fully predictable inputs.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from autojudge_base import Request
from autojudge_base.report import Report, ReportMetaData
from autojudge_base.report_sentences import NeuclirReportSentence

from judges.corroborate.judge import CorroborateJudge
from judges.corroborate.measures import MEASURE_NAMES
from judges.corroborate.pool import build_topic_pool
from judges.corroborate.scoring import harmonic, within_topic_percentile
from judges.corroborate.textutil import is_scoreable, normalize, scoreable_sentences

DIM = 64


class StubEncoders:
    """Deterministic hashing bag-of-words stand-in for the real encoders."""

    def __init__(self):
        self.embed_calls = 0
        self.cross_calls = 0

    def embed(self, texts):
        self.embed_calls += 1
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in normalize(t).casefold().split():
                out[i, hash(tok) % DIM if False else _slot(tok)] += 1.0
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out

    def cross(self, pairs):
        # Token-overlap Jaccard as an entailment proxy: deterministic and in [0, 1].
        self.cross_calls += 1
        scores = []
        for premise, hypothesis in pairs:
            a = set(normalize(premise).casefold().split())
            b = set(normalize(hypothesis).casefold().split())
            scores.append(len(a & b) / len(a | b) if (a | b) else 0.0)
        return np.asarray(scores, dtype=np.float32)

    def close(self):
        pass


def _slot(token: str) -> int:
    """Stable slot for a token. `hash()` is salted per process, so not that."""
    h = 0
    for ch in token:
        h = (h * 131 + ord(ch)) % 1000003
    return h % DIM


def _report(run_id: str, team_id: str, topic_id: str, sentences, citations=None):
    cites = citations or [[] for _ in sentences]
    return Report(
        metadata=ReportMetaData(team_id=team_id, run_id=run_id, topic_id=topic_id),
        responses=[
            NeuclirReportSentence(text=t, citations=c) for t, c in zip(sentences, cites)
        ],
    )


TOPIC = Request(request_id="t1", title="solar panel efficiency and cost")

# Two teams agree on two facts. team_c says something only it says.
AGREE_1 = "solar panels convert sunlight into electricity efficiently"
AGREE_2 = "the cost of solar panels has fallen sharply over the last decade"
UNIQUE = "purple giraffes orbit the moon every wednesday afternoon"


def _pool_reports():
    return [
        _report("run_a", "team_a", "t1", [AGREE_1, AGREE_2]),
        _report("run_b", "team_b", "t1", [AGREE_1, AGREE_2]),
        _report("run_c", "team_c", "t1", [UNIQUE]),
    ]


# --------------------------------------------------------------------- textutil


def test_normalize_folds_unicode_and_whitespace():
    assert normalize("  a  b\t\nc ") == "a b c"


def test_boilerplate_and_short_sentences_are_not_scoreable():
    assert not is_scoreable("In conclusion, this is the end of the report.", 5)
    assert not is_scoreable("Yes.", 5)
    assert is_scoreable(AGREE_1, 5)


def test_repeated_sentences_counted_once():
    # A run repeating a sentence must not get several votes for it, nor pad its
    # own NPREC denominator.
    got = scoreable_sentences([AGREE_1, AGREE_1, AGREE_2], 5)
    assert [t for _i, t in got] == [normalize(AGREE_1), normalize(AGREE_2)]


# ------------------------------------------------------------------- clustering


def test_clustering_is_deterministic():
    enc = StubEncoders()
    a = build_topic_pool(_pool_reports(), TOPIC, enc, tau=0.8)
    b = build_topic_pool(_pool_reports(), TOPIC, enc, tau=0.8)
    assert [c.cluster_id for c in a.clusters] == [c.cluster_id for c in b.clusters]
    assert a.assignment == b.assignment


def test_clustering_ignores_input_order():
    """Reports are sorted by run_id, so file order cannot change the pool."""
    enc = StubEncoders()
    straight = build_topic_pool(_pool_reports(), TOPIC, enc, tau=0.8)
    shuffled = _pool_reports()
    random.Random(0).shuffle(shuffled)
    other = build_topic_pool(shuffled, TOPIC, enc, tau=0.8)
    assert [c.cluster_id for c in straight.clusters] == [c.cluster_id for c in other.clusters]


def test_agreeing_teams_share_a_cluster():
    enc = StubEncoders()
    pool = build_topic_pool(_pool_reports(), TOPIC, enc, tau=0.8)
    shared = [c for c in pool.clusters if len(c.teams) >= 2]
    assert shared, "identical sentences from two teams must cluster together"
    assert all(c.teams == frozenset({"team_a", "team_b"}) for c in shared)


def test_vote_unit_run_treats_each_run_as_its_own_team():
    enc = StubEncoders()
    reports = _pool_reports() + [_report("run_a2", "team_a", "t1", [AGREE_1, AGREE_2])]
    by_team = build_topic_pool(reports, TOPIC, enc, tau=0.8, vote_unit="team")
    by_run = build_topic_pool(reports, TOPIC, enc, tau=0.8, vote_unit="run")
    shared_team = max(len(c.teams) for c in by_team.clusters)
    shared_run = max(len(c.teams) for c in by_run.clusters)
    # team_a submitted two runs: run counting inflates the vote, team counting does not.
    assert shared_team == 2
    assert shared_run == 3


# -------------------------------------------------------------------- LOTO maths


def test_loto_weight_excludes_the_scored_team():
    enc = StubEncoders()
    pool = build_topic_pool(_pool_reports(), TOPIC, enc, tau=0.8)
    c = next(c for c in pool.clusters if len(c.teams) == 2)
    assert c.loto_weight(None) == 2
    assert c.loto_weight("team_a") == 1
    assert c.loto_weight("team_c") == 2  # not a member, so nothing is removed


def test_bank_excludes_single_team_clusters():
    """min_teams >= 2 is what stops a team inventing its own bank entries."""
    enc = StubEncoders()
    pool = build_topic_pool(_pool_reports(), TOPIC, enc, tau=0.8)
    bank = pool.bank(specificity_floor=-1.0, min_teams=2, max_clusters=40)
    assert bank, "two teams agreed on something, so the bank cannot be empty"
    assert all(len(c.teams) >= 2 for c in bank)
    assert all(UNIQUE not in c.leader for c in bank)
    # And therefore no bank cluster can ever have a zero LOTO weight.
    for c in bank:
        for team in ("team_a", "team_b", "team_c"):
            assert c.loto_weight(team) >= 1


def test_bank_cap_is_deterministic_under_ties():
    enc = StubEncoders()
    pool = build_topic_pool(_pool_reports(), TOPIC, enc, tau=0.8)
    first = pool.bank(specificity_floor=-1.0, min_teams=2, max_clusters=1)
    second = pool.bank(specificity_floor=-1.0, min_teams=2, max_clusters=1)
    assert [c.cluster_id for c in first] == [c.cluster_id for c in second]


# ---------------------------------------------------------------------- scoring


def test_harmonic_is_zero_when_either_side_is_zero():
    assert harmonic(0.0, 0.9) == 0.0
    assert harmonic(0.9, 0.0) == 0.0
    assert harmonic(0.5, 0.5) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "scores, expected",
    [
        ({"a": 0.1, "b": 0.5, "c": 0.9}, {"a": 0.0, "b": 0.5, "c": 1.0}),
        ({"a": 0.5, "b": 0.5, "c": 0.5}, {"a": 0.5, "b": 0.5, "c": 0.5}),
        ({"a": 0.3}, {"a": 0.5}),
        ({}, {}),
    ],
)
def test_percentile_edge_cases(scores, expected):
    assert within_topic_percentile(scores) == expected


def test_percentile_is_bounded_and_worst_is_zero():
    """0 must mean worst, because the framework fills a missing cell with 0.0."""
    got = within_topic_percentile({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0})
    assert min(got.values()) == 0.0 and max(got.values()) == 1.0
    assert got["a"] == 0.0


# ------------------------------------------------------------- the judge phases


def _judge_with_stub(reports, topics, **overrides):
    judge = CorroborateJudge()
    judge._encoders = StubEncoders()
    judge._enc_sig = ("stub", None, "None", None)
    settings = dict(
        tau=0.8, specificity_floor=-1.0, min_teams=2, max_clusters=40,
        min_other_teams=1, attribution_mode="pool", use_cross_encoder=True,
    )
    settings.update(overrides)
    return judge, judge.judge(reports, topics, llm_config=None, **settings)


def test_every_measure_is_present_for_every_row():
    reports = _pool_reports()
    _judge, lb = _judge_with_stub(reports, [TOPIC])
    per_topic = [e for e in lb.entries if e.topic_id == "t1"]
    assert len(per_topic) == 3
    for e in per_topic:
        assert set(e.values) == set(MEASURE_NAMES)


def test_empty_report_is_scored_not_dropped():
    """A run that skipped a topic must still appear, scored at the bottom.

    Note what "bottom" means for NF1_RANK. It is a percentile, so an empty
    report ties for last with any other run that also scored zero rather than
    being pushed below it. Asserting a literal 0.0 there would demand the judge
    claim the empty run is strictly worse than an equally-zero run, which is a
    fact it does not have.
    """
    reports = _pool_reports() + [_report("run_empty", "team_d", "t1", [])]
    _judge, lb = _judge_with_stub(reports, [TOPIC])
    rows = {e.run_id: e.values for e in lb.entries if e.topic_id == "t1"}
    assert "run_empty" in rows, "an empty report must be scored, never dropped"
    for m in ("NCOV", "NPREC", "NF1", "CITE"):
        assert rows["run_empty"][m] == 0.0, m
    worst = min(v["NF1_RANK"] for v in rows.values())
    assert rows["run_empty"]["NF1_RANK"] == worst


def test_report_for_an_unknown_topic_is_dropped():
    reports = _pool_reports() + [_report("run_x", "team_x", "not_a_topic", [AGREE_1])]
    _judge, lb = _judge_with_stub(reports, [TOPIC])
    assert {e.topic_id for e in lb.entries} == {"t1", lb.all_topic_id}
    assert "run_x" not in {e.run_id for e in lb.entries if e.topic_id == "t1"}


def test_uncorroborated_run_scores_below_corroborated_ones():
    """The whole point: saying things nobody else says earns no credit."""
    _judge, lb = _judge_with_stub(_pool_reports(), [TOPIC])
    agg = {
        e.run_id: e.values
        for e in lb.entries
        if e.topic_id == lb.all_topic_id
    }
    assert agg["run_c"]["NPREC"] == 0.0
    assert agg["run_a"]["NPREC"] > 0.0 and agg["run_b"]["NPREC"] > 0.0
    assert agg["run_c"]["NCOV"] < agg["run_a"]["NCOV"]


def test_aggregate_rows_exist_for_every_measure():
    """Correlations are computed only from the topic_id == 'all' rows."""
    _judge, lb = _judge_with_stub(_pool_reports(), [TOPIC])
    agg = [e for e in lb.entries if e.topic_id == lb.all_topic_id]
    assert agg
    for e in agg:
        assert set(e.values) == set(MEASURE_NAMES)


def test_qrels_creation_is_disabled():
    """Emitting qrels beside the leaderboard breaks tira's upload format check."""
    judge = CorroborateJudge()
    assert judge.create_qrels([], [], llm_config=None) is None


# ------------------------------------------------------- opt-in real-model test


@pytest.mark.skipif(
    not Path("data/kiddie/runs/repgen").is_dir(),
    reason="kiddie data not present",
)
def test_real_models_end_to_end_on_kiddie():
    """Runs only when the encoder weights are reachable. Skipped on CI runners."""
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("torch")
    from autojudge_base import load_requests_from_file
    from autojudge_base.io import load_runs_failsave
    from judges.corroborate.encoders import Encoders

    try:
        enc = Encoders("BAAI/bge-base-en-v1.5", None, local_files_only=True, device="cpu")
        enc.embed(["probe"])
    except Exception as exc:
        pytest.skip(f"encoder weights unavailable: {exc}")

    reports = load_runs_failsave(Path("data/kiddie/runs/repgen"))
    topics = load_requests_from_file(Path("data/kiddie/topics/kiddie-topics.jsonl"))
    judge = CorroborateJudge()
    lb = judge.judge(
        reports, topics, llm_config=None,
        device="cpu", use_cross_encoder=False, min_other_teams=1,
    )
    runs = {e.run_id for e in lb.entries}
    assert runs == {"run1", "run2", "run3", "run4"}
    agg = {e.run_id: e.values for e in lb.entries if e.topic_id == lb.all_topic_id}
    assert len({round(agg[r]["NCOV"], 9) for r in runs}) > 1, "NCOV must separate runs"
