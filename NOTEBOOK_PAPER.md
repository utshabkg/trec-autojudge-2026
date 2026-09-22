# TREC 2026 AutoJudge: Team IRIS-MST

**Draft notebook paper. Last updated 2026-09-22.**

This is the short version, written to become the TREC notebook paper and to explain
the work to someone who has not followed it. The full engineering record is in
`WORK_LOG.md`.

Status markers: **[done]** finished and measured, **[pending]** not yet measured.

---

## Abstract

We built two automatic judges for TREC 2026 AutoJudge and found that they succeed on
different kinds of task. *Corroborate* needs no LLM. It builds its reference answer
from what many independent teams agree on, and it judges long reports well. On short
RAG answers it fails. *NuggetRecall* uses an LLM to write an answer key from the
question alone, then checks each response against it. It judges short answers and
long reports well, and fails when the "question" is really a document. On the pilot
data NuggetRecall reached the field median on RAG and ranked 32 of 96 on RAGTIME, so
it is our main submission.

---

## 1. The task

A normal TREC track asks systems to answer questions. AutoJudge asks the reverse: given
many systems' answers, **rank the systems**. A judge is scored by how closely its
ranking matches the ranking made by human assessors, using Kendall's tau and a
top-weighted variant, tau AP-b.

We judged two 2026 test sets:

| Test set | Source | Systems | Topics |
|---|---|---|---|
| rag26 | TREC RAG, short cited answers | 83 runs, 25 teams | 119 |
| ragtime26 | TREC RAGTIME, long cited reports | 49 runs, 10 teams | 103 |

Two things shaped every decision:

- **No human answers are available**, not even for the older pilot datasets. So we
  can only measure our judges by uploading results and letting the organizers score
  them against assessments we never see.
- **The official score takes the best of all the measures a judge reports.** So a
  judge may report a few different measures and is credited with its best one.

---

## 2. Method

### Judge 1: Corroborate, no LLM

The idea comes from how search engines are evaluated, where the pooled results of many
systems stand in for the set of relevant documents. Here we pool the *answers*.

1. For each topic, split every system's answer into sentences and group sentences that
   say the same thing, using sentence embeddings.
2. Give each group a weight equal to the number of **different teams** that said it.
   Counting teams rather than runs matters, because teams submit several near-identical
   runs.
3. Score a system on how many of the well-supported groups it covers.
4. **Leave-one-team-out:** when scoring a team, ignore its own votes. Otherwise a team
   gets credit for facts only it stated. This is the novel part.

### Judge 2: NuggetRecall, with an LLM

1. For each topic, an LLM writes an **answer key** of 8 to 20 short facts that a good
   answer should contain, using only the question. Each fact is marked vital or okay.
2. For each system's answer, the LLM labels every fact as supported, partly supported,
   or missing.
3. The score is the share of facts covered, with vital facts counted separately.

Building the key from the question alone means it cannot be biased by what any system
happened to write.

---

## 3. Experimental setup

- **Validation:** we ran both judges on three pilot datasets from 2025, uploaded the
  results, and read back the organizers' scores.
- **LLM for NuggetRecall:** Qwen3-30B-A3B-Instruct (FP8), served locally with vLLM on
  one RTX 6000 Ada GPU. The organizers re-run submitted code with
  Qwen2.5-7B-Instruct and gpt-oss-20b, so we also tested NuggetRecall on those.
- **Embeddings for Corroborate:** bge-base-en-v1.5, plus nli-deberta-v3-base to decide
  borderline matches.

---

## 4. Results

### Pilot results **[done]**

Kendall's tau against human assessments, with our rank among all entries on each
pilot's leaderboard.

| Pilot | Answer type | Corroborate | NuggetRecall | Field median | Best entry |
|---|---|---|---|---|---|
| rag25-gen | short RAG answers | 0.323 (87 of 92) | **0.788 (45 of 92)** | 0.787 | 0.862 |
| ragtime25 | long reports | 0.691 (59 of 96) | **0.739 (32 of 96)** | 0.724 | 0.802 |
| dragun-repgen | reports about a news article | **0.755 (7 of 61)** | 0.141 (57 of 61) | 0.650 | 0.821 |

rag25-gen is the pilot most like rag26, and ragtime25 the one most like ragtime26.
NuggetRecall wins on both, so it is our main submission.

### What we submitted **[done]**

- **Data submission:** NuggetRecall on rag26 and ragtime26, complete for every run and
  topic. Scores on the test sets come out after TREC.
- **[pending]** Corroborate as a second submission, and the code submissions for both
  judges.

---

## 5. Findings worth reporting

1. **Each judge fails exactly where its assumption breaks.** Corroborate assumes that
   what many teams agree on is what matters. That holds for long reports but not for
   short answers, where the best systems stand out by stating things others missed.
   NuggetRecall assumes the request is a question. That fails on DRAGUN, where the
   request is a whole news article, so the answer key lists facts from the article
   rather than what a good report should say.

2. **The two judges measure different things.** Their rankings agree only weakly
   (Kendall 0.50 on ragtime25, 0.23 on rag25-gen), so they are complementary rather
   than redundant.

3. **Answer length is a strong shortcut on some tracks and useless on others.** Simply
   ranking systems by answer length scores 0.558 on ragtime25 but only 0.152 on
   rag25-gen. A judge that looks good on RAGTIME may just be rewarding long answers, so
   we checked that none of our measures is mostly length.

4. **A bigger LLM makes a better judge, but a small one still works.** On rag25-gen,
   the same NuggetRecall judge scores **0.788 with Qwen3-30B** and **0.654 with
   Qwen2.5-7B**, a drop of about 0.13 and 20 ranking places. The smaller model
   follows the output format just as reliably (no failed or unreadable answers), but
   it judges less accurately, mostly by giving partial credit too freely. Even so it
   stays far above Corroborate (0.323) and the length shortcut (0.152).
   **[pending]** gpt-oss-20b.

---

## 6. Limitations

- Corroborate cannot reward a correct fact that no other team found.
- NuggetRecall depends on the LLM's knowledge to write the answer key, and on the
  request being a real question.
- We tuned only on the 2025 pilots, as the rules require, so the 2026 results may
  differ.

---

## 7. Reproducibility

Code: `github.com/utshabkg/trec-autojudge-2026`, in `judges/corroborate/` and
`judges/nuggetrecall/`. Both judges are submitted to TIRA as code, so the organizers
can re-run them. Two of the 83 rag26 runs are expected to be our own TREC RAG 2026
runs. We made no attempt to identify them, and Corroborate's leave-one-team-out rule
stops any team from gaining from its own answers.
