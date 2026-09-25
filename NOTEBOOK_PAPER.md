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
| rag25-gen | short RAG answers | 0.323 (92 of 98) | **0.801 (42 of 98)** | 0.787 | 0.862 |
| ragtime25 | long reports | 0.691 (67 of 103) | **0.739 (36 of 103)** | 0.726 | 0.802 |
| dragun-repgen | reports about a news article | **0.755 (7 of 61)** | 0.141 (57 of 61) | 0.650 | 0.821 |

The NuggetRecall column is the five-measure judge. Ranks move as other teams upload,
so they are read on the date shown.

rag25-gen is the pilot most like rag26, and ragtime25 the one most like ragtime26.
NuggetRecall wins on both, so it is our main submission.

### What we submitted

Every submission covers every run and every topic of its test set, which the upload
guard checks before anything is sent, since a partial submission is rejected. Test-set
scores come out after TREC.

| System | Judge and LLM | rag26 | ragtime26 | Why |
|---|---|---|---|---|
| `nuggetrecall-best` | NuggetRecall, Qwen3-30B | **[done]** | **[done]** | the main submission |
| `nuggetrecall-rank` | same judgments, plus the rank measure | **[done]** | **[done]** | better on the rag pilot, never worse |
| `nuggetrecall-gptoss` | NuggetRecall, gpt-oss-20b | | **[done]** | best single judge on the ragtime pilot |
| `ensemble-qwen-gptoss` | both LLMs, equal weight | **[done]** | | best top-weighted score on the rag pilot |
| `corroborate-best` | Corroborate, no LLM | **[done]** | **[done]** | the LLM-free method, for comparison |

**Code submissions [done].** Both judges are registered on TIRA, NuggetRecall as
`rank-dioptre` and Corroborate as `lead-racer`, so the organizers can re-run them,
NuggetRecall against their own choice of LLM. NuggetRecall is registered a second time
as `grave-node`, with the reply-budget fix from finding 11, because the organizers grade
with gpt-oss-20b and without that fix it left 12.4% of rag26 judgments unreadable. Getting there took more than the
judges did. The rehearsal first gave the container no LLM at all, so the judge fell
back to an all-zero leaderboard and every TIRA check still passed. Then two
independent reviews of the real submission path found that it would have failed
outright on deadline day, because TIRA refuses an untidy working folder only on a
real submission, never on a rehearsal. Both are fixed, and the rehearsal now fails
wherever the real run would.

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

4. **A small reasoning model judged long reports better than a larger instruct model.**
   Same judge, same prompts, only the LLM changes:

   | Judge LLM | rag25-gen (short answers) | ragtime25 (long reports) |
   |---|---|---|
   | Qwen3-30B-A3B-Instruct (FP8) | 0.788 | 0.739 (rank 37 of 105) |
   | gpt-oss-20b (reasoning) | 0.783 | **0.789 (rank 10 of 105)** |
   | Qwen2.5-7B-Instruct | 0.654 | not run |

   On short answers the two larger models tie, within our noise of about 0.01. On long
   reports the reasoning model is 0.05 better, five times the noise, even though 2.9%
   of its replies came back unreadable and 5 of 122 topics fell back to a title-only
   answer key. Checking twenty facts against a long report seems to reward a model
   that reasons before it answers. The 7B model follows the output format just
   as reliably as the others but judges less accurately, mostly by giving partial
   credit too freely, and still stays far above Corroborate (0.323) and the length
   shortcut (0.152).

5. **An LLM judge can fail silently and completely, and the failure looks like a bad
   judge.** Running the same judge on gpt-oss-20b scored **0.000**, last of 94. The
   model was not at fault. The LLM client times a call out after 60 seconds, which a
   reasoning model does not always meet, and it then abandons the entire batch once
   25 calls have failed. So 26 slow calls out of 5,302 destroyed all 5,302, every
   measure came out the same for every system, and the leaderboard ranked nothing.
   The score alone gives no hint of this. What caught it was a check we run before
   every upload, which refuses any leaderboard whose measures do not vary between
   systems. Anyone building an LLM judge should have that check and should not trust
   a low score to mean a weak judge. After raising the timeout and turning the batch
   breaker off, the same configuration scores **0.783**, which is within 0.005 of the
   much larger model. This mattered beyond the pilot: the organizers re-run submitted
   code against gpt-oss-20b, so our code submission would have scored zero.

6. **Reproducibility checks can be satisfied by two runs that both do nothing.** TIRA
   proves an LLM judge is reproducible by re-running the container with the endpoint
   removed and confirming the cached prompts give the same leaderboard. Our first
   rehearsal passed that check while making zero LLM calls: the first run cached
   nothing, so the replay reproduced nothing, and the two agreed exactly. A sameness
   check needs a liveness check beside it. Ours is now that the container's own
   output must rank systems apart before the job can pass.

7. **The same silent shape keeps coming back, so we went looking for it.** After two
   of these in one day we audited both judges specifically for failure modes that
   yield a plausible-looking but meaningless leaderboard. That found a third, and the
   worst: the judge configures its LLM client once per phase in one process, using
   settings that stick. A single missed connection while the model server was still
   loading pinned a five-second timeout onto the whole judging phase. Because the
   calls that exceed five seconds are the long answers, the zeros land unevenly, the
   measures still vary, and the constancy check cannot see it. That one would have
   ranked systems partly by brevity, and answer length correlates +0.558 with human
   truth on ragtime25, so the submitted score could have gone negative.

8. **How you average matters as much as what you measure.** Our judge scored each
   response for how many key facts it covered, then averaged that over topics. That
   lets an easy topic, where every system scores high, count as much as one that
   actually separates systems. Ranking systems within each topic first, then
   averaging the ranks, uses the identical judgments and scores **0.801** on
   rag25-gen against 0.788, and is better on the top-weighted measure too. It costs
   no extra LLM calls. On the other pilot it is slightly worse, which costs nothing,
   because the official score is the best of our measures, so we emit both.

9. **A threshold on a cosine is meaningless without knowing the encoder's scale, and
   a parameter sweep will not tell you.** Corroborate filters pooled claims by how
   specific they are to the topic, measured as cosine similarity under
   `bge-base-en-v1.5`. We set that floor to 0.15 and swept it over 0.0 to 0.20. Then
   we measured what the encoder actually produces: specific facts score 0.587 to
   0.753, boilerplate hedges 0.361 to 0.472, and deliberately nonsense sentences
   still score 0.266. Our floor and its entire sweep sat below all of it, so the
   filter kept everything, on every dataset, for the whole of development. The sweep
   could not reveal this, because every swept value gave an identical result and so
   read as "this parameter does not matter".

   This is not a harmless no-op. The bank is ranked by how many separate teams state
   a claim, and a hedge is exactly what every team writes, so hedges took the top
   slots and then counted as corroborated in the measure whose job is to penalise
   padding.

   We then re-ran Corroborate with a floor that actually filters, and it made **no
   measurable difference**: 0.323 on rag25-gen before and after, to three decimal
   places, and 0.691 against 0.690 on ragtime25. The measures themselves moved by at
   most 0.015. So we had guessed that the dead filter might explain Corroborate's
   weakness on short answers, and the measurement says it does not. The weakness is
   the one in finding 1: agreement between teams is simply not what separates good
   short answers from bad ones. The general lesson still stands and is worth more
   than the number: warn whenever a filter keeps everything or nothing, and report
   the range actually observed. **[done]**

10. **Two different LLM judges agree best about the top of the leaderboard.** We
    combined the Qwen3-30B and gpt-oss-20b judgments with equal weight, ranking systems
    within each topic by each model and averaging the two ranks. No weights were
    fitted. On the pilots:

    | | rag25-gen Kendall | rag25-gen **tau AP-b** | ragtime25 Kendall | ragtime25 tau AP-b |
    |---|---|---|---|---|
    | Qwen3-30B, best single measure | 0.801 | 0.628 | 0.739 | 0.599 |
    | gpt-oss-20b | 0.783 | 0.622 | **0.789** | **0.617** |
    | Both, equal weight | **0.812** | **0.695** | 0.769 | 0.605 |

    On short answers the ensemble's Kendall gain is within noise, but its top-weighted
    score rises by 0.067, far more than the run-to-run variation we saw on that
    measure. tau AP-b mostly rewards getting the best systems right, so two judges
    that err differently seem to cancel out each other's mistakes at the top. On long
    reports gpt-oss alone is clearly better, and averaging it with a weaker judge only
    dilutes it. So we submit the ensemble for rag26 and gpt-oss alone for ragtime26.
    Choosing per track from two pilots risks some overfitting, and we say so.

11. **A reasoning model can run out of room to think, and a retry must give it more.**
    On rag26, gpt-oss-20b left 27% of its first-pass judgments unreadable. The LLM
    server's own counters showed why: 5,020 of 13,275 requests stopped because they
    hit the 4,096-token reply limit, so the model was still reasoning and never wrote
    its answer. Our retries used to reword the prompt and send the same budget, which
    cannot help. With the budget fixed at 4,096 on every attempt, 12.4% of judgments
    stayed unreadable after three tries. Giving retries 8,192 and then 16,384 tokens
    cut that to **2.7%**, 264 of 9,838, with no other change. The first attempt keeps
    its original budget, so every cached reply still counts. The same model on the
    rag25-gen pilot had needed almost no retries, so this only showed up on the test
    data, which is a reason to watch retry counts rather than trust a pilot.

---

## 5b. A caution on every number above

Our judge is not deterministic run to run. vLLM does not produce bit-identical
output across different batch compositions, so the same configuration re-run gives
slightly different judgments. We measured the size of this by accident: re-running
NuggetRecall on ragtime25 with nothing changed except an **added** measure moved the
score by -0.010, which adding a measure cannot do, because the official score is the
best of our measures.

So roughly 0.01 Kendall is noise. We do not treat differences below about 0.02 as
real, and we withdrew one earlier result on that basis: combining the two judges
appeared to gain +0.012 on ragtime25, which is inside the noise, so we do not claim
it. The gain from rank aggregation (+0.013 on the five-measure judge, +0.021 in an
isolated re-aggregation of the identical leaderboard) sits at or above that line and
is consistent in direction, which is why we do claim that one.

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
