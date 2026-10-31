# TREC 2026 AutoJudge, Team IRIS-MST (plain-language version)

**Last updated 2026-10-23.** The paper itself is LaTeX, in `paper/`. Open it in Overleaf
from `results/paper-overleaf.zip`. This file says the same things in plain words, for the
professor and for anyone new. Every number here matches the paper. The full record
behind it is `docs/WORK_LOG.md`.

The numbers come from TIRA, the platform that runs and scores submitted judges. We read
them on 2026-10-19 (rag25-gen, ragtime25) and 2026-10-21 (our extra DRAGUN runs). The
organizers re-scored the pilots (datasets with human judgments) on 2026-10-09. So any
older number in the work log or the September update uses the old scoring and is out of
date.

---

## Still open before the paper is final

- Acknowledgments (funding, compute). Authors are in place.
- A public link to the code. The GitHub repository is still private.
- Test-set results. The organizers have not released them yet.
- Whether TIRA's test-set topics match the released files. Laura's reply on 2026-10-23
  covered only DRAGUN. It implies rag26 and ragtime26 needed no conversion but does not
  say so. If the topics differ, the paper needs one more paragraph.
- The 2026 notebook deadline. Not published yet. The conference runs 2026-11-16 to 18,
  so expect it in early November.

---

## Abstract

AutoJudge asks for automatic judges that rank the systems of a host track the way human
assessors would. A host track is the track whose answers get judged. We built two
judges with opposite assumptions.

- **NuggetRecall** asks an LLM to write an answer key of 8 to 20 facts, often called
  nuggets, from the question alone. Then it asks which facts each response states.
- **Corroborate** uses no LLM. It gathers all responses to a topic and counts how many
  different teams state each claim. A team's own vote never counts for itself.

On the pilots, the two judges do best on different human measures. NuggetRecall is our
better judge for nugget coverage. For F1 on long reports, Corroborate is our better
judge. It fails on short answers. The choice of LLM mattered more than anything else.
We also found six ways a judge can fail without any error message.

---

## 1. The task

A normal TREC track asks systems to answer questions. AutoJudge turns this around. Given
many systems' answers, the judge has to **rank the systems**. A judge is scored by how
closely its ranking matches the human ranking. The match is measured with Kendall's tau,
a rank correlation over pairs of systems. A top-weighted version, tau AP-b, weights
disagreements near the top more heavily.

| Dataset | Kind | Runs | Teams | Topics | Responses |
|---|---|---|---|---|---|
| rag26 (test) | short cited answers | 83 | 25 | 119 | 9,877 |
| ragtime26 (test) | long cited reports | 49 | 10 | 103 | 5,047 |
| rag25-gen (pilot) | short cited answers | 51 | 18 | 105 | 5,332 |
| ragtime25 (pilot) | long cited reports | 59 | 13 | 122 | 7,198 |
| dragun-repgen (pilot) | reports on a news article | 28 | 9 | 30 | 840 |

- **The pilots have human truth, but only TIRA has it.** We test a judge by running it on
  a pilot, uploading the result and reading back the score.
- **TIRA reports the best of a judge's measures** for each human measure. So a judge may
  report several measures. A judge with many of them gets a small free advantage.
- **Since 2026-10-09 each human measure is scored separately.** rag25-gen has Strict
  Vital Score and Sub Coverage. ragtime25 has Nugget Coverage and F1. Its Citation Support
  is 0.000 for everyone, so we ignore it. DRAGUN still reports one number.

---

## 2. The two judges

### NuggetRecall (with an LLM)

1. For each topic, the LLM writes an **answer key** of 8 to 20 facts using only the
   question. Each fact is marked VITAL (an answer must have it) or OKAY (good to have).
2. For each response, one LLM call labels every fact as support, partial or no.
3. The scores are the share of vital facts covered, the share of all facts, a strict
   version without partial credit and an importance-weighted version.
4. A fifth score, **NR_RANK**, first ranks the runs within each topic and then averages.
   Same judgments, different averaging.

Why build the key from the question alone? That way a long-winded system cannot shape it.
The key can also hold facts every system missed.

A reply must label at least 80% of the facts, or it is retried. A reasoning model
thinks out loud before answering, which uses up its reply budget, the cap on tokens it
may write. Each retry gives it a bigger budget (4,096, then 8,192, then 16,384 tokens). Every
call is cached (each prompt and reply is stored), so TIRA can replay the judge without
the LLM.

### Corroborate (no LLM)

1. For each topic, split every response into sentences and group the ones that say the
   same thing. Two sentences match when their embedding similarity is at least 0.80. An
   embedding is a vector that puts similar meanings close together.
2. A group enters the "bank" only if at least **two different teams** said it. We count
   teams rather than runs because teams submit several near-copies.
3. NCOV scores a response on how many bank groups it covers. NPREC scores it on how many
   of its own sentences other teams also said, which punishes padding.
4. **Leave-one-team-out.** When we score team t, a group's weight is the number of
   *other* teams that said it. The idea comes from the classic pool-bias test in
   retrieval evaluation. That test removes from the judging pool whatever one run alone
   contributed, then re-scores that run.

---

## 3. Setup

- **LLMs we ran.** Qwen3-30B-A3B-Instruct (FP8, a low-precision weight format),
  gpt-oss-20b and Qwen2.5-7B-Instruct. They ran on one RTX 6000 Ada GPU on compute node
  54, served by vLLM (an open-source LLM serving engine).
- **LLMs the organizers ran our code with.** Qwen2.5-7B, gpt-oss-20b and gpt-oss-120b. On
  the test sets they also used Qwen3.8-27B.
- **Corroborate.** It uses bge-base-en-v1.5 for embeddings. Borderline matches go to
  nli-deberta-v3-base, a natural language inference model that decides whether one
  sentence follows from another. No LLM, no network.

---

## 4. Results on the pilots

Scores are Kendall's tau, with tau AP-b where it matters. "Field" means every entry on
the TIRA leaderboard. There are 385 on rag25-gen, 206 on ragtime25 and 139 on DRAGUN.
Many entries are variants from the same team, so ranks are a loose guide.

### rag25-gen (short answers, like rag26)

| Judge | Strict Vital Score | Sub Coverage |
|---|---|---|
| Length only (baseline) | 0.152 | 0.094 |
| Field median / best | 0.804 / 0.906 | 0.772 / 0.866 |
| NuggetRecall, Qwen3-30B | 0.783 | 0.788 |
| NuggetRecall, Qwen3-30B, rank only | 0.792 | **0.809** |
| NuggetRecall, gpt-oss-20b | 0.783 | 0.736 |
| NuggetRecall, Qwen2.5-7B | 0.604 | 0.654 |
| NuggetRecall, gpt-oss-120b (organizers' run) | **0.839** (tau AP-b 0.710, 20th of 385) | 0.776 |
| Ensemble of Qwen3-30B and gpt-oss-20b | 0.812 (tau AP-b 0.695) | 0.761 |
| Corroborate | 0.257 | 0.323 |

### ragtime25 (long reports, like ragtime26)

| Judge | Nugget Coverage | F1 |
|---|---|---|
| Length only (baseline) | 0.193 | 0.155 |
| Field median / best | 0.726 / 0.825 | 0.583 / 0.808 |
| NuggetRecall, Qwen3-30B | 0.697 | 0.500 |
| NuggetRecall, gpt-oss-20b | **0.728** | 0.517 |
| Ensemble | 0.709 | 0.493 |
| Corroborate (floor 0.15) | 0.598 | **0.662** (tau AP-b 0.524, 32nd of 206) |

### dragun-repgen

Corroborate scores 0.755 and places 8th of 139. NuggetRecall swings from 0.04 to 0.80
across runs, which is noise (finding 6).

---

## 5. Findings

1. **The two judges do best on different human measures.** NuggetRecall tracks nugget
   coverage. Corroborate tracks F1 on long reports, probably because NPREC punishes
   padding, which a recall measure never does. On short answers Corroborate fails (0.26
   to 0.32). The likely reason is that good short answers stand out by saying what
   others missed. A consensus cannot reward that. The two judges also rank systems differently,
   with Kendall's tau of 0.51 between them on ragtime25 and 0.25 on rag25-gen.

2. **The LLM matters most.** On rag25-gen with the same prompts, Qwen2.5-7B scores 0.604,
   Qwen3-30B 0.783, gpt-oss-20b 0.783 and gpt-oss-120b 0.839. The 7B model follows the
   format fine but seems to give partial credit too easily. On long reports,
   gpt-oss-20b beats Qwen3-30B on Nugget Coverage by 0.031.

3. **Ranking within each topic helps on short answers but not on long ones.** It reuses
   the same judgments and needs no extra LLM calls. The score rises from 0.788 to
   0.809 on rag25-gen but falls from 0.697 to 0.681 on ragtime25. TIRA takes the best
   measure, so we report both.

4. **Averaging two LLMs helps at the top, but only for one human measure.** On rag25-gen
   Strict Vital Score, the ensemble's tau AP-b is 0.695, against 0.634 for the best
   single model. On Sub Coverage it gets 0.562, worse than either model. We chose the
   ensemble for rag26 under the old scoring, which hid this.

5. **Answer length is now a weak shortcut.** The length baseline scored 0.558 on
   ragtime25 under the old scoring. Under the new one it scores 0.193. How the human
   truth is computed changes what looks like a shortcut.

6. **DRAGUN cannot rank LLM settings for NuggetRecall.** Our first run scored 0.141. On
   2026-10-21 we checked six more runs.
   - The code was not the cause. The current code scores 0.041.
   - TIRA gives the judge only the article headline. The released file gives a question,
     a claims summary and a fact-checker background. We confirmed this by reproducing
     the organizers' runs.
   - Even with the same model and headline input, our run scored 0.190 and the
     organizers' run 0.587. Each run writes new answer keys and the pilot has
     only 28 systems, so the scores swing from 0.04 to 0.80.
   - On 2026-10-23 the organizers confirmed that DRAGUN topics do not fit the AutoJudge
     format and were converted by hand. The same reply confirmed that the original
     article topics are in the release. A judge could look up the original article by its
     request id. We did not try this.
   - The lesson is to check that what you validate on locally is what the platform
     feeds your judge.

7. **A filter that never fires looks like a filter that does not matter.** Corroborate's
   topic-specificity floor, a cutoff meant to drop generic filler, was set to 0.15. That
   is below every value the encoder ever produces. Facts score 0.59 to 0.75 and filler
   0.36 to 0.47. So the filter kept everything. Trying values from 0 to 0.20 showed
   "no effect". We moved the floor to 0.50. That did not help. On DRAGUN it
   lowered Corroborate from 0.755 to 0.739 (tau AP-b 0.608 to 0.516). The value we
   picked matters less than the check. A judge should warn when a filter keeps everything.

8. **Run-to-run noise.** Running the same NuggetRecall setup twice on ragtime25 moved the
   scores by 0.010 to 0.019. We treat differences under 0.02 as noise there and on
   rag25-gen. DRAGUN is far noisier (finding 6). Corroborate gives the same output
   every time.

---

## 6. Silent failures, the most reusable part

In six cases a judge failed without raising an error, or would have.

| What failed | What happened | What caught it |
|---|---|---|
| LLM client defaults (60 s timeout, batch abort after 25 failures) | gpt-oss-20b scored 0.000 because 26 slow calls wiped out all 5,302 judgments | the 0.000, then a check that measures vary |
| Client settings carrying over between phases | would have zeroed only long answers, so the judge would quietly reward short ones | a code audit |
| Reasoning model running out of reply budget | 27% of rag26 judgments unreadable on the first try | vLLM's counters. Fixed with a bigger budget per retry (12.4% down to 2.7%) |
| A whole batch failing at once | every judgment in the batch lost | the run log |
| Replay check with nothing to replay | rehearsal "passed" while making no LLM calls | reading the output |
| Specificity floor below every value | filter kept 100% | measuring the encoder's range |

In general, a check that two runs agree must sit next to a check that the run did
something.

---

## 7. What we submitted

| System | Judge | LLM | rag26 | ragtime26 |
|---|---|---|---|---|
| nuggetrecall-best (main) | NuggetRecall | Qwen3-30B | yes | yes |
| nuggetrecall-rank | NuggetRecall + NR_RANK | Qwen3-30B | yes | yes |
| nuggetrecall-gptoss | NuggetRecall | gpt-oss-20b | | yes |
| ensemble-qwen-gptoss | NuggetRecall, both LLMs | both | yes | |
| corroborate-best | Corroborate | none | yes | yes |

Our code submissions on TIRA are **rank-dioptre** (NuggetRecall), **grave-node**
(NuggetRecall with the bigger retry budget, our main one) and **lead-racer**
(Corroborate).

---

## 8. Limitations

- We tuned only on the 2025 pilots, which have 28 to 59 systems each.
- The pilots were re-scored after we chose our submissions.
- NuggetRecall depends on the LLM's own knowledge and never sees citations.
- Corroborate cannot reward a correct fact no other team stated.
- We do not know whether TIRA's test-set topics match the released files.

## 9. Data handling

We followed the AutoJudge data policy. On the test sets we read response text only for
the first ten topics. Every choice based on scores came from the pilots. One fix, the
bigger retry budget, came after a count of unreadable judgments on rag26. We saw only
counts, never text or truth. The paper says so. Two of the 83 rag26 runs are probably
our own TREC RAG runs. We never tried to identify them.
