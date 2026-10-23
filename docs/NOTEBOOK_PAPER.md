# TREC 2026 AutoJudge: Team IRIS-MST, plain-language version

**Last updated 2026-10-23.** The paper itself is LaTeX, in `paper/` (open it in
Overleaf from `results/paper-overleaf.zip`). This file says the same things in plain
words, for the professor and for anyone new. Every number here matches the paper.
The detailed record behind it is `docs/WORK_LOG.md`.

Where numbers come from: TIRA, read 2026-10-19 (rag25-gen, ragtime25) and 2026-10-21
(our extra DRAGUN runs). The organizers re-scored the pilots on 2026-10-09, so any
older number in the work log or in the September update uses the old scoring and is
out of date.

---

## Still open before the paper is final

- Author surname, advisor name and email, acknowledgments.
- A public link to the code. The GitHub repository is private now.
- Test-set results. The organizers have not released them yet.
- Whether TIRA's test-set topics match the released files. Laura's reply
  (2026-10-23) covered DRAGUN only. It implies rag26 and ragtime26 needed no
  conversion, but does not say so. If they differ, the paper needs one more paragraph.
- The 2026 notebook deadline. Not published yet. The conference is 2026-11-16 to 18,
  so expect it in early November.

---

## Abstract

AutoJudge asks for automatic judges that rank the systems of a host track the way
human assessors would. We built two judges with opposite assumptions.

- **NuggetRecall** asks an LLM to write an answer key of 8 to 20 facts from the
  question alone. Then, for each response, it asks which facts the response states.
- **Corroborate** uses no LLM. It pools all responses to a topic and counts how many
  different teams state each claim. A team's own vote never counts for itself.

On the pilots, the two judges do best on different human measures. NuggetRecall is
our better judge for nugget coverage. Corroborate is our better judge for F1 on long
reports, and it fails on short answers. The choice of LLM mattered more than
anything else. We also found six ways a judge can fail without any error message.

---

## 1. The task

A normal TREC track asks systems to answer questions. AutoJudge asks the reverse:
given many systems' answers, **rank the systems**. A judge is scored by how closely
its ranking matches the human ranking, using Kendall's tau and a top-weighted
version, tau AP-b.

| Dataset | Kind | Runs | Teams | Topics | Responses |
|---|---|---|---|---|---|
| rag26 (test) | short cited answers | 83 | 25 | 119 | 9,877 |
| ragtime26 (test) | long cited reports | 49 | 10 | 103 | 5,047 |
| rag25-gen (pilot) | short cited answers | 51 | 18 | 105 | 5,332 |
| ragtime25 (pilot) | long cited reports | 59 | 13 | 122 | 7,198 |
| dragun-repgen (pilot) | reports on a news article | 28 | 9 | 30 | 840 |

- **The pilots have human truth, but only TIRA has it.** We test a judge by running
  it on a pilot, uploading the result, and reading back the score.
- **TIRA reports the best of a judge's measures** for each human measure. So a judge
  may report several measures. It also means a judge with many measures gets a
  small free advantage.
- **Since 2026-10-09 each human measure is scored separately.** rag25-gen has Strict
  Vital Score and Sub Coverage. ragtime25 has Nugget Coverage and F1 (and Citation
  Support, which is 0.000 for everyone, so we ignore it). DRAGUN still reports one
  number.

---

## 2. The two judges

### NuggetRecall (with an LLM)

1. For each topic, the LLM writes an **answer key** of 8 to 20 facts from the question
   only. Each fact is VITAL or OKAY.
2. For each response, one LLM call labels every fact: support, partial, or no.
3. Scores: share of vital facts covered, share of all facts, a strict version that
   ignores partial credit, and an importance-weighted version.
4. A fifth score, **NR_RANK**, ranks the runs within each topic first and then
   averages. Same judgments, different averaging.

Why build the key from the question only? A long-winded system cannot shape it, and
it can include facts every system missed.

Details that matter: a reply must label at least 80% of the facts or it is retried.
A reasoning model gets a bigger reply budget on each retry (4,096, then 8,192, then
16,384 tokens). Every call is cached, so TIRA can replay the judge without the LLM.

### Corroborate (no LLM)

1. For each topic, split every response into sentences and group sentences that say
   the same thing (embedding similarity of at least 0.80).
2. A group enters the "bank" only if at least **two different teams** said it. Teams,
   not runs, because teams submit several near-copies.
3. Score a response on how many bank groups it covers (NCOV), and on how many of its
   own sentences other teams also said (NPREC, which punishes padding).
4. **Leave-one-team-out:** when scoring team t, a group's weight is the number of
   *other* teams that said it. The idea comes from the classic pool-bias test in
   retrieval evaluation.

---

## 3. Setup

- **LLMs we ran:** Qwen3-30B-A3B-Instruct (FP8), gpt-oss-20b, and Qwen2.5-7B-Instruct,
  on one RTX 6000 Ada with vLLM, on compute node 54.
- **LLMs the organizers ran our code with:** Qwen2.5-7B, gpt-oss-20b, gpt-oss-120b, and
  on the test sets also Qwen3.8-27B.
- **Corroborate:** bge-base-en-v1.5 for embeddings, nli-deberta-v3-base for
  borderline matches. No LLM, no network.

---

## 4. Results on the pilots

Kendall's tau (and tau AP-b where it matters). "Field" is every entry on the TIRA
leaderboard: 385 on rag25-gen, 206 on ragtime25, 139 on DRAGUN. Many entries are
variants from the same team, so ranks are a loose guide.

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

Corroborate scores 0.755, 8th of 139. NuggetRecall ranges from 0.04 to 0.80 across
runs, which is noise (finding 6).

---

## 5. Findings

1. **The two judges do best on different human measures.** NuggetRecall follows
   nugget coverage. Corroborate follows F1 on long reports, probably because NPREC
   punishes padding, which a recall measure never does. On short answers Corroborate
   fails (0.26 to 0.32), likely because good short answers stand out by saying what
   others missed, and a consensus cannot reward that. The two judges rank systems
   differently: Kendall 0.51 between them on ragtime25 and 0.25 on rag25-gen.

2. **The LLM matters most.** On rag25-gen, same prompts: Qwen2.5-7B 0.604, Qwen3-30B
   0.783, gpt-oss-20b 0.783, gpt-oss-120b 0.839. The 7B model follows the format fine
   but seems to give partial credit too easily. On long reports, gpt-oss-20b beats
   Qwen3-30B on Nugget Coverage by 0.031.

3. **Ranking within each topic helps on short answers, not on long ones.** Same
   judgments, no extra LLM calls: rag25-gen up from 0.788 to 0.809, ragtime25 down
   from 0.697 to 0.681. Since TIRA takes the best measure, we report both.

4. **Averaging two LLMs helps at the top, but only for one human measure.** On
   rag25-gen Strict Vital Score the ensemble's tau AP-b is 0.695, against 0.634 for
   the best single model. On Sub Coverage it is 0.562, worse than either model. The
   old scoring hid this, and we chose the ensemble for rag26 under it.

5. **Answer length is a weak shortcut now.** The length baseline was 0.558 on
   ragtime25 under the old scoring and is 0.193 under the new one. How the human
   truth is computed changes what looks like a shortcut.

6. **DRAGUN cannot rank LLM settings for NuggetRecall.** Our first run scored 0.141.
   We checked six more runs (2026-10-21):
   - The code was not the cause: the current code scores 0.041.
   - TIRA gives the judge only the article headline, while the released file gives a
     question, a claims summary and a fact-checker background. We confirmed this by
     reproducing the organizers' runs.
   - Even with the same model and the same headline input, our run scored 0.190 and
     the organizers' scored 0.587. Each run writes new answer keys and the pilot has
     only 28 systems, so the scores swing from 0.04 to 0.80.
   - The organizers confirmed (2026-10-23) that DRAGUN topics do not fit the AutoJudge
     format and were converted by hand, and that the original article topics are in
     the release. A judge could look up the original article by its request id. We
     did not try this.
   - Lesson: check that what you validate on locally is what the platform feeds your
     judge.

7. **A filter that never fires looks like a filter that does not matter.**
   Corroborate's topic-specificity floor was set to 0.15, below every value the
   encoder ever produces (facts score 0.59 to 0.75, filler 0.36 to 0.47), so it kept
   everything. A sweep from 0 to 0.20 showed "no effect". We moved it to 0.50. That
   did not help: on DRAGUN it lowered Corroborate from 0.755 to 0.739 (tau AP-b 0.608
   to 0.516). The lesson is the check, not the number: warn when a filter keeps
   everything.

8. **Run-to-run noise.** The same NuggetRecall setup run twice on ragtime25 moved by
   0.010 to 0.019. We treat differences under 0.02 as noise there and on rag25-gen.
   DRAGUN is far noisier (finding 6). Corroborate gives identical output every time.

---

## 6. Silent failures, the most reusable part

Six times a judge failed without an error, or would have:

| What failed | What happened | What caught it |
|---|---|---|
| LLM client defaults (60 s timeout, batch abort after 25 failures) | gpt-oss-20b scored 0.000: 26 slow calls wiped all 5,302 judgments | the 0.000, then a check that measures vary |
| Client settings sticking across phases | would have zeroed long answers only, so the judge would quietly reward short ones | a code audit |
| Reasoning model out of reply budget | 27% of rag26 judgments unreadable on the first try | vLLM's counters. Fix: bigger budget per retry, 12.4% down to 2.7% |
| A whole batch failing at once | all judgments of the batch lost | the run log |
| Replay check with nothing to replay | rehearsal "passed" while making no LLM calls | reading the output |
| Specificity floor below every value | filter kept 100% | measuring the encoder's range |

General lesson: a check that two runs agree must sit next to a check that the run
did something.

---

## 7. What we submitted

| System | Judge | LLM | rag26 | ragtime26 |
|---|---|---|---|---|
| nuggetrecall-best (main) | NuggetRecall | Qwen3-30B | yes | yes |
| nuggetrecall-rank | NuggetRecall + NR_RANK | Qwen3-30B | yes | yes |
| nuggetrecall-gptoss | NuggetRecall | gpt-oss-20b | | yes |
| ensemble-qwen-gptoss | NuggetRecall, both LLMs | both | yes | |
| corroborate-best | Corroborate | none | yes | yes |

Code submissions on TIRA: **rank-dioptre** (NuggetRecall), **grave-node** (NuggetRecall
with the bigger retry budget, our main one), **lead-racer** (Corroborate).

---

## 8. Limitations

- We tuned only on the 2025 pilots, which have 28 to 59 systems each.
- The pilots were re-scored after we chose our submissions.
- NuggetRecall depends on the LLM's own knowledge and never sees citations.
- Corroborate cannot reward a correct fact no other team stated.
- We do not know if TIRA's test-set topics match the released files.

## 9. Data handling

We followed the AutoJudge data policy: on the test sets we read response text only
for the first ten topics, and every choice by score came from the pilots. One fix,
the bigger retry budget, followed a count of unreadable judgments on rag26. We saw
only counts, never text or truth, and the paper says so. Two of the 83 rag26 runs
are probably our own TREC RAG runs. We never tried to identify them.
