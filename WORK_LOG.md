# TREC 2026 AutoJudge, team `iris-mst`: working document

**Last updated: 2026-09-22 (official data submission lodged on both test sets).** This file is the detailed engineering record of the
project: every decision with its reasoning, every measurement with its provenance,
and every problem hit along the way. It is kept current as work proceeds.

**For the paper, or to explain the work to someone new, read `NOTEBOOK_PAPER.md`
instead.** That is the short, plain-language version, shaped like a TREC notebook
paper. This file is what it is built from, and what to consult when reproducing or
debugging.

Two companion documents, deliberately narrower:

- `README.md` is the method description that ships with the submission. The
  organizers read it in place of a run description, so it stays short and
  external-facing.
- `judges/corroborate/README.md` is implementation notes for whoever touches the
  code.

A note on evidence, which matters for the paper: everything below is labelled
either **measured** or **not yet measured**. As of this writing the judge has
been validated only on the synthetic `kiddie` set, whose ground truth is
literally named `kiddie_fake`. Those numbers demonstrate that the pipeline works.
They are not evidence about judging quality, and are not quoted as such.

---

## 1. The task

TREC 2026 AutoJudge inverts the usual track. We do not build a retrieval or
generation system. We build the *judge*: given anonymized runs submitted by other
teams to a host track, produce a leaderboard ranking those systems. The score is
how well our leaderboard agrees with the human leaderboard, which is held back
until after the conference.

| Dataset | Host track | Systems | Topics | Reports to score |
|---|---|---|---|---|
| `rag26-generation` | TREC 2026 RAG (generation) | 83 runs, 25 teams | 119 | 9,877 |
| `ragtime26-repgen` | TREC 2026 RAGTIME (report generation) | 49 runs, 10 teams | 103 | 5,047 |

Deadline **2026-09-30**. A complete submission is two parts: a *data submission*
(the leaderboards we produce) and a *code submission* (the judge itself, so the
organizers can re-run it, including against their own choice of LLM).

### How we are scored, precisely

This shaped the design more than anything else, so it is worth stating exactly.
The TIRA evaluator runs the official `meta-evaluate` over the cross product of
(human truth measures x our measures) x {kendall, pearson, spearman, tauap_b,
kendall@10}, and reports **`Max (kendall)`, `Min (kendall)`, `Max (tauap_b)`,
`Min (tauap_b)`**. Three consequences:

1. **Emitting several complementary measures is upside**, because the maximum is
   taken. We emit five.
2. **`Min` is reported too**, so every measure must correlate positively with the
   truth. A deliberately inverted diagnostic measure would be pure downside. This
   is why there is no length-penalty measure in the output.
3. **Correlations use only the rows with `topic_id == "all"`**, the per-run
   aggregate. Per-topic rows are carried but are not what is correlated.

`tauap_b` is **top-heavy**. From the evaluation package's own test fixtures,
swapping the top two systems costs 0.222 while swapping the bottom two costs
0.025, roughly a factor of five. It also breaks ties by input order, which is
effectively random, so scores must stay continuous and must never be thresholded.

### The bar to beat

**Measured**, read from TIRA's public evaluations API for all four pilot
datasets. These are real correlations against the held-out human assessments, so
they are the actual competitive landscape, not estimates.

| Pilot | naive (length) Kendall / tauap_b | median submission | best submission | scored entries |
|---|---|---|---|---|
| `ragtime25` | 0.558 / 0.373 | 0.732 | 0.802 | 79 |
| `rag25-gen` | **0.152** / 0.119 | 0.737 | 0.849 | 68 |
| `dragun-repgen` | 0.391 / 0.184 | 0.700 | 0.821 | 47 |
| `rag24` | 0.376 / 0.244 | 0.592 | 0.716 | 66 |

Two conclusions, and the first corrects an assumption we had been designing
against.

**The length confound is track-specific, not universal.** Length reaches 0.558 on
ragtime25 report generation but only **0.152** on rag25-gen, the pilot most
analogous to our main target rag26. So on RAG generation, answer length is close
to uninformative, and the worry that a coverage measure might merely re-discover
length applies mainly to the RAGTIME side. We keep the length diagnostics, because
on ragtime26 they matter, but "beats length" is a weak bar on rag26 rather than
the central hurdle.

**The real bar is much higher than beating length.** The median submission sits
around 0.70 to 0.74 and the best around 0.80 to 0.85 on every pilot. Being merely
better than a word count would put us near the bottom of a field of 47 to 79
entries. That is the number to design against.

Worth noting for method choice: the top entry on rag25-gen is
`trema-unh/prefnugget-queryonly` at 0.849, where "queryonly" means nuggets derived
from the query alone rather than from the responses. On RAG generation, query-only
nuggets outperform response-grounded ones for that team. Our `query-only` variant
exists as a fallback and this suggests it deserves to be run as a serious
contender rather than an ablation.

---

## 2. Method: Corroborate

Pooled corroboration with leave-one-team-out weighting.

Neither test dataset ships nuggets, qrels, or the retrieval corpus. So a
nugget-recall judge has to construct its own reference standard, and a
citation-support judge has to work from whatever document text the runs
themselves carry.

**The core idea.** The pool of submitted answers is the resource nobody has to
download. For any topic we hold 83 answers (rag26) or 49 (ragtime26). A fact that
many *independent teams* state is, with high probability, a real and important
fact about that topic. This is the pooling assumption from retrieval evaluation,
transplanted from documents to answers.

**The governing cost constraint.** Nothing may scale as topics x runs x nuggets.
A design with one model call per (nugget, report) pair would need tens of millions
of calls for 15,000 reports. Every expensive step here is either per topic or is
an encoder pass that batches.

### Stage 1: pool clustering, which is also the matching

For each topic, take every sentence of every report. Participants already
sentence-segment their answers, so `Report.responses` is a list of sentences.
Embed with `BAAI/bge-base-en-v1.5`. Cluster greedily by cosine similarity
(leader algorithm), processing sentences in fixed `(run_id, sentence_index)`
order so the result is deterministic and independent of file order.

The clustering **is** the matching step. A report covers a cluster because one of
its own sentences was assigned to that cluster, so there is no separate
retrieve-then-entail pass over every (nugget, report) pair. Only pairs whose
cosine falls in an ambiguous band get a cross-encoder verdict. This is what takes
the cross-encoder workload from tens of millions of pairs to an estimated ~400k
on rag26.

### Stage 2: votes are per team, not per run

A cluster's weight is the number of **distinct teams** asserting it. rag26 is 83
runs from 25 teams, about 3.3 runs per team, and teams typically submit close
variants of one pipeline. Counting runs would measure herd size and let a single
team cast several votes for its own phrasing. `metadata.team_id` is present in the
anonymized data, so this costs nothing.

### Stage 3: leave-one-team-out, which is the contribution

A report's own sentences are in the bank it is scored against. Without a
correction, every report automatically covers whatever only it said, and a
verbose or idiosyncratic system can manufacture clusters that inflate its own
score. We call this bank capture.

Corroborate stores the set of contributing team ids per cluster. When scoring a
report from team `t`, cluster weights become `|teams(c) - {t}|`, and clusters that
fall to zero leave that report's denominator. Covering a nugget therefore means
stating something that **at least one other team independently stated**. Nothing a
team says alone earns it credit.

This is the leave-one-group-out pool-bias correction from retrieval evaluation
(the standard treatment for a system benefiting from its own contribution to a
judged pool), applied to answer pooling. It is the part of this work we would
present as novel.

A bank cluster additionally requires `min_teams >= 2`, which guarantees no bank
entry is a single team's private invention, so a leave-one-team-out weight can
never reach zero and the recall denominator is always well defined.

### Stage 4: corroboration precision

Coverage measures alone tend to re-discover length, because a longer answer covers
more of anything. `NPREC` is the fraction of a report's **own** substantive
sentences that are corroborated by at least one other team, gated by topical
specificity so that boilerplate is not treated as universally corroborated.

A sentence no other team asserts is either novel and unverifiable, or padding.
`NPREC` is length-normalised by construction, which makes it the one component
that cannot be a restatement of length.

### Stage 5: within-topic percentile rank

We also report each run's percentile rank within each topic, macro-averaged. This
removes topic-difficulty variance, which is the dominant noise term in system
ordering, and it is the cheapest available lever on the top-heavy `tauap_b`.

Percentile rank rather than a z-score, deliberately: the framework fills a missing
cell with `0.0`, which is the correct "worst" for a percentile but would wrongly
mean "average" for a z-score.

### Stage 6: citation support, two paths chosen at runtime

- **inline**, when the runs carry `Report.documents`: each cited sentence is
  tested for entailment from the best-matching chunk of the documents it cites.
  This is real attribution.
- **pooled**, when they do not: for each cited document, the leave-one-team-out
  share of other teams citing that same document for that topic, times citation
  density. A pseudo-qrel built from the pool. Weaker, because it measures
  agreement with other teams' retrieval rather than truth of support, but it
  needs no corpus.

Which path applies is **not yet measured** and is the first thing to settle once
the data is in hand. `scripts/probe_schema.py` answers it with aggregate-only
structural statistics, which the data policy permits on any topic.

### Measures emitted

| Measure | Definition | Targets |
|---|---|---|
| `NCOV` | leave-one-team-out, team-weighted share of the topic's bank covered | nugget recall |
| `NPREC` | share of own sentences corroborated by other teams | precision, padding |
| `NF1` | harmonic mean of the two | a combined truth measure |
| `NF1_RANK` | within-topic percentile of `NF1`, macro-averaged | `tauap_b`, `kendall@10` |
| `CITE` | citation support, inline or pooled | attribution |

All five are oriented higher-is-better and are continuous. Raw `LENGTH` is
deliberately **not** emitted: it would let `Max()` credit us with the naive
baseline's performance, which is gaming the aggregation rather than judging.

### Why this is not circular

Two arguments we consider load bearing, both of which belong in the paper.

1. **The human reference is itself pool-derived.** TREC RAG's nugget bank is
   extracted from pooled retrieved material and then vetted by assessors. A
   pool-derived bank is therefore approximating a pool-derived target, not
   substituting a different one.
2. **Universal misses are rank-preserving.** A fact no system covers cannot enter
   the bank, but it also contributes the same zero to every system, so it cannot
   change the ordering. The bank can only fail to discriminate on facts nobody
   found, which is exactly where no discrimination is available from the responses
   anyway.

The residual risk we do take seriously is that consensus regresses everyone
towards the median system, which would hurt most at the top of the leaderboard
where `tauap_b` is most sensitive. Leave-one-team-out and per-team voting reduce
it. The ablations in section 6 are designed to quantify what is left.

---

## 2b. Second judge: NuggetRecall

Built after the pilots showed Corroborate is strong on long reports and weak on short
RAG answers (section 5). It follows the approach behind the strongest public judge
on rag25-gen.

**Step 1, per topic: an answer key from the request alone.** One LLM call generates
8 to 20 atomic, checkable facts, each marked VITAL or OKAY. The workflow sets
`nugget_depends_on_responses: false`, so the nugget phase receives no responses at
all and the key cannot depend on what any system said.

**Step 2, per report: one call labels every fact** `support`, `partial` or `no`.
One call per report rather than per (report, fact) pair keeps the test sets at about
15,000 calls instead of hundreds of thousands. The prompt and the all-or-nothing
parser are the ones already validated in our TREC RAG 2026 work.

**Measures**, all nugget-recall variants: `NR_VITAL` (vital facts, partial half
credit), `NR_ALL`, `NR_VITAL_STRICT` (full support only) and `NR_WEIGHTED` (vital
weight 1, okay 0.5).

Three details that are easy to get wrong, all handled and tested:

- **A retry must change the prompt.** The prompt cache keys on the exact messages,
  so re-sending an identical prompt after an unparsable reply replays the same bad
  reply forever. Retries append fixed reminder strings, which gives them a new cache
  key while keeping a replay deterministic.
- **No endpoint must not mean a crash.** The LLM client raises when
  `OPENAI_BASE_URL` is unset, which is exactly the state of the code-submission dry
  run. A missing endpoint now becomes a closed local port, so cached judgments are
  still served and only misses fall back.
- **The question lives in a different field in every dataset.** rag25 keeps it only
  in `problem_statement`, rag26 only in `title` (the full narrative, median 553
  characters), and ragtime26 across `title`, `problem_statement` and `background`.
  A judge reading one fixed field would have produced meaningless keys on at least
  one dataset. The request text concatenates whatever is present.

**Answer-key quality on rag25-gen, measured:** 105 topics, 15 to 20 facts each
(median 18), 48% marked vital, no topic fell back to a placeholder. The facts are
specific and checkable, for example named statutes with their years.

## 3. Decision log

The reasoning behind each choice, which is the part hardest to reconstruct later.

| # | Decision | Why |
|---|---|---|
| D1 | The primary judge makes **no LLM calls** | Not purity. It removes the failure modes most likely to cost us a submission: no network in the TIRA sandbox, no prompt-cache determinism proof, no dependence on GPUs other users are saturating, and a container that runs offline. An LLM variant is a stretch goal, not the critical path. |
| D2 | Votes per **team**, not per run | 83 runs from 25 teams. Run counting measures herd size. |
| D3 | **Leave-one-team-out** weighting | Removes bank capture, and is also the technical answer to our own runs being in the pool. |
| D4 | Clustering **is** the matching | Avoids a separate retrieve-then-entail pass, cutting cross-encoder work by orders of magnitude. |
| D5 | Percentile rank, **not** z-score | The framework's missing-cell default of `0.0` means "worst" for a percentile and "average" for a z-score. |
| D6 | **No raw `LENGTH`** measure | `Max()` is taken over measures, so emitting length would credit us with the naive baseline. |
| D7 | `create_qrels: false`, permanently | A qrels file beside the leaderboard breaks the upload format check. Measured, see section 5. |
| D8 | Container builds its own **Python 3.12** | The base image ships 3.11.0rc1, on which torch cannot import `_dynamo`. Measured, see section 5. |
| D9 | `transformers<5` pinned | Not needed once D8 is in place, but kept so the container runs the same versions our numbers were produced with. |
| D10 | Code submission runs from **GitHub Actions** | This machine has no container runtime and no route to one. See section 7. |
| D11 | Five measures, not one | `Max()` over measures is free upside, and the organizers evaluate per truth measure. |
| D12 | **Add an LLM judge, NuggetRecall, alongside Corroborate** | Revisits D1 on evidence. Corroborate ranks 7 of 60 on report generation but 86 of 91 on short RAG answers, where the strongest public judges derive nuggets from the question. D1 bought robustness and cost a great deal of rank on rag26. Corroborate stays for ragtime26, where it is competitive. |
| D13 | Answer keys from the **request alone** (`nugget_depends_on_responses: false`) | The approach behind the top rag25-gen judge. It makes the key independent of what systems said, and keeps the bank readable under the data policy for every topic. |
| D14 | Local judge model **Qwen3-30B-A3B-Instruct-2507, FP8**, served by vLLM on the node | Mixture of experts with about 3B active parameters, so fast, and 30 GB so it fits one 48 GB card with room for batching. Stronger than the Qwen2.5-7B the organizers use. The code submission takes whatever model TIRA injects. |

---

## 4. What is built

| Component | State |
|---|---|
| `judges/corroborate/` (8 modules) | complete |
| `judges/corroborate/workflow.yml`, 7 variants, 2 sweeps | complete |
| Test suite | **39 passing**, 2 xfailed by design |
| `scripts/probe_schema.py` | complete, verified on kiddie |
| `scripts/make_dev_subset.py` | complete, verified on kiddie |
| `scripts/validate_pilots.py` | complete, verified on kiddie |
| `scripts/check_upload_dir.py` | complete, catches the qrels trap |
| `scripts/fetch_data.sh` | complete, waiting on the password |
| `Dockerfile` | complete, builds and runs offline |
| CI: tests on Python 3.10 to 3.14 | passing |
| CI: TIRA code submission | **dry run green, tira-verified** |
| Data downloaded | done, 4.8 GiB, on the main PC and the compute node |
| Compute node (machine 54) | set up: env, models, data, 47 tests passing |
| Pilot runs | **running** on rag25-gen, ragtime25, dragun-repgen |
| TIRA pilot uploads, the real correlation | done, both judges on three pilots |
| **Data submission, NuggetRecall, rag26 and ragtime26** | **lodged 2026-09-22** |
| Data submission, Corroborate | next, second submission |
| Code submission, Corroborate | dry run green, not yet submitted |
| Code submission, NuggetRecall | dry run in progress |
| NuggetRecall tested on the organizers' models | next |

Unit tests run against a stub encoder with no GPU, no network and no model
weights, because `tira-cli` runs pytest on the submitting machine and the weights
live in the container image rather than the repo.

---

## 5. Findings so far

Everything here is measured. The provenance of each is given, because the
difference between "verified on real data" and "verified on a synthetic toy"
matters.

### The datasets, now downloaded and probed

Both releases fetched on 2026-09-19, 1.54 GiB compressed and 4.8 GiB extracted,
stored on the large volume and symlinked as `local-data`.

Counts match the announced figures exactly: rag26 **83 generation runs, 25 teams,
119 topics, 9,877 reports**, and ragtime26 **49 repgen runs, 10 teams, 103 topics,
5,047 reports**.

**Task zero is resolved: documents are inline in both.** rag26 99.1% of reports,
median 13 documents each, median 7,549 characters per document. ragtime26 99.7%,
median 10 documents, median 4,128 characters. So **attribution path A works and no
corpus is needed**, which was the single largest open risk in the plan. The
inference from file sizes that led us to expect this was correct.

**Both releases are English.** ragtime26 documents carry `lang: eng` where the
field is present and are entirely Latin script on inspection, and rag26 likewise.
The concern that ragtime26 would be multilingual NeuCLIR text, which would have
made an English NLI model return noise rather than an error, does not apply to
this release. The organizers appear to have released the English subset. The
`multilingual` variant stays as a documented fallback but is not needed, which
also means the 2.16 GiB reranker is dead weight in the container image.

**rag26 does not use the citation format its own track spec describes.**
`track_specs.yml` declares rag26 as `sentence_type: [rag24, neuclir]`, meaning
integer indices into `references`. The released data is entirely
`{doc_id: confidence}` dictionaries, the RAGTIME format, and its metadata keys are
identical to ragtime26's. The organizers evidently harmonised both tracks when
building the release. Our code is unaffected because it reads citations only
through `Report.get_sentences_with_citations()`, which normalises all three
conventions. Had we touched `.responses` directly, as the obvious implementation
would, every citation in rag26 would have been silently misread.

**Runs per team, which is what the leave-one-team-out design turns on:** rag26
3.32, ragtime26 **4.90**. Counting runs rather than teams would therefore have
let a single RAGTIME team cast nearly five votes for its own phrasing.

**Undocumented extras in the releases.** rag26 additionally ships 118 *retrieval*
runs, and ragtime26 ships 37 `mlir` runs and a **`nuggets/` directory with 15
participant-contributed nugget banks**. The nugget banks are the ones the
organizers mentioned, with a warning about circularity if a nugget-based judge
consumes banks produced by systems it is judging. We are not using them, and our
bank is mined from the pool instead, but they are an available signal.

### No ground truth exists locally, and what we do instead

**Nothing in any release contains human ground truth.** Established by
elimination:

- The pilot tarballs (`v0.2.1`) contain `runs/`, `topics/`, `metadata/` and a
  `dataset.yml`, with no `eval/` directory and no `truth:` key in any manifest.
  The release page lists exactly what is included and truth is not among it.
- The 2026 test tarballs ship only `eval/*/random.eval.jsonl`, the placeholder the
  howto describes.
- `tira-cli download --dataset X --truths` is refused for every confidential
  dataset, pilots included.
- Both release pages offer only the anonymized-runs tarballs, nothing else.

An earlier reading of the documentation had suggested the pilots ship
`*.official.eval.jsonl`. They do not. That filename appears in TIRA's *evaluator
command lines*, referring to files that live on TIRA's server.

**So local meta-evaluation against human truth is impossible, and the route is
TIRA.** The pilot datasets have real truth held server side and public
leaderboards. We run our judge on a pilot locally, upload the leaderboard, and
read back `Max (Kendall)` and `Max (Tauap B)` computed by the organizers'
evaluator against assessments we never see. This is how the track is designed to
work, and it is better evidence than a local proxy would have been. What it costs
us is diagnostic richness: we cannot compute discordant-pair accuracy without the
truth leaderboard, so the length check falls back to correlating each measure
against length directly, which needs no truth.

### Verified against the real infrastructure

**The qrels upload trap.** tira's `TrecEvalLeaderboard.matching_files` accepts any
file whose lines split into four whitespace-separated tokens, regardless of
extension, and a TREC qrels file (`topic 0 doc grade`) matches. We reproduced this
deliberately: adding a qrels file beside the leaderboard turns a valid upload into
`No unique file in trec-eval -q format was found`. Hence D7, and hence
`check_upload_dir.py` runs tira's own checker so it cannot drift from the server.

**The base image cannot run modern transformers.** Its `/venv` is Python
**3.11.0rc1**, a release candidate. Any modern transformers reaches
`torch._dynamo` during `from transformers import PreTrainedModel`, and torch's
dynamo polyfills do not survive on that interpreter. Two distinct failures, both
observed in CI:

- transformers 5.x: `modeling_utils` to `integrations/deepgemm.py` to
  `@torch._dynamo.assume_constant_result`, failing on the missing
  `sys.get_int_max_str_digits` (added in Python 3.11.1).
- transformers 4.x: `modeling_utils` to `integrations/flex_attention.py` to
  `@torch.compiler.disable`, failing with
  `Duplicate dispatch rule for sys.intern`.

Capping transformers is therefore not a fix. The interpreter is the problem, hence
D8. This is worth a sentence in the paper's reproducibility section: the shipped
base image cannot run a current encoder stack unmodified.

**TIRA is not a route to the data.** `tira-cli download` serves only
non-confidential datasets. `kiddie` returns HTTP 200, while `rag25-gen` and both
2026 test sets return 405. The UNH release behind HTTP basic auth is the only way
in.

**Baked model weights are 8.5 GiB if taken naively.** `snapshot_download` takes
everything a repo publishes, including ONNX exports and a `.bin` duplicate of
every `.safetensors`. `cross-encoder/nli-deberta-v3-base` alone ships 4.8 GiB of
ONNX variants. Restricting to safetensors gives 3.2 GiB, confirmed as `3.3G` in
the built image.

### Performance, measured on the rag25-gen pilot

The first real pilot run made a shared 120-core machine unusable, load average 123,
while making slow progress. The work itself was not large. The clustering issued
one matrix-vector product per sentence, about 300,000 of them per dataset, and
OpenBLAS defaults to 64 threads on that machine, so each product launched a
60-plus thread parallel region for a few milliseconds of arithmetic. Nearly all the
time went into starting and joining threads.

Two fixes: thread pools capped at 8 (through `threadpoolctl`, because OpenBLAS
ignores environment variables set after import), and clustering batched into one
matrix product per block of 512 sentences. The batched version is **identical**
to the sequential one, not approximately so: a test asserts equality of every
cluster id, leader, team set, run set, size and assignment across block sizes 1 to
1000.

| | before | after |
|---|---|---|
| pooling, per topic | 13.0 s | **1.04 s** |
| load average during the run | 123 | **1.5** |
| full judge run, 10 topics of rag25-gen | not measured | 85 s |

### Both judges compared on three pilots, the central result

**Measured**, all scores from TIRA against held-out human assessments. NuggetRecall
used Qwen3-30B-A3B-Instruct-2507 (FP8) served by vLLM.

| pilot | task | Corroborate Max(K) | rank | NuggetRecall Max(K) | rank | field median |
|---|---|---|---|---|---|---|
| `rag25-gen` | short RAG answers | 0.323 | 87 of 92 | **0.788** | **45 of 92** | 0.787 |
| `ragtime25` | report generation | 0.691 | 59 of 96 | **0.739** | **32 of 96** | 0.724 |
| `dragun-repgen` | reports on a news article | **0.755** | **7 of 61** | 0.141 | 57 of 61 | 0.650 |

Tauap B, the top-weighted correlation, follows the same pattern: NuggetRecall
0.621 and 0.599 on rag25-gen and ragtime25, Corroborate 0.608 on dragun-repgen.

**The two judges are complementary, and NuggetRecall wins on both of our actual
targets.** rag25-gen is the analogue of rag26 and ragtime25 of ragtime26, and
NuggetRecall leads on both, above the field median on ragtime25. NuggetRecall is
therefore the primary submission for both test sets, and Corroborate a second
submission, since it captures a different signal and its code submission is already
proven. The choice rests only on pilot evidence, as the data policy requires.

**Why NuggetRecall collapses on dragun, which matters as a limitation.** In DRAGUN
the request is not a question: it is the news article itself (the topics file is
349 KB for 30 topics, roughly 11 KB each). An answer key generated from a request
like that lists facts from the article, not what a good report about it should
contain. Query-derived nuggets assume the request is an information need. Both
2026 test sets are, so the failure does not transfer, but it bounds the method.

**The two judges measure different things.** Kendall between them is only about
0.50 on ragtime25 and 0.23 on rag25-gen, while NuggetRecall's four measures agree
with each other at 0.87 to 0.97.

**Reading Min correctly.** Min is taken over every (truth measure, our measure)
pair, not over our measures alone. NuggetRecall's four measures move almost in
lockstep, yet its Min on ragtime25 is -0.291 while its Max is 0.739, so the negative
value cannot come from one inconsistent measure of ours. ragtime25's human data must
contain a second kind of truth measure that recall-style judges score against,
plausibly citation support, which RAGTIME emphasises. Min on ragtime is therefore
bounded by the structure of the truth rather than by what we emit. On rag25-gen every
pair is positive (Min +0.721), so its truth is recall-like throughout. This revises
the earlier reading below, which attributed Corroborate's negative Min on ragtime25
to `CITE` alone: on rag25-gen `CITE` is still the likely culprit, but on ragtime25 at
least part of it is the truth itself.

### NuggetRecall across judge models, rag25-gen

**Measured**, TIRA, 2026-09-22. Same judge code and prompts, only the model changes.

| model | Max(K) | Max(Tauap B) | Min(K) | rank |
|---|---|---|---|---|
| Qwen3-30B-A3B-Instruct-2507 (FP8) | 0.788 | 0.621 | +0.721 | 45 of 93 |
| Qwen2.5-7B-Instruct | 0.654 | 0.479 | +0.480 | 65 of 93 |
| gpt-oss-20b, stock batch settings | 0.000 | -0.500 | 0.000 | 93 of 94 |

The gpt-oss row is a harness failure, not a judgment of the model, and it is the
most useful thing measured today. Every one of the 5,302 judgments was lost, so all
four measures came out constant and the leaderboard carried no ranking at all. Two
minima-llm defaults caused it. `TIMEOUT_S` is 60 seconds, which a reasoning model
with a 4096-token reply budget and 32 outstanding requests does not meet on a long
judge prompt. `BATCH_MAX_FAILURES` is 25, and it aborts the *whole* batch rather
than the failing calls, so 26 timeouts out of 5,302 destroyed all of them. All three
retry attempts died the same way. The timeout is now 900 seconds and the breaker is
off, because our own handling of a failed call (retry, then score zero, and count
both in the log) is strictly better than losing the batch. Neither setting touches a
cache key, so TIRA's replay check is unaffected.

This was worth finding before the deadline rather than after it: the organizers run
submitted code against gpt-oss-20b, so the code submission would have scored zero.
The run also showed that the upload guard works. It refused the directory, naming
all four measures as constant, which is exactly the silent-zero it exists to catch.

The 7B model had zero failed calls and zero unreadable judgments over 5,302 calls,
so format compliance is not the problem. On kiddie it ranked the worst run at 38%
recall where the 30B model gave 0%, and swapped the order of the top runs, which
points to over-generous partial credit. This is also the realistic preview of the
code-submission score on RAG-style data, since the organizers run submitted code
with Qwen2.5-7B-Instruct. Run time 906 s against 672 s for the 30B model, because
the 7B run happened at lower client concurrency before that was raised.

A possible improvement that would not spend held-out truth: use the 30B model's
judgments as a silver standard and measure how often the 7B model agrees with it
under a revised prompt. Not yet tried.

### The first real scores, from TIRA, against held-out human assessments

**Measured.** The `best` configuration run on three full pilots, uploaded to TIRA,
and scored by the organizers' own evaluator. Ranks are against every entry on that
pilot's public leaderboard at the time of reading.

| pilot | task | our Max (Kendall) | Max (Tauap B) | Min (Kendall) | rank | field median |
|---|---|---|---|---|---|---|
| `dragun-repgen` | report generation | **0.755** | 0.608 | 0.500 | **7 of 60** | 0.650 |
| `ragtime25` | report generation | 0.691 | 0.574 | -0.358 | 58 of 95 | 0.724 |
| `rag25-gen` | short RAG answers | 0.323 | 0.278 | -0.188 | 86 of 91 | 0.787 |

**The result is split by task, and the split is the finding.** On long-form report
generation the method is competitive, top 12% on dragun and just under the median
on ragtime25. On short RAG answers it is near the bottom. That maps directly onto
our two targets: ragtime26 is report generation and rag26 is RAG generation.

A plausible mechanism, not yet tested: short answers have few sentences, so the
pool yields few clusters and coverage of the consensus is a thin signal, while what
separates good short answers from mediocre ones is precise facts that most teams
did not state, which a consensus bank by construction cannot reward. Long reports
give the consensus much more to be measured against. This is the "consensus
regresses everyone to the median" risk identified at design time, and it appears
to bite in exactly the regime where answers are short.

**Why Min goes negative.** TIRA reports only the maximum and minimum across our
five measures, not each one, so the offending measure is inferred rather than
measured. Correlations between our own measures, which need no truth:

| pilot | Min (Kendall) | how `CITE` relates to the other four |
|---|---|---|
| `dragun-repgen` | +0.500 | moves with them, 0.48 to 0.54 |
| `ragtime25` | -0.358 | weakly, 0.24 to 0.38 |
| `rag25-gen` | -0.188 | not at all, -0.10 to -0.03 |

Where every measure scored positive, all five move together. Where Min is negative,
`CITE` is the one that has detached from the coverage cluster. So citation support
is almost certainly what drags Min below zero. It is also plausible on its face: the
truth measure is nugget *recall*, and a system that cites conservatively can write
fewer claims. Dropping `CITE` should lift Min. It will not lift Max, and on rag25-gen
Max is the problem.

### First signal on real data, rag25-gen, first 10 topics, truth-free

rag25 is a pilot, so it is unrestricted under the data policy and can be inspected
freely.

| measure | distinct values over 51 runs | Kendall vs LENGTH |
|---|---|---|
| `NCOV` | 49 | 0.188 |
| `NPREC` | 49 | -0.258 |
| `NF1` | 50 | 0.070 |
| `NF1_RANK` | 50 | 0.101 |
| `CITE` | 50 | 0.023 |

Every measure separates nearly every system, so ties will not decide the
`tauap_b` ordering, and none is a restatement of length. `NPREC` runs slightly
against length, which is what it is designed to do: longer answers carry more
uncorroborated padding. **What this does not show is agreement with human
assessors.** Carrying information beyond length is necessary, not sufficient. The
TIRA upload is what answers that.

### Calibration, on the synthetic kiddie set only

These come from a 5-topic, 4-run synthetic dataset. Treat them as order-of-
magnitude guidance for where to centre the real sweeps, nothing more.

**Embedding similarity scale for `bge-base-en-v1.5`:** a genuine paraphrase pair
sat at cosine **0.83**, while two unrelated sentences about the same topic sat at
**0.61**. So the useful discrimination band is narrow and sits high. This is why
the clustering threshold default is 0.80 and the cross-encoder band floor is 0.62.

**Clustering threshold sensitivity**, counting clusters corroborated by two or
more teams:

| tau | 0.70 | 0.74 | 0.78 | 0.82 | 0.86 | 0.90 |
|---|---|---|---|---|---|---|
| clusters in bank | 11 | 12 | **15** | 13 | 5 | 2 |
| mean teams per cluster | 1.40 | 1.29 | 1.25 | 1.16 | 1.06 | 1.02 |

The bank peaks near 0.78 and collapses above 0.85, consistent with the paraphrase
similarity above. Default set to 0.80, swept over 0.72 to 0.88 on the pilots.

**`min_other_teams` has a degenerate region.** At 2 a cluster needs three teams,
which on a small pool zeroes `NPREC` for every run and silently takes `NF1` and
`NF1_RANK` down with it. At 1 the measure correctly ordered kiddie's runs by their
designed quality (run2 0.65, run1 0.48, run3 and run4 0.00). Default is 1, swept
over 1 to 3. The judge now warns whenever any measure is constant or
low-cardinality, so this class of silent failure cannot recur unnoticed.

**The NLI model's entailment head is usable, its contradiction head is not.**
`cross-encoder/nli-deberta-v3-base` gave P(entailment) 0.996 for a true paraphrase
and 0.000 for an unrelated pair, which is clean. But it labelled an unrelated pair
`contradiction` at 0.999 where `neutral` was correct. We therefore use
P(entailment) only and never the contradiction or neutral outputs.

**Pipeline check on kiddie** (fake truth, 4 runs, 6 system pairs, so **not
evidence of quality**): `Max (kendall)` 0.913 from `NCOV`, `Max (tauap_b)` 0.667,
`Min (kendall)` 0.333, all five measures positive. The naive baseline scores
Kendall -0.333 on the same file. What this establishes is that the whole chain
runs and produces a well-formed, upload-valid leaderboard.

### Not yet measured

- Whether `Report.documents` is populated in the 2026 releases, which decides the
  attribution path. Indirect evidence: our own raw TREC RAG 2026 submission
  carries no `documents` and is 454 KB for 119 topics, while the released rag26
  run files are 10 to 65 MB, suggesting the organizers enriched them.
- Whether Corroborate beats the length baseline. **This is the question that
  decides whether the method is worth submitting**, and it cannot be answered
  until the pilot data is downloaded.
- All real parameter values.

---

## 6. Validation protocol, fixed in advance

Stated before seeing the data, so that the numbers we report later are not the
product of searching.

**Revised on 2026-09-19**, because the original plan assumed the pilots ship
human truth and they do not (section 5). Validation now has two halves: what we
measure locally without truth, and what we get back from TIRA.

The 2026 test releases ship a **random placeholder** truth file, so local
correlations on the test data are meaningless by construction and using them to
choose anything would be tuning on the test set.

### Half one, local and truth-free

`scripts/validate_pilots.py` without `--truth` reports, per measure:

- **correlation against answer LENGTH**, the confound. Above about 0.85 in
  absolute value the measure is length wearing a hat. How much this matters is
  dataset dependent: length scores 0.558 on ragtime25 but only 0.152 on rag25-gen.
- **number of distinct aggregate values**, because a measure that separates few
  systems has its order decided by input order under `tauap_b`, effectively at
  random.

Both are computable from our own output plus the run files, so they gate every
run before it is worth uploading anything.

### Half two, the correlation that counts, from TIRA

Run on a pilot locally, upload the leaderboard to that pilot's TIRA dataset, read
back `Max (Kendall)` and `Max (Tauap B)` as computed by the organizers' evaluator
against held-out assessments. Pilot leaderboards are public, so our score sits in
a field of 47 to 79 comparable entries, which is a far more honest read than any
number we could produce ourselves.

Two constraints on using it. Uploads are public submissions, so this is for a
handful of considered runs and not for sweeping parameters. And it gives only the
summary correlations, not per-pair detail, so design iteration has to lean on the
local half.

### The length gate, which is the decisive test

For every measure on every pilot, `scripts/validate_pilots.py` reports four things
beyond the correlation:

1. `conf(len)`, Kendall of the measure against `LENGTH`. Above about 0.85 the
   measure is length wearing a hat.
2. **`disc-acc`, discordant-pair accuracy.** Of the system pairs that `LENGTH`
   orders the *wrong* way against truth, the share this measure orders right.
   Length scores 0.0 here by construction, so this isolates exactly the signal
   length cannot explain. This is the number we will report in the paper.
3. `partial`, rank-partial correlation with truth controlling for length.
4. `n_disc`, how many discordant pairs existed, so a good accuracy on three pairs
   is not mistaken for evidence.

Correlations come from the official evaluation package, so they match what the
organizers compute.

### Go and no-go bars

Set against the measured field rather than against zero.

| Gate | Bar | If missed |
|---|---|---|
| Not length in disguise | every emitted measure below 0.85 absolute correlation with LENGTH, checked locally | drop or rework that measure |
| Separates systems | every measure has distinct aggregate values for most runs | find the rounding or the degenerate setting |
| Beats the naive baseline | `Max (Kendall)` from TIRA clearly above 0.558 on ragtime25 and above 0.152 on rag25-gen | the method does not work, say so plainly |
| **Competitive** | **at or above 0.70, the median of the field** | acceptable to submit, but report the standing honestly |
| Strong | at or above 0.80 on ragtime25 or rag25-gen | a real result |
| `topheavy` variant | beats `best` on `tauap_b` on the pilots we test | do not ship it |

The competitive bar is the one that matters. Beating a word count on ragtime25
would still leave us below every one of the 79 entries on that leaderboard.

### Avoiding pilot overfitting

Parameters are fitted using the local truth-free diagnostics and on
`dragun-repgen`, and `rag25-gen` (the rag26 analogue) and `ragtime25` (the
ragtime26 analogue) are held out for a single TIRA upload each. Since every TIRA
upload is public and reveals a score, the held-out sets can only be used once
without turning into a tuning signal. A parameter value is accepted only if its sweep neighbours are within 0.02
Kendall on all four, so we take a plateau rather than an argmax by a hair. With
around 10 systems per pilot leaderboard, a Kendall difference below about 0.08 is
noise, and we do not choose between variants on smaller differences.

### Ablations, which are the empirical content of the paper

| Variant | Question |
|---|---|
| `no-loto` | how much does the leave-one-team-out correction change the ranking |
| `run-votes` | how much does per-team voting matter versus per-run |
| `coverage-only` | is corroboration precision contributing beyond coverage |
| `no-cross-encoder` | how much does band adjudication buy over pure embedding similarity |

---

## 7. Infrastructure, and why it looks unusual

**This machine cannot build a container.** docker, podman, buildah and nerdctl are
all absent, `/etc/subuid` and `/etc/subgid` have no entry for our user, and sudo
requires a password, so even rootless podman cannot be configured. The code
submission therefore runs from GitHub Actions, which has Docker.

That is only cheap because the judge declares `uses_llm: false`. The local test
that `tira-cli` runs inside the submission needs no network and no LLM endpoint,
so there is no `--allow-network`, no cache determinism proof, and no prompt cache
pushed to GitHub.

**One open infrastructure risk.** The real code submission must push the image to
`registry.webis.de`, authenticating through `git.webis.de`. That round trip timed
out once from a GitHub runner during `verify-installation`. It succeeded on the
next attempt, so it appears transient, but if it recurs on the real submission it
is a question for the organizers rather than something we can fix in the
repository. Trimming the image from 8.5 to 3.2 GiB reduces the exposure.

**Compute.** Two RTX 6000 Ada, shared with other users who typically hold about
38 GB on each card, so plan for roughly 10 GiB free rather than 48. That headroom
is not stable: the first pilot run died to a CUDA OOM when the neighbouring job
grew mid-run, and a batch of 256 through `bge-base` wants 1.3 GiB in one
activation. The encoders now halve the batch on OOM and finally move to CPU, so
contention costs time rather than the run, and the default batch sizes are 64 and
32 so that ladder is a safety net rather than the normal path. The driver is
CUDA 12.9, which caps torch at `2.11.0+cu128` and vLLM at 0.13.0. Note that
`torch.cuda.is_available()` returns True on this host even when CUDA
initialisation has failed, so GPU availability must be verified with a real
operation.

**The main machine's GPU driver wedged on 2026-09-20.** Every process touching
the GPU blocked in uninterruptible sleep on `os_acquire_rwlock_write`, a lock
inside the NVIDIA kernel module. That included the neighbouring vLLM job's workers,
28 `nvidia-smi` calls, several of the owner's own `side-paper-pypots` workers, and
`gnome-shell`, which is why the desktop froze. Processes in that state ignore
`SIGKILL`, so it cannot be cleared from a user account and needs a reboot or a root
GPU reset. Two lessons we now apply: never poll `nvidia-smi` on a machine whose
driver may be hung, since each call joins the pile, and gate any GPU run on a
short-timeout probe.

**Compute moved to a second machine ("54"):** one idle RTX 6000 Ada with 48 GB,
64 cores, 125 GB RAM, driver 560.35 (CUDA 12.6). The environment there is built
with miniforge rather than anaconda, because anaconda's default channels require a
Terms of Service acceptance and conda-forge does not. Versions match the main PC
exactly: torch 2.11.0 (the cu126 build, to suit that driver), transformers 4.57.6,
sentence-transformers 5.7.0. The main PC remains the single source of truth for
code and git, so no credentials are copied to the node.
`scripts/sync_remote.sh` pushes code and data out and pulls results back, and
`scripts/run_on_node.sh` runs jobs detached so a dropped connection cannot kill
them.

**Data handling.** The policy permits inspecting report content only for the first
ten topics of each dataset's topics file, in file order, plus structural
diagnostics on any topic. Our mined nugget bank is response-derived and therefore
restricted outside that window, so all qualitative inspection of nuggets happens
on the pilots. We submitted two generation runs to TREC RAG 2026 under team id
`iris`, so two of the 83 anonymized rag26 runs are expected to be ours. We make no
attempt to identify them. Leave-one-team-out is the technical guarantee that no
team can gain from its own runs.

---

## 8. Progress log

### 2026-09-23 and 24

- **The specificity-gate fix changed nothing measurable, which refutes our guess.**
  Corroborate re-run on rag25-gen and ragtime25 with the floor at 0.50 instead of the
  inert 0.15. The config confirms it took effect. Measures moved by at most 0.015
  (NPREC most, CITE least), and TIRA's scores did not move: 0.323 on rag25-gen to
  three decimals on all three numbers, 0.690 against 0.691 on ragtime25. We had
  written that the dead filter was a plausible cause of Corroborate's collapse on
  short answers. It is not. The cause is the design assumption in finding 1. The
  fix stays because the gate should do what it says, but it is not a result.
- **Those Corroborate runs took 8.5 and 7.9 hours, on CPU.** The log says "No GPU with
  >= 6.0 GiB free, using CPU": vLLM held 44.8 of the single card's 49 GB, idle, for
  two days. Worth remembering both for scheduling and because it is a preview of a
  CPU-only TIRA worker.
- **NuggetRecall's code-submission rehearsal is now genuinely green.** Run 35789234299
  on commit 40f64f6. The judge ran on the runner against the stub and warmed the
  cache, the container then ran with no reachable endpoint, served every judgment
  from that cache, and the guard confirmed the container's own leaderboard ranks
  systems. TIRA's replay reproduced it.
- **Checked that the stub replies cannot ship.** Read tira's submit_code: on a real
  submission it uploads only the pre-upload output directory as try-run metadata
  (`upload_run_anonymous(tmp_dir)` with `tmp_dir` being `.../output`). The cache
  sits in a sibling `CACHE_DIR/`, and for it TIRA registers only the variable name
  and `rw` mode. So the organizers get an empty cache, not ours.

- **gpt-oss-20b on long reports: 0.789 on ragtime25, rank 10 of 105**, against 0.739
  and rank 37 for Qwen3-30B, the model our lodged submission used. That is 0.05, five
  times the noise floor. It got there despite 209 of 7,159 judgments (2.9%) staying
  unreadable after three attempts, against 0.75% on rag25-gen, and 5 of 122 topics
  falling back to a title-only key: the longer the report, the more often the
  reasoning model's reply is not usable. Zero failed calls, average latency 40 s, 99
  minutes. On rag25-gen the two models tie. Since the choice rests on the pilot, which
  is what the pilots are for, we started gpt-oss on ragtime26 for an additional data
  submission.
- **Two-model ensemble.** Equal-weight rank fusion of NR_VITAL from Qwen3-30B and
  gpt-oss-20b (`fuse_leaderboards.py --weight 0.5`), no weights fitted. rag25-gen:
  0.812 Kendall and **0.695 tau AP-b**, rank 33 of 113, against 0.801 and 0.628 for
  the best single configuration. The Kendall gain is inside the noise, the tau gain of
  0.067 is not. ragtime25: 0.769 and 0.605, worse than gpt-oss alone (0.789, 0.617),
  since averaging dilutes the stronger judge there. Decision: ensemble for rag26 (needs
  a gpt-oss rag26 run first), gpt-oss alone for ragtime26 (already submitted as
  `nuggetrecall-gptoss`). A per-track choice from two pilots, which is a real
  overfitting risk, stated in the paper notes.
- **Additional data submission on ragtime26: `nuggetrecall-gptoss`**, gpt-oss-20b,
  all 49 runs over 103 topics, 5 measures. 163 of 5,041 judgments unreadable (3.2%)
  and 3 of 103 topics on a title-only key, the same profile as the pilot that ranked
  10th. TIRA records the model this time.
- **The sync script deleted results a third time**, `upload-rank/` (regenerated
  exactly, and already on TIRA), and its pull would have stranded most result
  folders. Both fixed structurally rather than by extending a list: push deletes only
  inside git-tracked directories, pull fetches every directory git does not track.
  Proven with planted probes before use.
- **Two more reviews before submitting.** The first, of the real submission path,
  found the dirty-tree failure, the replay breaker, the four-second patience,
  Corroborate's missing cache in the graded run, and a guard that skipped
  Corroborate. The second, of those fixes, found that my scratch cache could itself
  crash Corroborate, that a 503 gateway bypassed the new patience, and that my
  patience test passed under the old code too. All fixed in e1b7c48 and 86317c3,
  with five regression tests checked to fail against the old judge.py.
- **NuggetRecall's code submission is registered on TIRA as `rank-dioptre`.** CI
  run 36072255644 on commit 86317c3, dispatched only after dry runs of both judges
  on that exact commit came back green. The log confirms each stage: the git
  repository is clean (the check that would have killed it two days earlier), the
  image produced valid output on kiddie and the guard confirmed it ranks systems,
  the replay from cache reproduced it, the metadata was uploaded, the image pushed,
  and the software registered.

### 2026-09-22

- The main machine's GPU driver wedged (section 7). Moved compute to machine 54.
- Built the environment there with miniforge, matching the main PC's versions
  exactly, fetched the models, pushed 5.1 GiB of data.
- Validated the performance commit that had been written while the main machine was
  hung: 47 passed, 2 xfailed on the node.
- Measured the fix on real data: pooling 12 times faster, load 1.5 instead of 123.
- First truth-free signal on rag25-gen: all five measures separate the systems and
  none is length in disguise.
- Launched full runs on three pilots: rag25-gen, ragtime25, dragun-repgen.
- **First TIRA scores for Corroborate**: strong on report generation, near the
  bottom on short RAG answers. Built **NuggetRecall** in response, which scored 0.788
  on rag25-gen (median) and 0.739 on ragtime25 (rank 32 of 96).
- Organizer Maik Froebe confirmed code submissions will be run against
  `Qwen/Qwen2.5-7B-Instruct` and `openai/gpt-oss-20b`, so NuggetRecall must be tested
  on both, not only on the model used for the data submission.
- Fixed intermittent CI failures, which were the runner running out of disk rather
  than anything in the code, and trimmed the test matrix to 3.10 and 3.12.
- Started NuggetRecall on the rag26 and ragtime26 test sets.
- **Lodged the official data submission**: NuggetRecall on both test sets, system
  `nuggetrecall-best`, model Qwen3-30B-A3B-Instruct-2507 (FP8). rag26 took 30 minutes
  and ragtime26 14, with zero failed calls, zero unparsable judgments and zero
  fallback answer keys. Both are complete, every one of the 83 and 49 runs over all
  119 and 103 topics, which the upload guard checked before anything was sent,
  because a partial submission is rejected. The judgment counts reconcile exactly:
  rag26 made 9,838 calls for 9,877 reports, and the 39-report gap is the 39 reports
  the schema probe found with no answer text, which are scored zero without a call.
- On the test sets only the upload guard was run, never the length diagnostics used
  on the pilots, because counting words reads report text, and outside the
  permitted window that is restricted.
- Found and fixed a code-submission bug that would have been silent: the workflow
  never forwarded the LLM endpoint names, so a NuggetRecall code submission could
  never have received a model on TIRA.
- Scored NuggetRecall on the organizers' smaller model, `Qwen/Qwen2.5-7B-Instruct`:
  0.654 max Kendall on rag25-gen, rank 65 of 93, against 0.788 and rank 45 for
  Qwen3-30B. The judge works on the small model but gives up about 0.13 Kendall.
- **NuggetRecall's code submission builds and boots.** The first attempt stalled for
  50 minutes and was killed. The cause was minima-llm's `MAX_ATTEMPTS`, whose default
  is 50 rather than the 6 we assumed, and our own fast-fail path, which only
  triggered when the endpoint variable was unset. CI sets a placeholder, so it
  never triggered. Added an endpoint preflight that opens a socket once and, if
  nothing answers, caps attempts and timeouts before any call is made. The dry run
  then finished in 11 minutes with every TIRA check green.

  **Correction, same day.** We first read that green run as proof the judge works in
  the container. It was not, and the mistake is worth recording because it is the
  same shape as the gpt-oss zero. The endpoint in CI was a closed port, so inside
  the container every call failed, the judge fell back, and the pre-upload test
  wrote an all-zero leaderboard. TIRA accepted it, because its format check only
  looks at the shape of the file and never asks whether a measure varies between
  systems. The deterministic-cache check passed too, for the worst possible reason:
  the first run cached nothing, so the replay reproduced that nothing exactly. Two
  runs that both did nothing agree perfectly. So the rehearsal proved the image
  builds, boots and emits a well-formed file, and proved nothing about judging.

  Fixed by giving the container something real to call. A stub OpenAI endpoint now
  runs on the runner, the container reaches it over the docker bridge with
  `--allow-network`, and a new step runs `check_upload_dir.py` over the container's
  own output so a degenerate leaderboard fails the job. A unit test pins the stub's
  replies to the real parsers, so a prompt change cannot quietly put us back in the
  all-zero case. The general lesson: **a reproducibility check compares two runs, so
  it is satisfied by two runs that both fail identically.** It needs a liveness
  check beside it.
- Fixed a failure mode the gpt-oss-20b run exposed. `run_batched` can raise for a
  whole batch rather than per item, and it did so on the first call of the judge
  phase, taking all 5302 judgments with it. The caller then retried under a changed
  prompt, which is right for an unparsable reply but wrong here, because changing
  the prompt defeats the cache and forces every answer to be re-sent. Had the throw
  landed on the final attempt, every judgment would have scored zero behind a single
  warning line. A batch-level throw is now resent unchanged, twice, so the cache
  replays what already succeeded. The two kinds of retry are now separate by
  design, and two tests pin the distinction.

- **Audited both judges for silent-failure modes**, after two in one day with the
  same signature. Six independent reviewers over the judge code, the leaderboard
  construction, the submission path, dataset differences, and policy and
  determinism, with two adversarial refuters per candidate. 23 candidates. The
  strongest signal was convergence rather than any single verdict: five reviewers
  independently found the sticky LLM tuning, three found the dead specificity gate.
  Three were confirmed by direct measurement and fixed, all in this log above.
  Worth noting for method: requiring both refuters to uphold a finding killed all
  23, including the three that were real. Independent rediscovery was the useful
  signal, and adversarial verification tuned that strictly is a filter for
  precision at a heavy cost in recall.

- **NR_RANK, and the measurement that produced it.** Fusing the two judges by
  within-topic rank looked like a small win on ragtime25 (0.751 against 0.739), so
  we ran the control that separates "fusion helped" from "rank aggregation helped":
  fuse at weight 1.0, which is pure NuggetRecall re-aggregated. That scored 0.809 on
  rag25-gen against 0.788, better on all three reported numbers, and 19 places
  higher. The gain was never fusion. It was the aggregation. Added as a fifth
  measure, NR_RANK, costing no LLM calls.
- **Measured our own noise floor, by accident.** Re-running NuggetRecall on ragtime25
  with nothing changed but an added measure moved max Kendall by -0.010. Adding a
  measure cannot lower a maximum, so that is run-to-run variance: vLLM is not
  bitwise deterministic across batch compositions. About 0.01 Kendall. We withdrew
  the fusion result on that basis, since +0.012 is inside it.
- **Derived NR_RANK onto the lodged test-set submissions rather than re-judging**,
  so the judgments already submitted are reused exactly and the noise above cannot
  bite. Verified bit for bit against the judge's own output: strip NR_RANK from the
  rag25-gen leaderboard, recompute, and all 51 aggregates agree to twelve decimal
  places. That took two attempts, because the two halves use different conventions.
  Per topic, rank over the runs present on that topic, since leaderboards are ragged
  (51 runs on most rag25-gen topics, 50 on twenty, 48 on one). For the aggregate,
  divide by every topic and count an absence as 0.0, which is what
  LeaderboardBuilder does under fix_aggregate. Getting it backwards moved aggregates
  by up to 0.047, several times the noise floor, so it would have read as real.
  Uploaded to both test sets as `nuggetrecall-rank`, alongside the existing
  submission rather than replacing it.
- One provenance gap, recorded deliberately: the derived upload carries only the
  leaderboard, so TIRA's Model column is empty for it. The judgments are the same
  Qwen3-30B-A3B-Instruct-2507 (FP8) ones as `nuggetrecall-best`. We did not
  re-upload to fix a metadata field, because TREC results cannot be withdrawn and a
  duplicate entry on an official test set is worse than a blank column.
- **Rewrote README.md**, which the organizers read in place of a run description. It
  covered only Corroborate, did not mention NuggetRecall at all, still said "the
  submitted configuration makes no LLM calls", left Validation as a TODO, and
  claimed a `scrub-log.jsonl` that does not exist. We never needed `scrub`, because
  every bug reproduced on a pilot or on kiddie.
- **Two process failures worth recording, both mine.** I killed a running job, read
  `pgrep -c` output of "2" as success when it meant two processes were still alive,
  and launched a second judge beside the first. Both wrote the same log and the same
  output directory while I restarted vLLM under them, producing a leaderboard that
  was part one model and part another. Nothing was uploaded. `run_on_node.sh` now
  refuses to launch beside a live run and has a real `stop` that kills the parent
  loop first, because killing the python child only advances the loop to the next
  dataset and looks like success. Both list processes instead of counting them.
  Separately, the code push's `--delete` removed the node's `fusion/` directory,
  three built leaderboards, minutes before upload. Same trap that once took
  `vllm.log`. Excluded now, with a comment saying to add any new node-written path
  in the same commit that creates it.

### 2026-09-19 and 20

Project started from nothing.

- Researched the task, the framework, the evaluation package, the data releases
  and the TIRA submission mechanics from primary sources.
- Wrote the plan, including the scoring analysis in section 1 that drove the
  design.
- Cloned and customised the starter kit as `iris-mst-autojudge`, kept the
  `starterkit` remote (a version test needs it), deleted the example judges we did
  not write and the committed artefacts that would otherwise ship.
- Implemented Corroborate: 8 modules, 39 tests.
- Verified end to end on kiddie, including the official meta-evaluation.
- Built the tooling: schema probe, dev subsets, the length gate, the upload guard.
- Set up the repo, CI, and TIRA authentication. Team registered as `iris-mst`
  (TIRA lowercases it, and `iris` was already taken by another group).
- **Got the code-submission dry run green.** Five CI cycles, five distinct bugs:
  the release-candidate Python twice by different routes, a transient Webis
  timeout, our own `HF_HUB_CACHE` leaking a host path into the container, and
  5.6 GiB of model formats nothing loads. tira confirms the image produces valid
  outputs.
- Renamed the repo to `trec-autojudge-2026`.

**Resolved later the same day:** the credentials arrived, and we then spent real
time on a 401 that was our own mistake. We had one password and assumed it was
"the Active Participants password". It is not. Each TREC year has its own, both
sent round by Angela Ellis at NIST in that year's welcome mail. Using `trec2026`
with the 2025 password gives a 401, and we wrongly read that as a broken server,
because the 2025 pair *does* work on the UNH data host even though NIST rejects it.
Organizer Laura Dietz pointed out the two passwords differ, which resolved it.
Measured on 2026-09-22 after finding the 2026 mail:

| user and password | UNH data host | trec.nist.gov |
|---|---|---|
| `trec2026` + 2026 password | 200 | 200 |
| `trec2025` + 2025 password | 200 | 401 |

So the howto was right all along and there was no server fault. The lesson worth
keeping is that the data host is more permissive than NIST, so it cannot be used to
confirm you hold the right credentials. `scripts/fetch_data.sh` now defaults to
`trec2026` and its 401 message says to check the year rather than blame the server.
Credentials stay in `~/.config/trec-autojudge/`, outside the repo, mode 600.

Data fetched, 1.54 GiB compressed and 4.8 GiB extracted, stored under
`/media/12TB/shared/datasets/trec-autojudge-2026` (itself a symlink onto the 20 TB
volume) and reached through a gitignored `local-data` symlink so the tooling's
hardcoded path keeps working.

Then, in order:

- **Task zero resolved.** Documents are inline in both releases, so attribution
  path A works and no corpus is needed. The largest open risk in the plan is
  closed.
- **Both releases are English**, so the multilingual reranker is not needed. It is
  2.16 GiB of the 3.2 GiB baked into the image and is a candidate for removal.
- **rag26 ships the RAGTIME citation format, not the one its own track spec
  declares.** Reading citations through the framework accessor rather than
  touching `.responses` is what saved this from being a silent
  every-citation-misread bug.
- **Discovered that no release contains ground truth**, which invalidated the
  original validation plan. Replaced it with the TIRA-upload route, see section 6.
- **Read the real competitive field** off TIRA's public pilot leaderboards, which
  moved the target from "beat length" to "reach about 0.70", and revealed that
  length is nearly uninformative on rag25-gen (0.152) though strong on ragtime25
  (0.558).
- **Made the encoders survive a contended GPU** after the first pilot run died to
  an OOM.
- Started the first real pilot run, on `rag25-gen`, the closest analogue to rag26.

**Previously blocked on** the shared TREC Active Participants password. Recorded
here because the four dead ends are worth not retrying:

- Not our personal `ir.nist.gov/evalbase` account password. Tried against the
  release, returns 401.
- Not retrievable by us: evalbase now authenticates through Login.gov, so it needs
  the account holder and MFA.
- Not available through TIRA: `tira-cli download` serves only non-confidential
  datasets, so every pilot and both test sets return 405.
- Not already in our possession from TREC RAG 2026. That work used the public
  `TREC-RAG/trec-rag-data` repository and a Pyserini API token, so the participants
  password was never needed and is nowhere on this machine.

It is the shared password for `https://trec.nist.gov/act_part/act_part.html`
(confirmed to be HTTP basic auth protected), used with username `trec2026`. The
routes to it are the TREC 2026 registration mail from NIST, or asking the
organizers directly through the private TIRA chat opened at registration, the
`#trec2026-auto-judge` Slack channel, or dietz@cs.unh.edu.

**Next, in order, once the data arrives:** probe whether documents are inline,
reproduce the naive baseline on the pilots, run the length gate, fit parameters on
the two fitting pilots, confirm on the two held-out ones, then the full test-set
run and both submissions.

---

## 9. Reproducing this

```bash
conda create -n autojudge python=3.12 && conda activate autojudge
pip install -e '.[all]'
pip install "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128  # this host only
export HF_HOME=/media/20TB/shared-cache/huggingface

bash run_kiddie.sh        # framework smoke test
pytest                    # 39 tests, no GPU or network needed

# the judge, on the synthetic set
export CACHE_DIR=./cache/kiddie
auto-judge run --workflow judges/corroborate/workflow.yml --variant best \
    --rag-responses data/kiddie/runs/repgen/ \
    --rag-topics data/kiddie/topics/kiddie-topics.jsonl \
    --out-dir ./output-kiddie/
python scripts/check_upload_dir.py output-kiddie --expect-runs 4 --expect-topics 5
```

Once the data is available:

```bash
printf '%s' 'THE-PASSWORD' > ~/.config/trec-autojudge/trec_password
chmod 600 ~/.config/trec-autojudge/trec_password
bash scripts/fetch_data.sh --pilot --test-2026

python scripts/probe_schema.py local-data/rag26/runs/generation      # attribution path
python run_all_datasets.py --workflow judges/corroborate/workflow.yml \
    --variant best --dataset ragtime25 --meta-evaluate
python scripts/validate_pilots.py --truth local-data/ragtime25/eval/*.official.eval.jsonl \
    --eval output/ragtime25/corroborate/best-all-all/best.eval.txt \
    --runs local-data/ragtime25/runs/repgen/ --label ragtime25
```

---

## 10. Pointers for the paper

Concepts this work stands on, worth citing properly when writing up:

- **Pooling in retrieval evaluation**, and its known bias towards systems that
  contributed to the pool. Our leave-one-team-out weighting is the standard
  leave-one-group-out correction for that bias, applied to answers rather than
  documents.
- **Nugget-based evaluation** of long-form answers, and TREC RAG's AutoNuggetizer
  in particular, which is itself pool-derived. That is what makes our
  pool-derived bank an approximation of the target rather than a substitute for
  it.
- **Rank correlation for meta-evaluation**, specifically the tie-aware,
  top-weighted AP rank correlation that the organizers report alongside Kendall.
- **Topic-difficulty variance and score standardisation**, which motivates the
  within-topic percentile measure.
- **The length confound in long-form answer evaluation.** Our discordant-pair
  accuracy diagnostic is the contribution we would offer here: it isolates the
  part of a judge's agreement with human assessors that answer length cannot
  explain.
