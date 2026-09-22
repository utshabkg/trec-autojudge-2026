# TREC 2026 AutoJudge, team `iris-mst`: working document

**Last updated: 2026-09-22 (first real scores from TIRA on three pilots).** This file is the single running record of the
project: what we are doing, why, what is built, what we have measured, and what
is still open. It is kept current as work proceeds, so it can be read directly
for a progress update or used as the raw material for the paper.

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
| TIRA pilot upload, the real correlation | next |
| Test-set run and upload | after the pilot result |

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

**Resolved later the same day:** the credentials arrived. One documentation bug
worth reporting upstream: the howto states the download user is `trec2026`, which
returns 401. The working user is **`trec2025`** with the same password.

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
