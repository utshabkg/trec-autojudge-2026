# TREC 2026 AutoJudge, team `iris-mst`: working document

**Last updated: 2026-09-20.** This file is the single running record of the
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

**Measured, from the public TIRA leaderboards on the pilot datasets:**

| Judge | Kendall | tauap_b | Dataset |
|---|---|---|---|
| shipped naive baseline (scores by answer **length**, plus a random number) | 0.558 | 0.373 | ragtime25 |
| best public judge (`trema-unh` `prefnugget-best`, preference-based, LLM) | 0.781 | 0.625 | ragtime25 |
| shipped naive baseline | 0.376 | 0.244 | rag24 |

**Length alone reaching 0.558 is the single most important number in this
project.** It means raw Kendall cannot demonstrate that a judge works, because a
measure that merely re-expresses answer length will score around 0.55 while
adding nothing a word count does not already give. Our validation protocol
(section 6) is built around that problem.

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
| Data downloaded | **blocked** |
| Pilot validation | **blocked on data** |
| Test-set run and upload | **blocked on data** |

Unit tests run against a stub encoder with no GPU, no network and no model
weights, because `tira-cli` runs pytest on the submitting machine and the weights
live in the container image rather than the repo.

---

## 5. Findings so far

Everything here is measured. The provenance of each is given, because the
difference between "verified on real data" and "verified on a synthetic toy"
matters.

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

The 2026 test releases ship a **random placeholder** truth file, so local
correlations on the test data are meaningless by construction and using them to
choose anything would be tuning on the test set. Real validation happens on the
four pilot datasets that ship genuine human truth: `ragtime25`, `rag25-gen`,
`rag25-auggen`, `dragun-repgen`.

### Reproduce the anchors first

Before trusting any number from our own judge, run the shipped naive judge on the
same files and confirm roughly 0.558 Kendall on ragtime25. If that does not
reproduce, the harness is wrong rather than the judge.

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

| Gate | Bar | If missed |
|---|---|---|
| Beats length | best measure at or above **0.65** Kendall on ragtime25 **and** `disc-acc` at or above **0.60** on 3 of 4 pilots | we do not have a judge, fall back to a simpler configuration and say so plainly |
| Competitive | at or above 0.72 Kendall on ragtime25 | acceptable to submit |
| Stretch | at or above 0.781 Kendall or 0.625 `tauap_b` | claim it, with the pilot caveat |
| Per-measure emission | every emitted measure at or above **+0.30** Kendall against every truth measure on all four pilots | drop that measure, to protect `Min (kendall)` |
| `topheavy` variant | beats `best` on `tauap_b` on all four pilots | do not ship it |

### Avoiding pilot overfitting

Parameters are fitted on `dragun-repgen` and `rag25-auggen` only. `rag25-gen` (the
rag26 analogue) and `ragtime25` (the ragtime26 analogue) are held out and reported
once. A parameter value is accepted only if its sweep neighbours are within 0.02
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
38 GB on each card, so plan for roughly 10 GiB free rather than 48. The driver is
CUDA 12.9, which caps torch at `2.11.0+cu128` and vLLM at 0.13.0. Note that
`torch.cuda.is_available()` returns True on this host even when CUDA
initialisation has failed, so GPU availability must be verified with a real
operation.

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

**Blocked on:** the shared TREC Active Participants password, needed to download
any real data. Not our personal evalbase password, which returns 401, and evalbase
now authenticates through Login.gov so it cannot be scripted.

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
