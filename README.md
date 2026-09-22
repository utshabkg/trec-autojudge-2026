# Corroborate: a pooled-corroboration AutoJudge with leave-one-team-out weighting

TREC 2026 AutoJudge submission from team `IRIS-MST`.

Corroborate scores a RAG or report-generation run without human nuggets, without
qrels, and without access to the retrieval corpus. It builds its own nugget bank
from the pool of submitted answers, corrects that bank for pool bias by leaving
out the team being scored, and does all per-report scoring with encoder models
rather than a generative LLM.

**Status: work in progress. Numbers are not filled in yet.** Sections marked
TODO are completed once the judge has been validated on the pilot datasets.

`NOTEBOOK_PAPER.md` is the short, plain-language account of the method and results,
shaped like a TREC notebook paper. `WORK_LOG.md` is the detailed engineering record
behind it.

## The problem this solves

Neither `rag26-generation` nor `ragtime26-repgen` ships nuggets, qrels, or a
corpus. A nugget-recall judge therefore has to invent its own reference standard,
and a citation-support judge has to work from whatever document text the runs
themselves carry.

The pool of submitted answers is the resource nobody has to download. For any
topic there are 83 answers (rag26) or 49 answers (ragtime26). A fact that many
independent teams state is, with high probability, a real and important fact
about that topic. This is the pooling assumption from retrieval evaluation,
applied to answers instead of documents.

## Method

### 1. Pool clustering, which is also the matching

For each topic, every sentence of every run is embedded with
`BAAI/bge-base-en-v1.5` and clustered greedily by cosine similarity. Sentences
are processed in `(run_id, sentence_index)` order, so clustering is
deterministic and reproducible.

The clustering is also the matching step. A report covers a cluster because one
of its own sentences is in that cluster, so there is no separate
retrieve-then-entail pass over every nugget and report pair. Only pairs whose
cosine similarity falls in an ambiguous band get a cross-encoder verdict from
`cross-encoder/nli-deberta-v3-base`. This keeps the cross-encoder workload to
roughly 400 thousand pairs on rag26 rather than the tens of millions a naive
nugget-by-report design would need.

### 2. Votes are counted per team, not per run

A cluster's weight is the number of **distinct teams** that assert it. rag26 has
83 runs from 25 teams, so roughly 3 runs per team, and teams typically submit
close variants of a single pipeline. Counting runs would measure herd size and
would let one team cast several votes for its own phrasing. `metadata.team_id`
is available in the anonymized data.

### 3. Leave-one-team-out, which removes bank capture

A report's own sentences are in the bank it is scored against. Without a
correction, every report automatically covers at least the clusters that only it
created, and a verbose or idiosyncratic system can manufacture clusters that
inflate its own score.

Corroborate stores the set of contributing team ids per cluster. When scoring a
report from team `t`, cluster weights become `|teams(c) - {t}|` and clusters that
fall to zero are dropped from that report's denominator. Covering a nugget
therefore means stating something that at least one *other* team independently
stated. Nothing a team says alone earns it credit.

This is the leave-one-group-out pool-bias correction from retrieval evaluation,
transplanted from document pooling to answer pooling.

### 4. Corroboration precision

Coverage measures alone tend to rediscover answer length, because a longer
answer covers more of anything. `NPREC` is the fraction of a report's *own*
sentences that are corroborated by at least two other teams, gated by topical
specificity so that boilerplate is not treated as universally corroborated.

A sentence no other team asserts is either novel and unverifiable or it is
padding. `NPREC` is length-normalized by construction, which makes it the one
component that cannot be a restatement of length.

### 5. Within-topic percentile rank

Corroborate also reports each run's percentile rank within each topic,
macro-averaged across topics. This removes topic-difficulty variance from the
system ordering. Percentile rank is used rather than a z-score because the
framework fills a missing cell with `0.0`, which is correct for a percentile
(worst) and wrong for a z-score (average).

### 6. Citation support, with two paths

Selected at runtime according to whether the run files carry document text.

- **Inline path**, when `Report.documents` is populated: each cited sentence is
  tested for entailment from the best-matching chunk of the documents it cites.
- **Pooled path**, when it is not: for each cited document id, the share of
  other teams citing that same document for that topic, times citation density.
  This is a pseudo-qrel built from the pool and needs no corpus.

`ragtime26` documents come from NeuCLIR and are not all English, so the
multilingual reranker `BAAI/bge-reranker-v2-m3` is used there in place of the
English NLI model.

## Measures reported

| Measure | Definition | Range |
|---|---|---|
| `NCOV` | leave-one-team-out, team-weighted fraction of the topic's clusters this report covers | 0 to 1, higher is better |
| `NPREC` | fraction of this report's own sentences corroborated by at least two other teams | 0 to 1, higher is better |
| `NF1` | harmonic mean of `NCOV` and `NPREC` | 0 to 1, higher is better |
| `NF1_RANK` | within-topic percentile rank of `NF1`, macro-averaged | 0 to 1, higher is better |
| `CITE` | citation support, inline or pooled path | 0 to 1, higher is better |

Every measure is oriented so that higher is better, and every measure is
continuous rather than thresholded, so system scores stay distinct.

## Why this is not circular

Two arguments, both of which we consider load bearing.

First, the human reference this judge is compared against is itself pool
derived. TREC RAG's nugget bank is extracted from pooled retrieved material and
then vetted by assessors. A pool-derived bank is therefore approximating a
pool-derived target rather than substituting a different one.

Second, a fact that no system covers cannot enter the bank, but it also
contributes the same zero to every system, so it cannot change the ordering of
systems. Universal misses are rank preserving. The bank can only fail to
discriminate on facts that no one found, which is exactly the case where no
discrimination is available from the responses anyway.

The residual risk we do take seriously is that consensus regresses everyone
towards the median system, which would hurt most at the top of the leaderboard
where the top-heavy `tauap_b` measure is most sensitive. Leave-one-team-out and
per-team voting reduce it, and the ablations below quantify what is left.

## Validation

TODO. To be filled in from the four pilot datasets that ship human truth
(`ragtime25`, `rag25-gen`, `rag25-auggen`, `dragun-repgen`).

Validation protocol, fixed in advance:

- Parameters are fitted on `dragun-repgen` and `rag25-auggen` only.
  `rag25-gen` and `ragtime25` are held out and reported once.
- Because the shipped naive baseline scores by answer length and already reaches
  Kendall 0.558 on ragtime25, raw Kendall is not treated as evidence. Every
  measure is additionally reported with its correlation against length, its
  rank-partial correlation against truth controlling for length, and its
  **discordant-pair accuracy**, meaning accuracy on exactly those system pairs
  that length orders the wrong way. Length scores zero on that last diagnostic by
  construction.

Ablations, reported as workflow variants:

| Variant | Question it answers |
|---|---|
| `no-loto` | how much does the leave-one-team-out correction change the ranking |
| `run-votes` | how much does per-team voting matter versus per-run voting |
| `coverage-only` | is corroboration precision contributing beyond coverage |
| `no-cross-encoder` | how much does band adjudication buy over pure embedding similarity |

## Reproducibility

The submitted configuration makes **no LLM calls**. `workflow.yml` declares
`uses_llm: false`. All scoring runs on two encoder models plus one multilingual
reranker, all baked into the container image, so the judge runs with no network
access. Given the same inputs it produces byte-identical output.

Clustering order is fixed by `(run_id, sentence_index)`, so the bank does not
depend on filesystem iteration order.

Local runs use `torch 2.11.0+cu128`, which is the newest CUDA 12.8 build and the
newest that works against a CUDA 12.9 driver. The container image installs the
CPU build instead, since TIRA workers are not assumed to have a GPU.

## Running it

```bash
pip install -e '.[all]'
bash run_kiddie.sh                    # smoke test, no LLM or GPU needed

auto-judge run \
    --workflow judges/corroborate/workflow.yml \
    --variant best \
    --rag-responses local-data/rag26/runs/generation/ \
    --rag-topics   local-data/rag26/topics/topics.jsonl \
    --out-dir      ./output/rag26/
```

## Data handling and disclosure

Development followed the AutoJudge data policy. Report content was inspected
only for the first ten topics of each dataset's topics file, in file order.
Parameters were fitted on the pilot datasets, never on the 2026 test topics.
`scrub-log.jsonl` is included as the policy asks.

We submitted two generation runs to TREC RAG 2026 under the team id `iris`, so
two of the 83 anonymized `rag26` runs are expected to be ours. We made no attempt
to identify them, including by comparison against our own submissions. The judge treats all
runs identically, and leave-one-team-out removes the self-credit mechanism by
which a team's own runs could otherwise be advantaged.

## Limitations

- The nugget bank can only contain facts that at least one system stated, so the
  judge cannot reward a fact every system missed. As argued above, this is rank
  preserving.
- Consensus weighting is a proxy for importance, not a measure of it. A correct
  but rare fact is weighted low.
- The pooled citation path measures agreement with other teams' retrieval rather
  than truth of support, and is used only when the run files carry no document
  text.
