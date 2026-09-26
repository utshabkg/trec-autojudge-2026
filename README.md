# Two AutoJudges from team `iris-mst`

TREC 2026 AutoJudge. We submit two judges that work from different assumptions,
because on the pilot datasets each one fails where its assumption breaks.

**NuggetRecall is the primary submission.** It builds an answer key of facts from
the topic alone, then asks an LLM which of those facts each response covers.

**Corroborate is the secondary submission.** It makes no LLM calls. It pools every
system's sentences for a topic, treats a claim many independent teams make as
probably important, and corrects for the fact that a system is being scored
against a bank it helped build.

Everything below that carries a number was measured on three pilot datasets
(rag25-gen, ragtime25 and dragun-repgen) by uploading to TIRA and reading back the
correlation against the held-out human assessments. Nothing here is estimated.

## Where things are

| Path | What it holds |
|---|---|
| `judges/nuggetrecall/`, `judges/corroborate/` | the two judges |
| `scripts/` | running jobs on the compute node, syncing, upload checks, fusion |
| `tests/` | the test suite, which CI and tira-cli run before every code submission |
| `data/kiddie/` | the small synthetic dataset used for smoke tests |
| `docs/NOTEBOOK_PAPER.md` | the plain-language paper draft |
| `docs/WORK_LOG.md` | the full engineering record: every decision and measurement |
| `docs/updates/` | progress updates |
| `run_all_datasets.py`, `run_kiddie.sh`, `datasets.yml`, `fetch_datasets.py`, `Dockerfile` | from the starter kit, to run judges and build the image |
| `results/` | everything the judges produce: outputs, logs, caches. Not in git |
| `local-data/` | a link to the TREC data. Not in git |

---

## How we read the scoring, because it shaped both judges

TIRA correlates every measure we emit against every human truth measure and
reports `Max (Kendall)` and `Max (Tauap B)`, along with the corresponding `Min`.
Three consequences drove the design.

Several well-motivated measures are upside, since a maximum is taken. We emit five
per judge. `Min` is also reported, so every measure has to correlate positively
with truth. That is why neither judge emits a deliberately inverted diagnostic,
however useful one would be internally. Only rows with `topic_id == "all"` are
correlated, so per-topic rows are carried but are not what is scored.

`tauap_b` is top heavy. In the evaluation package's own fixtures, swapping the top
two systems costs 0.222 while swapping the bottom two costs 0.025. It also breaks
ties by input order, which is arbitrary. So every measure we emit is continuous and
never thresholded, and we check how many distinct aggregate values each produces.

One more thing worth stating plainly, because it sets the bar. The shipped naive
judge scores by answer length and reaches Kendall 0.558 on ragtime25, but only
0.152 on rag25-gen. The length confound is therefore specific to the track, not
universal, and beating length is a weak result on RAG generation. The median
submission across pilots sits near 0.73 and the best near 0.85. That is the real
bar.

---

## NuggetRecall, the primary submission

`judges/nuggetrecall/`, `uses_llm: true`.

### The answer key comes from the topic, not from the responses

For each topic the judge asks an LLM to list the facts a complete answer should
contain, marking each VITAL or OKAY. The responses are not shown. That matters for
two reasons. A key derived from the topic cannot be captured by whichever system
happens to be verbose, and it can contain a fact that every system missed, which a
pool-derived key cannot.

This is also what the strongest public judge on rag25-gen does, and the workflow
declares `nugget_depends_on_responses: false` so the framework does not pass
responses into the generation step at all.

### Judging is one call per response

Each response is judged against the whole key in a single call, which labels every
fact `support`, `partial` or `no`. Nothing scales as topics times runs times
nuggets, which is what makes the judge affordable: rag26 is 9,877 responses, so it
is 9,877 judging calls rather than tens of millions of pairwise ones.

A reply that does not label at least 80% of the facts is rejected outright rather
than back-filled, because filling unlabelled facts with `no` would invent misses.
Rejected replies are retried with a different prompt suffix, which changes the
cache key, and anything still unusable after three attempts scores zero and is
counted in the log. On our submitted runs that count was zero.

### Measures

| Measure | Definition |
|---|---|
| `NR_VITAL` | recall of VITAL facts, support 1 and partial 0.5 |
| `NR_ALL` | recall of all facts, same credit |
| `NR_VITAL_STRICT` | recall of VITAL facts counting full support only |
| `NR_WEIGHTED` | importance-weighted recall, vital 1 and okay 0.5 |
| `NR_RANK` | `NR_VITAL` re-expressed as a within-topic rank, then averaged |

`NR_RANK` is the one worth explaining, because it is the same judgments aggregated
differently and it is our best measure on the pilot that matters most. Averaging a
recall fraction over topics lets an easy topic, where every system scores high,
count as much as one that actually separates systems. Ranking within a topic first
removes topic difficulty. On rag25-gen that is worth 0.013 Kendall over the plain
mean and it also improves `tauap_b`. On ragtime25 it is slightly worse than the
mean, which costs nothing, because the reported score is a maximum over our
measures and the mean simply wins there instead.

---

## Corroborate, the secondary submission

`judges/corroborate/`, `uses_llm: false`.

Neither test dataset ships nuggets, qrels, or the retrieval corpus, so this judge
builds its reference from the only resource that needs no download: the pool of
submitted answers.

**Pool clustering is also the matching.** Every sentence of every response for a
topic is embedded with `BAAI/bge-base-en-v1.5` and clustered greedily in fixed
`(run_id, sentence_index)` order, so the result does not depend on file order. A
response covers a cluster because one of its own sentences landed there, so there
is no separate retrieve-then-entail pass. Only borderline pairs get a cross-encoder
verdict, which is what keeps the cross-encoder workload near 400k pairs on rag26
instead of millions.

**Votes are per team, not per run.** rag26 is 83 runs from 25 teams, and teams
submit close variants of one pipeline. Counting runs would measure herd size and
let one team vote repeatedly for its own phrasing.

**Leave-one-team-out removes bank capture.** A response's own sentences are in the
bank it is scored against, so without a correction every response automatically
covers whatever only it said. Cluster weights are recomputed per scored team as
the number of other teams asserting that cluster. This is the leave-one-group-out
pool-bias correction from retrieval evaluation, moved from documents to answers.
It is also the technical answer to the circularity objection rather than just a
disclosure of it.

| Measure | Definition |
|---|---|
| `NCOV` | leave-one-team-out, team-weighted coverage of the topic's clusters |
| `NPREC` | fraction of the response's own sentences corroborated by other teams |
| `NF1` | harmonic mean of the two |
| `NF1_RANK` | within-topic percentile rank of `NF1` |
| `CITE` | citation support, from inline documents where present, else pooled agreement |

RAGTIME draws on NeuCLIR, so we expected non-English documents and built a
`multilingual` variant for them. Both 2026 releases turned out to be English, so the
submitted `best` configuration uses the English NLI model on both test sets. The
multilingual reranker is not baked into the image, so that variant runs only
locally or with the model mounted.

### Why a pool-derived bank is not circular

The human reference is itself pool derived. TREC RAG's nugget bank is extracted
from pooled retrieved material and then vetted by assessors, so a pool-derived
bank approximates a pool-derived target rather than substituting a different one.

Separately, a fact no system covers cannot enter the bank, but it also contributes
the same zero to every system, so it cannot change their order. Universal misses
are rank preserving.

The residual risk we take seriously is that consensus pulls everyone toward the
median system, which would hurt most at the top of the leaderboard where `tauap_b`
is most sensitive. Leave-one-team-out and per-team voting reduce it. The ablation
variants quantify what is left.

---

## Results

Max Kendall against held-out human assessments, from TIRA. Rank is against every
scored entry on that pilot's public leaderboard.

| Pilot | Corroborate | NuggetRecall | NuggetRecall + `NR_RANK` | median | best |
|---|---|---|---|---|---|
| `rag25-gen` (short RAG answers) | 0.323 | 0.788 | **0.801** | 0.787 | 0.862 |
| `ragtime25` (long reports) | 0.691 | **0.739** | 0.729 | 0.726 | 0.802 |
| `dragun-repgen` (reports on a news article) | **0.755** | 0.141 | not run | 0.650 | 0.821 |

`rag25-gen` is the analogue of `rag26` and `ragtime25` of `ragtime26`, so
NuggetRecall is the primary submission on both test sets.

**Each judge fails exactly where its assumption breaks.** Corroborate assumes what
many teams agree on is what matters. That holds for long reports and fails for
short answers, where the best systems stand out by stating what others missed.
NuggetRecall assumes the request is a question. On DRAGUN the request is a whole
news article, so the key lists facts from the article rather than what a good
report should contain, and the judge collapses. The two rankings agree only weakly
with each other, Kendall 0.50 on ragtime25 and 0.23 on rag25-gen, so they are
complementary rather than redundant.

### Judge model

The submitted judgments use `Qwen/Qwen3-30B-A3B-Instruct-2507` (FP8). We also
measured the two models the organizers re-run submitted code against, on
rag25-gen:

| Judge model | Max Kendall |
|---|---|
| Qwen3-30B-A3B-Instruct-2507 (FP8) | 0.788 |
| openai/gpt-oss-20b | 0.783 |
| Qwen2.5-7B-Instruct | 0.654 |

A reasoning model works as well as the larger instruct model here. The 7B model
follows the output format just as reliably, with no failed or unreadable replies
over 5,302 calls, but judges less accurately, mostly by granting partial credit too
freely.

### Two cautions about reading any of these numbers

**Our run-to-run noise is about 0.01 Kendall.** We measured it by accident.
Re-running NuggetRecall on ragtime25 with nothing changed except an added measure
moved Max Kendall by -0.010, which adding a measure cannot do, since the score is
a maximum. vLLM is not bitwise deterministic across batch compositions. So we do
not treat differences below roughly 0.02 as real, and we withdrew an earlier
fusion result of +0.012 on that basis.

**A broken judge and a weak judge look identical from the score alone.** Running
NuggetRecall on gpt-oss-20b once scored exactly 0.000, last of 95 entries. The
model was fine. The LLM client timed calls out after 60 seconds, too short for a
reasoning model, and then abandoned the whole batch once 25 calls had failed, so 26
slow calls out of 5,302 destroyed all 5,302. Every measure came out constant, the
leaderboard ranked nothing, and no error was raised. What caught it was a check we
now run before every upload, which refuses any leaderboard whose measures do not
vary between systems. After fixing the timeout and the batch breaker the same
configuration scores 0.783. We would encourage anyone building an LLM judge to add
that check.

---

## Reproducibility

NuggetRecall routes every call through `minima-llm`, whose SQLite prompt cache is
keyed on the model and the exact messages and deliberately excludes the endpoint
URL. That is what lets TIRA re-execute the container with the endpoint removed and
reproduce the leaderboard from the cache alone. Prompts are built in sorted run
order, so they are byte identical across runs.

One caveat we would rather state than have discovered. A reproducibility check
compares two runs, so two runs that both fail identically will satisfy it. Our
first container rehearsal passed the cache check while making zero LLM calls,
because the endpoint was a closed port and the replay faithfully reproduced
nothing. The rehearsal now runs against a stub endpoint and fails if the emitted
leaderboard does not separate systems.

Corroborate makes no LLM calls and no network requests. Its models are baked into
the image, and given the same inputs it produces byte-identical output.

```bash
pip install -e '.[all]'
bash run_kiddie.sh                    # smoke test, no LLM or GPU needed

auto-judge run \
    --workflow judges/nuggetrecall/workflow.yml \
    --variant best \
    --rag-responses local-data/rag26/runs/generation/ \
    --rag-topics   local-data/rag26/topics/topics.jsonl \
    --out-dir      ./output/rag26/
```

Both judges run with `create_qrels: false`. This is deliberate and load bearing: a
TREC qrels file also parses as four columns, so emitting one beside the leaderboard
makes the upload fail its format check. We verified that empirically.

---

## Data handling and disclosure

Development followed the AutoJudge data policy. Response content was inspected only
for the first ten topics of each dataset's topics file, in file order. Parameters
were fitted on the pilot datasets and never on the 2026 test topics. We did not
need the `scrub` tool at any point, because every bug we hit reproduced on a pilot
or on the synthetic `kiddie` set, so there is no scrub log to include.

We submitted two generation runs to TREC RAG 2026 as team `iris`, so two of the 83
anonymized `rag26` runs are probably ours. We made no attempt to identify them,
including by comparison against our own submissions. Both judges treat all runs
identically, and Corroborate's leave-one-team-out weighting removes the mechanism
by which a team's own runs could otherwise gain credit.

---

## Limitations

- NuggetRecall depends on the LLM's own knowledge to write the answer key, and on
  the request being a real question. DRAGUN shows what happens when it is not.
- Corroborate cannot reward a fact that no other team stated. As argued above this
  is rank preserving, but it does mean the judge is blind to genuine novelty.
- Consensus weighting is a proxy for importance, not a measure of it. A correct but
  rare fact is weighted low.
- Corroborate's pooled citation path measures agreement with other teams'
  retrieval rather than whether a citation truly supports a claim. It is used only
  when the run files carry no document text.
- We tuned on the 2025 pilots, as the rules require, so the 2026 results may differ
  from what is reported here.
