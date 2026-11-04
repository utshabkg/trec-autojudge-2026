# judges/corroborate

Corroborate clusters the sentences of all responses to a topic and treats each
cluster as a claim. A claim needs at least two teams to enter the topic's reference
bank, which keeps out a solo claim. Its leave-one-team-out (LOTO) weight counts the
teams behind it, minus the team being scored. The repository README describes the
method. This file covers the code and how to run it.

## Modules

| File | Responsibility |
|---|---|
| `textutil.py` | NFKC normalization (Unicode cleanup), boilerplate and length filters and stable cache keys, using only the standard library so normalization cannot drift between machines. |
| `encoders.py` | Model loading, device selection and the on-disk score cache. |
| `pool.py` | Per-topic clustering, team votes and the LOTO weight. |
| `scoring.py` | `NCOV` (bank coverage), `NPREC` (share of sentences another team backs), their harmonic mean `NF1` and its within-topic percentile rank `NF1_RANK`. |
| `attribution.py` | Citation support, from entailment (the cited text backs the sentence) or citation agreement among teams. |
| `measures.py` | The `LeaderboardSpec`, with the reasoning for each measure. |
| `judge.py` | The `auto-judge` runner's interface for the nugget and judge phases, plus export to `NuggetBank`. |

Three code decisions look arbitrary but are not.

- **`pick_device` proves CUDA with a real matrix multiplication.** On our driver and
  PyTorch build, `torch.cuda.is_available()` returns True even when CUDA
  initialization has failed.
- **The cross-encoder, a model that scores two texts read together, runs through
  plain `transformers`.** The default output function of
  `sentence_transformers.CrossEncoder.predict` has changed between versions, so we
  compute the entailment probability from the raw outputs.
- **One class serves both phases.** The runner caches instances by class path, so
  declaring only `judge_class` means the pooled clustering runs once.

## Running it

```bash
export CACHE_DIR=./results/cache/rag26        # embedding and cross-encoder score cache
auto-judge run \
    --workflow judges/corroborate/workflow.yml --variant best \
    --rag-responses local-data/rag26/runs/generation/ \
    --rag-topics    local-data/rag26/topics/topics.jsonl \
    --out-dir       ./results/output/rag26/
```

Set `CACHE_DIR` even though the judge makes no LLM calls. It keeps the embedding and
cross-encoder scores, so an interrupted run is cheap to resume.

`ragtime26` needs no special variant. We feared non-English documents would give noise
from an English-only entailment model, but both 2026 releases are English. So `best`
fits both test sets.

## Variants

`best` is the submitted configuration, behind `corroborate-best` (data submission) and
`lead-racer` (code submission, a container run by TIRA, the evaluation platform). One
NuggetRecall code submission, `grave-node`, doubles its LLM reply budget (token limit)
on each retry, which does not apply here.

`no-loto`, `run-votes`, `coverage-only` and `no-cross-encoder` are ablations, copies
of the judge with one part changed. We ran no ablation of the LOTO weight or of
per-team voting, so their effect is unknown. `no-loto` only stops `NPREC` requiring
another team's agreement. The `NCOV` weight stays.

`fast` turns off the cross-encoder in coverage scoring, caps the bank at 25 clusters
and reads at most 2 chunks per cited document. It is faster on a CPU-only worker but
not time-bounded, because citation entailment still calls the cross-encoder. It is not
validated against truth. `multilingual` swaps in `bge-reranker-v2-m3`, which the
container lacks, so it runs only locally or with the model mounted.

## Two things that will bite you

**Never set `create_qrels: true`.** TIRA finds your leaderboard by looking for files
whose lines split into four whitespace-separated tokens. A TREC qrels file (relevance
labels) has that shape too, so the upload fails with "No unique file in trec-eval -q
format was found". Verified locally. Run `scripts/check_upload_dir.py` before every
upload.

**Watch for measures that stop varying.** `min_other_teams` sets how many other teams
must share a claim before `NPREC` counts it. Set it too high for the pool and `NPREC`
falls to zero for every run. `NF1` falls with it and every run ties on `NF1_RANK`,
silently. `judge()` warns when a measure is constant or takes few distinct values.
Treat either warning on real data as a reason not to submit.
