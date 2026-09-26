# judges/corroborate

Pooled corroboration with leave-one-team-out weighting. The method is described
in the repository README, and the full reasoning, measurements and decision log
are in `docs/WORK_LOG.md`. This file covers the code layout and how to run it.

## Modules

| File | Responsibility |
|---|---|
| `textutil.py` | NFKC normalisation, the boilerplate and length gates, stable cache keys. No dependencies beyond the standard library, so normalisation cannot drift between machines. |
| `encoders.py` | Model loading, device selection, and the on-disk score cache. |
| `pool.py` | Per-topic clustering, team votes, the LOTO weight, export to `NuggetBank`. |
| `scoring.py` | `NCOV`, `NPREC`, the harmonic mean, and within-topic percentile rank. |
| `attribution.py` | Citation support: inline entailment, or pooled citation agreement. |
| `measures.py` | The `LeaderboardSpec`. One place, with the reasoning for each measure. |
| `judge.py` | The protocol implementation. Both phases, one class. |

Three decisions in the code that look arbitrary and are not:

- **`pick_device` proves CUDA with a real matmul.** On this driver and wheel
  combination `torch.cuda.is_available()` returns True even when CUDA
  initialisation has failed, so only an actual operation settles it.
- **The cross-encoder runs through plain `transformers`, not
  `sentence_transformers.CrossEncoder`.** We need P(entailment) specifically, and
  which activation `CrossEncoder.predict` applies by default has changed between
  versions. Computing it from the logits is version-proof.
- **One class serves both phases.** The runner caches instances by class path, so
  declaring only `judge_class` gives the nugget phase and the judge phase the
  same object, and the pooled clustering is computed once.

## Running it

```bash
export CACHE_DIR=./cache/rag26        # embedding and cross-encoder score cache
auto-judge run \
    --workflow judges/corroborate/workflow.yml --variant best \
    --rag-responses local-data/rag26/runs/generation/ \
    --rag-topics    local-data/rag26/topics/topics.jsonl \
    --out-dir       ./output/rag26/
```

`CACHE_DIR` is worth setting even though the judge makes no LLM calls. It is
where the embedding and cross-encoder scores are cached, which is what makes an
interrupted run cheap to resume and the ablation variants nearly free.

`ragtime26` needs no special variant. RAGTIME draws on NeuCLIR, so we expected
non-English documents, where an English-only NLI model would return noise rather
than an error. Both 2026 releases turned out to be English, so `best` is right for
both test sets.

## Variants

`best` is the submitted configuration. `no-loto`, `run-votes`, `coverage-only`
and `no-cross-encoder` are the ablations that answer whether each component
earns its place. `fast` turns off cross-encoder adjudication in coverage scoring,
caps the bank at 25 clusters and reads at most 2 chunks per cited document, which
makes it faster on a CPU-only worker. It is not bounded, though: citation
attribution calls the cross-encoder unconditionally, and `fast` has not been
validated against truth. `multilingual` swaps in `bge-reranker-v2-m3`, which the image does not
bake in, so it runs only locally or with the model mounted.

## Two things that will bite you

**Never set `create_qrels: true`.** The TIRA evaluator finds your leaderboard by
scanning the output directory for files whose lines split into four
whitespace-separated tokens. A TREC qrels file matches that shape, so the
evaluator can no longer tell which file is the leaderboard and the upload fails
with "No unique file in trec-eval -q format was found". Verified locally. Run
`scripts/check_upload_dir.py` before every upload.

**Watch the degeneracy warnings.** If `min_other_teams` is too high for the pool,
`NPREC` goes to zero for every run and takes `NF1` and `NF1_RANK` down with it,
silently. `judge()` warns when a measure is constant or low-cardinality. Treat
either warning on real data as a reason not to submit that run.
