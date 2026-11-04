# Two AutoJudges from team `iris-mst`

[TREC](https://trec.nist.gov) 2026 AutoJudge asks for judges that rank the systems of
another TREC track the way human assessors would. We submit two judges built on opposite
assumptions, because they fail in different places.

**NuggetRecall is the primary submission.** A large language model (LLM) writes an answer
key of facts, or nuggets, from the topic alone. It then checks which of those facts each
response covers.

**Corroborate is the secondary submission.** It makes no LLM calls. It pools all responses
to a topic and treats a claim many independent teams make as probably important. A team's
own vote never counts toward the claims it is scored on.

The three pilot datasets (rag25-gen, ragtime25 and dragun-repgen) come from TREC 2025. Their
human assessments stay on TIRA, the task's evaluation platform, so we upload a leaderboard
of scores and read back the correlation. Scores on the test sets (rag26 and ragtime26) are
not out yet. The rag sets hold short answers and the ragtime sets hold long reports.

## Where things are

| Path | What it holds |
|---|---|
| `judges/nuggetrecall/`, `judges/corroborate/` | the two judges |
| `scripts/` | serving the LLM, upload checks, rank aggregation and fusion |
| `tests/` | the test suite, which GitHub Actions and tira-cli, TIRA's upload tool, run before every code submission |
| `data/kiddie/` | the small synthetic dataset used for quick end-to-end checks |
| `judges/naive/`, `run_all_datasets.py`, `run_kiddie.sh`, `datasets.yml`, `fetch_datasets.py`, `Dockerfile` | from the task's starter kit, to run judges and build the image |
| `results/` | judge outputs, logs and caches. Not in git |
| `local-data/` | a link to the TREC data. Not in git |

## How we read the scoring, because it shaped both judges

TIRA correlates the systems' aggregate scores on every measure we emit with each human truth
measure. For each truth measure it reports the best of our measures, as Kendall's tau and
`Tauap B` (`tauap_b`). The second weighs agreement at the top more heavily. Until 2026-10-09
TIRA reported one `Max` and one `Min` over all truth measures. Since then rag25-gen and
ragtime25 are scored per truth measure, while dragun-repgen still gets a single maximum. The
organizers have not detailed the revised measures, so we go by their names. We made our
design choices under the earlier scoring. Three of its features shaped both judges.

Only the best measure counts, so several well-motivated measures cost nothing. Each judge
emits four or five. The same rule flatters a judge with many measures, since the best one is
picked after the fact.

The earlier scoring also reported `Min`, so we oriented every measure to make higher better.

In the evaluation package's fixtures, swapping the top two systems costs 0.222 on `tauap_b`
while swapping the bottom two costs 0.025. The evaluator also breaks ties by input order, so
a coarse score loses agreement at random. Every measure we emit is therefore continuous.

Answer length is a weak shortcut under the current truth. The organizers' length baseline
scores 0.152 Kendall on rag25-gen and 0.193 on ragtime25 (0.558 under the old ragtime25
scoring).

## NuggetRecall, the primary submission

`judges/nuggetrecall/`, `uses_llm: true`.

### The answer key comes from the topic, not from the responses

For each topic the judge asks an LLM for 8 to 20 facts a complete answer should contain,
marking each VITAL or OKAY. The responses are not shown, so a verbose system cannot shape
the key. The key can also hold a fact every system missed, which a key mined from the
responses cannot.

The workflow declares `nugget_depends_on_responses: false`, so the framework passes no
responses into this step. If every attempt fails, the topic falls back to a single fact made
from its title. That never happened with Qwen3-30B, our main judge model. With gpt-oss-20b
it hit 3 of 103 ragtime26 topics.

### Judging is one call per response

Each response is judged against the whole key in one call, which labels every fact
`support`, `partial` or `no`. Calls grow with topics plus responses, not responses times
facts, which keeps the judge affordable. rag26 takes 9,838 judging calls.

A reply must label at least 80% of the facts, rounded down, or it is rejected. Filling a
shorter reply with `no` would invent misses. In an accepted reply the few unlabeled facts do
count as `no`. A rejected reply is retried up to twice with a fixed suffix, which changes
the cache key (the lookup key for a stored reply). A response still unreadable after three
attempts scores zero and is counted in the log. That count was zero for Qwen3-30B.
`nuggetrecall-gptoss` left 163 of 5,041 judgments unreadable on ragtime26.

A reasoning model such as gpt-oss-20b writes out its steps before answering. When it uses up
its reply budget, the most tokens it may generate, it never writes the labels. Each retry
therefore doubles the budget, from 4,096 tokens to 8,192 and then 16,384. On rag26 with
gpt-oss-20b this cut the final unreadable share from 12.4% to 2.7%.

### Measures

| Measure | Definition |
|---|---|
| `NR_VITAL` | recall of VITAL facts, support 1 and partial 0.5 |
| `NR_ALL` | recall of all facts, same credit |
| `NR_VITAL_STRICT` | recall of VITAL facts counting full support only |
| `NR_WEIGHTED` | importance-weighted recall, vital 1 and okay 0.5 |
| `NR_RANK` | `NR_VITAL` re-expressed as a within-topic rank, then averaged |

`NR_RANK` uses the same judgments, aggregated differently. Averaging recall over topics lets
an easy topic, where every system scores high, count as much as one that separates systems.
Ranking within a topic first removes that effect. On rag25-gen it raises Kendall from 0.783
to 0.792 on Strict Vital Score and from 0.788 to 0.809 on Sub Coverage. The judgments are
identical, so run-to-run noise cannot explain this. On ragtime25 it does worse (0.681
against 0.697 on Nugget Coverage). That costs nothing, because the reported score is a
maximum. `nuggetrecall-rank` adds this measure to the judgments of `nuggetrecall-best`.

## Corroborate, the secondary submission

`judges/corroborate/`, `uses_llm: false`.

This judge builds its reference from the one resource every dataset has, the responses
themselves. It needs no document collection and no network access.

**Pool clustering is also the matching.** Apart from boilerplate and sentences under five
words, every sentence for a topic is embedded with `BAAI/bge-base-en-v1.5`, which maps
similar meanings to nearby vectors. Each sentence joins the closest cluster or starts a new
one, in fixed `(run_id, sentence_index)` order, so the result does not depend on file order.
A response covers a cluster when one of its sentences landed there, so no separate retrieval
step is needed. For coverage, only borderline pairs reach a cross-encoder, a model that
reads both texts together.

**Votes are per team, not per run.** rag26 is 83 runs from 25 teams. Counting runs would let
a team with five close variants vote five times for its own phrasing.

**Leave-one-team-out drops the scored team's own vote.** The claim bank is the reference
each response is scored against. A cluster needs at least two teams to enter it, so a claim
made by one team alone is never scored. It must also pass a specificity floor, a minimum
cosine similarity of 0.50 with the topic text. Its weight is then the number of other teams
asserting it, recomputed for each scored team. This is a team-level version of the
leave-one-run-out test from retrieval evaluation. That test measures pool bias. We use the
same idea to correct for it.

| Measure | Definition |
|---|---|
| `NCOV` | leave-one-team-out, team-weighted coverage of the topic's clusters |
| `NPREC` | fraction of the response's own sentences corroborated by other teams |
| `NF1` | harmonic mean of the two |
| `NF1_RANK` | within-topic percentile rank of `NF1` |
| `CITE` | citation support, from inline documents where present, else pooled agreement |

We also built a `multilingual` variant for RAGTIME's cross-language NeuCLIR collection, but
both 2026 releases turned out to be English. So the submitted `best` configuration uses the
English cross-encoder `nli-deberta-v3-base`. `judges/corroborate/README.md` covers the
variant.

### Why a pool-derived bank is not circular

The human reference also starts from a pool. In TREC RAG, retrieved documents are pooled,
nugget banks are drafted from them and assessors vet the banks. That pool holds documents
rather than answers, so the analogy is loose. Still, both references come from what systems
produced.

A fact no system states would give every system the same zero, so leaving it out of the bank
cannot change their order.

The real risk is that consensus pulls everyone toward the median. That would hurt most at
the top of the leaderboard, where `tauap_b` is most sensitive. We have not tested this
directly. We also ran no ablation (switching one part off to see what it adds) of the
leave-one-team-out weight or of per-team voting. Their effect on the ranking is unknown.

## Results

Kendall against held-out human assessments, from TIRA, read 2026-10-19 after the organizers
re-scored the pilots. Field figures cover every entry on that pilot's leaderboard (385, 206
and 139 entries, many of them variants from one team). The full tables will be in our TREC
notebook paper. These comparisons with the field are informal. None claims to beat another
entry. Bold marks our best stable score in each row.

| Pilot, truth measure | Corroborate | NuggetRecall (Qwen3-30B) | best NuggetRecall | field median | field best |
|---|---|---|---|---|---|
| `rag25-gen`, Strict Vital Score | 0.257 | 0.783 | **0.839** (gpt-oss-120b, organizers' run) | 0.804 | 0.906 |
| `rag25-gen`, Sub Coverage | 0.323 | 0.788 | **0.809** (`NR_RANK` only) | 0.772 | 0.866 |
| `ragtime25`, Nugget Coverage | 0.585 | 0.697 | **0.728** (gpt-oss-20b) | 0.726 | 0.825 |
| `ragtime25`, F1 | **0.653** | 0.500 | 0.517 (gpt-oss-20b) | 0.583 | 0.808 |
| `dragun-repgen` | **0.739** | 0.04 to 0.14 across runs | 0.04 to 0.80 across all runs, see below | 0.668 | 0.821 |

Corroborate scores use our submitted specificity floor of 0.50. An earlier floor of 0.15
scored 0.598 and 0.662 on ragtime25 and 0.755 on dragun-repgen.

**The two judges do best on different truth measures.** NuggetRecall is our stronger judge
on nugget coverage, the share of a topic's key facts a response covers. Corroborate does
better on F1 for long reports, perhaps because `NPREC` punishes padding. It fails on short
answers. One reading is that the best systems there state precise facts others missed. The
two rankings agree only weakly (Kendall 0.51 on ragtime25 and 0.25 on rag25-gen between
`NCOV` and `NR_VITAL`).

**NuggetRecall's DRAGUN scores cannot rank models.** Each dragun-repgen topic is a news
article. The released file holds a converted request with a question, a summary of the
article's claims and a fact-checker background. TIRA gives judges only the article's
headline. Even with the same model and input, two runs scored 0.190 and 0.587. Each run
writes new answer keys and the pilot has only 28 systems, so scores swing widely.

### Judge model

Same prompts, rag25-gen Strict Vital Score. The runs also differ in code version, reply
budget and server, so this is not a fully controlled comparison.

| Judge model | Kendall |
|---|---|
| Qwen2.5-7B-Instruct | 0.604 |
| Qwen3-30B-A3B-Instruct-2507 (FP8) | 0.783 |
| openai/gpt-oss-20b | 0.783 |
| openai/gpt-oss-120b (organizers' run of our code) | 0.839 |

FP8 is a low-precision weight format. The 7B model follows the output format just as
reliably, with no unreadable replies over 5,302 calls. It seems to grant partial credit too
freely.

### Two cautions about reading any of these numbers

**Our run-to-run noise is about 0.01 to 0.02 Kendall, far larger on DRAGUN.** A ragtime25
re-run of Qwen3-30B added one measure, which cannot lower a maximum, plus a larger reply
budget no Qwen3-30B reply had reached. Kendall still fell by 0.010 to 0.019. vLLM, the
engine serving our models, gives slightly different output when the mix of requests in a
batch changes. We therefore treat differences below 0.02 as noise on rag25-gen and
ragtime25.

**A broken judge and a weak judge look identical from the score alone.** NuggetRecall on
gpt-oss-20b once scored exactly 0.000 on rag25-gen. The LLM client timed calls out after 60
seconds, too short for a reasoning model. It abandoned the whole batch after 25 failures, so
26 slow calls destroyed all 5,302. Every measure came out constant and no error was raised.
Our pre-upload check now refuses a leaderboard whose measures do not vary between systems.
After fixing the timeout and the batch abort, the same configuration scores 0.783.

## Reproducibility

Code submissions are containers the organizers rerun on TIRA with models they choose.
`lead-racer` is Corroborate. `grave-node` and `rank-dioptre` are NuggetRecall. Only
`grave-node` carries the growing reply budget, so we asked the organizers to treat it as our
main code submission.

Data submissions are our own leaderboards. `nuggetrecall-best` and `nuggetrecall-rank` use
Qwen3-30B on both test sets. `nuggetrecall-gptoss` uses gpt-oss-20b on ragtime26 only, with
4,096 tokens on every attempt. `ensemble-qwen-gptoss` covers rag26 only and averages
within-topic ranks of `NR_VITAL` from both models. `corroborate-best` shares its
configuration with `lead-racer`. `nuggetrecall-best` used older reply caps of 1,536 and
1,024 tokens, which no Qwen3-30B reply reached.

NuggetRecall routes every call through the organizers' `minima-llm` client. Its SQLite
prompt cache, a stored record of each prompt and reply, is keyed on the model, the messages,
the temperature and the reply budget. The LLM endpoint, the server that answers those calls,
is not part of the key. So TIRA can rerun the container with the endpoint switched off and
replay the leaderboard from the cache. Sorting responses by run id keeps prompts byte
identical between executions.

A reproducibility check compares two runs, so two runs that both fail identically will pass
it. Our first container trial passed the cache check while making zero LLM calls, because
the endpoint was a closed port. The trial now runs against a stub endpoint, a stand-in that
replies like a real model. It fails if the leaderboard does not separate systems.

Corroborate's models are baked into the image. Given the same inputs and hardware it
produces byte-identical output.

```bash
pip install -e '.[all]'
bash run_kiddie.sh                    # smoke test, no LLM or GPU needed

auto-judge run \
    --workflow judges/nuggetrecall/workflow.yml \
    --variant best \
    --rag-responses local-data/rag26/runs/generation/ \
    --rag-topics   local-data/rag26/topics/topics.jsonl \
    --out-dir      ./results/output/rag26/
```

Both judges run with `create_qrels: false` on purpose. A TREC qrels file, the standard file
of relevance labels, also parses as four columns. Writing one beside the leaderboard makes
the upload fail its format check.

## Data handling and disclosure

Development followed the AutoJudge data policy. On the test sets we read response content
only for the first ten topics of each topics file, in file order. Every choice we made by
score came from the pilots.

All bugs but one reproduced on a pilot or on the synthetic `kiddie` set. That is why we
never needed the `scrub` tool, which masks a failing record's content. The exception is the
exhausted reply budget. It also occurs on the pilots at lower rates, but we noticed it on
rag26 and saw it there only as counts. These were counts of unreadable judgments in our logs
(27% on the first pass with gpt-oss-20b) and of requests vLLM stopped at the length limit.
We read no response text to diagnose it. The growing budget and `grave-node` followed from
those counts.

We also submitted two generation runs to TREC 2026 RAG, so two of the 83 `rag26` runs are
probably ours. We did not try to identify them, by comparison with our own submissions or by
any other means. Both judges treat every run alike. NuggetRecall judges each response on its
own, against a key built from the topic. Corroborate drops a team's own vote from each
claim's weight, so a team's runs cannot raise the weight of the claims they are scored on.

## Limitations

- NuggetRecall relies on what the LLM already knows to write the answer key. It never sees
  citations, so it cannot judge attribution.
- Corroborate cannot reward a correct fact that no other team stated. Agreement among teams
  is a proxy for importance, not a measure of it, so a correct but rare fact gets little
  weight.
- When fewer than half the responses carry document text, unlike the test sets, `CITE`
  measures agreement with other teams rather than real support.
- We tuned on the 2025 pilots, as the rules require, so the 2026 results may differ. The
  pilots were re-scored after we chose.
- TIRA's DRAGUN topics differ from the release. So we do not know whether its test-set
  copies match the released files, which our data submissions used.
