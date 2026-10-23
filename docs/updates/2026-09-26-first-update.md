# TREC 2026 AutoJudge: first update

> **Note, 2026-10-23.** This is a dated record and is kept as sent. Its numbers use the
> scoring before the organizers re-scored the pilots on 2026-10-09, and its DRAGUN
> explanation turned out wrong. Current numbers: `docs/NOTEBOOK_PAPER.md`.

*team iris-mst, 2026-09-26*

Our first update on this project, so I am start from the scratch.

**In short:** I built two automatic judges and submitted everything by Sept 25. The deadline is Oct 14. On the matching practice sets, my main judge is at or above the median entry.

## The two test sets
- **rag26** (RAG track): short answers that cite documents. 83 systems from 25 teams, 119 questions.
- **ragtime26** (RAGTIME track): long reports that cite documents. 49 systems from 10 teams, 103 questions.
- TREC calls a question a *topic* and a system a *run*.

## How do I test a judge?
- The test sets have no human judgments yet. So I cannot check my judge on them.
- I used three 2025 practice sets (*pilots*) that do have them:
  - **rag25-gen**: short answers, like rag26.
  - **ragtime25**: long reports, like ragtime26.
  - **dragun**: reports about a news article. It matches neither, but it shows where a judge breaks.
- Their human judgments stay hidden on TIRA, the organizers' upload and scoring site.
- I upload my judge's ranking of a pilot. TIRA sends back a score.
- The score is Kendall's tau against the human ranking. 1.0 means the same order. 0 means no relation.
- My judge gives each system a few scores. TIRA keeps the best match.
- TIRA also reports tau AP-b. It is like Kendall's tau, but mistakes at the top cost more.

## The plan and the judges
- I started on Sept 19. Everything here took one week.
- My first judge, Corroborate, uses no LLM. The organizers also run each team's code on their machines. With no LLM, less can break there.
- The pilots showed it was weak on short answers. So I added NuggetRecall, which uses an LLM. It became my main judge.

### NuggetRecall (main, uses an LLM)
- The LLM sees only the question. It writes 8 to 20 key facts a good answer should contain. It marks each one *vital* or *okay*.
- This list is made before the LLM sees any answer. So no system can shape what counts as important.
- Then the LLM reads each answer. For each fact, it says if the answer states it: yes, partly, or no.
- An answer's score is the share of facts it states. A system's score is its average over all questions.
- One extra score ranks systems within each question first, then averages the ranks. So easy questions do not dominate.
- Checking key facts like this is a known idea. It ignores citations.
- Two open LLMs, on one lab GPU:
  - **Qwen3-30B** answers directly. My main submissions use it.
  - **gpt-oss-20b** (from OpenAI) is a reasoning model. It thinks step by step before it answers.

### Corroborate (no LLM)
- It borrows an IR idea. If many systems return a document, treat it as relevant. Corroborate does this with sentences.
- Sentence embeddings group sentences that say the same thing.
- A group's weight is how many different teams wrote it. Teams, not systems, since teams send near-copies.
- A system scores well if it covers the heavy groups.
- When scoring a system, it ignores that system's own team. So no team gets credit for something only it said. This trick comes from IR pool bias work. Using it on answers is the new part.
- One of its scores checks if the cited documents back each sentence.

## Does it work?
Only pilot scores exist. Test-set scores come after TREC.
- The rules forbid tuning on the test sets. So I made every choice, like which LLM to use, on the pilots.
- The LLM does not give identical output every time. So a re-run moves a score by about 0.01. I treat gaps under 0.02 as noise.
- TIRA has a public board of every team's judges per pilot. Ranks below are from today, Sept 26, and still move.

**Short answers (rag25-gen)**: 115 entries. Median 0.791, best 0.862. Ranking by answer length alone: 0.152.
- NuggetRecall, Qwen3-30B: 0.801, 53rd.
- Two-model mix (Qwen3-30B and gpt-oss-20b, see below): 0.812, 34th.
- Corroborate: 0.323, 108th. My best guess: good short answers stand out by saying what others missed. Agreement cannot reward that.

**Long reports (ragtime25)**: 107 entries. Median 0.726, best 0.802. Length alone: 0.558.
- NuggetRecall, gpt-oss-20b: 0.789, 10th.
- NuggetRecall, Qwen3-30B: 0.739, 39th.
- Corroborate: 0.691, 70th.

**Reports on a news article (dragun)**: 75 entries. Median 0.63, best 0.821.
- Corroborate: 0.755, 8th.
- NuggetRecall, Qwen3-30B: 0.141, 71st. My guess: the "question" is the whole article. So the facts come from the article, not from what a good report should say.
- But the organizers ran my NuggetRecall code with their LLMs. It got 0.804 with gpt-oss-120b (4th), and about 0.59 with gpt-oss-20b and Qwen2.5-7B. So my guess may be wrong.
- gpt-oss-120b is a much larger model. With it, my code also got 0.839 on short answers, 6th. So a bigger LLM may help.

### Why keep two judges?
- Their rankings agree only weakly. Kendall's tau between them: 0.23 on short answers, 0.50 on long reports.
- NuggetRecall wins on both pilots that match the test sets. So it is the main one. Corroborate measures something different and needs no LLM.

## What did I submit?
TIRA takes two kinds of submission.
- **Data**: rankings of the test-set systems, computed on our GPU. Four per test set.
  - Both sets: NuggetRecall with Qwen3-30B, the same judgments plus the rank score, and Corroborate.
  - rag26 also gets the two-model mix. ragtime26 also gets NuggetRecall with gpt-oss-20b. Each was my own best on the matching pilot.
- **Code**: the judge programs. The organizers run them with their own LLMs. TIRA gives each a random name.
  - rank-dioptre: NuggetRecall.
  - lead-racer: Corroborate.
  - grave-node: NuggetRecall with a retry fix (see below). I asked them to run it too and treat it as our main NuggetRecall.
- The organizers have started running them. Corroborate should take about 3 days. It runs on CPU there, where its sentence models are slow.
- One disclosure: our team also sent two runs to TREC RAG 2026. So two rag26 systems are probably ours. They are anonymized. The rules forbid trying to find them, and I did not.

## What did I learn?
Three findings for our TREC notebook paper.

### 1. LLM judges fail quietly.
- My first gpt-oss-20b pass on short answers scored exactly 0, last of all entries. The model was fine.
- The organizers' LLM library has two default settings. It gives up on a call after 60 seconds. After 25 such failures, it throws away the whole batch.
- So 26 slow calls wiped out all 5,302 judgments. Every system tied, and an all-tie ranking scores 0.
- A check I run before every upload refuses all-tie rankings. That is what flagged it.
- The fix: a 15-minute limit and no batch cutoff. The same setup then scored 0.783.
- The organizers run our code with gpt-oss-20b. Unfixed, it could have scored 0 there too.

### 1b. A second quiet failure, on rag26.
- gpt-oss-20b often spent its whole reply limit (4,096 tokens) on thinking. It never wrote the verdict.
- Now the first try is unchanged, but retries get 8,192 and then 16,384 tokens.
- Replies with no usable verdict fell from 12.4% to 2.7%. The rest still count as zero. This fix is grave-node.

### 2. A smaller reasoning model judged long reports better.
- Same judge, same prompts. On long reports gpt-oss-20b got 0.789, Qwen3-30B 0.739.
- The gap is 0.05, five times the noise. On short answers they tie (0.783 vs 0.788).
- It is one pilot, so I treat it as a hint.

### 3. On short answers, mixing two LLMs helps most at the top.
- Per question, I rank the systems with each model, then average the two ranks. Equal weight, no tuning.
- Against Qwen3-30B alone, Kendall's tau barely moved (0.801 to 0.812). tau AP-b rose from 0.628 to 0.695, far above its re-run noise.
- So the two models seem to fix each other's mistakes about the best systems.
- On long reports the mix did worse than gpt-oss-20b alone (0.769 vs 0.789). So I used it only on rag26.
- Picking per track from two pilots risks some overfitting.

## What is next?
- Official test-set scores come after the TREC conference.
- I am drafting the notebook paper now. The draft is [docs/NOTEBOOK_PAPER.md](../NOTEBOOK_PAPER.md) in this repo, and the full engineering record is [docs/WORK_LOG.md](../WORK_LOG.md). I can send it as a PDF.
- Your feedback on the draft would help a lot.
