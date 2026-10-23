# Notebook paper, TREC 2026 AutoJudge

The LaTeX source of our TREC notebook paper. It uses the ACM `acmart` class
(`sigconf`, with the ACM copyright block turned off), which Overleaf has built in.

## Open it in Overleaf

1. Zip this folder. On the main PC a ready zip is at `results/paper-overleaf.zip`.
2. In Overleaf: **New Project, then Upload Project**, and pick the zip.
3. Leave the compiler on pdfLaTeX. XeLaTeX also works.

If the Overleaf account has GitHub sync, linking this repository works too. The
paper lives in `paper/`, so set `paper/main.tex` as the main document.

## What is still open

Everything unfinished is printed in **red brackets**. In the source, search for
`\pending`. At the time of writing:

- author surname, advisor name and email, and the acknowledgments
- the test-set results table (Section 9.1), once the organizers publish scores
- the public URL of the code (Section 11). The GitHub repository is private now
- the organizers' answer on whether TIRA's test-set topics match the released
  files (asked 2026-10-21). If not, add a paragraph to Sections 7.3 and 10

Every bibliography entry was checked online on 2026-10-19 (Crossref, arXiv, DBLP),
and every number against TIRA and `docs/WORK_LOG.md`.

## Where the numbers come from

Every pilot score was read from TIRA's public evaluation API on 2026-10-19, after
the organizers re-scored the pilots on 2026-10-09. The older numbers in
`docs/NOTEBOOK_PAPER.md` predate that re-scoring. Everything else comes from
`docs/WORK_LOG.md`.

## Rules for editing

- Never name another participating team, in text or tables. Field median, field
  best and our rank among entries are fine, stated as informal.
- No claim of beating another entry without a significance test.
- Read response text only through what the data policy allows. The answer key in
  Figure 1 is generated from the topic alone, so it is safe to show.

## Building it locally

No TeX installation is needed. With [tectonic](https://tectonic-typesetting.github.io)
installed:

```bash
cd paper && tectonic -X compile main.tex
```
