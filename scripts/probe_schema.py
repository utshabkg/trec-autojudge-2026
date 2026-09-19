#!/usr/bin/env python3
"""Task zero: does a release carry its cited documents inline?

This decides which attribution path the judge can use, and it has to be settled
before any attribution work is worth doing. Neither test dataset ships a corpus.
If `Report.documents` is populated the judge can test entailment of a cited
sentence against the document it cites. If it is not, the only option is pooled
citation agreement, because ClimbMix-400b and the RAGTIME collection cannot be
obtained and indexed in the time available.

Everything printed is an AGGREGATE over the dataset: counts, field presence,
sizes, and the scripts the text is written in. No report text is shown, and no
per-run or per-topic breakdown. The data policy permits structural diagnostics
(schema path, data type, array dimensions, presence or absence of fields) on any
topic, which is exactly what this is. Use `scrub` if you need to look at the
shape of an individual restricted record.

    python scripts/probe_schema.py local-data/rag26/runs/generation
    python scripts/probe_schema.py local-data/ragtime26/runs/repgen
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import median


def script_of(text: str, sample: int = 400) -> str:
    """Dominant Unicode script of a string, to catch non-English documents.

    ragtime26 documents come from NeuCLIR, so Chinese, Persian and Russian. An
    English-only NLI model returns noise on those rather than an error, which is
    the failure mode worth catching early.
    """
    counts: Counter = Counter()
    for ch in text[:sample]:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch).split()[0]
        except ValueError:
            continue
        counts[name] += 1
    return counts.most_common(1)[0][0].lower() if counts else "none"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_dir", type=Path)
    ap.add_argument("--max-files", type=int, default=None, help="probe only the first N run files")
    args = ap.parse_args()

    files = sorted(
        p for p in list(args.runs_dir.glob("*")) + list(args.runs_dir.glob("*/*"))
        if p.is_file()
    )
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        print(f"FAIL: no run files under {args.runs_dir}")
        return 2

    records = 0
    with_docs = 0
    doc_counts: list[int] = []
    doc_chars: list[int] = []
    with_refs = 0
    ref_counts: list[int] = []
    with_ranking = 0
    sentences_key = Counter()
    citation_shape = Counter()
    doc_scripts = Counter()
    doc_meta_keys = Counter()
    meta_keys = Counter()
    topics = set()
    runs = set()
    teams = set()

    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    records += 1
                    md = d.get("metadata") or {}
                    meta_keys.update(md.keys())
                    runs.add(str(md.get("run_id") or f.stem))
                    teams.add(str(md.get("team_id") or "?"))
                    topics.add(str(md.get("topic_id") or md.get("narrative_id")
                                   or md.get("request_id") or "?"))

                    for key in ("answer", "responses"):
                        if d.get(key):
                            sentences_key[key] += 1
                    sents = d.get("answer") or d.get("responses") or []
                    for s in sents[:5]:
                        c = s.get("citations")
                        if c is None:
                            citation_shape["null"] += 1
                        elif isinstance(c, dict):
                            citation_shape["dict[doc_id->conf] (ragtime)"] += 1
                        elif isinstance(c, list):
                            if not c:
                                citation_shape["empty list"] += 1
                            elif all(isinstance(x, int) for x in c):
                                citation_shape["list[int] index (rag24/rag26)"] += 1
                            else:
                                citation_shape["list[str] doc_id (neuclir)"] += 1

                    refs = d.get("references")
                    if refs:
                        with_refs += 1
                        ref_counts.append(len(refs))
                    if d.get("ranking"):
                        with_ranking += 1

                    docs = d.get("documents")
                    if docs:
                        with_docs += 1
                        doc_counts.append(len(docs))
                        for doc in list(docs.values())[:3]:
                            if not isinstance(doc, dict):
                                continue
                            doc_meta_keys.update(doc.keys())
                            text = doc.get("text") or ""
                            doc_chars.append(len(text))
                            doc_scripts[script_of(text)] += 1
        except Exception as exc:
            print(f"  warn: {f.name}: {exc}", file=sys.stderr)

    def stats(xs: list[int]) -> str:
        return f"min={min(xs)} median={int(median(xs))} max={max(xs)}" if xs else "n/a"

    print(f"\n=== {args.runs_dir} ===")
    print(f"run files            : {len(files)}")
    print(f"records              : {records}")
    print(f"distinct run_ids     : {len(runs)}")
    print(f"distinct team_ids    : {len(teams)}")
    print(f"distinct topic_ids   : {len(topics)}")
    if runs and teams:
        print(f"runs per team        : {len(runs) / max(len(teams), 1):.2f}")
    print(f"sentences key        : {dict(sentences_key)}")
    print(f"citation shapes      : {dict(citation_shape)}")
    print(f"metadata keys        : {sorted(meta_keys)}")
    print(f"has references       : {with_refs}/{records}   counts: {stats(ref_counts)}")
    print(f"has ranking          : {with_ranking}/{records}")
    print(f"\nhas documents INLINE : {with_docs}/{records}"
          f"  ({100.0 * with_docs / max(records, 1):.1f}%)")
    print(f"  documents per report: {stats(doc_counts)}")
    print(f"  document text chars : {stats(doc_chars)}")
    print(f"  document keys       : {sorted(doc_meta_keys)}")
    print(f"  dominant scripts    : {dict(doc_scripts.most_common(6))}")

    frac = with_docs / max(records, 1)
    print("\nVERDICT")
    if frac >= 0.5:
        print("  documents are inline -> attribution path A (entailment). No corpus needed.")
        non_latin = sum(v for k, v in doc_scripts.items() if k not in ("latin", "none"))
        if non_latin and non_latin / max(sum(doc_scripts.values()), 1) > 0.1:
            print("  WARNING: a significant share of documents are not Latin script.")
            print("           Use the `multilingual` variant (bge-reranker-v2-m3). An")
            print("           English-only NLI model returns noise here, not an error.")
        if "translation" in " ".join(doc_meta_keys).lower():
            print("  NOTE: a translation-looking field is present on documents, prefer it.")
    else:
        print("  documents are NOT inline -> attribution path B (pooled citation")
        print("  agreement). Do not attempt to obtain the corpus, there is no time.")
        print("  Record this as a stated limitation in the README.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
