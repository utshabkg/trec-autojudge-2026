"""Citation support, by two paths chosen at runtime.

Neither test dataset ships a corpus, so how citations can be judged depends
entirely on whether the released run files carry the cited document text inline.

- `inline`: the runs carry `Report.documents`. Each cited sentence is tested for
  entailment from the best-matching chunk of the documents it cites. This is
  real attribution.
- `pool`: they do not. For each cited document, we measure the share of *other*
  teams that cite the same document for the same topic, times how much of the
  report is cited at all. This is a pseudo-qrel built from the pool. It measures
  agreement with other teams' retrieval rather than truth of support, which is
  weaker, but it needs no corpus and is the only option available in that case.

`auto` picks by inspecting how many reports actually have documents. That check
is a structural diagnostic (field presence and count), which the data policy
permits on any topic.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .textutil import normalize, scoreable_sentences

try:
    # The two functions Document.get_text_chunks is built from. Calling them
    # directly gives byte-identical chunks without the framework's
    #   print(f"Breaking documents {self.id} into ...")
    # which writes a document id to stdout for every document split. Document ids
    # are part of the Report, so outside the permitted ten-topic window they are
    # restricted, and a run log full of them is a surface we must never read.
    from autojudge_base.document.text_chunker import (
        get_limit_length_chunks,
        get_sentence_chunks_on_newline,
    )
except ImportError:  # pragma: no cover - older framework layouts
    get_limit_length_chunks = None
    get_sentence_chunks_on_newline = None


def _doc_chunks(doc, limit: int):
    """Chunk a document the way the framework does, silently."""
    if get_limit_length_chunks is not None:
        return get_limit_length_chunks(
            get_sentence_chunks_on_newline(doc.get_text()), limit=limit
        )
    text = normalize(getattr(doc, "text", "") or "")
    return [text[i : i + limit] for i in range(0, len(text), limit)] or [text]

log = logging.getLogger(__name__)

# Track rules cap citations at 3 per sentence, and get_sentences_with_citations
# returns them in priority order, so looking past the first few buys nothing.
_MAX_CITED_DOCS_PER_SENTENCE = 3


def detect_mode(reports: Sequence, requested: str = "auto") -> str:
    """Return 'inline' or 'pool'.

    Structural only: counts how many reports have a non-empty `documents` map.
    """
    if requested in ("inline", "pool"):
        return requested
    if not reports:
        return "pool"
    with_docs = sum(1 for r in reports if getattr(r, "documents", None))
    frac = with_docs / len(reports)
    mode = "inline" if frac >= 0.5 else "pool"
    log.info(
        "attribution: %d/%d reports carry inline documents (%.1f%%) -> mode=%s",
        with_docs, len(reports), 100.0 * frac, mode,
    )
    return mode


def _cited_sentences(report) -> List[Tuple[str, List[str]]]:
    """(sentence_text, cited_doc_ids) using the format-normalising accessor.

    `get_sentences_with_citations` resolves rag26 integer indices into
    `references` and orders ragtime26 confidence dicts, so both tracks arrive
    here as a plain list of doc ids.
    """
    out: List[Tuple[str, List[str]]] = []
    try:
        for s in report.get_sentences_with_citations():
            text = normalize(s.text or "")
            if not text:
                continue
            cites = [c for c in (s.citations or []) if c]
            out.append((text, cites[:_MAX_CITED_DOCS_PER_SENTENCE]))
    except Exception as exc:
        # Type only, never the message: an exception raised while reading a report
        # can quote the offending citation or sentence, and outside the permitted
        # window that is restricted content. The data policy is explicit that a
        # traceback's type and location are ours but a value in its message is not.
        log.warning("run %s: could not read citations (%s)",
                    report.metadata.run_id, type(exc).__name__)
    return out


def score_inline(
    report,
    encoders,
    *,
    doc_chunk_chars: int,
    max_doc_chunks: int,
    min_sentence_words: int,
) -> Tuple[float, int, int]:
    """Entailment of each cited sentence from its cited documents.

    Returns (CITE, judged_sentences, cited_sentences). A sentence whose cited
    documents are all missing from the run file is excluded from the mean rather
    than scored zero: that is a gap in the released data, not a fault of the
    system being judged.
    """
    docs = getattr(report, "documents", None) or {}
    pairs_text: List[Tuple[str, str]] = []
    slot_of: List[int] = []
    per_sentence: List[List[int]] = []
    cited = 0

    for text, cites in _cited_sentences(report):
        if len(text.split()) < min_sentence_words or not cites:
            per_sentence.append([])
            continue
        cited += 1
        slots: List[int] = []
        for doc_id in cites:
            doc = docs.get(doc_id)
            if doc is None:
                continue
            try:
                chunks = _doc_chunks(doc, doc_chunk_chars)[:max_doc_chunks]
            except Exception:
                chunks = [normalize(getattr(doc, "text", "") or "")[:doc_chunk_chars]]
            chunks = [c for c in chunks if c and c.strip()]
            if not chunks:
                continue
            # Pick the chunk most similar to the sentence, then ask the
            # cross-encoder only about that one.
            cvecs = encoders.embed(chunks)
            svec = encoders.embed([text])[0]
            best = chunks[int(np.argmax(cvecs @ svec))]
            slots.append(len(pairs_text))
            pairs_text.append((best, text))
        per_sentence.append(slots)

    if not pairs_text:
        return 0.0, 0, cited

    scores = encoders.cross(pairs_text)
    sent_scores = [max(float(scores[i]) for i in slots) for slots in per_sentence if slots]
    if not sent_scores:
        return 0.0, 0, cited
    return float(np.mean(sent_scores)), len(sent_scores), cited


def build_citation_index(reports: Sequence, *, vote_unit: str = "team") -> Dict[str, set]:
    """doc_id -> set of teams citing it, for one topic."""
    index: Dict[str, set] = {}
    for rep in reports:
        run_id = str(rep.metadata.run_id)
        team_id = str(rep.metadata.team_id or run_id)
        if vote_unit == "run":
            team_id = run_id
        for _text, cites in _cited_sentences(rep):
            for doc_id in cites:
                index.setdefault(doc_id, set()).add(team_id)
    return index


def score_pooled(
    report,
    citation_index: Dict[str, set],
    n_teams: int,
    *,
    min_sentence_words: int,
    vote_unit: str = "team",
) -> float:
    """Pooled citation agreement times citation density.

    Agreement is leave-one-team-out for the same reason coverage is: a team must
    not corroborate its own retrieval. Density multiplies it because a report
    that cites almost nothing has not supported its claims, however good the few
    citations it does give are.
    """
    run_id = str(report.metadata.run_id)
    team_id = str(report.metadata.team_id or run_id)
    if vote_unit == "run":
        team_id = run_id

    sents = _cited_sentences(report)
    scoreable = [(t, c) for t, c in sents if len(t.split()) >= min_sentence_words]
    if not scoreable:
        return 0.0

    denom = max(n_teams - 1, 1)
    shares: List[float] = []
    n_cited = 0
    for _text, cites in scoreable:
        if not cites:
            continue
        n_cited += 1
        for doc_id in cites:
            others = citation_index.get(doc_id, set()) - {team_id}
            shares.append(len(others) / denom)

    if not shares:
        return 0.0
    agreement = float(np.mean(shares))
    density = n_cited / len(scoreable)
    return agreement * density
