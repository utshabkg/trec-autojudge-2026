"""Text normalisation and gating shared by the pool builder and the scorers.

Everything here is deterministic and depends on nothing but the standard
library. Prompt- and cache-key stability depends on that: the same sentence must
normalise to the same string on every machine and every run.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable, List, Sequence, Tuple

_WS = re.compile(r"\s+")

# Openers and closers that carry no topical content. A report full of these
# would otherwise look "corroborated by everybody", which is exactly the
# boilerplate NPREC must not reward. The embedding-based specificity gate in
# pool.py is the main defence; this is a cheap pre-filter for the obvious cases.
_BOILERPLATE = re.compile(
    r"^(in conclusion|to conclude|to summari[sz]e|in summary|overall"
    r"|this report|this answer|the following|here is|here are"
    r"|based on the (provided |above )?(documents?|sources?|information)"
    r"|according to the (provided |above )?(documents?|sources?)"
    r"|i (am |a)?(sorry|cannot|can't|do not|don't)"
    r"|as an ai|i hope this|let me know|no (relevant )?(answer|information))",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    """NFKC-fold, collapse whitespace, strip.

    NFKC matters for ragtime26, whose length limit is counted in NFKC
    characters and whose documents are not all Latin script.
    """
    if not text:
        return ""
    return _WS.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def word_count(text: str) -> int:
    return len(text.split())


def is_scoreable(text: str, min_words: int) -> bool:
    """Is this sentence substantive enough to cluster or to score?

    Rejects the empty string, fragments below `min_words`, and obvious
    boilerplate openers. Rejecting a sentence removes it from both the numerator
    and the denominator of NPREC, so this gate cannot be gamed by padding with
    filler: filler is simply not counted.
    """
    t = normalize(text)
    if word_count(t) < min_words:
        return False
    if _BOILERPLATE.match(t):
        return False
    return True


def dedupe_preserving_order(texts: Iterable[str]) -> List[str]:
    """Drop repeated sentences, keeping first occurrence.

    A run that repeats the same sentence ten times must not get ten votes for
    it, nor ten sentences in its own NPREC denominator.
    """
    seen: set[str] = set()
    out: List[str] = []
    for t in texts:
        k = normalize(t).casefold()
        if k and k not in seen:
            seen.add(k)
            out.append(t)
    return out


def scoreable_sentences(sentences: Sequence[str], min_words: int) -> List[Tuple[int, str]]:
    """Return (original_index, normalised_text) for sentences worth scoring."""
    out: List[Tuple[int, str]] = []
    seen: set[str] = set()
    for i, raw in enumerate(sentences):
        t = normalize(raw)
        if not is_scoreable(t, min_words):
            continue
        k = t.casefold()
        if k in seen:
            continue
        seen.add(k)
        out.append((i, t))
    return out


def topic_text(title: str | None, background: str | None, problem_statement: str | None) -> str:
    """The topic side of the specificity gate.

    Only `title` is guaranteed present on a Request, so the others are optional.
    Order is fixed so the embedding cache key is stable.
    """
    parts = [p for p in (title, problem_statement, background) if p]
    return normalize(" ".join(parts))


def cache_key(*parts: str) -> str:
    """Stable content hash. Used for the embedding and cross-encoder caches.

    Joined with a NUL so that ("ab", "c") and ("a", "bc") cannot collide.
    """
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()
