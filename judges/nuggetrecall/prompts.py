"""Prompts and parsers for query-derived nugget recall.

Two LLM steps, both batched:

1. Per topic, generate an answer key of atomic facts from the request alone. This
   is "query-only" nugget generation. A bank built from the request is also readable
   under the data policy for every topic, unlike one mined from responses.
2. Per report, label every fact in the key as support, partial or no, in a single
   call. One call per report rather than per (report, fact) pair keeps this at
   about 15,000 calls for both test sets instead of hundreds of thousands.

The coverage prompt and its all-or-nothing parser follow the ones already
validated in our TREC RAG 2026 work (trec-rag-2026/scripts/eval/evaluate_nuggets.py).

Both parsers are deliberately strict about completeness. A reply that labels only
a few of the facts is a truncated or confused generation, and back-filling the rest
as "no" would silently invent misses and understate the system, invisibly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

# Reasoning models (gpt-oss, Qwen3 in thinking mode, DeepSeek-R1 style) can emit a
# visible reasoning block before the answer. The organizers re-run code against
# gpt-oss-20b, so this has to be handled rather than assumed away.
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def strip_reasoning(text: str) -> str:
    text = _THINK.sub("", text or "")
    # An unterminated block means the reply was cut off inside the reasoning.
    # Keep only what follows the last close tag, if any.
    if "<think>" in text.lower():
        tail = text.lower().rsplit("</think>", 1)
        text = text[len(text) - len(tail[-1]):] if len(tail) > 1 else ""
    return text.strip()


@dataclass(frozen=True)
class Nugget:
    text: str
    vital: bool


# --------------------------------------------------------------- generation

GENERATE_PROMPT = """\
You are building the answer key for evaluating responses to an information request.

REQUEST:
{request}

List the key facts that a complete, high-quality response to this request should \
contain. Each item must be:
- one self-contained factual statement, a single idea per line
- specific enough that you could check whether a response states it
- relevant to what the request actually asks for

Mark an item VITAL if a response that omits it is clearly incomplete, or OKAY if it \
is useful supporting detail.

Give between {min_n} and {max_n} items, most important first. Output only the list, \
one item per line, in exactly this format and nothing else:
1. [VITAL] <fact>
2. [OKAY] <fact>
"""

# Tolerates markdown bold around the tag, "4. **[VITAL]** fact", which instruction
# models add unprompted. The first version of this pattern silently dropped such
# lines, shrinking the answer key without any error.
_GEN_LINE = re.compile(
    r"^\s*\d+\s*[.):]\s*\**\s*\[\s*(VITAL|OKAY|OK)\s*\]\s*\**\s*(.+?)\s*$", re.I | re.M
)


def request_text(title: Optional[str], problem_statement: Optional[str],
                 background: Optional[str]) -> str:
    """The request as the model sees it. Field order is fixed so prompts, and
    therefore cache keys, are byte-identical across runs."""
    parts = []
    if title:
        parts.append(title.strip())
    if problem_statement and problem_statement.strip() != (title or "").strip():
        parts.append(problem_statement.strip())
    if background:
        parts.append("Background: " + background.strip())
    return "\n".join(parts)


def generation_messages(request: str, min_n: int = 8, max_n: int = 20) -> List[Dict[str, str]]:
    return [{"role": "user",
             "content": GENERATE_PROMPT.format(request=request, min_n=min_n, max_n=max_n)}]


def parse_nuggets(raw: str, max_n: int = 20) -> List[Nugget]:
    """Parse the answer key. Returns [] if the reply is not a usable list.

    Duplicates are dropped case-insensitively, and the list is capped, so a model
    that ignores the requested range cannot inflate the denominator.
    """
    out: List[Nugget] = []
    seen: set[str] = set()
    for m in _GEN_LINE.finditer(strip_reasoning(raw)):
        tag, fact = m.group(1).upper(), m.group(2).strip().strip("*").strip()
        key = re.sub(r"\s+", " ", fact).casefold()
        if len(fact.split()) < 2 or key in seen:
            continue
        seen.add(key)
        out.append(Nugget(text=fact, vital=(tag == "VITAL")))
        if len(out) >= max_n:
            break
    # A key with no vital item is still usable, but a key of one item is not.
    return out if len(out) >= 2 else []


# ----------------------------------------------------------------- coverage

JUDGE_PROMPT = """\
You are checking which facts a response covers. Judge only what the response says. \
Never use outside knowledge.

REQUEST:
{request}

RESPONSE TO JUDGE:
{answer}

FACTS TO CHECK (one per line, numbered):
{facts}

For every numbered fact, decide whether the RESPONSE conveys it:
  support  - the response clearly states this fact
  partial  - the response states part of it, gestures at it, or hedges it
  no       - the response does not convey it

Judge meaning, not wording: the same fact stated in different words is support. A \
fact the response merely names as a topic, without asserting it, is partial.

Output exactly one line per fact, in order, and nothing else:
1: support
2: no
3: partial

Output {count} lines, numbered 1 to {count}.
"""

_JUDGE_LINE = re.compile(
    r"^\s*(\d+)\s*[:.)]\s*\**\s*(support|partial|no|not[_ ]support|none)\b", re.I | re.M
)


def judge_messages(request: str, answer: str, nuggets: List[Nugget]) -> List[Dict[str, str]]:
    facts = "\n".join(f"{i}. {n.text}" for i, n in enumerate(nuggets, 1))
    return [{"role": "user", "content": JUDGE_PROMPT.format(
        request=request, answer=answer, facts=facts, count=len(nuggets))}]


def parse_judgments(raw: str, count: int, min_coverage: float = 0.8) -> Optional[Dict[int, str]]:
    """Map fact index to support, partial or no. None if the reply is unusable.

    All or nothing on coverage: fewer than `min_coverage` of the facts labelled
    means a truncated or confused reply, and back-filling the gap as "no" would
    invent misses. The caller retries, and only then gives up.
    """
    labels: Dict[int, str] = {}
    for m in _JUDGE_LINE.finditer(strip_reasoning(raw)):
        idx = int(m.group(1))
        lab = m.group(2).lower().replace(" ", "_")
        lab = {"not_support": "no", "none": "no"}.get(lab, lab)
        if 1 <= idx <= count and idx not in labels:
            labels[idx] = lab
    if count == 0 or len(labels) < max(1, int(count * min_coverage)):
        return None
    return labels


def score(labels: Dict[int, str], nuggets: List[Nugget]) -> Dict[str, float]:
    """Nugget recall, four ways. Partial is half credit in the lenient forms.

    Missing labels, possible when coverage is between min_coverage and 100%, count
    as "no". That is conservative but bounded, since at most a fifth can be absent.
    """
    credit = {"support": 1.0, "partial": 0.5, "no": 0.0}
    strict = {"support": 1.0, "partial": 0.0, "no": 0.0}
    vital = [i for i, n in enumerate(nuggets, 1) if n.vital]
    every = list(range(1, len(nuggets) + 1))
    # With no vital item in the key, vital recall falls back to all-item recall
    # rather than being undefined.
    vital = vital or every

    def rec(idx: List[int], table: Dict[str, float]) -> float:
        return sum(table.get(labels.get(i, "no"), 0.0) for i in idx) / len(idx) if idx else 0.0

    w = [1.0 if nuggets[i - 1].vital else 0.5 for i in every]
    weighted = (sum(w[i - 1] * credit.get(labels.get(i, "no"), 0.0) for i in every) / sum(w)) if w else 0.0
    return {
        "NR_VITAL": rec(vital, credit),
        "NR_ALL": rec(every, credit),
        "NR_VITAL_STRICT": rec(vital, strict),
        "NR_WEIGHTED": weighted,
    }
