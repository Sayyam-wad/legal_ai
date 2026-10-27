"""Text utilities tuned for Indian Supreme Court judgments.

- sentence splitting that doesn't break on "No.", "S.", "v.", "Cr.P.C." etc.
- token-safe, sentence-aligned chunking
- case header parsing (title, court, date) from AILA case files
- outcome (disposition) extraction
- statute / article citation extraction
"""

from __future__ import annotations

import re
from collections import Counter

# ── Sentence splitting ────────────────────────────────────────────────────────
_ABBREVIATIONS = {
    "no", "nos", "s", "ss", "sec", "secs", "art", "arts", "cl", "v", "vs",
    "mr", "mrs", "ms", "dr", "j", "jj", "cj", "ltd", "co", "pvt", "corpn",
    "govt", "i.e", "e.g", "etc", "viz", "cr", "crl", "p.c", "i.p.c", "cr.p.c",
    "para", "paras", "p", "pp", "vol", "ors", "anr", "sr", "jr", "st", "u.p",
    "m.p", "a.p", "h.p", "w.p", "s.l.p", "c.a", "a.i.r", "s.c.c", "scr", "hon",
}

_SPLIT_RE = re.compile(r"(?<=[.?!])\s+(?=[\"'(\[]?[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Split judgment text into sentences, keeping legal abbreviations intact."""
    sentences: list[str] = []
    # Judgments are paragraph-per-line; split lines first so paragraph numbers
    # ("12.  The appellant...") and header lines never glue onto a sentence.
    for line in (text or "").splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        line = re.sub(r"^\(?\d{1,3}[.)]\s+", "", line)
        if line:
            sentences.extend(_split_line(line))
    return sentences


def _split_line(text: str) -> list[str]:
    pieces = _SPLIT_RE.split(text)
    sentences: list[str] = []
    for piece in pieces:
        if sentences:
            prev = sentences[-1]
            last_token = prev.rsplit(" ", 1)[-1].rstrip(".").lower()
            # Merge when the previous piece ended in an abbreviation, a single
            # initial ("A."), or is just a paragraph number ("12.").
            if (last_token in _ABBREVIATIONS
                    or re.fullmatch(r"[a-z]", last_token)
                    or re.fullmatch(r"\(?\d{1,3}\)?", prev.strip(" .")) ):
                sentences[-1] = f"{prev} {piece}"
                continue
        sentences.append(piece)
    return [s.strip() for s in sentences if s.strip()]


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_sentences(sentences: list[str], max_words: int = 250,
                    overlap_sentences: int = 2) -> list[str]:
    """Group sentences into chunks of at most ~max_words words.

    Chunks never cut a sentence in half (except a single sentence that is
    itself longer than max_words, which is split on word boundaries), and
    consecutive chunks share `overlap_sentences` sentences of context.
    """
    # Break up pathological very long "sentences" first.
    units: list[str] = []
    for s in sentences:
        words = s.split()
        if len(words) <= max_words:
            units.append(s)
        else:
            for i in range(0, len(words), max_words):
                units.append(" ".join(words[i:i + max_words]))

    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for unit in units:
        n = len(unit.split())
        if current and count + n > max_words:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            count = sum(len(x.split()) for x in current)
            # If the overlap alone plus the new unit is still too big, drop it.
            if count + n > max_words:
                current, count = [], 0
        current.append(unit)
        count += n
    if current:
        chunks.append(" ".join(current))
    return chunks


# ── Case header parsing ───────────────────────────────────────────────────────
_YEAR_RE = re.compile(r"\b(18|19|20)\d{2}\b")


def parse_case_header(raw: str) -> dict:
    """AILA case files start with: title, court, (blank), date, case number."""
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    head = lines[:6]
    title = head[0] if head else ""
    court = head[1] if len(head) > 1 and "court" in head[1].lower() else ""
    year = ""
    for line in head[1:5]:
        m = _YEAR_RE.search(line)
        if m:
            year = m.group(0)
            break
    return {"title": title[:200], "court": court, "year": year}


_GOVT_RE = re.compile(
    r"^(the\s+)?(state\b|union of india|govt|government|central bureau|"
    r"c\.?b\.?i\b|directorate|commissioner|collector|municipal|"
    r"public prosecutor|enforcement)", re.I)


def appellant_type(title: str) -> str:
    """Rough guess at who brought the appeal, from the case title."""
    first_party = re.split(r"\s+v(?:s)?\.?\s+", title, maxsplit=1, flags=re.I)[0]
    return "State/Government" if _GOVT_RE.search(first_party.strip()) else "Private party"


# ── Outcome extraction ────────────────────────────────────────────────────────
OUTCOMES = ("Allowed", "Partly allowed", "Dismissed", "Disposed of", "Unclear")

_OUTCOME_PATTERNS = [
    ("Partly allowed", re.compile(
        r"\b(partly|partially)\s+allowed\b|\ballowed\s+(in\s+)?part\b", re.I)),
    ("Allowed", re.compile(
        r"\b(appeals?|petitions?|writ petitions?|applications?)\s+"
        r"(is|are|stands?|shall stand|must be|be|accordingly)?\s*(accordingly\s+)?allowed\b"
        r"|\bwe\s+allow\s+(this|the|these)\s+(appeals?|petitions?)\b"
        r"|^(appeals?|petitions?)\s+allowed\b"
        r"|\bwe,?\s+(therefore,?\s+|accordingly,?\s+)?allow\s+(this|the|these)\s+(appeals?|petitions?|writ)"
        r"|\b(appeal|petition)\s+succeeds\b", re.I)),
    ("Dismissed", re.compile(
        r"\b(appeals?|petitions?|writ petitions?|applications?)\s+"
        r"(is|are|stands?|shall stand|must be|be|accordingly)?\s*(accordingly\s+)?dismissed\b"
        r"|\bwe\s+dismiss\s+(this|the|these)\s+(appeals?|petitions?)\b"
        r"|^(appeals?|petitions?)\s+dismissed\b"
        r"|\b(appeals?|petitions?)\s+(fails?|must fail)\b"
        r"|\b(fail and the same are|therefore,? fail)\b"
        r"|\bwhich\s+(is|are)\s+(accordingly\s+)?dismissed\b|\bdeserves?\s+to\s+be\s+dismissed\b"
        r"|\bno\s+(merit|substance)\s+in\s+(this|the|these)\s+(appeals?|petitions?)\b", re.I)),
    ("Disposed of", re.compile(r"\bdisposed\s+of\b|^order\s+accord(?:i)?ngly\b", re.I)),
]


def extract_outcome(raw: str) -> str:
    """Return the disposition of a judgment.

    AILA case files usually end with an editorial line such as
    "Appeal dismissed" or "Appeals partly allowed."; we check that first and
    fall back to the last outcome phrase in the final part of the judgment.
    """
    lines = [l.strip() for l in (raw or "").splitlines() if l.strip()]
    if not lines:
        return "Unclear"
    last = lines[-1]
    if len(last) < 80:
        for label, pattern in _OUTCOME_PATTERNS:
            if pattern.search(last):
                return label

    tail = " ".join(lines[-6:])[-2500:]
    best_label, best_pos = "Unclear", -1
    for label, pattern in _OUTCOME_PATTERNS:
        for m in pattern.finditer(tail):
            # "Partly allowed" also matches the "Allowed" pattern; prefer it on ties.
            if m.start() > best_pos or (m.start() == best_pos and label == "Partly allowed"):
                best_label, best_pos = label, m.start()
    return best_label


# ── Citation extraction ───────────────────────────────────────────────────────
_ACT_ALIASES = [
    (re.compile(r"i\.?\s?p\.?\s?c|indian penal code|penal code", re.I), "IPC"),
    (re.compile(r"cr\.?\s?p\.?\s?c|code of criminal procedure|criminal procedure code", re.I), "CrPC"),
    (re.compile(r"c\.?\s?p\.?\s?c|code of civil procedure|civil procedure code", re.I), "CPC"),
    (re.compile(r"evidence act", re.I), "Evidence Act"),
    (re.compile(r"n\.?d\.?p\.?s|narcotic", re.I), "NDPS Act"),
    (re.compile(r"prevention of corruption", re.I), "PC Act"),
    (re.compile(r"dowry prohibition", re.I), "Dowry Prohibition Act"),
    (re.compile(r"arms act", re.I), "Arms Act"),
    (re.compile(r"constitution", re.I), "Constitution"),
]

_SECTION_RE = re.compile(
    r"\b(?:sections?|secs?\.|ss?\.|u/s\.?)\s*(\d{1,4}[A-Z]{0,2})"
    r"(?:\s*\(\w{1,4}\))*"
    r"(?=(?P<tail>[^.;]{0,60}))", re.I)
_ARTICLE_RE = re.compile(r"\barticles?\s+(\d{1,3}[A-Z]?)", re.I)


def extract_citations(text: str, top_n: int = 6) -> list[str]:
    """Most frequently cited provisions, e.g. ['Section 302 IPC', 'Article 21']."""
    counts: Counter[str] = Counter()
    for m in _SECTION_RE.finditer(text or ""):
        act = ""
        for pattern, name in _ACT_ALIASES:
            if pattern.search(m.group("tail")):
                act = name
                break
        if act:  # Bare "section 5" with no act is too ambiguous to report.
            counts[f"Section {m.group(1).upper()} {act}"] += 1
    for m in _ARTICLE_RE.finditer(text or ""):
        counts[f"Article {m.group(1).upper()}"] += 1
    return [c for c, _ in counts.most_common(top_n)]


# ── Lexical tokenization (BM25) ───────────────────────────────────────────────
_BM25_STOP = set("""a an the and or of to in on for with by at from as is are was were be been being this that these those
it its into over under about which who whom when where why how can could should would may might will shall do does did not
no any all some such he she they them his her their him i we you our your said also has have had upon there than then if so
learned counsel court appellant appellants respondent respondents petitioner petitioners case""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens for BM25, keeping section numbers like 302 / 498a."""
    return [t for t in re.findall(r"[a-z]+|\d+[a-z]?", (text or "").lower())
            if t not in _BM25_STOP and len(t) > 1]
