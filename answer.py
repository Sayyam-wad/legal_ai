"""Grounded answer synthesis (retrieval-augmented generation) with citation checks.

Given the judgments the assistant retrieved, write a short answer to the
user's question in which every claim cites the case it came from, e.g.
"A dying declaration can alone sustain a conviction if it is reliable [C833]."

Providers (choose with LEGAL_AI_LLM or the `provider` argument)
  anthropic   Claude via the Messages API          needs ANTHROPIC_API_KEY
  openai      any OpenAI-compatible endpoint       needs OPENAI_API_KEY (+ OPENAI_BASE_URL)
              e.g. a local Ollama server: OPENAI_BASE_URL=http://localhost:11434/v1
  extractive  no LLM: stitches the top key passages together, with citations
  (default)   anthropic/openai if a key is set, otherwise extractive

Every answer is checked after generation: citations to documents that were not
retrieved are removed and reported, and sentences without any citation are
counted, so hallucinated sources can't slip through silently.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import asdict, dataclass, field

CITATION_RE = re.compile(r"\[((?:C|S)\d+)\]")

SYSTEM_PROMPT = """You are a legal research assistant for Indian law. Answer the user's question using ONLY the numbered source passages provided. Rules:
- After every factual or legal claim, cite its source as [C123] or [S45] using the IDs shown.
- Never cite an ID that is not in the sources. Never use outside knowledge of cases.
- If the sources do not answer the question, say so plainly.
- Be concise: 3-6 sentences, plain English, no headings.
- Do not give advice on what the user should do; describe what the courts held."""


@dataclass
class GroundedAnswer:
    text: str
    provider: str
    cited: list[str]
    removed_citations: list[str] = field(default_factory=list)
    uncited_sentences: int = 0

    @property
    def grounded(self) -> bool:
        return bool(self.cited) and not self.removed_citations

    def to_dict(self) -> dict:
        d = asdict(self)
        d["grounded"] = self.grounded
        return d


def build_context(cases: list[dict], statutes: list[dict], max_chars: int = 1200) -> str:
    blocks = []
    for c in cases:
        passages = " ".join(p["text"] for p in c.get("key_passages", [])) or c.get("passage", "")
        body = (passages + " " + c.get("passage", ""))[:max_chars]
        year = f", {c['year']}" if c.get("year") else ""
        blocks.append(f"[{c['case_id']}] {c['title']}{year} — outcome: {c['outcome']}\n{body}")
    for s in statutes:
        blocks.append(f"[{s['statute_id']}] Statute: {s['title']}\n{s['text'][:600]}")
    return "\n\n".join(blocks)


def validate_citations(text: str, allowed: set[str]) -> tuple[str, list[str], list[str], int]:
    """Strip citations to unknown documents; return (text, cited, removed, uncited_sentences)."""
    removed = sorted({c for c in CITATION_RE.findall(text) if c not in allowed})
    for bad in removed:
        text = text.replace(f"[{bad}]", "")
    text = re.sub(r"\s+([.,;])", r"\1", re.sub(r"[ \t]{2,}", " ", text)).strip()
    cited = list(dict.fromkeys(c for c in CITATION_RE.findall(text)))
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) > 4]
    uncited = sum(1 for s in sentences if not CITATION_RE.search(s))
    return text, cited, removed, uncited


def _post_json(url: str, payload: dict, headers: dict, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _call_anthropic(question: str, context: str) -> str:
    data = _post_json(
        "https://api.anthropic.com/v1/messages",
        {"model": os.getenv("LEGAL_AI_LLM_MODEL", "claude-sonnet-5-5"), "max_tokens": 600,
         "system": SYSTEM_PROMPT,
         "messages": [{"role": "user", "content": f"Sources:\n\n{context}\n\nQuestion: {question}"}]},
        {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"})
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


def _call_openai(question: str, context: str) -> str:
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    data = _post_json(
        f"{base}/chat/completions",
        {"model": os.getenv("LEGAL_AI_LLM_MODEL", "gpt-4o-mini"), "temperature": 0, "max_tokens": 600,
         "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": f"Sources:\n\n{context}\n\nQuestion: {question}"}]},
        {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'ollama')}"})
    return data["choices"][0]["message"]["content"]


def extractive_answer(cases: list[dict], max_points: int = 4) -> str:
    """LLM-free answer: the strongest key passage from each top case, cited."""
    lines = []
    for c in cases:
        # Only quote sentences that carry a substantive role (ratio, ruling,
        # analysis...), never a bare opening line such as the case header.
        substantive = [p for p in c.get("key_passages", []) if set(p["roles"]) - {"PREAMBLE", "STA"}]
        ranked = sorted(substantive,
                        key=lambda p: ("RATIO" in p["roles"], "RPC" in p["roles"], len(p["roles"])), reverse=True)
        if ranked:
            sentence = ranked[0]["text"].rstrip(". ")
            lines.append(f"In {c['title']}, the court noted: \"{sentence}\" [{c['case_id']}].")
        if len(lines) >= max_points:
            break
    return " ".join(lines) if lines else "The retrieved judgments do not contain a clear statement on this question."


def default_provider() -> str:
    choice = os.getenv("LEGAL_AI_LLM", "").lower()
    if choice:
        return choice
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_BASE_URL"):
        return "openai"
    return "extractive"


def synthesize(question: str, cases: list[dict], statutes: list[dict] | None = None,
               provider: str | None = None) -> GroundedAnswer:
    statutes = statutes or []
    provider = provider or default_provider()
    allowed = {c["case_id"] for c in cases} | {s["statute_id"] for s in statutes}
    if not cases:
        return GroundedAnswer("No relevant judgments were found for this question.", provider, [])

    raw = None
    if provider in ("anthropic", "openai"):
        context = build_context(cases, statutes)
        try:
            raw = (_call_anthropic if provider == "anthropic" else _call_openai)(question, context)
        except Exception as e:  # network/auth errors: degrade gracefully
            raw, provider = None, f"extractive (LLM unavailable: {type(e).__name__})"
    if raw is None:
        raw = extractive_answer(cases)
        provider = provider if provider.startswith("extractive") else "extractive"

    text, cited, removed, uncited = validate_citations(raw, allowed)
    return GroundedAnswer(text, provider, cited, removed, uncited)
