"""AI Legal Assistant for Indian case law (AILA 2019 corpus).

Pipeline
  1. intent detection      -> which rhetorical-role indexes to emphasise
  2. dense retrieval       -> fine-tuned InLegalBERT over the ALL index + role indexes
  3. chunk-level reranking -> cross-encoder scores each candidate passage, a case
                              is scored by its best passage
  4. explanation           -> key sentences per case, attributed to that case
  5. outcome summary       -> actual disposition of each retrieved judgment

Usage
  python assistant.py                                  # interactive
  python assistant.py -q "bail in a dowry death case"  # one-shot
  python assistant.py -q "..." --json                  # machine-readable
  python assistant.py --no-rerank                      # faster, bi-encoder only
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

import config
from legal_text import split_sentences
from tag_roles import sentence_roles

DISCLAIMER = ("Research aid only, not legal advice. Results come from a fixed corpus of "
              "2,914 Supreme Court judgments (AILA 2019) and may be incomplete or outdated.")

# ── Intent detection ──────────────────────────────────────────────────────────
# Each intent lists the roles whose indexes get an extra search. The ALL index
# is always searched too, so a wrong intent can no longer hide relevant cases.
INTENTS: list[tuple[str, re.Pattern, list[str]]] = [
    ("bail", re.compile(r"\b(anticipatory bail|bail|custody|arrest(ed)?|remand|section 438|section 439)\b", re.I),
     ["RATIO", "RPC", "ANALYSIS"]),
    ("homicide", re.compile(r"\b(murder|homicide|culpable|killed|killing|section 30[24]|death sentence|life imprisonment)\b", re.I),
     ["RATIO", "FAC", "ANALYSIS", "RPC"]),
    ("sexual offences", re.compile(r"\b(rape|sexual|pocso|outrag\w* (the )?modesty|section 376|section 354)\b", re.I),
     ["RATIO", "FAC", "ANALYSIS"]),
    ("dowry / cruelty", re.compile(r"\b(dowry|498 ?a|304 ?b|cruelty|harass\w*|bride burning)\b", re.I),
     ["RATIO", "FAC", "ANALYSIS"]),
    ("fraud / cheating", re.compile(r"\b(fraud\w*|cheat\w*|section 420|forg\w+|misappropriat\w*|breach of trust|section 40[69])\b", re.I),
     ["RATIO", "STA", "ANALYSIS"]),
    ("narcotics", re.compile(r"\b(ndps|narcotic\w*|ganja|heroin|opium|charas|contraband|psychotropic)\b", re.I),
     ["RATIO", "STA", "ANALYSIS"]),
    ("corruption", re.compile(r"\b(bribe\w*|corrupt\w*|illegal gratification|disproportionate assets)\b", re.I),
     ["RATIO", "STA", "ANALYSIS"]),
    ("service law", re.compile(r"\b(dismiss\w* from service|terminat\w*|promotion|seniority|pension|disciplinary|departmental (inquiry|enquiry|proceedings?)|suspension)\b", re.I),
     ["RATIO", "ISSUE", "RLC"]),
    ("property / land", re.compile(r"\b(property|land acquisition|acquisition|tenan\w*|evict\w*|possession|partition|title deed|easement)\b", re.I),
     ["RATIO", "ISSUE", "FAC"]),
    ("constitutional", re.compile(r"\b(article \d+|fundamental rights?|writ|habeas corpus|constitution\w*)\b", re.I),
     ["RATIO", "ISSUE", "STA"]),
    ("evidence", re.compile(r"\b(confession|eye ?witness\w*|circumstantial|dying declaration|hostile witness|recovery|identification parade|last seen)\b", re.I),
     ["RATIO", "ANALYSIS"]),
    ("statute lookup", re.compile(r"\b(what (is|does) (section|article)|meaning of (section|article)|define (section|article)|ingredients of (section|the offence))\b", re.I),
     ["STA", "ISSUE"]),
]

_STOPWORDS = set("""a an the and or of to in on for with by at from as is are was were be been being this that these those
it its into over under about whether what which who whom when where why how can could should would may might will shall do does
did not no any all some such case cases court law legal my our your their his her him he she they them i we you there than then
if so also has have had upon against between after before during within without more most other""".split())


@dataclass
class Intent:
    names: list[str]
    roles: list[str]

    @property
    def label(self) -> str:
        return ", ".join(self.names) if self.names else "general"


def detect_intent(query: str) -> Intent:
    names, roles = [], []
    for name, pattern, intent_roles in INTENTS:
        if pattern.search(query):
            names.append(name)
            roles += [r for r in intent_roles if r not in roles]
    if not roles:
        roles = ["RATIO", "ISSUE", "FAC"]
    return Intent(names, roles)


def query_terms(query: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", query.lower())
    return {w for w in words if w not in _STOPWORDS and (len(w) > 2 or w.isdigit())}


# ── Result types ──────────────────────────────────────────────────────────────
@dataclass
class CaseResult:
    case_id: str
    title: str
    year: str
    outcome: str
    appellant: str
    relevance: float           # 0-1, cross-encoder probability (or bi-encoder proxy)
    roles: list[str]
    citations: list[str]
    key_passages: list[dict]   # [{"text": ..., "roles": [...]}]
    passage: str               # best-matching passage


@dataclass
class StatuteResult:
    statute_id: str
    title: str
    relevance: float
    text: str


@dataclass
class AssistantResponse:
    query: str
    intent: str
    cases: list[CaseResult]
    statutes: list[StatuteResult]
    outcome_summary: dict
    disclaimer: str = DISCLAIMER
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Assistant ─────────────────────────────────────────────────────────────────
class LegalAssistant:
    def __init__(self, index_dir=config.INDEX_DIR, bi_encoder: str | None = None,
                 cross_encoder: str | None = config.CROSS_ENCODER, device: str | None = None,
                 verbose: bool = True):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.index_dir = index_dir
        self.verbose = verbose
        manifest_path = index_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No indexes in {index_dir}. Build them first:\n"
                "  python preprocess.py && python tag_roles.py && python build_role_index.py")
        manifest = json.loads(manifest_path.read_text())
        bi_encoder = bi_encoder or manifest["model"]
        if bi_encoder != manifest["model"]:
            self._log(f"WARNING: indexes were built with {manifest['model']} but querying with "
                      f"{bi_encoder}. Rebuild the indexes or results will be meaningless.")
        device = device or config.get_device()

        self._log(f"Loading bi-encoder {bi_encoder} on {device}...")
        self.bi_encoder = SentenceTransformer(bi_encoder, device=device)
        self.bi_encoder.max_seq_length = min(self.bi_encoder.max_seq_length or 512, 512)
        self.cross_encoder = None
        if cross_encoder:
            from sentence_transformers import CrossEncoder
            self._log(f"Loading cross-encoder {cross_encoder}...")
            self.cross_encoder = CrossEncoder(cross_encoder, device=device, max_length=512)

        self._log("Loading indexes...")
        self.meta = pd.read_csv(index_dir / f"meta_{config.ALL_INDEX}.csv", keep_default_na=False)
        self.indexes, self.positions = {}, {}
        for name in manifest["indexes"]:
            self.indexes[name] = faiss.read_index(str(index_dir / f"index_{name}.bin"))
            if name != config.ALL_INDEX:
                self.positions[name] = np.load(index_dir / f"positions_{name}.npy")

        cases = pd.read_csv(config.CASES_CSV, keep_default_na=False,
                            usecols=["case_id", "title", "year", "outcome", "appellant", "citations"])
        self.case_info = cases.set_index("case_id").to_dict("index")
        statutes = pd.read_csv(config.STATUTES_CSV, keep_default_na=False)
        self.statute_info = statutes.set_index("statute_id").to_dict("index")
        self._log(f"Ready: {len(self.meta)} chunks, {len(self.indexes)} indexes, "
                  f"{len(self.case_info)} cases, {len(self.statute_info)} statutes.\n")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=sys.stderr)

    # ── retrieval ─────────────────────────────────────────────────────────────
    def _retrieve(self, query_emb: np.ndarray, roles: list[str], k_all: int, k_role: int) -> dict[int, float]:
        """Return {row in meta_ALL: L2 distance} for the candidate chunks."""
        candidates: dict[int, float] = {}
        dist, idx = self.indexes[config.ALL_INDEX].search(query_emb, min(k_all, len(self.meta)))
        for d, i in zip(dist[0], idx[0]):
            if i >= 0:
                candidates[int(i)] = float(d)
        for role in roles:
            if role not in self.indexes:
                continue
            index = self.indexes[role]
            dist, idx = index.search(query_emb, min(k_role, index.ntotal))
            for d, i in zip(dist[0], idx[0]):
                if i < 0:
                    continue
                pos = int(self.positions[role][i])
                candidates[pos] = min(float(d), candidates.get(pos, math.inf))
        return candidates

    def _score(self, query: str, rows: list[int], distances: dict[int, float]) -> np.ndarray:
        if self.cross_encoder is None:
            # Turn L2 distances into a 0-1 score relative to the best candidate.
            d = np.array([distances[r] for r in rows])
            return np.exp(-(d - d.min()) / (d.std() + 1e-6))
        pairs = [(query, self.meta.at[r, "text"]) for r in rows]
        raw = np.asarray(self.cross_encoder.predict(pairs, batch_size=32, show_progress_bar=False),
                         dtype=np.float64).reshape(-1)
        if raw.min() < 0 or raw.max() > 1:  # model returned logits
            raw = 1.0 / (1.0 + np.exp(-raw))
        return raw

    def ask(self, query: str, top_k: int = 5, n_statutes: int = 3,
            k_all: int = 50, k_role: int = 15, max_rerank: int = 100) -> AssistantResponse:
        query = " ".join(query.split())
        intent = detect_intent(query)
        q_emb = self.bi_encoder.encode([query], convert_to_numpy=True).astype("float32")

        distances = self._retrieve(q_emb, intent.roles, k_all, k_role)
        rows = sorted(distances, key=distances.get)[:max_rerank]
        scores = self._score(query, rows, distances)

        # Aggregate chunk scores into documents: a document is as relevant as
        # its best passage (MaxP), with role tags pooled across its passages.
        best: dict[str, tuple[float, int]] = {}
        doc_roles: dict[str, set] = defaultdict(set)
        for r, s in zip(rows, scores):
            doc_id = self.meta.at[r, "doc_id"]
            doc_roles[doc_id] |= {x for x in self.meta.at[r, "roles"].split("|") if x}
            if doc_id not in best or s > best[doc_id][0]:
                best[doc_id] = (float(s), r)
        ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)

        terms = query_terms(query)
        cases, statutes = [], []
        for doc_id, (score, row) in ranked:
            text = self.meta.at[row, "text"]
            if self.meta.at[row, "doc_type"] == "statute":
                if len(statutes) < n_statutes:
                    info = self.statute_info.get(doc_id, {})
                    statutes.append(StatuteResult(doc_id, info.get("title", doc_id), round(score, 3),
                                                  info.get("description", text)[:600]))
                continue
            if len(cases) >= top_k:
                continue
            info = self.case_info.get(doc_id, {})
            cases.append(CaseResult(
                case_id=doc_id,
                title=info.get("title") or doc_id,
                year=str(info.get("year", "")),
                outcome=info.get("outcome", "Unclear"),
                appellant=info.get("appellant", ""),
                relevance=round(score, 3),
                roles=sorted(doc_roles[doc_id]),
                citations=[c for c in str(info.get("citations", "")).split("|") if c][:4],
                key_passages=self._key_passages(text, terms),
                passage=text,
            ))

        notes = []
        if self.cross_encoder is None:
            notes.append("Reranker disabled: relevance is relative to the best match, not absolute.")
        elif cases and max(c.relevance for c in cases) < 0.2:
            notes.append("Low relevance overall: the corpus may not contain a close match for this query. "
                         "Try describing the facts or the legal question more specifically.")
        return AssistantResponse(query, intent.label, cases, statutes, self._outcome_summary(cases), notes=notes)

    # ── explanation ───────────────────────────────────────────────────────────
    @staticmethod
    def _key_passages(text: str, terms: set[str], n: int = 3) -> list[dict]:
        """Pick the sentences that best explain why this passage matched."""
        weights = {"RATIO": 2.0, "RPC": 1.5, "ISSUE": 1.2, "ANALYSIS": 1.0, "PRE_RELIED": 0.8, "STA": 0.5}
        scored = []
        for i, s in enumerate(split_sentences(text)):
            if len(s) < 50:
                continue
            words = set(re.findall(r"[a-z0-9]+", s.lower()))
            overlap = len(terms & words)
            roles = sentence_roles(s)
            score = overlap + sum(weights.get(r, 0.0) for r in roles)
            if score > 0:
                scored.append((score, i, s, sorted(roles)))
        top = sorted(scored, key=lambda x: (-x[0], x[1]))[:n]
        if not top:  # nothing stands out: show the opening of the passage instead
            top = [(0, i, s, []) for i, s in enumerate(split_sentences(text)) if len(s) >= 50][:2]
        return [{"text": s if len(s) <= 400 else s[:400].rsplit(" ", 1)[0] + " ...", "roles": roles}
                for _, _, s, roles in sorted(top, key=lambda x: x[1])]

    @staticmethod
    def _outcome_summary(cases: list[CaseResult]) -> dict:
        counts = Counter(c.outcome for c in cases)
        decided = {k: v for k, v in counts.items() if k in ("Allowed", "Partly allowed", "Dismissed")}
        total = sum(decided.values())
        if total == 0:
            trend = "No clear pattern: the retrieved judgments were disposed of without a clear win/loss."
        else:
            success = decided.get("Allowed", 0) + decided.get("Partly allowed", 0)
            share = success / total
            if share >= 0.67:
                trend = "Most similar appeals/petitions succeeded (fully or partly)."
            elif share <= 0.33:
                trend = "Most similar appeals/petitions were dismissed."
            else:
                trend = "Mixed outcomes: the result turned on the facts of each case."
        state_appeals = sum(1 for c in cases if c.appellant == "State/Government")
        caveat = ("'Allowed' means the party that appealed won; in "
                  f"{state_appeals} of these {len(cases)} cases that party was the State.") if state_appeals else \
                 "'Allowed' means the party that appealed won."
        return {"counts": dict(counts), "trend": trend, "caveat": caveat}


# ── CLI formatting ────────────────────────────────────────────────────────────
def format_response(resp: AssistantResponse) -> str:
    out = [f"\nQuery intent: {resp.intent}", "=" * 70]
    if not resp.cases and not resp.statutes:
        out.append("No results found.")
        return "\n".join(out)
    out.append("RELEVANT JUDGMENTS")
    for i, c in enumerate(resp.cases, 1):
        year = f" ({c.year})" if c.year else ""
        out.append(f"\n{i}. {c.title}{year}  [{c.case_id}]")
        out.append(f"   Relevance: {c.relevance:.0%}  |  Outcome: {c.outcome}  |  Appellant: {c.appellant}")
        if c.citations:
            out.append(f"   Provisions: {', '.join(c.citations)}")
        for p in c.key_passages:
            tag = f" [{', '.join(p['roles'])}]" if p["roles"] else ""
            out.append(f"   • {p['text']}{tag}")
    if resp.statutes:
        out.append("\nRELEVANT STATUTES")
        for s in resp.statutes:
            out.append(f"  • {s.title} [{s.statute_id}]  ({s.relevance:.0%})")
    o = resp.outcome_summary
    out.append("\nOUTCOMES IN SIMILAR CASES")
    out.append("  " + ", ".join(f"{k}: {v}" for k, v in o["counts"].items()))
    out.append(f"  {o['trend']}")
    out.append(f"  Note: {o['caveat']}")
    for n in resp.notes:
        out.append(f"\n! {n}")
    out.append(f"\n{resp.disclaimer}")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="AI legal assistant for Indian case law")
    parser.add_argument("-q", "--query", help="ask one question and exit")
    parser.add_argument("-k", "--top-k", type=int, default=5)
    parser.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder (faster)")
    parser.add_argument("--model", default=None, help="bi-encoder (defaults to the one the indexes were built with)")
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = parser.parse_args()

    assistant = LegalAssistant(bi_encoder=args.model, cross_encoder=None if args.no_rerank else config.CROSS_ENCODER)

    def run(q: str):
        resp = assistant.ask(q, top_k=args.top_k)
        print(json.dumps(resp.to_dict(), indent=2) if args.json else format_response(resp))

    if args.query:
        run(args.query)
        return
    while True:
        try:
            q = input("\nEnter your legal query (or 'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"quit", "exit", "q"}:
            break
        if q:
            run(q)


if __name__ == "__main__":
    main()
