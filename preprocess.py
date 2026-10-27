"""Turn the raw AILA 2019 files into clean CSVs and retrieval chunks.

Outputs
  data/processed/cases.csv              one row per judgment (+ title, year, outcome, citations)
  data/processed/statutes.csv           one row per statute
  data/processed/queries.csv            the 50 AILA query situations
  data/processed/relevance_cases.csv    query_id, case_id, relevant (0/1)
  data/processed/relevance_statutes.csv query_id, statute_id, relevant (0/1)
  data/chunks/chunks.csv                sentence-aligned, token-safe chunks of cases + statutes
  data/stats/summary.json               dataset statistics

Run: python preprocess.py
"""

import csv
import json
from pathlib import Path

import config
from legal_text import (
    appellant_type,
    chunk_sentences,
    extract_citations,
    extract_outcome,
    parse_case_header,
    split_sentences,
)


def read_txt(filepath: Path) -> str:
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return Path(filepath).read_text(encoding=enc)
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    return ""


def flatten(text: str) -> str:
    """Single-line version of a text for CSV storage."""
    return " ".join(text.split())


def save_csv(filepath: Path, rows: list[dict]) -> None:
    if not rows:
        print(f"      WARNING: nothing to save for {filepath}")
        return
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"      Saved {len(rows):>7} rows -> {filepath.relative_to(config.ROOT)}")


def load_relevance(filepath: Path, id_field: str) -> list[dict]:
    """Parse TREC-style qrels: `<query-id> Q0 <doc-id> <relevance>`.

    (The previous version treated every token after the query id as a relevant
    document, so "Q0" and "0" became document ids and every pair was marked
    relevant.)
    """
    rows = []
    for line in read_txt(filepath).splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        query_id, _, doc_id, label = parts
        rows.append({"query_id": query_id, id_field: doc_id, "relevant": int(label)})
    return rows


def load_cases(raw_dir: Path):
    cases, chunks = [], []
    for fp in sorted(raw_dir.glob("*.txt"), key=lambda p: int(p.stem[1:]) if p.stem[1:].isdigit() else 0):
        raw = read_txt(fp)
        if not raw.strip():
            continue
        header = parse_case_header(raw)
        sentences = split_sentences(raw)
        body = " ".join(sentences)
        case_chunks = chunk_sentences(sentences, config.CHUNK_WORDS, config.CHUNK_OVERLAP_SENTENCES)
        cases.append({
            "case_id": fp.stem,
            "title": header["title"],
            "court": header["court"],
            "year": header["year"],
            "outcome": extract_outcome(raw),
            "appellant": appellant_type(header["title"]),
            "citations": "|".join(extract_citations(body)),
            "word_count": len(body.split()),
            "n_chunks": len(case_chunks),
            "text": body,
        })
        for i, text in enumerate(case_chunks):
            chunks.append({
                "chunk_id": f"{fp.stem}_c{i}",
                "doc_id": fp.stem,
                "doc_type": "case",
                "chunk_idx": i,
                "n_chunks": len(case_chunks),
                "word_count": len(text.split()),
                "text": text,
            })
    return cases, chunks


def load_statutes(raw_dir: Path):
    statutes, chunks = [], []
    for fp in sorted(raw_dir.glob("*.txt"), key=lambda p: int(p.stem[1:]) if p.stem[1:].isdigit() else 0):
        raw = read_txt(fp)
        if not raw.strip():
            continue
        title, desc = "", []
        for line in raw.splitlines():
            if line.startswith("Title:"):
                title = line[len("Title:"):].strip()
            elif line.startswith("Desc:"):
                desc.append(line[len("Desc:"):].strip())
            elif line.strip():
                desc.append(line.strip())
        description = flatten(" ".join(desc))
        statutes.append({
            "statute_id": fp.stem,
            "title": title,
            "description": description,
            "word_count": len(description.split()),
        })
        sentences = [f"{title}."] + split_sentences(description)
        parts = chunk_sentences(sentences, config.CHUNK_WORDS, 1)
        for i, text in enumerate(parts):
            chunks.append({
                "chunk_id": f"{fp.stem}_c{i}",
                "doc_id": fp.stem,
                "doc_type": "statute",
                "chunk_idx": i,
                "n_chunks": len(parts),
                "word_count": len(text.split()),
                "text": text if i == 0 else f"{title}. {text}",
            })
    return statutes, chunks


def load_queries(query_file: Path) -> list[dict]:
    queries = []
    for line in read_txt(query_file).splitlines():
        if "||" not in line:
            continue
        query_id, text = line.split("||", 1)
        text = flatten(text)
        if text:
            queries.append({"query_id": query_id.strip(), "text": text,
                            "word_count": len(text.split())})
    return queries


def main():
    raw = config.RAW_DIR
    if not (raw / "Object_casedocs").exists():
        raise SystemExit(f"AILA data not found in {raw}. Run `python download_data.py` first.")

    print("[1/4] Cases")
    cases, case_chunks = load_cases(raw / "Object_casedocs")
    save_csv(config.CASES_CSV, cases)

    print("[2/4] Statutes")
    statutes, statute_chunks = load_statutes(raw / "Object_statutes")
    save_csv(config.STATUTES_CSV, statutes)

    print("[3/4] Queries + relevance judgments")
    queries = load_queries(raw / "Query_doc.txt")
    save_csv(config.QUERIES_CSV, queries)
    rel_cases = load_relevance(raw / "relevance_judgments_priorcases.txt", "case_id")
    rel_statutes = load_relevance(raw / "relevance_judgments_statutes.txt", "statute_id")
    save_csv(config.REL_CASES_CSV, rel_cases)
    save_csv(config.REL_STATUTES_CSV, rel_statutes)

    print("[4/4] Chunks (queries are intentionally NOT indexed)")
    all_chunks = case_chunks + statute_chunks
    save_csv(config.CHUNKS_CSV, all_chunks)

    outcome_counts = {}
    for c in cases:
        outcome_counts[c["outcome"]] = outcome_counts.get(c["outcome"], 0) + 1

    stats = {
        "data_source": "AILA FIRE 2019",
        "total_cases": len(cases),
        "total_statutes": len(statutes),
        "total_queries": len(queries),
        "total_chunks": len(all_chunks),
        "case_chunks": len(case_chunks),
        "statute_chunks": len(statute_chunks),
        "relevant_case_pairs": sum(r["relevant"] for r in rel_cases),
        "judged_case_pairs": len(rel_cases),
        "relevant_statute_pairs": sum(r["relevant"] for r in rel_statutes),
        "judged_statute_pairs": len(rel_statutes),
        "avg_case_words": round(sum(c["word_count"] for c in cases) / max(len(cases), 1), 1),
        "avg_query_words": round(sum(q["word_count"] for q in queries) / max(len(queries), 1), 1),
        "chunk_words": config.CHUNK_WORDS,
        "chunk_overlap_sentences": config.CHUNK_OVERLAP_SENTENCES,
        "case_outcomes": outcome_counts,
    }
    config.STATS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.STATS_DIR / "summary.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("\nPREPROCESSING COMPLETE")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
