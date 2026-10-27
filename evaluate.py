"""Measure retrieval quality on the AILA 2019 prior-case task.

For each of the 50 AILA queries, rank all 2,914 case documents and compare
with the official relevance judgments (195 relevant query-case pairs).

Methods
  bm25     lexical baseline over full case documents (no GPU / model needed)
  dense    bi-encoder over the ALL chunk index, case score = best chunk (MaxP)
  hybrid   reciprocal-rank fusion of bm25 + dense
  rerank   hybrid candidates re-scored by the cross-encoder

Metrics: MAP, MRR, P@10, R@10, R@100, nDCG@10 (macro-averaged over queries).

IMPORTANT: if the bi-encoder was fine-tuned on AILA queries, evaluating it on
those same queries is leakage. Use `--split test` with the held-out queries
written by prepare_training_data.py, and fine-tune with that split.

Run: python evaluate.py --methods bm25 dense hybrid [--split test|all]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict

import numpy as np
import pandas as pd

import config


# ── metrics ───────────────────────────────────────────────────────────────────
def average_precision(ranked: list[str], relevant: set[str]) -> float:
    hits, total = 0, 0.0
    for i, doc in enumerate(ranked, 1):
        if doc in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant) if relevant else 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1) if d in relevant)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_run(run: dict[str, list[str]], qrels: dict[str, set[str]]) -> dict[str, float]:
    m = defaultdict(list)
    for qid, relevant in qrels.items():
        ranked = run.get(qid, [])
        m["MAP"].append(average_precision(ranked, relevant))
        rr = next((1 / i for i, d in enumerate(ranked, 1) if d in relevant), 0.0)
        m["MRR"].append(rr)
        m["P@10"].append(len(set(ranked[:10]) & relevant) / 10)
        m["R@10"].append(len(set(ranked[:10]) & relevant) / len(relevant))
        m["R@100"].append(len(set(ranked[:100]) & relevant) / len(relevant))
        m["nDCG@10"].append(ndcg_at_k(ranked, relevant, 10))
    return {k: float(np.mean(v)) for k, v in m.items()}


# ── retrieval methods ─────────────────────────────────────────────────────────
def bm25_run(queries: dict[str, str], cases: pd.DataFrame, depth: int) -> dict[str, list[str]]:
    from lexical import BM25Index
    print("  building BM25 over full case texts...")
    bm25 = BM25Index().fit(cases["text"].tolist())
    ids = cases["case_id"].tolist()
    return {qid: [ids[i] for i in bm25.top_k(q, depth)[0]] for qid, q in queries.items()}


def dense_run(queries: dict[str, str], model_name: str | None, depth: int) -> dict[str, list[str]]:
    import faiss
    from sentence_transformers import SentenceTransformer

    manifest = json.loads((config.INDEX_DIR / "manifest.json").read_text())
    model_name = model_name or manifest["model"]
    if model_name != manifest["model"]:
        raise SystemExit(f"Index was built with {manifest['model']}; rebuild it with "
                         f"`python build_role_index.py --model {model_name}` first.")
    meta = pd.read_csv(config.INDEX_DIR / f"meta_{config.ALL_INDEX}.csv", usecols=["doc_id", "doc_type"])
    is_case = (meta["doc_type"] == "case").to_numpy()
    doc_ids = meta["doc_id"].to_numpy()
    index = faiss.read_index(str(config.INDEX_DIR / f"index_{config.ALL_INDEX}.bin"))
    model = SentenceTransformer(model_name, device=config.get_device())
    model.max_seq_length = min(model.max_seq_length or 512, 512)

    qids = list(queries)
    q_emb = model.encode([queries[q] for q in qids], convert_to_numpy=True, show_progress_bar=True).astype("float32")
    _, idx = index.search(q_emb, min(3000, index.ntotal))
    run = {}
    for qid, row in zip(qids, idx):
        seen, ranked = set(), []
        for i in row:
            if i < 0 or not is_case[i]:
                continue
            d = doc_ids[i]
            if d not in seen:
                seen.add(d)
                ranked.append(d)
            if len(ranked) >= depth:
                break
        run[qid] = ranked
    return run


def rrf(runs: list[dict[str, list[str]]], k: int = 60, depth: int = 1000) -> dict[str, list[str]]:
    fused = {}
    for qid in runs[0]:
        scores = defaultdict(float)
        for run in runs:
            for rank, d in enumerate(run.get(qid, []), 1):
                scores[d] += 1 / (k + rank)
        fused[qid] = sorted(scores, key=scores.get, reverse=True)[:depth]
    return fused


def rerank_run(queries, base_run, n_candidates: int = 50) -> dict[str, list[str]]:
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(config.CROSS_ENCODER, device=config.get_device(), max_length=512)
    chunks = pd.read_csv(config.CHUNKS_CSV, usecols=["doc_id", "doc_type", "chunk_idx", "text"])
    first_chunks = chunks[(chunks["doc_type"] == "case")].groupby("doc_id")["text"].apply(lambda s: list(s)[:4])
    run = {}
    for qid, ranked in base_run.items():
        cands = ranked[:n_candidates]
        pairs, owners = [], []
        for d in cands:
            for t in first_chunks.get(d, []):
                pairs.append((queries[qid], t))
                owners.append(d)
        scores = ce.predict(pairs, batch_size=32, show_progress_bar=False)
        best = defaultdict(lambda: -1e9)
        for d, s in zip(owners, scores):
            best[d] = max(best[d], float(s))
        run[qid] = sorted(cands, key=lambda d: best[d], reverse=True) + ranked[n_candidates:]
    return run


# ── main ──────────────────────────────────────────────────────────────────────
def load_queries_and_qrels(split: str):
    queries = pd.read_csv(config.QUERIES_CSV).set_index("query_id")["text"].to_dict()
    rel = pd.read_csv(config.REL_CASES_CSV)
    rel = rel[(rel["relevant"] == 1) & rel["case_id"].str.startswith("C")]
    qrels = rel.groupby("query_id")["case_id"].apply(set).to_dict()
    if split != "all":
        if not config.SPLIT_JSON.exists():
            raise SystemExit(f"{config.SPLIT_JSON} not found; run prepare_training_data.py or use --split all")
        keep = set(json.loads(config.SPLIT_JSON.read_text())[split])
        qrels = {q: r for q, r in qrels.items() if q in keep}
    queries = {q: queries[q] for q in qrels if q in queries}
    return queries, qrels


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", default=["bm25", "dense", "hybrid"],
                        choices=["bm25", "dense", "hybrid", "rerank"])
    parser.add_argument("--split", default="all", choices=["all", "train", "test"])
    parser.add_argument("--model", default=None, help="bi-encoder (must match the built index)")
    parser.add_argument("--depth", type=int, default=1000)
    parser.add_argument("--save", default=None, help="write results JSON here")
    args = parser.parse_args()

    queries, qrels = load_queries_and_qrels(args.split)
    print(f"Evaluating on {len(qrels)} queries ({args.split}), "
          f"{sum(len(v) for v in qrels.values())} relevant pairs\n")
    cases = pd.read_csv(config.CASES_CSV, usecols=["case_id", "text"], keep_default_na=False)

    runs, results = {}, {}
    need_bm25 = any(m in args.methods for m in ("bm25", "hybrid", "rerank"))
    need_dense = any(m in args.methods for m in ("dense", "hybrid", "rerank"))
    for name, needed, fn in [("bm25", need_bm25, lambda: bm25_run(queries, cases, args.depth)),
                             ("dense", need_dense, lambda: dense_run(queries, args.model, args.depth))]:
        if needed:
            t = time.time()
            print(f"[{name}]")
            runs[name] = fn()
            print(f"  done in {time.time() - t:.1f}s")
    if "hybrid" in args.methods or "rerank" in args.methods:
        runs["hybrid"] = rrf([runs["bm25"], runs["dense"]], depth=args.depth)
    if "rerank" in args.methods:
        print("[rerank]")
        runs["rerank"] = rerank_run(queries, runs["hybrid"])

    for m in args.methods:
        results[m] = evaluate_run(runs[m], qrels)

    cols = ["MAP", "MRR", "P@10", "R@10", "R@100", "nDCG@10"]
    print("\n" + f"{'method':<10}" + "".join(f"{c:>9}" for c in cols))
    for m, r in results.items():
        print(f"{m:<10}" + "".join(f"{r[c]:>9.4f}" for c in cols))
    if args.save:
        with open(args.save, "w") as f:
            json.dump({"split": args.split, "n_queries": len(qrels), "results": results}, f, indent=2)
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
