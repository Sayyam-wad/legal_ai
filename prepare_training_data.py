"""Build bi-encoder training triplets from the AILA relevance judgments.

Fixes vs. the first version
  * No hard-coded Windows paths: reads the CSVs produced by preprocess.py.
  * Query-level train/test split, saved to data/training/query_split.json.
    Previously triplets were shuffled and split at random, so the same query
    appeared in training and evaluation and the reported ~96.7% triplet
    accuracy was inflated.
  * Positives are the passages of each relevant case that best match the query
    (BM25), not just the first 300 words (which is mostly the case header).
  * Hard negatives: passages from non-relevant cases that BM25 ranks highly,
    which teach the model far more than random cases.

Run: python prepare_training_data.py [--test-queries 10] [--seed 42]
"""

import argparse
import json
import random

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm

import config
from legal_text import tokenize


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-queries", type=int, default=10, help="queries held out for evaluation")
    p.add_argument("--positives-per-case", type=int, default=2)
    p.add_argument("--negatives-per-positive", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = random.Random(args.seed)

    queries = pd.read_csv(config.QUERIES_CSV).set_index("query_id")["text"].to_dict()
    rel = pd.read_csv(config.REL_CASES_CSV)
    rel = rel[(rel["relevant"] == 1) & rel["case_id"].str.startswith("C")]
    positives = rel.groupby("query_id")["case_id"].apply(list).to_dict()

    # ── query-level split ────────────────────────────────────────────────────
    qids = sorted(positives, key=lambda q: int(q.split("Q")[-1]))
    rng.shuffle(qids)
    test = sorted(qids[:args.test_queries], key=lambda q: int(q.split("Q")[-1]))
    train = sorted(qids[args.test_queries:], key=lambda q: int(q.split("Q")[-1]))
    config.TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    config.SPLIT_JSON.write_text(json.dumps({"seed": args.seed, "train": train, "test": test}, indent=2))
    print(f"Queries with relevant cases: {len(qids)}  ->  train {len(train)} / test {len(test)}")

    # ── BM25 over case chunks ────────────────────────────────────────────────
    chunks = pd.read_csv(config.CHUNKS_CSV, usecols=["doc_id", "doc_type", "chunk_idx", "text"])
    chunks = chunks[chunks["doc_type"] == "case"].reset_index(drop=True)
    print(f"Indexing {len(chunks)} case chunks with BM25...")
    bm25 = BM25Okapi([tokenize(t) for t in tqdm(chunks["text"], desc="tokenize")])
    chunk_doc = chunks["doc_id"].to_numpy()
    by_doc = chunks.groupby("doc_id").indices

    def build(split_qids):
        triplets = []
        for qid in tqdm(split_qids, desc="triplets"):
            query = queries[qid]
            scores = bm25.get_scores(tokenize(query))
            rel_set = set(positives[qid])
            # Hard negatives: top BM25 passages from cases NOT judged relevant.
            neg_pool, seen_docs = [], set()
            for i in np.argsort(-scores):
                d = chunk_doc[i]
                if d in rel_set or d in seen_docs:
                    continue
                seen_docs.add(d)
                neg_pool.append(chunks.at[i, "text"])
                if len(neg_pool) >= 30:
                    break
            for case_id in positives[qid]:
                rows = by_doc.get(case_id)
                if rows is None:
                    continue
                best_rows = sorted(rows, key=lambda i: -scores[i])[:args.positives_per_case]
                for r in best_rows:
                    for neg in rng.sample(neg_pool, min(args.negatives_per_positive, len(neg_pool))):
                        triplets.append({"query_id": qid, "query": query,
                                         "positive": chunks.at[r, "text"], "negative": neg})
        return triplets

    train_triplets = build(train)
    test_triplets = build(test)
    out = config.TRAINING_DIR / "training_pairs.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"train": train_triplets, "test": test_triplets,
                   "stats": {"train_queries": len(train), "test_queries": len(test),
                             "train_triplets": len(train_triplets), "test_triplets": len(test_triplets)}},
                  f, indent=1)
    print(f"\nTrain triplets: {len(train_triplets)} | Test triplets: {len(test_triplets)}")
    print(f"Saved to {out.relative_to(config.ROOT)}; split in {config.SPLIT_JSON.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
