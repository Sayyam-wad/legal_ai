"""Fast sparse BM25 (Okapi) built on scipy.

`rank_bm25` scores every document in pure Python, which takes ~1-2 s per
query over 59k chunks. Here the per-term BM25 weights are precomputed into a
sparse matrix once, so a query is a single sparse column-slice + sum
(a few milliseconds), and the index can be saved next to the FAISS indexes.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from legal_text import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.vectorizer: CountVectorizer | None = None
        self.weights: sparse.csc_matrix | None = None  # docs x terms, BM25 term weights

    def fit(self, texts: list[str]) -> BM25Index:
        self.vectorizer = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                          dtype=np.float32)
        tf = self.vectorizer.fit_transform(texts).tocsr()          # docs x terms counts
        n_docs = tf.shape[0]
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avg_len = doc_len.mean() if n_docs else 0.0
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        # BM25 term weight: idf * tf*(k1+1) / (tf + k1*(1-b+b*len/avg))
        norm = self.k1 * (1 - self.b + self.b * doc_len / max(avg_len, 1e-9))
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        data = tf.data
        tf.data = (idf[tf.indices] * data * (self.k1 + 1) / (data + norm[rows])).astype(np.float32)
        self.weights = tf.tocsc()
        return self

    def scores(self, query: str) -> np.ndarray:
        vocab = self.vectorizer.vocabulary_
        cols = [vocab[t] for t in tokenize(query) if t in vocab]
        if not cols:
            return np.zeros(self.weights.shape[0], dtype=np.float32)
        # Repeated query terms count once per occurrence, as in Okapi BM25.
        return np.asarray(self.weights[:, cols].sum(axis=1)).ravel()

    def top_k(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        s = self.scores(query)
        k = min(k, len(s))
        idx = np.argpartition(-s, k - 1)[:k]
        idx = idx[np.argsort(-s[idx])]
        return idx, s[idx]

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"k1": self.k1, "b": self.b, "vectorizer": self.vectorizer,
                         "weights": self.weights}, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls(d["k1"], d["b"])
        obj.vectorizer, obj.weights = d["vectorizer"], d["weights"]
        return obj
