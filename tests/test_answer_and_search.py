"""Tests for grounded answers, hybrid fusion and sparse BM25 (no models needed)."""

import answer
from assistant import rrf_fuse
from lexical import BM25Index

CASES = [
    {"case_id": "C833", "title": "A v State", "year": "1999", "outcome": "Dismissed", "passage": "...",
     "key_passages": [{"text": "A dying declaration can form the sole basis of conviction if it is truthful",
                       "roles": ["RATIO"]}]},
    {"case_id": "C12", "title": "B v State", "year": "2004", "outcome": "Allowed", "passage": "...",
     "key_passages": [{"text": "The witness turned hostile during trial", "roles": []}]},
]


def test_validation_removes_hallucinated_citations():
    text = "A dying declaration alone can sustain a conviction [C833]. This was also held in [C9999]."
    cleaned, cited, removed, uncited = answer.validate_citations(text, {"C833", "C12"})
    assert removed == ["C9999"] and cited == ["C833"]
    assert "[C9999]" not in cleaned
    assert uncited == 1  # the second sentence lost its only citation


def test_extractive_answer_is_cited_and_grounded():
    a = answer.synthesize("dying declaration", CASES, provider="extractive")
    assert a.provider == "extractive"
    assert a.cited[0] == "C833" and a.grounded


def test_llm_path_with_mocked_api(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(answer, "_post_json", lambda url, payload, headers, timeout=60: {
        "content": [{"type": "text", "text": "Courts accept a reliable dying declaration [C833]. See also [C1]."}]})
    a = answer.synthesize("dying declaration", CASES, provider="anthropic")
    assert a.provider == "anthropic"
    assert a.removed_citations == ["C1"] and not a.grounded


def test_llm_failure_falls_back_to_extractive(monkeypatch):
    def boom(*a, **k):
        raise TimeoutError
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr(answer, "_post_json", boom)
    a = answer.synthesize("q", CASES, provider="openai")
    assert a.provider.startswith("extractive") and a.cited


def test_rrf_fuse_rewards_agreement():
    fused = rrf_fuse([[1, 2, 3], [3, 1]], weights=[1.0, 1.0])
    assert max(fused, key=fused.get) == 1
    assert fused[3] > fused[2]


def test_sparse_bm25_ranks_exact_terms(tmp_path):
    docs = ["the accused was convicted under section 302 for murder",
            "land acquisition compensation was enhanced",
            "a dying declaration recorded by the magistrate was relied upon"]
    bm25 = BM25Index().fit(docs)
    idx, scores = bm25.top_k("dying declaration", 2)
    assert idx[0] == 2 and scores[0] > 0
    bm25.save(tmp_path / "b.pkl")
    assert BM25Index.load(tmp_path / "b.pkl").top_k("section 302", 1)[0][0] == 0
