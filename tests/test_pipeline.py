"""Fast unit tests (no models or indexes needed). Run: pytest -q"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant import detect_intent, query_terms  # noqa: E402
from evaluate import average_precision, evaluate_run, ndcg_at_k  # noqa: E402
from legal_text import (appellant_type, chunk_sentences, extract_citations,  # noqa: E402
                        extract_outcome, parse_case_header, split_sentences)
from preprocess import load_relevance  # noqa: E402
from tag_roles import sentence_roles, tag_chunk  # noqa: E402

SAMPLE = """Anil Kumar v State of U.P.
Supreme Court of India

13 February 2003
Appeal (Crl.) 139 of 1996
The Judgment was delivered by : S. N. Variava, J.
1.  This Appeal is against a Judgment dated 22nd November, 1994. He was convicted under Section 302 read with Section 34 IPC.
2.  We are of the view that the evidence of P.W. 2 is reliable. The Appeal stands dismissed.
Appeal dismissed
"""


def test_sentence_split_keeps_abbreviations():
    s = split_sentences("He relied on S. 302 of the I.P.C. and Cr.P.C. No. 5 v. State. The appeal failed.")
    assert len(s) == 2
    assert s[1] == "The appeal failed."


def test_split_drops_paragraph_numbers():
    s = split_sentences(SAMPLE)
    assert "This Appeal is against a Judgment dated 22nd November, 1994." in s
    assert not any(x.startswith("1.") for x in s)


def test_chunks_respect_word_limit():
    sentences = [f"Sentence number {i} has exactly seven words." for i in range(200)]
    chunks = chunk_sentences(sentences, max_words=50, overlap_sentences=2)
    assert all(len(c.split()) <= 50 for c in chunks)
    assert chunks[1].startswith(sentences[5])  # two-sentence overlap with chunk 0 (7 sentences)


def test_header_and_outcome():
    h = parse_case_header(SAMPLE)
    assert h == {"title": "Anil Kumar v State of U.P.", "court": "Supreme Court of India", "year": "2003"}
    assert extract_outcome(SAMPLE) == "Dismissed"
    assert extract_outcome("...\nAppeals partly allowed.") == "Partly allowed"
    assert extract_outcome("We, therefore, allow the appeal and set aside the conviction.") == "Allowed"
    assert extract_outcome("Order accordngly.") == "Disposed of"


def test_appellant_type():
    assert appellant_type("State of Punjab v Gurmit Singh") == "State/Government"
    assert appellant_type("Union Of India v Raj Kumar") == "State/Government"
    assert appellant_type("Anil Kumar v State of U.P.") == "Private party"


def test_citations():
    c = extract_citations("convicted under Section 302 read with Section 34 IPC; Article 21 of the Constitution")
    assert {"Section 302 IPC", "Section 34 IPC", "Article 21"} <= set(c)


def test_relevance_parser(tmp_path):
    f = tmp_path / "qrels.txt"
    f.write_text("AILA_Q1 Q0 C1 0\nAILA_Q1 Q0 C2 1\n")
    rows = load_relevance(f, "case_id")
    assert rows == [{"query_id": "AILA_Q1", "case_id": "C1", "relevant": 0},
                    {"query_id": "AILA_Q1", "case_id": "C2", "relevant": 1}]


def test_role_tagging():
    assert "RATIO" in sentence_roles("We are of the view that the conviction cannot be sustained.")
    assert "FAC" in sentence_roles("An FIR was lodged at the police station the same night.")
    assert "STA" not in sentence_roles("The fact remains that the witness turned hostile.")
    # outcome language in an early chunk of a long judgment is not the final ruling
    assert "RPC" not in tag_chunk("The appeal is dismissed by the High Court.", 0, 10, "case")
    assert "RPC" in tag_chunk("The appeal is dismissed.", 9, 10, "case")


def test_intent_uses_word_boundaries():
    assert detect_intent("contract of sale for a flat").names == []  # "act" inside "contract"
    assert "bail" in detect_intent("anticipatory bail under section 438").names
    assert "dowry / cruelty" in detect_intent("harassment for dowry, 498A").names
    assert "section" not in query_terms("what is the law")


def test_metrics():
    assert average_precision(["a", "x", "b"], {"a", "b"}) == (1 + 2 / 3) / 2
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 10) == 1.0
    r = evaluate_run({"q": ["x", "a"]}, {"q": {"a"}})
    assert r["MRR"] == 0.5 and r["R@10"] == 1.0
