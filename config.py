"""Shared configuration for the legal_ai pipeline.

Every script imports paths and model names from here, so changing a model or
a folder only needs to happen in one place. Values can be overridden with
environment variables (useful on Colab / servers).
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── Data paths ────────────────────────────────────────────────────────────────
RAW_DIR = Path(os.getenv("LEGAL_AI_RAW_DIR", ROOT / "data" / "raw"))
PROCESSED_DIR = ROOT / "data" / "processed"
CHUNKS_DIR = ROOT / "data" / "chunks"
STATS_DIR = ROOT / "data" / "stats"
TRAINING_DIR = ROOT / "data" / "training"
INDEX_DIR = Path(os.getenv("LEGAL_AI_INDEX_DIR", ROOT / "faiss_indexes"))

CASES_CSV = PROCESSED_DIR / "cases.csv"
STATUTES_CSV = PROCESSED_DIR / "statutes.csv"
QUERIES_CSV = PROCESSED_DIR / "queries.csv"
REL_CASES_CSV = PROCESSED_DIR / "relevance_cases.csv"
REL_STATUTES_CSV = PROCESSED_DIR / "relevance_statutes.csv"
CHUNKS_CSV = CHUNKS_DIR / "chunks.csv"
TAGGED_CSV = CHUNKS_DIR / "chunks_tagged.csv"
SPLIT_JSON = PROCESSED_DIR / "query_split.json"  # committed, so everyone evaluates on the same queries

# ── Models ────────────────────────────────────────────────────────────────────
BI_ENCODER = os.getenv("LEGAL_AI_BI_ENCODER", "Sayyam9/legal-bert-aila-finetuned")
BASE_MODEL = os.getenv("LEGAL_AI_BASE_MODEL", "law-ai/InLegalBERT")
CROSS_ENCODER = os.getenv("LEGAL_AI_CROSS_ENCODER", "cross-encoder/ms-marco-MiniLM-L-12-v2")
FINETUNED_DIR = ROOT / "models" / "finetuned_legal_bert"

# ── Chunking ──────────────────────────────────────────────────────────────────
# BERT models see at most 512 *tokens*. Legal English runs ~1.2-1.3 WordPiece
# tokens per word, so 512-word chunks (~600-670 tokens) had their last
# 15-25% silently dropped before embedding. ~250 words (~300 tokens) fits.
CHUNK_WORDS = 250
CHUNK_OVERLAP_SENTENCES = 2

# ── Retrieval ─────────────────────────────────────────────────────────────────
ROLES = ["RPC", "RATIO", "FAC", "STA", "ISSUE", "PRE_RELIED", "PRE_NOT_RELIED",
         "RLC", "ARG_PETITIONER", "ARG_RESPONDENT", "ANALYSIS", "PREAMBLE"]
ALL_INDEX = "ALL"


def get_device() -> str:
    """Use the GPU when one is available, otherwise the CPU."""
    forced = os.getenv("LEGAL_AI_DEVICE")
    if forced:
        return forced
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
