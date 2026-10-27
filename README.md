# AI Legal Assistant for Indian Case Law

Semantic search and decision support over 2,914 Indian Supreme Court judgments (AILA FIRE 2019), combining a fine-tuned InLegalBERT bi-encoder, rhetorical role-aware indexes, cross-encoder reranking and outcome analysis.

> Research aid only — not legal advice. The corpus is a fixed snapshot and does not include recent judgments.

---

## What it does

Ask in plain language ("Is a dying declaration alone enough to convict for murder?") or paste a fact situation. For each query the assistant returns:

- **The most relevant judgments**, with title, year and a relevance score from the cross-encoder.
- **Key passages** from each judgment, attributed to that case and tagged with their rhetorical role (ratio, final ruling, issue…).
- **Provisions cited** in each judgment (e.g. Section 302 IPC, Article 21).
- **Relevant statutes** from the AILA statute set.
- **Outcomes in similar cases** — the actual disposition of each retrieved judgment (allowed / partly allowed / dismissed / disposed of), read from the judgment itself, plus who the appellant was.

Use it from the command line, as a JSON API, or through the web UI.

---

## Pipeline

```
query
  └─ intent detection ─────────── which role indexes to emphasise (bail, homicide, dowry, NDPS, service law…)
  └─ dense retrieval ──────────── fine-tuned InLegalBERT over the ALL index + selected role indexes
  └─ chunk-level reranking ────── cross-encoder scores each candidate passage; a case = its best passage
  └─ explanation ──────────────── key sentences per case (query overlap + rhetorical role)
  └─ outcome summary ──────────── dispositions of the retrieved judgments
```

**Indexing:** judgments are split into sentence-aligned ~250-word chunks (fits BERT's 512-token window), every sentence is tagged with rhetorical roles, each chunk is encoded once, and FAISS indexes are built for all chunks and for each role.

| Component | Implementation |
|---|---|
| Bi-encoder | InLegalBERT fine-tuned on AILA 2019 ([Sayyam9/legal-bert-aila-finetuned](https://huggingface.co/Sayyam9/legal-bert-aila-finetuned)) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-12-v2` |
| Vector search | FAISS (L2, matching the triplet-loss training) |
| Role tagging | Rule-based, sentence-level, 12 roles, multi-label per chunk |
| Lexical baseline | BM25 (`rank-bm25`) |
| Dataset | AILA FIRE 2019 — 2,914 cases, 197 statutes, 50 queries |

---

## Setup

```bash
git clone https://github.com/Sayyam-wad/legal_ai.git
cd legal_ai
conda create -n legal_ai python=3.11 -y
conda activate legal_ai
pip install -r requirements.txt
```

The raw AILA files are already in `data/raw/`. To fetch them again: `python download_data.py` (needs a Kaggle API token).

### Build the indexes

```bash
python preprocess.py        # ~15 s  → cases/statutes/queries CSVs, chunks, outcomes, citations
python tag_roles.py         # ~1 min → rhetorical roles per chunk
python build_role_index.py  # encodes ~59k chunks once; ~10 min on a GPU, much longer on CPU
```

Embeddings are cached in `faiss_indexes/embeddings.npy`, so re-tagging and rebuilding does not re-encode.

### Run

```bash
python assistant.py                                   # interactive
python assistant.py -q "anticipatory bail in a dowry harassment case"
python assistant.py -q "..." --json                   # structured output
python assistant.py --no-rerank                       # faster, skips the cross-encoder

python app.py                                         # web UI at http://127.0.0.1:5000
```

API: `POST /api/ask` with `{"query": "...", "top_k": 5}`.

---

## Evaluation

`evaluate.py` scores retrieval on the official AILA Task 1 judgments (rank all 2,914 cases for each query):

```bash
python evaluate.py --methods bm25 dense hybrid rerank --split test
```

Metrics: MAP, MRR, P@10, R@10, R@100, nDCG@10. `hybrid` fuses BM25 and dense rankings with reciprocal-rank fusion.

BM25 baseline from this repo:

| Split | MAP | MRR | P@10 | R@100 |
|---|---|---|---|---|
| all 50 queries | 0.114 | 0.207 | 0.052 | 0.320 |
| 10 held-out test queries | 0.072 | 0.209 | 0.070 | 0.360 |

For context, the best AILA 2019 Task 1 systems scored around 0.15 MAP — this is a hard task with only 195 relevant pairs.

**Leakage warning.** A model fine-tuned on AILA queries must be evaluated on queries it never saw. `prepare_training_data.py` writes a fixed query-level split to `data/processed/query_split.json` (committed); fine-tune on `train`, evaluate with `--split test`. The published Hugging Face model may have been trained on all 50 queries, in which case its scores on any AILA query are optimistic.

---

## Fine-tuning (optional)

```bash
python prepare_training_data.py   # query-level split, BM25-mined positives + hard negatives
python finetune.py                # TripletLoss, reports held-out accuracy before/after
python build_role_index.py --model models/finetuned_legal_bert
python evaluate.py --split test
```

The held-out triplet accuracy uses *hard* negatives from *unseen* queries, so it is much lower than — and not comparable to — the earlier ~96.7% figure, which came from a random triplet split where test queries also appeared in training.

---

## Project structure

```
legal_ai/
├── config.py                 Paths, model names, chunk size, device selection
├── legal_text.py             Sentence splitting, chunking, case headers, outcomes, citations, BM25 tokens
├── preprocess.py             Raw AILA files → CSVs + chunks
├── tag_roles.py              Sentence-level rhetorical role tagging
├── build_role_index.py       Encode once → ALL + per-role FAISS indexes
├── assistant.py              LegalAssistant class + CLI
├── app.py                    Flask web UI + JSON API
├── evaluate.py               MAP / MRR / P@10 / Recall / nDCG on AILA
├── prepare_training_data.py  Triplets with query-level split and hard negatives
├── finetune.py               Bi-encoder fine-tuning
├── download_data.py          Kaggle download helper
├── tests/                    Unit tests (pytest -q)
└── data/
    ├── raw/                  AILA 2019 files
    ├── processed/            cases, statutes, queries, relevance judgments
    └── chunks/               generated, not committed
```

Settings can be overridden with environment variables: `LEGAL_AI_BI_ENCODER`, `LEGAL_AI_CROSS_ENCODER`, `LEGAL_AI_INDEX_DIR`, `LEGAL_AI_DEVICE`.

---

## Known limitations

- The reranker is a general-domain MS MARCO model; a legal cross-encoder fine-tuned on AILA pairs would likely help.
- Role tagging is rule-based (~57% of chunks get at least one role); a trained classifier on the LREC / SemEval rhetorical-role corpora would be more accurate.
- "Outcome" is the disposition of the appeal. "Allowed" means the appellant won — which is sometimes the State — so it is not the same as "relief for the accused".
- `data/processed/metadata.csv` and `ner_entities.csv` are not produced from the AILA files and are not used by the pipeline.

---

## References

- AILA FIRE 2019: Artificial Intelligence for Legal Assistance
- Paul et al., *Pre-trained Language Models for the Legal Domain: A Case Study on Indian Law* (InLegalBERT)
- Chalkidis et al., *LEGAL-BERT: The Muppets straight out of Law School*
- Kalamkar et al., *Corpus for Automatic Structuring of Legal Documents* (LREC 2022); SemEval-2023 Task 6

## Contributors

Vishwass Khattarr · Sayyam Wad · RaghuRumble · Saanvi09
