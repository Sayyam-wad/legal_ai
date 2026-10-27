"""Build FAISS indexes over the tagged chunks.

Every chunk is encoded exactly once. The ALL index holds every chunk; each
role index holds the subset of chunks carrying that role and a positions file
mapping its rows back to the ALL metadata. (The old version encoded the whole
corpus twice: once split across role indexes and again for ALL.)

Embeddings are cached in faiss_indexes/embeddings.npy together with the model
name, so re-running after changing the role tagger does not re-encode.

Run: python build_role_index.py [--model NAME_OR_PATH] [--batch-size 64]
"""

import argparse
import json

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import config


def encode(model_name: str, texts: list[str], batch_size: int, cache_dir) -> np.ndarray:
    emb_path = cache_dir / "embeddings.npy"
    info_path = cache_dir / "embeddings.json"
    if emb_path.exists() and info_path.exists():
        info = json.loads(info_path.read_text())
        if info.get("model") == model_name and info.get("count") == len(texts):
            print(f"Reusing cached embeddings ({len(texts)} x {info['dim']}) from {emb_path.name}")
            return np.load(emb_path)

    device = config.get_device()
    print(f"Loading {model_name} on {device}...")
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = min(model.max_seq_length or 512, 512)
    print(f"Encoding {len(texts)} chunks (max_seq_length={model.max_seq_length})...")
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                       convert_to_numpy=True).astype("float32")
    np.save(emb_path, emb)
    info_path.write_text(json.dumps({"model": model_name, "count": len(texts), "dim": int(emb.shape[1])}))
    return emb


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=config.BI_ENCODER)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    df = pd.read_csv(config.TAGGED_CSV, keep_default_na=False)
    print(f"Tagged chunks: {len(df)}")
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    emb = encode(args.model, df["text"].astype(str).tolist(), args.batch_size, config.INDEX_DIR)
    dim = emb.shape[1]

    # Fine-tuning used TripletLoss (Euclidean distance), so L2 search matches training.
    full = faiss.IndexFlatL2(dim)
    full.add(emb)
    faiss.write_index(full, str(config.INDEX_DIR / f"index_{config.ALL_INDEX}.bin"))
    df.drop(columns=["role"], errors="ignore").to_csv(
        config.INDEX_DIR / f"meta_{config.ALL_INDEX}.csv", index=False)

    role_sets = df["roles"].astype(str).str.split("|")
    summary = {config.ALL_INDEX: len(df)}
    for role in config.ROLES:
        positions = np.flatnonzero(role_sets.apply(lambda rs: role in rs).to_numpy())
        if len(positions) == 0:
            print(f"  skip {role}: no chunks")
            continue
        index = faiss.IndexFlatL2(dim)
        index.add(emb[positions])
        faiss.write_index(index, str(config.INDEX_DIR / f"index_{role}.bin"))
        np.save(config.INDEX_DIR / f"positions_{role}.npy", positions.astype("int64"))
        summary[role] = int(len(positions))

    (config.INDEX_DIR / "manifest.json").write_text(json.dumps(
        {"model": args.model, "dim": int(dim), "indexes": summary}, indent=2))

    print("\n══════════ INDEX SUMMARY ══════════")
    for role, n in summary.items():
        print(f"  {role:<16} {n:>7} chunks")
    print(f"\nSaved to {config.INDEX_DIR}/")


if __name__ == "__main__":
    main()
