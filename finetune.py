"""Fine-tune InLegalBERT as a bi-encoder on AILA triplets.

Trains only on the training queries from prepare_training_data.py and reports
triplet accuracy on the held-out test queries, so the number reflects
generalisation to unseen situations rather than memorised ones.

Run: python finetune.py [--epochs 4] [--batch-size 8]
Then: python build_role_index.py --model models/finetuned_legal_bert
      python evaluate.py --split test --methods bm25 dense hybrid
"""

import argparse
import json
import math
import random

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
try:  # sentence-transformers >= 6
    from sentence_transformers.sentence_transformer.evaluation import TripletEvaluator
except ImportError:
    from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader

import config


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default=config.BASE_MODEL)
    p.add_argument("--output", default=str(config.FINETUNED_DIR))
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8, help="8 fits in 8GB of VRAM")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--max-seq-length", type=int, default=384)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = json.loads((config.TRAINING_DIR / "training_pairs.json").read_text(encoding="utf-8"))
    train, test = data["train"], data["test"]
    print(f"Train triplets: {len(train)} | held-out test triplets: {len(test)}")

    device = config.get_device()
    model = SentenceTransformer(args.base_model, device=device)
    model.max_seq_length = args.max_seq_length
    print(f"Loaded {args.base_model} on {device}")

    examples = [InputExample(texts=[t["query"], t["positive"], t["negative"]]) for t in train]
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    # Euclidean TripletLoss: build_role_index.py uses L2 search to match.
    loss = losses.TripletLoss(model=model)
    evaluator = TripletEvaluator(
        anchors=[t["query"] for t in test],
        positives=[t["positive"] for t in test],
        negatives=[t["negative"] for t in test],
        name="aila_heldout",
    )

    print(f"Held-out triplet accuracy BEFORE fine-tuning: {_score(evaluator(model)):.4f}")
    steps = len(loader) * args.epochs
    model.fit(
        train_objectives=[(loader, loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=math.ceil(steps * args.warmup_ratio),
        optimizer_params={"lr": args.lr},
        evaluation_steps=len(loader),
        output_path=args.output,
        save_best_model=True,
        show_progress_bar=True,
    )
    best = SentenceTransformer(args.output, device=device)
    print(f"Held-out triplet accuracy AFTER fine-tuning:  {_score(evaluator(best)):.4f}")
    print(f"\nSaved to {args.output}")
    print(f"Next: python build_role_index.py --model {args.output}")


def _score(result) -> float:
    """TripletEvaluator returns a float (old versions) or a dict (v3+)."""
    if isinstance(result, dict):
        return float(next((v for k, v in result.items() if "accuracy" in k), next(iter(result.values()))))
    return float(result)


if __name__ == "__main__":
    main()
