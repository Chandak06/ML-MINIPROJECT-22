from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate hotel ABSA or predict without retraining.")
    parser.add_argument("--dataset", choices=["real", "synthetic"], default="real")
    parser.add_argument("--data", type=Path, help="Custom CSV; use separate --out and --models directories")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--review", help="Predict with saved models; does not load a dataset or train")
    parser.add_argument("--include-unmentioned", action="store_true", help="Return model estimates for all trained aspects")
    args = parser.parse_args()
    models = args.models or ROOT / "models" / args.dataset
    try:
        if args.review is not None:
            from src.predict import predict_review
            print(json.dumps(predict_review(args.review, models, args.include_unmentioned), indent=2))
            return
        from src.train import train_and_evaluate
        from src.utils import load_dataset
        data = args.data or ROOT / "data" / ("hotel_reviews_real.csv" if args.dataset == "real" else "hotel_reviews_sample.csv")
        output = args.out or ROOT / "outputs" / args.dataset
        if not data.exists():
            parser.error(f"Dataset missing: {data}. Run python prepare_data.py first.")
        metrics = train_and_evaluate(load_dataset(data), output, models, args.seed, data.name)
        print(f"\nCompleted {len(metrics)} targets. Results: {output / 'metrics.csv'}")
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
