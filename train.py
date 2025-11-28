import argparse
import logging
from pathlib import Path
from typing import Sequence

from sms_spam.training import TrainingConfig, run_training


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an SMS spam classifier.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("sms_spam_no_header.csv"),
        help="Path to the SMS spam CSV (no header, label,text).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts"),
        help="Directory to store model, vectorizer, metrics, and plots.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to use for validation (0-1).",
    )
    parser.add_argument(
        "-C",
        "--regularization",
        type=float,
        default=1.0,
        help="Inverse regularization strength for logistic regression.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting and model initialization.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=200,
        help="Maximum iterations for logistic regression convergence.",
    )
    parser.add_argument(
        "--remove-stopwords",
        action="store_true",
        help="Apply English stopword removal during preprocessing.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = TrainingConfig(
        data_path=args.data,
        output_dir=args.output,
        test_size=args.test_size,
        random_state=args.seed,
        regularization=args.regularization,
        max_iter=args.max_iter,
        remove_stopwords=args.remove_stopwords,
    )

    result = run_training(config)
    logging.info("Artifacts saved: model=%s, vectorizer=%s", result.model_path, result.vectorizer_path)
    logging.info("Metrics written to %s", result.metrics_path)
    logging.info("Confusion matrix plot written to %s", result.plot_path)
    logging.info("Validation metrics: %s", result.metrics)


if __name__ == "__main__":
    main()
