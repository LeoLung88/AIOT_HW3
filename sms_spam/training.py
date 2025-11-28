import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from sms_spam.preprocess import clean_corpus

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    data_path: Path
    output_dir: Path
    test_size: float = 0.2
    random_state: int = 42
    regularization: float = 1.0
    max_iter: int = 200
    remove_stopwords: bool = False


@dataclass
class TrainingResult:
    metrics: Dict[str, float]
    model_path: Path
    vectorizer_path: Path
    metrics_path: Path
    plot_path: Path
    config: TrainingConfig


def validate_config(config: TrainingConfig) -> None:
    if not 0 < config.test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if config.regularization <= 0:
        raise ValueError("regularization must be positive")


def load_dataset(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path, names=["label", "text"])
    if "label" not in df or "text" not in df:
        raise ValueError("Dataset must have two columns: label,text")
    return df


def split_data(
    labels: Iterable[str],
    texts: Iterable[str],
    test_size: float,
    random_state: int,
) -> Tuple[Iterable[str], Iterable[str], Iterable[str], Iterable[str]]:
    return train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )


def vectorize_texts(
    train_texts: Iterable[str],
    val_texts: Iterable[str],
    remove_stopwords: bool,
) -> Tuple[TfidfVectorizer, Any, Any]:
    cleaned_train = clean_corpus(train_texts, remove_stopwords=remove_stopwords)
    cleaned_val = clean_corpus(val_texts, remove_stopwords=remove_stopwords)
    vectorizer = TfidfVectorizer()
    train_features = vectorizer.fit_transform(cleaned_train)
    val_features = vectorizer.transform(cleaned_val)
    return vectorizer, train_features, val_features


def train_classifier(
    features: Any,
    labels: Iterable[str],
    regularization: float,
    max_iter: int,
) -> LogisticRegression:
    classifier = LogisticRegression(
        C=regularization,
        max_iter=max_iter,
        class_weight="balanced",
        n_jobs=None,
        solver="liblinear",
    )
    classifier.fit(features, labels)
    return classifier


def compute_metrics(
    model: LogisticRegression,
    val_features: Any,
    y_val: Iterable[str],
) -> Tuple[Dict[str, float], list[str]]:
    predictions = model.predict(val_features)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, predictions, average="binary", pos_label="spam", zero_division=0
    )
    metrics = {
        "accuracy": accuracy_score(y_val, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return metrics, list(predictions)


def save_confusion_matrix(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    output_path: Path,
) -> Path:
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Validation Confusion Matrix")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def save_metrics(
    metrics: Dict[str, float],
    config: TrainingConfig,
    output_path: Path,
) -> Path:
    payload = {
        "metrics": metrics,
        "config": {
            **asdict(config),
            "data_path": str(config.data_path),
            "output_dir": str(config.output_dir),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path


def save_artifacts(
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    output_dir: Path,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    vectorizer_path = output_dir / "vectorizer.joblib"
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    return model_path, vectorizer_path


def run_training(config: TrainingConfig) -> TrainingResult:
    validate_config(config)
    logger.info("Loading dataset from %s", config.data_path)
    df = load_dataset(config.data_path)
    labels = df["label"].astype(str)
    texts = df["text"].astype(str)

    logger.info("Splitting dataset with test_size=%s, seed=%s", config.test_size, config.random_state)
    X_train_texts, X_val_texts, y_train, y_val = split_data(
        labels=labels,
        texts=texts,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    logger.info("Vectorizing %d training and %d validation samples", len(y_train), len(y_val))
    vectorizer, train_features, val_features = vectorize_texts(
        X_train_texts, X_val_texts, remove_stopwords=config.remove_stopwords
    )

    logger.info("Training logistic regression (C=%s, max_iter=%s)", config.regularization, config.max_iter)
    model = train_classifier(
        train_features,
        y_train,
        regularization=config.regularization,
        max_iter=config.max_iter,
    )

    metrics, predictions = compute_metrics(model, val_features, y_val)
    logger.info("Validation metrics: %s", metrics)

    model_path, vectorizer_path = save_artifacts(model, vectorizer, config.output_dir)
    metrics_path = save_metrics(metrics, config, config.output_dir / "metrics.json")
    plot_path = save_confusion_matrix(y_val, predictions, config.output_dir / "plots" / "confusion_matrix.png")

    return TrainingResult(
        metrics=metrics,
        model_path=model_path,
        vectorizer_path=vectorizer_path,
        metrics_path=metrics_path,
        plot_path=plot_path,
        config=config,
    )
