import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from sms_spam.preprocess import clean_text


@dataclass
class InferenceArtifacts:
    model: LogisticRegression
    vectorizer: TfidfVectorizer
    metrics: dict
    remove_stopwords: bool = False
    plot_path: Path | None = None


def load_artifacts(
    model_path: Path,
    vectorizer_path: Path,
    metrics_path: Path,
    plot_path: Path | None = None,
) -> InferenceArtifacts:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    metrics_payload = json.loads(metrics_path.read_text())
    metrics = metrics_payload.get("metrics", metrics_payload)
    config = metrics_payload.get("config", {})
    remove_stopwords = bool(config.get("remove_stopwords", False))
    return InferenceArtifacts(
        model=model,
        vectorizer=vectorizer,
        metrics=metrics,
        remove_stopwords=remove_stopwords,
        plot_path=plot_path if plot_path and plot_path.exists() else None,
    )


def predict_labels(
    texts: Iterable[str],
    artifacts: InferenceArtifacts,
) -> Tuple[list[str], list[float]]:
    cleaned = [
        clean_text(text or "", remove_stopwords=artifacts.remove_stopwords)
        for text in texts
    ]
    features = artifacts.vectorizer.transform(cleaned)
    proba = artifacts.model.predict_proba(features)
    classes = list(artifacts.model.classes_)
    spam_idx = classes.index("spam")
    spam_probs = proba[:, spam_idx]
    predictions = ["spam" if p >= 0.5 else "ham" for p in spam_probs]
    return predictions, spam_probs.tolist()
