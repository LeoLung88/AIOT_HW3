import os
from pathlib import Path

from sms_spam.inference import load_artifacts, predict_labels
from sms_spam.training import TrainingConfig, run_training


def write_dataset(path: Path) -> Path:
    rows = [
        ("ham", "Let's grab lunch soon."),
        ("spam", "Win money now!!!"),
        ("ham", "Are we still on for tonight?"),
        ("spam", "Claim your free prize today"),
    ]
    lines = [f'"{label}","{text}"\n' for label, text in rows]
    path.write_text("".join(lines), encoding="utf-8")
    return path


def test_inference_predicts_probabilities(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplcache"))
    data_file = write_dataset(tmp_path / "mini.csv")
    output_dir = tmp_path / "artifacts"

    config = TrainingConfig(
        data_path=data_file,
        output_dir=output_dir,
        test_size=0.5,
        random_state=123,
        regularization=1.0,
        max_iter=50,
    )

    run_training(config)

    artifacts = load_artifacts(
        model_path=output_dir / "model.joblib",
        vectorizer_path=output_dir / "vectorizer.joblib",
        metrics_path=output_dir / "metrics.json",
        plot_path=output_dir / "plots" / "confusion_matrix.png",
    )

    messages = ["Win cash now!!!", "Let's meet later"]
    labels, probs = predict_labels(messages, artifacts)

    assert labels[0] in {"spam", "ham"}
    assert labels[1] in {"spam", "ham"}
    assert 0.0 <= probs[0] <= 1.0
    assert 0.0 <= probs[1] <= 1.0
    assert probs[0] >= probs[1]
