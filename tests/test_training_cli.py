from pathlib import Path

from sms_spam.training import TrainingConfig, run_training


def write_mini_dataset(path: Path) -> Path:
    rows = [
        ('ham',"Let's grab lunch soon."),
        ("spam", "Win money now!!!"),
        ("ham", "Are we still on for tonight?"),
        ("spam", "Claim your free prize today"),
    ]
    lines = [f'"{label}","{text}"\n' for label, text in rows]
    path.write_text("".join(lines), encoding="utf-8")
    return path


def test_run_training_creates_artifacts(tmp_path: Path):
    data_file = write_mini_dataset(tmp_path / "mini.csv")
    output_dir = tmp_path / "artifacts"

    config = TrainingConfig(
        data_path=data_file,
        output_dir=output_dir,
        test_size=0.5,
        random_state=123,
        regularization=1.0,
        max_iter=50,
    )

    result = run_training(config)

    assert result.model_path.exists()
    assert result.vectorizer_path.exists()
    assert result.metrics_path.exists()
    assert result.plot_path.exists()
    assert set(result.metrics.keys()) == {"accuracy", "precision", "recall", "f1"}
