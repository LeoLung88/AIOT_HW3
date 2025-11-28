# Project Context

## Purpose
- Detect SMS/text spam with a lightweight ML pipeline and surface results in an interactive Streamlit app.
- Use the bundled `sms_spam_no_header.csv` dataset, with reproducible preprocessing → training → evaluation steps.
- Produce a simple, local-first workflow for both training and inference with clear metrics and artifacts.

## Tech Stack
- Python 3.10+; recommended tooling via `pip`/`venv`.
- Data/ML: pandas, scikit-learn (TF-IDF + linear/logistic models), optional NLTK for text cleaning/stopwords, joblib for model persistence.
- Visualization/UI: Streamlit for the frontend; matplotlib or seaborn for training charts.

## Project Conventions

### Code Style
- Python: Black (PEP8, 88-120 columns acceptable), isort for imports, and type hints on public functions where practical.
- Modules and files in snake_case; functions/variables snake_case; classes PascalCase.
- Keep notebooks out of main flow; prefer scripts/modules for reproducibility.

### Architecture Patterns
- Baseline pipeline:
  - Load `ssms_spam_no_header.csv` → clean text (lowercase, strip punctuation, collapse whitespace, optional stopword removal) → split train/validation → vectorize (TF-IDF) → train linear/logistic classifier → persist model + vectorizer with joblib.
  - Track metrics: accuracy, precision, recall, F1; plot confusion matrix and PR/ROC if available.
- Artifacts: store trained model/vectorizer under `artifacts/` (e.g., `model.joblib`, `vectorizer.joblib`) and derived figures under `reports/` or `artifacts/plots/`.
- Streamlit app: load artifacts if present; display training metrics/plots; provide a text input to classify new messages and show probabilities/confidence.
- Keep config (paths, split ratios, model params, random seed) centralized in a small settings module or dataclass.

### Testing Strategy
- Unit tests with pytest for preprocessing utilities and model wrapper functions; include a tiny fixture dataset.
- Validate metrics on a held-out split; consider stratified 80/20 split by default.
- Add smoke tests for Streamlit helpers (e.g., model load, predict) to ensure app start-up without downloading extras.

### Git Workflow
- Branches: `main` for stable; feature branches `feature/<short-topic>`; fix branches `fix/<issue>`.
- Commits: short, imperative messages; keep related changes together (tests + code).
- Open PRs with a brief summary, checklist of tests run, and mention of any model artifact updates.

## Domain Context
- Binary SMS spam detection; dataset may be imbalanced—monitor recall on the spam class.
- Messages can include punctuation/emojis; cleaning should avoid stripping meaningful alphanumerics.

## Important Constraints
- Offline/local-first workflows only; no external data fetches needed beyond the provided CSV.
- Reproducibility: fix random seeds for splitting/training; log parameters used for each run.
- Keep training lightweight enough for laptops (TF-IDF + linear model preferred over heavy deep models).

## External Dependencies
- None beyond Python packages; NLTK stopwords download may be required once for preprocessing.
- Streamlit runs locally; no third-party APIs are expected.
