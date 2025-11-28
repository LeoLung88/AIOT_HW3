# SMS Spam Classifier

Train a lightweight SMS spam classifier on the bundled `sms_spam_no_header.csv` dataset and produce reproducible artifacts/metrics.

## Setup
- Python 3.10+ recommended.
- Install dependencies (ideally in a virtualenv):
  ```bash
  pip install -r requirements.txt
  ```

## Training
Run the CLI (defaults shown):
```bash
python train.py \
  --data sms_spam_no_header.csv \
  --output artifacts \
  --test-size 0.2 \
  --regularization 1.0 \
  --seed 42 \
  --max-iter 200 \
  [--remove-stopwords]
```

The dataset must have two columns without a header: `label,text` where labels are `ham`/`spam`.

Outputs (under `--output`, default `artifacts/`):
- `model.joblib` and `vectorizer.joblib` — fitted logistic regression and TF-IDF vectorizer.
- `metrics.json` — validation metrics plus the parameters used.
- `plots/confusion_matrix.png` — validation confusion matrix.

## Testing
```bash
pytest
```

## Streamlit app
After training, launch the demo UI:
```bash
streamlit run streamlit_app.py
```

Features:
- Displays validation metrics and the confusion matrix plot.
- Text input for ad-hoc predictions with spam probability.
- Buttons to auto-fill spam/ham example messages.
