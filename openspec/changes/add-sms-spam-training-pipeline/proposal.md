# Change: Add SMS spam training pipeline

## Why
- Establish a reproducible, lightweight training path for the bundled dataset so we can generate metrics and persisted artifacts before building the UI.
- Provide a single CLI entry point that configures preprocessing, model selection, and outputs for easy iteration.

## What Changes
- Add a training CLI that loads `sms_spam_no_header.csv`, cleans text, splits train/validation with a fixed seed, vectorizes with TF-IDF, and trains a linear/logistic classifier.
- Persist fitted model and vectorizer artifacts (joblib) plus metrics and a confusion-matrix plot under an artifacts directory chosen via CLI flags.
- Surface configurable parameters (output dir, test split size, regularization strength, random seed) through the CLI and document usage.

## Impact
- Affected specs: sms-spam-detection
- Affected code: training module/CLI entry point, artifact/plots output structure, preprocessing utilities, and corresponding tests/docs.
