## 1. Implementation
- [x] 1.1 Scaffold training CLI entry point (argparse) that accepts data path, output dir, test split, regularization strength, and random seed.
- [x] 1.2 Implement preprocessing (lowercase, strip punctuation, collapse whitespace, optional stopword removal) and TF-IDF vectorization.
- [x] 1.3 Train a linear/logistic classifier with stratified train/validation split; log accuracy, precision, recall, and F1.
- [x] 1.4 Persist the fitted model and vectorizer with joblib; emit metrics report and confusion-matrix plot under the artifacts directory.
- [x] 1.5 Add smoke/unit tests for preprocessing and CLI wiring; document CLI usage and expected artifacts layout.
