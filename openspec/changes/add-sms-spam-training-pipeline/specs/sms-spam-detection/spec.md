## ADDED Requirements
### Requirement: Trainable SMS spam classifier pipeline
The system SHALL provide a command-line training pipeline for the bundled SMS spam dataset that outputs reproducible artifacts and metrics.

#### Scenario: Default training run
- **WHEN** a user runs `python train.py --data sms_spam_no_header.csv --output artifacts/` with default parameters
- **THEN** the run loads the dataset, cleans text (lowercase, punctuation stripped, whitespace collapsed), and performs a stratified train/validation split with a fixed seed
- **AND** it vectorizes text with TF-IDF and trains a linear/logistic classifier
- **AND** it reports accuracy, precision, recall, and F1 for the validation split
- **AND** it saves the fitted model and vectorizer to the output directory in joblib format
- **AND** it writes a metrics report and a confusion-matrix plot under the output directory (a `plots/` subfolder is acceptable)

#### Scenario: Configurable parameters
- **WHEN** a user passes CLI flags for output dir, test split size, regularization strength, and random seed
- **THEN** the training run honors those values and records them in the metrics/report output

#### Scenario: Reproducible runs
- **WHEN** a user reruns training with the same dataset, parameters, and seed
- **THEN** the split and resulting metrics remain stable within floating-point tolerance, and the run logs the seed/parameters used
