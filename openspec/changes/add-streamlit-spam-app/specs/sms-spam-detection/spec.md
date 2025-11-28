## ADDED Requirements
### Requirement: Streamlit spam demo app
The system SHALL provide a Streamlit app that visualizes training results and allows interactive spam/ham predictions using trained artifacts.

#### Scenario: Show training results
- **WHEN** the app starts with access to `artifacts/metrics.json` and `artifacts/plots/confusion_matrix.png`
- **THEN** it displays validation metrics in a chart and renders the confusion-matrix image

#### Scenario: Ad-hoc prediction input
- **WHEN** a user enters text into the input bar
- **THEN** the app shows the predicted label (spam/ham) and the model's probability/confidence

#### Scenario: Example buttons
- **WHEN** a user clicks the "Spam example" or "Ham example" button
- **THEN** the input is populated with a representative spam/ham message that can be submitted for prediction
