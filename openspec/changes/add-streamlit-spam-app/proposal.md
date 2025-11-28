# Change: Add Streamlit SMS spam demo app

## Why
- Provide an interactive way to visualize training results and test the spam classifier without writing code.
- Enable quick manual validation with canned examples and custom user input.

## What Changes
- Add a Streamlit app that loads trained artifacts, shows validation metrics with a chart/plot, and displays the confusion matrix image.
- Include a text input for ad-hoc predictions plus buttons to auto-fill spam/ham examples.
- Wire the app to use existing model/vectorizer artifacts and emit probability/confidence per prediction.

## Impact
- Affected specs: sms-spam-detection
- Affected code: new Streamlit app entrypoint, model loading/prediction utilities, requirements/docs.
