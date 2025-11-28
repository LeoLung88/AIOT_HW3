from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from sms_spam.inference import InferenceArtifacts, load_artifacts, predict_labels
from sms_spam.preprocess import clean_text


ARTIFACT_DIR = Path("artifacts")
SPAM_EXAMPLE = "Congratulations! You have won a free prize. Click here to claim now!"
HAM_EXAMPLE = "Hey, just checking in. Are we still meeting later tonight?"


@st.cache_resource(show_spinner=False)
def get_artifacts() -> InferenceArtifacts:
    model_path = ARTIFACT_DIR / "model.joblib"
    vectorizer_path = ARTIFACT_DIR / "vectorizer.joblib"
    metrics_path = ARTIFACT_DIR / "metrics.json"
    plot_path = ARTIFACT_DIR / "plots" / "confusion_matrix.png"
    return load_artifacts(model_path, vectorizer_path, metrics_path, plot_path)


def render_metrics(artifacts: InferenceArtifacts) -> None:
    st.subheader("Validation metrics")
    if artifacts.metrics:
        df = pd.DataFrame(
            {"metric": list(artifacts.metrics.keys()), "value": list(artifacts.metrics.values())}
        ).set_index("metric")
        st.bar_chart(df)
    else:
        st.info("No metrics available.")

    if artifacts.plot_path:
        st.image(str(artifacts.plot_path), caption="Validation confusion matrix")
    else:
        st.info("Confusion matrix plot not found. Run training to generate it.")


def predict_and_store(text: str, artifacts: InferenceArtifacts) -> None:
    if not text.strip():
        st.session_state["prediction"] = None
        return
    labels, probs = predict_labels([text], artifacts)
    spam_prob = probs[0]
    st.session_state["prediction"] = {
        "label": labels[0],
        "spam_prob": spam_prob,
        "normalized": clean_text(text, remove_stopwords=artifacts.remove_stopwords),
    }


def render_prediction() -> None:
    result = st.session_state.get("prediction")
    if not result:
        return

    label = result["label"]
    spam_prob = result["spam_prob"]
    color = "#198754" if label == "ham" else "#dc3545"
    st.markdown(
        f"<div style='padding:0.75rem 1rem;background:{color}22;border:1px solid {color};"
        f"border-radius:6px;color:{color};font-weight:600;'>"
        f"Prediction: {label.upper()} | spam-prob = {spam_prob:.3f} (threshold = 0.50)"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Probability bar with threshold marker
    df = pd.DataFrame({"spam_prob": [spam_prob]})
    chart = (
        alt.Chart(df)
        .mark_bar(size=40, color="#4c78a8")
        .encode(
            x=alt.X("spam_prob:Q", scale=alt.Scale(domain=[0, 1]), title="spam probability"),
            y=alt.value(30),
        )
    )
    rule = alt.Chart(pd.DataFrame({"threshold": [0.5]})).mark_rule(strokeDash=[6, 3], color="black").encode(
        x="threshold:Q"
    )
    text = (
        alt.Chart(df)
        .mark_text(dy=-10, fontSize=14, color="#000")
        .encode(x="spam_prob:Q", text=alt.Text("spam_prob:Q", format=".2f"))
    )
    st.altair_chart((chart + rule + text).properties(height=120), use_container_width=True)

    with st.expander("Show normalized text"):
        st.code(result["normalized"])


def main() -> None:
    st.set_page_config(page_title="SMS Spam Classifier", page_icon="✉️")
    st.title("SMS Spam Classifier")
    st.write("Visualize training results and try the model on your own messages.")

    try:
        artifacts = get_artifacts()
    except FileNotFoundError:
        st.error("Artifacts not found. Run `python train.py --data sms_spam_no_header.csv --output artifacts` first.")
        return
    except Exception as exc:
        st.error(f"Failed to load artifacts: {exc}")
        return

    render_metrics(artifacts)

    st.subheader("Try a message")
    if "message_input" not in st.session_state:
        st.session_state["message_input"] = ""
    if "prediction" not in st.session_state:
        st.session_state["prediction"] = None

    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        spam_clicked = st.button("Spam example")
    with col3:
        ham_clicked = st.button("Ham example")

    if spam_clicked:
        st.session_state["message_input"] = SPAM_EXAMPLE
    if ham_clicked:
        st.session_state["message_input"] = HAM_EXAMPLE

    with col1:
        st.text_area("Message", key="message_input", height=150, placeholder="Type or paste an SMS to classify...")

    predict = st.button("Predict", type="primary")
    if predict:
        predict_and_store(st.session_state.get("message_input", ""), artifacts)

    render_prediction()


if __name__ == "__main__":
    main()
