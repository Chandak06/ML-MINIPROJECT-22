from pathlib import Path
import json
import pandas as pd
import streamlit as st
from src.predict import ReviewPredictor

def show_table(df):
    st.markdown(
        df.to_html(index=False, escape=True),
        unsafe_allow_html=True
    )

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Hotel Review ABSA", page_icon="🏨", layout="wide")
st.title("Hotel review sentiment analysis")
st.caption("Team 22 · Harshit Chandak and Chinmayee C M · UE24CS352A")
dataset = st.selectbox("Model dataset", ["Real reviews (ArguAna)", "Synthetic demonstration"])
mode = "real" if dataset.startswith("Real") else "synthetic"
if mode == "real":
    st.info("Models trained on published hotel reviews. Sleep-quality ratings are unavailable in this dataset. Predictions are estimates, not measured customer ratings.")
else:
    st.warning("DEMONSTRATION ONLY: synthetic template-based data demonstrates all seven aspects. Its scores are not real-world performance.")
review = st.text_area("Enter a hotel review", value="The staff were friendly and helpful. The room was dirty, but the location was close to the station.", height=140, max_chars=20000)
include_unmentioned = st.checkbox("Show estimates for aspects not mentioned", value=False)
st.caption("Rules identify aspect clauses and preserve negation. Validation chooses routed text or the full review separately for each target. Evidence shows the actual model input. Implicit aspects can still be missed.")


@st.cache_resource
def load_predictor(folder, manifest_version):
    return ReviewPredictor(folder)


if st.button("Predict aspects", type="primary"):
    try:
        folder = ROOT / "models" / mode
        version = (folder / "manifest.json").stat().st_mtime_ns if (folder / "manifest.json").exists() else 0
        predictor = load_predictor(str(folder), version)
        results = predictor.predict(review, include_unmentioned)
        rows = [{"Aspect": aspect, "Relevant evidence": value["evidence"] or "—",
                 "Rating": str(value["rating"]) if value["rating"] is not None else "—",
                 "Polarity": value["polarity"] or "—",
                 "Confidence": f"{value['confidence']:.1%}" if value["confidence"] is not None else "—",
                 "Status": ("unavailable in real dataset" if aspect == "Sleep Quality" and mode == "real" else value["status"].replace("_", " ")) + (" (estimate)" if value["estimate"] else "")}
                for aspect, value in results.items()]
        show_table(pd.DataFrame(rows))
        st.caption("Confidence estimates the predicted polarity, not exact stars. NB/LR probabilities are uncalibrated; SVM probabilities use development-only calibration. Below a validation-derived threshold, the app abstains as uncertain. Percentages are not guarantees.")
        with st.expander("Evidence sources and tentative candidates"):
            show_table(pd.DataFrame([{"Aspect": aspect, "Text strategy": v["routing_strategy"], "Evidence source": v["evidence_source"],
                                      "Relevant clauses": v["routed_aspect_text"] or "—", "Model": v["selected_model"] or "—",
                                      "Tentative rating": v["candidate_rating"] if v["status"] == "uncertain" else "—",
                                      "Fallback reason": v["fallback_reason"] or "—"} for aspect, v in results.items()]))
        st.download_button("Download predictions", json.dumps(results, indent=2), "predictions.json", mime="application/json")
    except (ValueError, FileNotFoundError) as exc:
        st.error(str(exc))
        st.info(f"For missing models, run: python run_project.py --dataset {mode}")

with st.expander("Evaluation and model limitations"):
    path = ROOT / "outputs" / mode / "metrics.csv"
    if path.exists():
        metrics = pd.read_csv(path)
        show_table(
    metrics[
        [
            "target_display",
            "test_n",
            "test_star_accuracy",
            "test_within_one_star_accuracy",
            "test_star_mae",
            "test_polarity_accuracy",
            "test_polarity_macro_f1",
            "baseline_polarity_accuracy"
        ]
    ]
)
    st.write("Model and text strategy are selected on validation macro-F1. Test metrics score every observed label before abstention, so uncertain or unmentioned rows cannot inflate accuracy. Hotels remain separate across real-data splits. Missing labels are excluded per target. See confidence_analysis.csv for coverage after abstention.")
