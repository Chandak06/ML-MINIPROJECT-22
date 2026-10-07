from pathlib import Path
import json
import pandas as pd
import streamlit as st
from src.predict import ReviewPredictor

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Hotel Review ABSA", page_icon="🏨", layout="wide")
st.title("Hotel review sentiment analysis")
st.caption("Team 22 · Harshit Chandak and Chinmayee C M · UE24CS352A")
dataset = st.selectbox("Model dataset", ["Real reviews (ArguAna)", "Synthetic demonstration"])
mode = "real" if dataset.startswith("Real") else "synthetic"
if mode == "real":
    st.info("Models trained on published hotel reviews. Sleep-quality ratings are unavailable in this dataset. Predictions are estimates, not measured customer ratings.")
else:
    st.warning("Synthetic template-based data: use this mode to demonstrate all seven aspects, not to claim real-world accuracy.")
review = st.text_area("Enter a hotel review", value="The staff were friendly and helpful. The room was dirty, but the location was close to the station.", height=140, max_chars=20000)
include_unmentioned = st.checkbox("Show estimates for aspects not mentioned", value=False)
st.caption("The aspect mention filter uses keywords and can miss paraphrases. Ratings use the entire review; this is not a learned aspect-span extractor.")


@st.cache_resource
def load_predictor(folder, manifest_version):
    return ReviewPredictor(folder)


if st.button("Predict aspects", type="primary"):
    try:
        folder = ROOT / "models" / mode
        version = (folder / "manifest.json").stat().st_mtime_ns if (folder / "manifest.json").exists() else 0
        predictor = load_predictor(str(folder), version)
        results = predictor.predict(review, include_unmentioned)
        rows = [{"Aspect": aspect, "Rating": value["rating"], "Sentiment": value["polarity"] or "—", "Status": value["status"].replace("_", " ")} for aspect, value in results.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.download_button("Download predictions", json.dumps(results, indent=2), "predictions.json", mime="application/json")
    except (ValueError, FileNotFoundError) as exc:
        st.error(str(exc))
        st.info(f"For missing models, run: python run_project.py --dataset {mode}")

with st.expander("Evaluation and model limitations"):
    path = ROOT / "outputs" / mode / "metrics.csv"
    if path.exists():
        metrics = pd.read_csv(path)
        st.dataframe(metrics[["target_display", "test_n", "test_star_accuracy", "test_polarity_accuracy", "test_polarity_macro_f1", "baseline_polarity_accuracy"]], hide_index=True, width="stretch")
    st.write("Models are selected on validation macro-F1 and evaluated on a held-out test set. Real-data splits keep hotels separate. Missing labels are excluded per target. High polarity accuracy can coexist with poor neutral-class recall or imprecise star ratings.")
