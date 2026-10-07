# Demo Guide

Use the real-review mode for the evaluated project. Five sentences were fixed before prediction; all five outputs, including obvious mistakes, are retained in `outputs/demo_predictions.json`. These constructed inputs have no formal ground-truth labels and are not an accuracy benchmark.

## 1. Verify Project

From the final project root, with the Python 3.11 environment active:

```bash
python verify_project.py
```

The script must finish with `FINAL STATUS: READY FOR SUBMISSION`. Inspect every failed check if it does not. Logs are saved under `outputs/verification/`. This verifies local technical readiness, not faculty access or university submission.

## 2. CLI Prediction

```bash
python run_project.py --review "The room was clean and spacious, but the staff was rude and the breakfast was disappointing."
```

Actual saved-model output for this sentence: Overall 4/positive, Service 4/positive, Rooms 3/neutral, Cleanliness 4/positive. Other trained aspects are `not_mentioned`; Sleep Quality is `unavailable`.

**Explain the failure:** the Service prediction contradicts the explicit negative phrase about rude staff. The neutral Rooms score also misses the positive description. Breakfast/food has no model in this implementation. Do not imply that this example was classified perfectly, or add a fabricated food rating.

## 3. Streamlit Demo

```bash
python -m streamlit run streamlit_app.py --browser.gatherUsageStats false
```

Open `http://localhost:8501`. Choose **Real reviews (ArguAna)**, enter a sentence and click **Predict aspects**. Leave **Show estimates for aspects not mentioned** unchecked at first. Explain the rating, polarity and status columns. Open the evaluation expander and show the held-out metrics. Enter `!!!` to demonstrate input rejection. Predictions load existing models and never retrain them.

## 4. Five pre-tested sentences

| # | Sentence | Actual output and what to explain |
|---|---|---|
| 1 | The room was clean and spacious, but the staff was rude and the breakfast was disappointing. | Overall 4, Service 4, Rooms 3, Cleanliness 4. Service is an obvious error; food is unsupported. |
| 2 | The staff were friendly and helpful. The room was dirty, but the location was close to the station. | Overall 5, Service 5, Rooms 4, Location 5, Cleanliness 5. Cleanliness is an obvious error despite the word dirty. Mixed opinions are difficult. |
| 3 | The room was noisy all night and I could not sleep. | Overall 2 and Rooms 2, both negative. Sleep Quality stays unavailable because there are no real labels. |
| 4 | Beautiful view. | Overall 5/positive. All six trained aspects are not mentioned by the keyword filter; Sleep Quality is unavailable. This illustrates limited aspect coverage. |
| 5 | The wifi was unusable and the conference room was dirty. | Overall 1, Rooms 1, Cleanliness 1 and Business Service 2, all negative. Other aspects abstain. |

The numbers above are predictions, not hand-assigned correct labels. The automated suite checks that these documented outputs match the included models. Outputs can change after legitimate retraining or dependency changes; regenerate the evidence before updating any claims.

## 5. What to Explain

- Standard sentiment analysis yields an overall label. This project adds separately trained review-level aspect-rating models; it does not learn aspect spans.
- The real corpus has 2,100 reviews from 122 hotels. Overall and six aspect labels are supported; real sleep-quality supervision is absent.
- Normalization preserves negation. Count or TF-IDF unigrams/bigrams feed NB, LR or Linear SVM. Vocabulary and IDF are fitted only on each training partition.
- Hotel-disjoint splitting prevents the same hotel appearing in train and test: 1,342/388/370 reviews, seed 42. Incomplete supplied hotel IDs are rejected.
- Validation polarity macro-F1 selects each pipeline. It gives negative, neutral and positive F1 equal weight. Final models use train plus validation, never test reviews.
- Overall polarity accuracy is 74.6%; exact star accuracy is 51.1%. Location accuracy is below its majority baseline. Show error examples as well as aggregate scores.
- Whole-review features can mix sentiments; implicit aspects and unseen vocabulary remain difficult. The keyword mention filter is not an evaluated learned detector.

## 6. Optional synthetic demonstration

The original synthetic mode remains available to preserve working functionality and regression tests. Clearly identify it before use. It demonstrates all seven aspects but cannot establish real-world accuracy. The final report and presentation are based on the real dataset; do not present synthetic scores as the main result.

## 7. Demo recovery

- Missing packages: run `python -m pip install -r requirements.txt` in the correct environment.
- Missing or incompatible models: first restore the included files; otherwise run `python run_project.py` with the pinned versions. Do not download arbitrary joblib files.
- Port 8501 occupied: add `--server.port 8502` and open the corresponding local URL.
- Internet unavailable: installed dependencies, included CSV files and saved models support offline demos. `prepare_data.py` downloads only when the source archive is missing.
- A wrong prediction is not a software crash. Show it honestly and discuss the limitation.

## 8. Before submission

Complete `CONTRIBUTIONS.md` truthfully; both verified names/SRNs are already filled. Publish only this final folder to a private GitHub repository, add the actual faculty/TA collaborators, verify access and submit its URL on the university portal. These are manual actions and have not been claimed as completed.
