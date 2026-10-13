# Demo Guide

Use **Real reviews (ArguAna)** for the evaluated project. All examples below are actual saved-model outputs, including mistakes. These constructed sentences are qualitative diagnostics, not an independent accuracy benchmark. No review-specific prediction override exists.

## Exact commands

From the final project directory with the Python 3.11 environment active:

```powershell
python verify_project.py
python run_project.py --review "The location was excellent, the staff were helpful and the hotel was great value for money, but the room was extremely dirty."
python -m streamlit run streamlit_app.py --browser.gatherUsageStats false
```

Open http://localhost:8501. Choose real mode, paste a review and click **Predict aspects**. Keep **Show estimates for aspects not mentioned** unchecked. Explain evidence, stars, polarity, confidence and status. Open details to distinguish actual model input from routed evidence. Open evaluation results to show all-row metrics. Enter `!!!` to demonstrate input rejection. Prediction never retrains models.

The verifier must report READY only after every required check passes. Its readiness scope is the local technical package; faculty access, portal submission and truthful contribution confirmation remain manual.

## Nine requested hard cases

Numbers on uncertain rows are **tentative candidates only**. The main UI/JSON rating is blank/null when it abstains. Sleep Quality is unavailable in all real cases. Unlisted aspects are not mentioned unless the explicit estimate option is enabled.

| Case | Review | Actual candidates and statuses | Discussion |
| --- | --- | --- | --- |
| 1 | The location was excellent and the staff were helpful, but the room was dirty. | Overall: 4 (uncertain); Service: 5 (predicted); Rooms: 2 (uncertain); Location: 5 (predicted); Cleanliness: 2 (predicted) | Cleanliness receives the dirty-room clause and predicts negative. Rooms uses the full review and abstains. |
| 2 | The room was spotless, but the receptionist was extremely rude. | Overall: 1 (uncertain); Service: 2 (uncertain); Rooms: 2 (uncertain); Cleanliness: 5 (predicted) | Cleanliness predicts positive. Service preserves rude evidence but abstains. Rooms remains uncertain despite spotless. |
| 3 | The hotel was expensive but perfectly located near the station. | Overall: 4 (predicted); Location: 4 (predicted); Value: 4 (predicted) | Location is positive. Value remains incorrectly positive for expensive. |
| 4 | The staff were friendly. | Overall: 5 (predicted); Service: 5 (predicted) | Only Service is mentioned among the six trained aspects. Other aspects are not_mentioned. |
| 5 | The room was not clean. | Overall: 1 (uncertain); Rooms: 2 (uncertain); Cleanliness: 3 (uncertain) | The router preserves not clean, but Cleanliness has a tentative neutral candidate and abstains. |
| 6 | The room was not bad. | Overall: 3 (uncertain); Rooms: 2 (predicted) | Rooms incorrectly predicts negative for not bad. Preserved tokens do not solve compositional negation. |
| 7 | The location was terrible, but everything else about the hotel was fantastic. | Overall: 5 (uncertain); Location: 5 (predicted) | Location uses the isolated terrible-location clause yet incorrectly predicts positive. |
| 8 | The room was filthy although the staff, location and food were excellent. | Overall: 3 (uncertain); Service: 4 (predicted); Rooms: 4 (predicted); Location: 5 (predicted); Cleanliness: 1 (predicted) | Cleanliness is negative; the full-review Rooms model remains incorrectly positive. |
| 9 | The location was excellent, the staff were helpful and the hotel was great value for money, but the room was extremely dirty. | Overall: 5 (predicted); Service: 5 (predicted); Rooms: 4 (predicted); Location: 5 (predicted); Value: 5 (predicted); Cleanliness: 1 (predicted) | Central case: Cleanliness is negative from extremely dirty. Rooms remains incorrectly positive with its full-review winner. |

## Old versus new examples

The legacy system forced a rating for mentioned aspects without confidence abstention. Candidate comparisons below separate a changed model prediction from a new abstention. Full evidence and numeric confidence are in `outputs/real/difficult_review_comparison.json`.

| Case | Old outputs | New outputs |
| --- | --- | --- |
| 1 | Overall: 4 (predicted); Service: 5 (predicted); Rooms: 2 (predicted); Location: 5 (predicted); Cleanliness: 5 (predicted) | Overall: 4 (uncertain); Service: 5 (predicted); Rooms: 2 (uncertain); Location: 5 (predicted); Cleanliness: 2 (predicted) |
| 2 | Overall: 1 (predicted); Service: 2 (predicted); Rooms: 2 (predicted); Cleanliness: 5 (predicted) | Overall: 1 (uncertain); Service: 2 (uncertain); Rooms: 2 (uncertain); Cleanliness: 5 (predicted) |
| 3 | Overall: 4 (predicted); Location: 3 (predicted); Value: 4 (predicted) | Overall: 4 (predicted); Location: 4 (predicted); Value: 4 (predicted) |
| 4 | Overall: 5 (predicted); Service: 5 (predicted) | Overall: 5 (predicted); Service: 5 (predicted) |
| 5 | Overall: 1 (predicted); Rooms: 2 (predicted); Cleanliness: 1 (predicted) | Overall: 1 (uncertain); Rooms: 2 (uncertain); Cleanliness: 3 (uncertain) |
| 6 | Overall: 3 (predicted); Rooms: 2 (predicted) | Overall: 3 (uncertain); Rooms: 2 (predicted) |
| 7 | Overall: 5 (predicted); Location: 5 (predicted) | Overall: 5 (uncertain); Location: 5 (predicted) |
| 8 | Overall: 3 (predicted); Service: 4 (predicted); Rooms: 4 (predicted); Location: 5 (predicted); Cleanliness: 5 (predicted) | Overall: 3 (uncertain); Service: 4 (predicted); Rooms: 4 (predicted); Location: 5 (predicted); Cleanliness: 1 (predicted) |
| 9 | Overall: 5 (predicted); Service: 5 (predicted); Rooms: 4 (predicted); Location: 5 (predicted); Value: 5 (predicted); Cleanliness: 5 (predicted) | Overall: 5 (predicted); Service: 5 (predicted); Rooms: 4 (predicted); Location: 5 (predicted); Value: 5 (predicted); Cleanliness: 1 (predicted) |

## Original five demonstration inputs

The original inputs remain unchanged. Their outputs have been regenerated from current saved models.

| Review | Actual candidates and statuses |
| --- | --- |
| The room was clean and spacious, but the staff was rude and the breakfast was disappointing. | Overall: 4 (uncertain); Service: 4 (uncertain); Rooms: 3 (uncertain); Cleanliness: 4 (predicted) |
| The staff were friendly and helpful. The room was dirty, but the location was close to the station. | Overall: 5 (predicted); Service: 5 (predicted); Rooms: 4 (uncertain); Location: 5 (predicted); Cleanliness: 2 (predicted) |
| The room was noisy all night and I could not sleep. | Overall: 2 (predicted); Rooms: 2 (predicted) |
| Beautiful view. | Overall: 5 (predicted) |
| The wifi was unusable and the conference room was dirty. | Overall: 1 (uncertain); Rooms: 1 (uncertain); Cleanliness: 2 (predicted); Business Service: 2 (uncertain) |

## What the demonstration establishes

Routing tests check clause membership, multiple aspects, variants, fallback and negation preservation. Controlled training fixtures test known simple positive/negative examples. Real diagnostic snapshots test reproducibility, not perfect natural-language understanding. Correct routing does not guarantee correct sentiment, as the terrible-location and not-bad examples show.

All nine cases remain visible. The long central example improves Cleanliness but not Rooms. On held-out mixed-label rows, Cleanliness macro-F1 improves from 0.511 to 0.560, Location declines from 0.412 to 0.322 and Rooms stays at 0.460. Do not claim contamination is fully solved from one example.

## Viva explanation

- The real corpus contains 2,100 reviews from 122 hotels. Missing aspect labels stay missing, including all real Sleep Quality labels.
- The 1,342/388/370 split keeps hotels separate. Candidate vocabulary/IDF learn from training only. Selection uses validation macro-F1 and all final models exclude test rows.
- Six candidates per target compare three classical families with whole-review and aspect-routed text. Location and Cleanliness select routing; other real targets retain whole reviews.
- Polarity macro-F1 treats negative, neutral and positive classes equally. Exact stars, within-one-star accuracy and MAE give different views of ordinal errors.
- Confidence comes from model probabilities. Validation chooses abstention thresholds. The 74.6% overall test accuracy includes every scored row and must not be replaced with selective accuracy.
- Location is below its majority accuracy baseline and has poor minority recall. Business Service has fewer labels and only 13.8% UI prediction coverage on labelled test rows.
- Synthetic mode is DEMONSTRATION ONLY. It remains separate and demonstrates Sleep Quality without inventing real labels.

## Recovery and handoff

Install missing dependencies with `python -m pip install -r requirements.txt`. Restore trusted saved models or retrain with `python run_project.py`; never use arbitrary downloaded joblib files. If port 8501 is occupied, add `--server.port 8502`. The included CSV/models work offline after setup. The UI can show a wrong prediction without a software failure.

Fill `CONTRIBUTIONS.md` truthfully. Push this project folder to a private GitHub repository, invite the actual faculty/TA accounts, verify access and submit its URL. No repository access or portal submission has been claimed as completed.
