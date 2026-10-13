# Verified upgrade report

## 1. Problems found

Every original aspect model consumed full-review text, allowing unrelated sentiment to affect predictions. Mention detection was separate from fitting and too narrow. No prediction probabilities, validation-derived uncertainty or abstention existed. Ordinal reporting lacked within-one-star accuracy. The user had changed Streamlit tables to escaped HTML, leaving one outdated dataframe assertion (41/42 tests passed before the upgrade). Original overall metrics and hotel separation were sound. Location minority performance and Business Service sample size were weak. Demo outputs and documentation needed regeneration whenever models changed.

## 2. Changes and files

Added a stateless router inside every training/inference pipeline, six candidates per target, validation-only strategy selection, probability methods, threshold fitting, abstention, transparent evidence and richer error analysis. Added hotel bootstrap, paired baselines, confidence coverage and explicit mixed-label diagnostics. The verifier now starts Streamlit and independently recomputes saved results, selection and thresholds. Training automatically exports the raw confidence evidence and current demos.

Created source/test/document files:

- `.gitattributes` (preserve fingerprinted Python/CSV bytes across Git platforms)
- `src/aspect_router.py`
- `src/confidence.py`
- `src/evaluation.py`
- `src/verification.py`
- `src/diagnostics.py`
- `tests/test_routing.py`
- `tests/test_artifacts.py`
- `docs/UPGRADE_REPORT.md`

Modified source/test/document files:

- `src/train.py`
- `src/predict.py`
- `run_project.py`
- `streamlit_app.py`
- `verify_project.py`
- `tests/test_project.py`
- `README.md`
- `DEMO_GUIDE.md`
- `docs/ABSA_Hotel_Reviews_Final_Writeup.pdf`
- `docs/ABSA_Hotel_Reviews_Final_Presentation.pptx`

Regenerated all seven real and eight synthetic model files, both manifests and corresponding evaluation artifacts. Added the meaningful CSV/JSON output families listed in README. A complete hash-based inventory is in `outputs/verification/change_inventory.json`. Dataset bytes, labels, dependency pins, preprocessing utility, data converter and contributions template remain unchanged. The original full-review models and results were preserved outside the final package in the workspace audit backup.

## 3. Architecture, routing, mentions and confidence

Old: normalize full review, vectorize, aspect classifier, stars/polarity, keyword display gate.

New: segment raw review and detect aspect evidence, select full-review/routed transformer inside the fitted pipeline, vectorize, classify, expose model probabilities, apply the frozen validation threshold and mention policy. Actual input evidence and extracted relevant text are separate fields. `whole_review_selected` differs from `whole_review_fallback`; fallback records the reason. `aspect_text` identifies successful routing. No model receives an inference-only preprocessing change.

Sentence punctuation and contrast/independent clauses separate opinions while shared noun lists remain together. Simple pronoun carry-over and spelling variants are supported. Negations remain in text/features. This is a deterministic heuristic without annotated mention accuracy. Lexicon and candidate settings were frozen before inspecting new held-out results.

NB/LR native probabilities are uncalibrated. SVM uses out-of-fold sigmoid calibration, grouping hotels when available. Threshold selection uses only training-fit validation correctness, maximizing balanced correctness-detection accuracy and preferring coverage on ties. Final refit uses development data only. Raw threshold rows and calibration memberships are exported. Refit probability shift and imperfect calibration limit uncertainty guarantees. Current real winners use NB/LR; selected synthetic SVMs and group-aware fixtures exercise calibration.

## 4. Old and new validation/test comparison

| Target | Old val. F1 | New val. F1 | Old test F1 | New test F1 |
| --- | --- | --- | --- | --- |
| Overall | 0.666 | 0.666 | 0.709 | 0.709 |
| Service | 0.635 | 0.635 | 0.628 | 0.628 |
| Rooms | 0.588 | 0.588 | 0.560 | 0.560 |
| Location | 0.411 | 0.413 | 0.423 | 0.383 |
| Value | 0.600 | 0.600 | 0.615 | 0.615 |
| Cleanliness | 0.504 | 0.567 | 0.539 | 0.566 |
| Business Service | 0.548 | 0.548 | 0.558 | 0.558 |

Both experiments use identical 1,342/388/370 hotel-disjoint partitions and seed 42. The newly fitted full-review baseline exactly reproduces the archived original scores. Test rows are never used in model fitting, vocabulary, IDF, strategy selection, threshold fitting or calibration. The original test results were already known, so this remains a descriptive comparison on a reused holdout.

## 5. Final real per-target metrics

| Target | n | Model / text | Star acc. | Within 1 | MAE | Polarity acc. | Macro-F1 | Majority polarity / star |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | 370 | NB / full_review | 51.1% | 92.2% | 0.608 | 74.6% | 0.709 | 32.4% / 14.9% |
| Service | 271 | LR / full_review | 44.6% | 84.5% | 0.764 | 68.3% | 0.628 | 45.0% / 22.1% |
| Rooms | 276 | LR / full_review | 39.1% | 80.8% | 0.873 | 60.9% | 0.560 | 47.8% / 28.3% |
| Location | 223 | LR / aspect_text | 42.2% | 83.0% | 0.807 | 72.2% | 0.383 | 78.9% / 43.5% |
| Value | 279 | NB / full_review | 46.2% | 85.3% | 0.710 | 62.4% | 0.615 | 37.3% / 20.4% |
| Cleanliness | 278 | LR / aspect_text | 43.9% | 81.7% | 0.784 | 65.1% | 0.566 | 61.5% / 33.5% |
| Business Service | 130 | LR / full_review | 38.5% | 76.9% | 0.877 | 55.4% | 0.558 | 39.2% / 28.5% |

All rows with observed test labels are included before abstention. Star macro-F1 and baseline MAE/within-one metrics are additionally in metrics.csv. No real Sleep Quality model is trained.

## 6. Per-class polarity evidence

| Target | Class | n | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| Overall | negative | 170 | 0.829 | 0.800 | 0.814 |
| Overall | neutral | 80 | 0.488 | 0.500 | 0.494 |
| Overall | positive | 120 | 0.806 | 0.833 | 0.820 |
| Service | negative | 98 | 0.793 | 0.745 | 0.768 |
| Service | neutral | 51 | 0.358 | 0.373 | 0.365 |
| Service | positive | 122 | 0.738 | 0.762 | 0.750 |
| Rooms | negative | 86 | 0.562 | 0.686 | 0.618 |
| Rooms | neutral | 58 | 0.380 | 0.328 | 0.352 |
| Rooms | positive | 132 | 0.744 | 0.682 | 0.711 |
| Location | negative | 20 | 0.333 | 0.100 | 0.154 |
| Location | neutral | 27 | 0.160 | 0.148 | 0.154 |
| Location | positive | 176 | 0.807 | 0.881 | 0.842 |
| Value | negative | 114 | 0.865 | 0.561 | 0.681 |
| Value | neutral | 61 | 0.339 | 0.607 | 0.435 |
| Value | positive | 104 | 0.760 | 0.702 | 0.730 |
| Cleanliness | negative | 56 | 0.560 | 0.500 | 0.528 |
| Cleanliness | neutral | 51 | 0.319 | 0.451 | 0.374 |
| Cleanliness | positive | 171 | 0.833 | 0.760 | 0.795 |
| Business Service | negative | 42 | 0.692 | 0.643 | 0.667 |
| Business Service | neutral | 37 | 0.400 | 0.649 | 0.495 |
| Business Service | positive | 51 | 0.677 | 0.412 | 0.512 |

Five-star class reports and both confusion matrices accompany every target. No class was removed because it was difficult. Balanced weights for LR/SVM remain unchanged.

## 7. Location investigation

Location has 794/221/223 observed training/validation/test labels. The new LR aspect-text strategy wins validation macro-F1 by only 0.002026 over the best full-review SVM. On test, accuracy falls 74.4% → 72.2% and macro-F1 0.423 → 0.383. The majority positive baseline reaches 78.9% accuracy. Test classes are 20 negative, 27 neutral and 176 positive. Negative recall falls 15.0% → 10.0%; neutral recall falls 18.5% → 14.8%. Positive recall is 88.1%.

The new polarity confusion matrix (rows actual negative/neutral/positive, columns predicted negative/neutral/positive) is saved in location_polarity_confusion_matrix.csv. Location has explicit routed evidence in 551/794 training rows and 150/223 test rows; other labelled rows fall back to the full review. Correct segmentation therefore cannot eliminate all contamination. The fixed terrible-location probe is wrongly positive even without the unrelated fantastic clause. This points to a learned lexical/class-distribution problem beyond segmentation, but the probe alone cannot establish causality. No rule or hyperparameter was changed to repair this test outcome.

The saved linear-model decomposition in `outputs/real/location_feature_probe.json` makes this feature problem concrete. For “The location was terrible.”, the star-5 intercept is +0.758 and the features `location` and `location was` add +0.431 and +0.182. The negative features `terrible` and `was terrible` subtract only 0.088 and 0.107 from that class score. The final star-5 score is +1.063 versus -0.976 for star 1. These are additive decision scores, not probabilities or a causal explanation of the dataset. The negative phrase is in vocabulary; its learned weight is insufficient to overcome the other contributions. This inspection occurred after evaluation and did not change the model.

## 8. Business Service investigation

Only 772 reviews provide labels (499 train, 143 validation, 130 test). Test support is 42 negative, 37 neutral and 51 positive. The unchanged balanced LR whole-review winner gives 55.4% accuracy and 0.558 macro-F1, compared with 39.2% majority polarity accuracy. Per-class F1 is 0.667/0.495/0.512. Positive recall is only 41.2%, while neutral recall is 64.9%.

Mention rules detect 171/499 labelled training reviews, of which 149 permit confident routed evidence. Bare desk mentions trigger uncertainty/fallback rather than a claim of business facilities. At test, 49/130 have mentions and 40/130 confident routed evidence. Review-level ratings often exist without an explicit business clause. Full-review retention follows validation evidence. No fabricated samples or post-test resampling were introduced. The business_service polarity confusion CSV records exact errors.

## 9. Cross-aspect contamination and uncertainty

On mixed-label rows (at least two observed aspect polarities disagree), Cleanliness macro-F1 improves 0.511 → 0.560, Location declines 0.412 → 0.322 and Rooms is unchanged at 0.460. Counts are 190, 162 and 189 respectively. This subset is descriptive and is not gold annotation of sentiment spans.

A fixed dirty-room counterfactual changes the old Cleanliness candidate from 1 to 5 when unrelated positive clauses are added; the new routed candidate stays 2. The Rooms candidate still changes 1 to 2 because validation retained full reviews. Invariance is not correctness: the isolated and augmented terrible-location sentences both receive an incorrect positive candidate. All five probes are saved, including failures. The central long example yields Cleanliness 1/negative but Rooms 4/positive. Contamination is reduced selectively, not solved globally.

Hotel-cluster bootstrap (500 seeded replicates) gives Cleanliness F1 change +0.027 with 95% interval [-0.064, +0.083], and Location -0.039 with [-0.122, +0.088]. Both intervals include zero. They measure uncertainty conditional on the fitted models/test hotels, not variation from retraining or a new population.

| Target | Threshold | Mean confidence | Accepted coverage | Accepted accuracy | UI coverage | UI accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| Overall | 0.548 | 0.598 | 61.6% | 88.2% | 61.6% | 88.2% |
| Service | 0.587 | 0.589 | 48.3% | 89.3% | 41.0% | 90.1% |
| Rooms | 0.549 | 0.558 | 50.0% | 79.7% | 44.6% | 78.9% |
| Location | 0.658 | 0.661 | 52.5% | 94.0% | 45.3% | 93.1% |
| Value | 0.515 | 0.513 | 52.0% | 84.1% | 26.2% | 86.3% |
| Cleanliness | 0.495 | 0.592 | 66.9% | 79.0% | 34.9% | 78.4% |
| Business Service | 0.495 | 0.438 | 36.2% | 80.9% | 13.8% | 88.9% |

Selective accuracy is reported only with coverage. It does not replace primary all-row scores. Business UI coverage is 13.8%; probabilities can be confident and wrong. Confidence correctness-Brier scores and mean probabilities are in confidence_analysis.csv. Native probability labels explicitly disclose lack of calibration.

## 10. All difficult examples

| Case | Review | Old candidates | New candidates/status | Interpretation |
| --- | --- | --- | --- | --- |
| 1 | The location was excellent and the staff were helpful, but the room was dirty. | Overall: 4 (predicted); Service: 5 (predicted); Rooms: 2 (predicted); Location: 5 (predicted); Cleanliness: 5 (predicted) | Overall: 4 (uncertain); Service: 5 (predicted); Rooms: 2 (uncertain); Location: 5 (predicted); Cleanliness: 2 (predicted) | Cleanliness receives the dirty-room clause and predicts negative. Rooms uses the full review and abstains. |
| 2 | The room was spotless, but the receptionist was extremely rude. | Overall: 1 (predicted); Service: 2 (predicted); Rooms: 2 (predicted); Cleanliness: 5 (predicted) | Overall: 1 (uncertain); Service: 2 (uncertain); Rooms: 2 (uncertain); Cleanliness: 5 (predicted) | Cleanliness predicts positive. Service preserves rude evidence but abstains. Rooms remains uncertain despite spotless. |
| 3 | The hotel was expensive but perfectly located near the station. | Overall: 4 (predicted); Location: 3 (predicted); Value: 4 (predicted) | Overall: 4 (predicted); Location: 4 (predicted); Value: 4 (predicted) | Location is positive. Value remains incorrectly positive for expensive. |
| 4 | The staff were friendly. | Overall: 5 (predicted); Service: 5 (predicted) | Overall: 5 (predicted); Service: 5 (predicted) | Only Service is mentioned among the six trained aspects. Other aspects are not_mentioned. |
| 5 | The room was not clean. | Overall: 1 (predicted); Rooms: 2 (predicted); Cleanliness: 1 (predicted) | Overall: 1 (uncertain); Rooms: 2 (uncertain); Cleanliness: 3 (uncertain) | The router preserves not clean, but Cleanliness has a tentative neutral candidate and abstains. |
| 6 | The room was not bad. | Overall: 3 (predicted); Rooms: 2 (predicted) | Overall: 3 (uncertain); Rooms: 2 (predicted) | Rooms incorrectly predicts negative for not bad. Preserved tokens do not solve compositional negation. |
| 7 | The location was terrible, but everything else about the hotel was fantastic. | Overall: 5 (predicted); Location: 5 (predicted) | Overall: 5 (uncertain); Location: 5 (predicted) | Location uses the isolated terrible-location clause yet incorrectly predicts positive. |
| 8 | The room was filthy although the staff, location and food were excellent. | Overall: 3 (predicted); Service: 4 (predicted); Rooms: 4 (predicted); Location: 5 (predicted); Cleanliness: 5 (predicted) | Overall: 3 (uncertain); Service: 4 (predicted); Rooms: 4 (predicted); Location: 5 (predicted); Cleanliness: 1 (predicted) | Cleanliness is negative; the full-review Rooms model remains incorrectly positive. |
| 9 | The location was excellent, the staff were helpful and the hotel was great value for money, but the room was extremely dirty. | Overall: 5 (predicted); Service: 5 (predicted); Rooms: 4 (predicted); Location: 5 (predicted); Value: 5 (predicted); Cleanliness: 5 (predicted) | Overall: 5 (predicted); Service: 5 (predicted); Rooms: 4 (predicted); Location: 5 (predicted); Value: 5 (predicted); Cleanliness: 1 (predicted) | Central case: Cleanliness is negative from extremely dirty. Rooms remains incorrectly positive with its full-review winner. |

These are fixed qualitative probes supplied before evaluation. Tests validate evidence routing independently from the model's learned sentiment. No function contains a review-specific correct-rating override. Uncertain candidate ratings are not displayed as definitive ratings.

## 11. Tests and reproducibility

The complete suite runs 78 tests with 0 failures, 0 errors and 0 skips. Original tests are retained. Two intentional changes preserve their purpose: delivered-model recomputation passes raw text because segmentation is now inside the pipeline; Streamlit assertions inspect escaped HTML table output because the user had already changed the presentation away from dataframes. The new suite covers routing, negation, evidence/fallback, mention statuses, uncertainty, real/synthetic Sleep Quality, model probabilities, group-aware calibration, bootstrap, candidate selection and saved-artifact consistency.

Saved checks recompute exact-star/polarity predictions and metrics, confirm all observed test rows remain, reproduce every candidate validation F1 and selected threshold, verify dataset/model/source hashes, constrain final vocabularies to development features and reject test/hotel overlap in calibration folds. A fixed-configuration training-only rebuild independently reproduced confidence thresholds. The export-only repeat reproduced primary metrics and test predictions without changing decisions.

One repeated real training run encountered an operating-system error while replacing location_model.joblib. It was not counted as successful; the identical run was repeated and its final artifacts verified. The final logs supersede intermediate failures. The final readiness command also checks pinned packages, actual CLI inference, a real Streamlit server health/application response, all tests and current document hashes.

## 12. Commands and readiness

```powershell
python run_project.py
python run_project.py --dataset real
python run_project.py --dataset synthetic
python run_project.py --review "The staff were helpful, but the room was dirty."
python -m streamlit run streamlit_app.py
python verify_project.py
```

Run from the project root with the pinned Python 3.11 environment. On this Windows machine use `.venv\Scripts\python.exe` if the environment is not active. Detailed commands and recovery steps are in README and DEMO_GUIDE. The final technical status is the actual `outputs/verification/verification_summary.json` result, generated after document review. Source, models, metrics, two-page PDF, editable presentation and tests are supplied. The ZIP contains one project folder and excludes the installed virtual environment.

## 13. Remaining limits and administrative actions

Unresolved errors include not-bad negation, expensive-value polarity, terrible-location polarity and positive Rooms candidates in dirty-room mixed reviews. Mention rules miss implicit aspects and complex reference; review-level labels are not clause labels. The small historical corpus, class imbalance, missing labels and reused split limit generalization. No food model, real Sleep Quality model, trained transformer or production-readiness claim is added. Future improvements require new labelled evidence or validation-only experimentation with a fresh final holdout.

CONTRIBUTIONS.md remains a truthful template for the team to confirm. Private GitHub publication, faculty/TA access and portal submission remain manual. No messages, repository permissions or submission actions were performed.
