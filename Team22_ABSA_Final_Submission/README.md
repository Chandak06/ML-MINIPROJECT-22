# Aspect-Based Sentiment Analysis on Hotel Reviews

**Team 22 | UE24CS352A Machine Learning | Project 1**  
Harshit Chandak (PES2UG24CS185) and Chinmayee C M (PES2UG24CS133)

This classical ML mini-project predicts overall and aspect-specific hotel ratings (1–5), then maps 1–2 to negative, 3 to neutral and 4–5 to positive. The upgraded version compares full-review and routed-clause features, shows the actual evidence used, and abstains when confidence falls below a validation-derived threshold. Cross-aspect contamination is reduced in some cases but remains unresolved for others.

## Setup and exact commands

From this project directory on Windows, use Python 3.11 and the pinned dependencies:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe verify_project.py
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open http://localhost:8501. On Linux/macOS substitute `.venv/bin/python`. The following commands assume the environment is active, so `python` resolves to its interpreter:

```powershell
python run_project.py
python run_project.py --dataset real
python run_project.py --dataset synthetic
python run_project.py --review "The staff were helpful, but the room was dirty."
python run_project.py --include-unmentioned --review "The staff were friendly."
python -m streamlit run streamlit_app.py --browser.gatherUsageStats false
python -m unittest discover -s tests -v
python verify_project.py
```

The first two training commands are equivalent. Prediction loads saved models without training. Training regenerates metrics, validation-confidence evidence, manifests and real demo predictions. The included datasets/models allow offline prediction after dependency setup. `python prepare_data.py` recreates the real CSV from the published archive and verifies its checksum. The ZIP excludes the local virtual environment.

## Data and provenance

| Data | Reviews | Purpose | Supported labels |
|---|---:|---|---|
| ArguAna TripAdvisor annotated v2 | 2,100 from 122 hotels | Primary evaluation | Overall and six aspects |
| Supplied synthetic sample | 320 | DEMONSTRATION ONLY | Overall and seven aspects |

The [ArguAna corpus](https://zenodo.org/records/3973241) accompanies Wachsmuth et al. (2014), *A Review Corpus for Argumentation Analysis*. Its usage terms allow scientific use, not commercial applications. Title and review text come from XMI Sofa text, overall rating from `Sentiment.score` (never hotel category), and aspect ratings from the `Aspects` fields. The conversion omits author metadata. Source -1 stays missing. Provenance records source and CSV checksums.

Observed labels: Overall 2,100; Service 1,608; Rooms 1,647; Location 1,238; Value 1,631; Cleanliness 1,651; Business Service 772. **Real Sleep Quality has zero usable labels and remains unavailable.** Missing labels are excluded per target, never filled from predictions. The real corpus is historical, English and balanced by overall sentiment. Synthetic templates have no supplied generator/provenance and are never combined with primary evaluation.

## Architecture and routing

Previously, every aspect classifier consumed the whole normalized review. Now a stateless transformer sits inside each candidate pipeline:

```text
Review → sentence/clause segmentation → aspect mentions and evidence
       → selected full_review OR aspect_text transformer
       → training-fitted count/TF-IDF features → classifier → stars/polarity
       → probabilities + validation threshold → UI/CLI status
```

`src/aspect_router.py` splits sentence punctuation and useful contrast/independent clauses. It preserves shared noun lists, multiple aspect assignments and useful negation including contractions. Rules support staff/reception/check-in, room/bathroom/AC, location/station, price/value, dirty/spotless and wifi/conference variants. “Room service” and “front desk” have contextual exceptions. Restrained pronoun carry-over helps simple adjacent clauses. The vocabulary is fixed from the task before evaluating new test results; the router learns no corpus parameters.

Both candidate training and inference use the same serialized transformer. Confident routing uses `aspect_text`. Ambiguous routing or a labelled row without explicit mention uses `whole_review_fallback` with a reason. A validation-selected full-review model honestly reports `whole_review_selected`; it does not pretend to use only the displayed relevant clause. Overall always uses the whole review. `routed_aspect_text` exposes the extracted clause separately from `evidence`, the actual model input.

Mention detection is a transparent heuristic, not a trained span detector. Coverage on rated rows and fixed routing tests are supplied; no mention precision/recall is claimed without gold mention annotations.

## Confidence, abstention and UI

NB/LR expose native, **uncalibrated** probabilities. Confidence sums class probabilities for the polarity of the base model's predicted rating; `rating_confidence` is the exact-star probability. SVM uses sigmoid calibration on out-of-fold development predictions, with hotel-aware folds when hotels exist. Calibration never uses test labels. See the [scikit-learn calibration method](https://scikit-learn.org/stable/modules/calibration.html). Current real winners are NB/LR; calibrated SVM is used by selected synthetic models and tested with hotel-group fixtures.

For each selected model, a training-only fit predicts validation rows. The confidence threshold maximizes balanced accuracy for distinguishing correct from incorrect validation polarity predictions, with ties favoring coverage. It is frozen before refitting on train + validation. Refit changes probability distributions, so thresholds are useful heuristics, not guaranteed risk control. Calibration fold memberships and raw validation confidence/correctness rows are saved for audit.

Statuses: `predicted`, `uncertain`, `not_mentioned`, `unavailable`, plus the preserved `unknown_vocabulary` state. Uncertain outputs have null `rating`/`polarity`; tentative `candidate_rating`/`candidate_polarity` remain in details. Empty, punctuation-only and excessive input is rejected. Sleep Quality remains unavailable even with estimates enabled. Unmentioned estimates are off by default and explicitly labelled when requested.

The Streamlit table shows Aspect, Relevant evidence, Rating, Polarity, Confidence and Status. Expand details to see selected strategy, relevant clauses, actual input source and tentative estimates. The existing escaped HTML table presentation is preserved. Confidence is model output, not an invented correctness percentage.

## Evaluation protocol

Seed 42 preserves the original hotel-disjoint **1,342 train / 388 validation / 370 test** split. Synthetic data have separate 192/64/64 review partitions. Candidate features and IDF learn only from training rows. Per target, compare three unchanged families across both text strategies (six candidates):

| Family | Features | Fixed settings |
|---|---|---|
| Multinomial NB | TF-IDF unigrams/bigrams | alpha 0.5 |
| Linear SVM | Count unigrams/bigrams | C 0.1, balanced classes |
| Logistic Regression | TF-IDF unigrams/bigrams | C 4, balanced classes |

Maximum vocabulary is 12,000. Validation polarity macro-F1 selects independently per target, then star accuracy, then full review and model name for deterministic ties. Overall's routing strategies are intentionally identical. Selected models refit on train + validation before test evaluation. Model/strategy/threshold selection completes before any new test scoring. Existing historical test scores were known, so the comparison is descriptive on a reused holdout, not a fresh confirmatory benchmark.

Every observed test label is scored **before mention gating or confidence abstention**. Accuracy, per-class precision/recall/F1, macro-F1, confusion matrices, exact stars, star MAE and within-one-star accuracy are retained/exported. Macro-F1 weights classes equally; MAE respects distance between stars; within-one counts absolute error ≤1. Majority star and majority polarity baselines are independently fitted on development labels. Classification reports cover all five star classes and all three polarity classes.

## Real held-out results

| Target | n | Model / text | Star acc. | Within 1 | MAE | Polarity acc. | Macro-F1 | Majority polarity / star |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | 370 | NB / full_review | 51.1% | 92.2% | 0.608 | 74.6% | 0.709 | 32.4% / 14.9% |
| Service | 271 | LR / full_review | 44.6% | 84.5% | 0.764 | 68.3% | 0.628 | 45.0% / 22.1% |
| Rooms | 276 | LR / full_review | 39.1% | 80.8% | 0.873 | 60.9% | 0.560 | 47.8% / 28.3% |
| Location | 223 | LR / aspect_text | 42.2% | 83.0% | 0.807 | 72.2% | 0.383 | 78.9% / 43.5% |
| Value | 279 | NB / full_review | 46.2% | 85.3% | 0.710 | 62.4% | 0.615 | 37.3% / 20.4% |
| Cleanliness | 278 | LR / aspect_text | 43.9% | 81.7% | 0.784 | 65.1% | 0.566 | 61.5% / 33.5% |
| Business Service | 130 | LR / full_review | 38.5% | 76.9% | 0.877 | 55.4% | 0.558 | 39.2% / 28.5% |

NB = Naive Bayes; LR = Logistic Regression. Baselines in the last column show polarity and exact-star accuracy respectively. No real Sleep Quality model exists.

| Target | Old val. F1 | New val. F1 | Old test F1 | New test F1 |
| --- | --- | --- | --- | --- |
| Overall | 0.666 | 0.666 | 0.709 | 0.709 |
| Service | 0.635 | 0.635 | 0.628 | 0.628 |
| Rooms | 0.588 | 0.588 | 0.560 | 0.560 |
| Location | 0.411 | 0.413 | 0.423 | 0.383 |
| Value | 0.600 | 0.600 | 0.615 | 0.615 |
| Cleanliness | 0.504 | 0.567 | 0.539 | 0.566 |
| Business Service | 0.548 | 0.548 | 0.558 | 0.558 |

Cleanliness polarity macro-F1 improves by 0.027, but exact-star accuracy and MAE worsen. Location's tiny validation gain (0.002) does not generalize: polarity accuracy falls from 74.4% to 72.2%, below the 78.9% majority baseline. All five full-review targets retain identical held-out predictions. Strategy selection has not been reversed using test results.

Hotel-level paired bootstrap uses 500 seeded replicates, resampling hotels with all their reviews. The 95% interval for Cleanliness macro-F1 change is [-0.064, +0.083]; Location is [-0.122, +0.088]. Both include zero. These conditional test intervals exclude model-fitting variability. Legacy Wilson intervals remain available but ignore hotel clustering.

## Class imbalance and mixed-aspect errors

Location has 176 positive, 27 neutral and 20 negative test ratings. New negative recall is 10.0%, neutral recall 14.8%, positive recall 88.1%. Positive dominance makes aggregate accuracy misleading. Routed training rows still need full-review fallback for 243/794 labelled Location reviews. A short isolated “location was terrible” remains wrongly positive, exposing a learned feature/model limitation even when routing is correct.

Business Service has only 772 labels (499/143/130). Its balanced LR full-review model beats the majority polarity baseline (55.4% versus 39.2%), but class F1 remains 0.667 negative, 0.495 neutral and 0.512 positive. Only 149/499 training rows have confident explicit routed evidence, making aggressive routing a poor substitute for review-level supervision. No resampling or fabricated labels were added.

On the descriptive subset where observed aspect polarities disagree, Cleanliness macro-F1 rises 0.511 → 0.560 (190 rows), Location falls 0.412 → 0.322 (162 rows), and Rooms stays 0.460 (189 rows). These labels are a mixed-opinion proxy, not annotated spans or a causal contamination measure. The fixed nine diagnostics include both improvements and remaining errors. See `DEMO_GUIDE.md` and `docs/UPGRADE_REPORT.md`.

## Selective prediction results

| Target | Threshold | Mean confidence | Accepted coverage | Accepted accuracy | UI coverage | UI accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| Overall | 0.548 | 0.598 | 61.6% | 88.2% | 61.6% | 88.2% |
| Service | 0.587 | 0.589 | 48.3% | 89.3% | 41.0% | 90.1% |
| Rooms | 0.549 | 0.558 | 50.0% | 79.7% | 44.6% | 78.9% |
| Location | 0.658 | 0.661 | 52.5% | 94.0% | 45.3% | 93.1% |
| Value | 0.515 | 0.513 | 52.0% | 84.1% | 26.2% | 86.3% |
| Cleanliness | 0.495 | 0.592 | 66.9% | 79.0% | 34.9% | 78.4% |
| Business Service | 0.495 | 0.438 | 36.2% | 80.9% | 13.8% | 88.9% |

Acceptance coverage considers the confidence threshold alone. UI coverage also requires an explicit mention and known vocabulary. Each denominator is all labelled test rows for that target. Selective accuracy must always be read together with coverage; it must never replace the all-row metrics above. Business Service's 13.8% UI coverage is a substantial limitation.

## Verification and reproducibility

**78 tests pass, with 0 failures, 0 errors and 0 skips.** The original 42 tests are preserved, with intentional interface corrections for raw text entering the new pipeline and HTML Streamlit tables. New tests distinguish deterministic routing/negation behavior from learned sentiment behavior. They check uncertainty, fallback, calibration, split isolation, validation selection, vocabulary, model integrity, CLI and Streamlit interactions in both modes. Diagnostic prediction snapshots are compared to actual saved models without asserting fabricated correct sentiments.

`verify_project.py` checks pinned packages and `pip check`, dataset counts/provenance, no hotel overlap, all 15 models, schemas, actual CLI inference, real Streamlit startup/health, recomputed saved predictions/metrics/probabilities, candidate selection, validation thresholds, fitted vocabulary bounds, calibration-fold isolation, current PDF/PPT hashes, document/metric consistency and the complete test suite. It returns nonzero on failure and writes evidence under `outputs/verification/`.

Full real and synthetic training was run, and the export-only repeat reproduced the evaluated results. The previous whole-review baseline exactly matches its archived metrics on the unchanged split. Raw selection, calibration and prediction evidence make these checks reviewable. Passing tests establishes software behavior and result consistency; it does not prove every English review receives the right sentiment.

`.gitattributes` preserves Python/CSV bytes across Git checkouts so source and dataset integrity fingerprints stay valid on Windows and Linux.

## Deliverables and output map

| File or group | Purpose |
|---|---|
| `docs/ABSA_Hotel_Reviews_Final_Writeup.pdf` | Two-page summary of verified results |
| `docs/ABSA_Hotel_Reviews_Final_Presentation.pptx` | Editable presentation with actual results and limitations |
| `docs/UPGRADE_REPORT.md`, `DEMO_GUIDE.md` | Detailed audit, all nine hard examples, commands and viva guidance |
| `metrics.csv`, `system_comparison.csv` | New metrics and paired old/new results |
| `validation_model_comparison.csv`, `validation_predictions.csv` | All six candidates and validation predictions |
| `validation_confidence_predictions.csv`, `confidence_analysis.csv` | Threshold reproduction, coverage and selective accuracy |
| `test_predictions.csv`, `error_examples.csv` | Every test prediction and errors with evidence/confidence |
| `per_class_metrics.csv`, `*_classification_report.*`, `*_confusion_matrix.csv` | Class precision/recall/F1 and star/polarity confusion matrices |
| `label_distribution.csv`, `aspect_routing_evaluation.csv`, `mention_detection_examples.csv` | Label support, rule coverage and example mentions |
| `hotel_bootstrap_metrics.csv`, `mixed_aspect_evaluation.csv` | Hotel uncertainty and mixed-label subset comparisons |
| `split_manifest.csv`, `selection_manifest.json`, `run_metadata.json`, `calibration_folds.csv` | Split/selection/fitting provenance |
| `difficult_review_comparison.json`, `counterfactual_diagnostics.csv` | All hard examples and controlled added-clause probes |
| `outputs/verification/` | Actual tests, reproducibility and final document/package audit |

CSV names above are under `outputs/real/`; synthetic outputs stay in `outputs/synthetic/`. Manifests include source hashes, routing version, model strategy, calibrated/native confidence method, row counts, seed, model hashes and fitted review-ID digest. Joblib files must come from a trusted source; integrity hashes detect corruption, not publisher authenticity.

## Custom data and remaining limits

```powershell
python run_project.py --data path/to/reviews.csv --out outputs/custom --models models/custom
python run_project.py --models models/custom --review "The staff were helpful."
```

Use the columns in `src/config.py`; ratings must be integer 1–5 or missing, with overall observed. Require at least 30 distinct reviews and five hotels when hotel IDs exist. Partial hotel IDs are rejected. Each trainable target needs at least five training and five validation labels and two training classes. A target with no test labels is explicitly unevaluated. Use distinct custom directories to retain the reference deliverables. Retraining after changing code/results requires refreshing and reviewing the PDF/PPT audit before a new submission-readiness claim.

Rules miss implicit aspects, complex coreference, sarcasm and unsupported food opinions. Negation tokens are preserved, but “not bad” can still receive a negative learned prediction. Whole-review winners remain vulnerable to cross-aspect contamination. Routed models retain fallback rows and noisy review-level labels. Native probabilities need not be calibrated, and abstention does not guarantee correctness. One historical corpus and one reused split limit generalization. A transformer, more real labels, improved mention annotation and development-only cross-validation are future work only; no transformer results or production-readiness claims are made.

## Individual Contributions

Detailed contribution information is available in `CONTRIBUTIONS.md`. Verified names/SRNs are prefilled. Each member must fill and confirm their actual responsibilities and files; no contribution history has been invented.

## Repository Access

This project must be submitted through a **PRIVATE GitHub repository**.

Before final submission:
1. Push this final folder to the private repository.
2. Add the required faculty/TA GitHub accounts as collaborators.
3. Confirm that repository access works.
4. Submit the repository URL through the official submission portal.

These are manual user actions. No repository URL, collaborator access or GitHub history is claimed. Required datasets and model files remain included and are not ignored; the final size audit checks that no file exceeds GitHub's 100 MiB file-size limit.

## Assignment requirements and remaining administrative steps

The guidelines explicitly ask for a two-page summary, despite the section heading saying one-page; the PDF follows the two-page instruction. Source, PDF, slides, live demo and testing evidence are supplied. The review period is 5-9 October 2026; the submission deadline is 10 October 2026 at 11:59 PM India time.

The mandatory **private GitHub repository and faculty/TA access are not completed locally**. Publish this project folder in a private repository, invite the actual faculty/TA accounts, verify access, and submit the repository link. Do not upload the entire surrounding semester repository. Record truthful individual contributions; none have been fabricated. No authenticated GitHub CLI session was available during preparation.

## Reference project

[Yangyang Yu, Aspect-based Sentiment Analysis on Hotel Reviews, Stanford CS229 (2016)](https://cs229.stanford.edu/proj2016spr/report/032.pdf) and its [poster](https://cs229.stanford.edu/proj2016spr/poster/032.pdf) motivate the task. This implementation uses a smaller different corpus, different partitions and logistic regression; it does not reproduce the paper's reported scores.
