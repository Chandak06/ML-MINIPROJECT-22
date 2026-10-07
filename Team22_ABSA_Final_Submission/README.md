# Aspect-Based Sentiment Analysis on Hotel Reviews

**UE24CS352A Machine Learning | Team 22 | Project 1**  
PES2UG24CS185 Harshit Chandak; PES2UG24CS133 Chinmayee C M

A working supervised text-classification project with real-data evaluation, a clearly separated synthetic demo, saved models, a Streamlit interface and automated verification.

## Problem Statement

Traditional sentiment analysis assigns one label to the whole review. A guest can praise the location while criticizing service or cleanliness, so an overall label can hide useful detail. This implementation predicts overall ratings and separate ratings for service, rooms, location, value, cleanliness, sleep quality and business service. The real corpus supports all of these except sleep quality. Food/breakfast and general amenities are not separate trained targets. Aspect mentions are detected by an explicit keyword heuristic; sentiment/rating classification uses trained whole-review text models, not a learned span extractor.

## Project Structure

```text
Team22_ABSA_Final_Submission/
  README.md, CONTRIBUTIONS.md, DEMO_GUIDE.md
  requirements.txt, setup.cmd, start_demo.cmd, .gitignore
  run_project.py          Training and prediction CLI
  prepare_data.py         Reproducible corpus conversion
  streamlit_app.py        Interactive demonstration
  verify_project.py      Full local submission verification
  src/                   Production model/preprocessing modules
  data/                  Required CSVs and real-data provenance
  models/real/           Evaluated real-data models and manifest
  models/synthetic/      Optional demo models used by existing tests
  tests/                 Original tests plus audit regression tests
  outputs/real/          Final real-data evaluation evidence
  outputs/synthetic/     Minimal evidence required by retained regression tests
  outputs/verification/  Actual verification and audit results
  outputs/demo_predictions.json   All five documented demo outputs
  docs/                  One final two-page PDF and one final slide deck
  .github/workflows/     Verification workflow for the future repository
```

No virtual environment, cached source archive, nested project, old report or old slide deck belongs in the final ZIP. Synthetic data/models are retained only because the working optional demo and existing regression tests use them; obsolete synthetic-only documents are excluded.

## Start here

Use Python 3.11. From this project directory:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe verify_project.py
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open http://localhost:8501. On macOS/Linux use `.venv/bin/python` in place of `.venv\Scripts\python.exe`. Activating the environment is optional. The remaining examples assume `python` refers to this environment. Exact tested package versions are pinned; dependencies need internet on first setup. Included models and converted datasets allow offline predictions, evaluation and tests afterward.

## Reproduce training

```powershell
python run_project.py
python run_project.py --dataset synthetic
```

The default uses real reviews; the second command is explicitly a synthetic demonstration. Outputs are isolated under `outputs/real` and `outputs/synthetic`; models are under `models/real` and `models/synthetic`. Paths resolve from the script location, so the CLI works from another directory. Expect training to take longer than prediction.

To recreate the real CSV from its published source:

```powershell
python prepare_data.py
```

The downloader validates the published ZIP checksum and converts only corpus XMI documents. It reads XML, never executes source archive content. Data are for scientific use only. The converted CSV and provenance JSON are included.

## Predict without retraining

```powershell
python run_project.py --review "The staff were helpful, but the room was dirty."
python run_project.py --dataset synthetic --review "The room was comfortable and sleep was quiet."
```

`--review` loads saved models only. By default unmentioned aspects have a null rating and `not_mentioned` status. `--include-unmentioned` returns model estimates for all trained aspects. A keyword heuristic detects mentions and can miss paraphrases; it is not a learned aspect extractor. The model uses the entire review for each aspect and can make mistakes.

Possible statuses: `predicted`, `not_mentioned`, `unavailable` (no trained labels), and `unknown_vocabulary`. Empty, punctuation-only and excessively long input are rejected. Load only trusted project model files: joblib is a Python serialization format, not a safe format for arbitrary model downloads. File hashes check corruption, not publisher authenticity.

## Data and provenance

| Dataset | Reviews | Purpose | Label coverage |
|---|---:|---|---|
| ArguAna TripAdvisor annotated v2 | 2,100 from 122 hotels | Primary real-data evaluation | Overall and six aspects |
| Supplied sample | 320 | Synthetic functional demonstration | Overall and seven aspects |

The real corpus comes from [Wachsmuth et al., A Review Corpus for Argumentation Analysis, CICLing 2014](https://zenodo.org/records/3973241). Review title and text are in `Sofa.sofaString`; overall rating comes from `Sentiment.score`, never `HotelData.stars`. `Aspects.room` maps to rooms and `Aspects.business` to business service. Other fields retain their meaning. Source -1 becomes missing. Reviewer author/date fields are omitted from the converted CSV. The corpus is scientifically usable, not for commercial applications.

Real label counts: overall 2,100; service 1,608; rooms 1,647; location 1,238; value 1,631; cleanliness 1,651; business service 772. **Sleep quality has no real labels and is unavailable in real mode.** No aspect ratings are invented. The source corpus is balanced by overall sentiment and does not represent current hotel traffic.

The supplied synthetic sample visibly reuses sentence templates. Its generator and external provenance were not provided. Its high scores must not be represented as real-world accuracy or as a reproduction of the Stanford paper.

## Evaluation method

- Normalize text while preserving negations; detect conflicting duplicate text and remove exact normalized duplicates before splitting.
- Use seed 42. Keep hotels disjoint across real-data partitions: 1,342 train, 388 validation, 370 test reviews. Split fractions refer to hotels, so review proportions vary. Synthetic data use 192/64/64 distinct review splits.
- Compare Multinomial NB with TF-IDF, balanced Linear SVM with counts, and balanced Logistic Regression with TF-IDF. All use unigrams/bigrams and a 12,000-feature cap. Pipeline features are fitted only on training data during selection.
- Select by validation polarity macro-F1, then star accuracy, then model name to break ties. Refit on train plus validation. Test labels never participate in fitting or selection.
- Omit missing labels separately per target. The saved final models exclude every test review.
- Evaluate exact-star accuracy, star macro-F1 and MAE, polarity accuracy and macro-F1. Stars 1-2 are negative, 3 neutral, 4-5 positive. Polarity macro-F1 includes all three labels equally.
- Train separate majority-star and majority-polarity baselines on development labels. Also compare aspect models to the overall model evaluated against aspect labels.

These are fixed pipeline configurations, not a classifier-only ablation or an exhaustive hyperparameter search. Wilson intervals in the CSV describe per-review accuracy and do not adjust for within-hotel dependence. Report the fixed split as a limited experiment, not an exact estimate of deployment performance.

**Metric definitions:** accuracy is the fraction of correct predictions. For each class, precision is the fraction of predicted members that are correct; recall is the fraction of actual members found. F1 is their harmonic mean. Macro-F1 averages class F1 scores equally, making minority-class errors visible. Per-class precision, recall and F1 are in the classification reports. Star MAE is the average absolute error in stars. None of these metrics evaluate the keyword mention filter.

## Real-data results

| Target | Test reviews | Star accuracy | Polarity accuracy | Polarity macro-F1 | Majority polarity accuracy |
|---|---:|---:|---:|---:|---:|
| Overall | 370 | 51.1% | 74.6% | 0.709 | 32.4% |
| Service | 271 | 44.6% | 68.3% | 0.628 | 45.0% |
| Rooms | 276 | 39.1% | 60.9% | 0.560 | 47.8% |
| Location | 223 | 47.5% | 74.4% | 0.423 | 78.9% |
| Value | 279 | 46.2% | 62.4% | 0.615 | 37.3% |
| Cleanliness | 278 | 46.4% | 64.7% | 0.539 | 61.5% |
| Business service | 130 | 38.5% | 55.4% | 0.558 | 39.2% |

Location loses to the majority polarity baseline on accuracy. Aspect-specific training beats the overall-model transfer only for rooms, location and cleanliness in this experiment. High polarity accuracy does not mean exact star ratings are equally accurate. See saved error examples and per-class reports.

## Testing and evidence

```powershell
python verify_project.py
# Alternative without writing a report:
python -m unittest discover -s tests -v
```

The original 33 tests are retained. Nine audit regression tests cover incomplete hotel IDs, false-positive readiness results, invalid/duplicate documents and reproducibility of the five demo outputs. The full suite contains **42 tests**; the actual latest run status is in `outputs/verification/test_summary.json`. Tests also cover invalid input, missing labels, duplicates, disjoint/reproducible splits, held-out vocabulary, validation-based selection, serialization, corrupted models, positive/negative fixture behavior, inference without retraining, independent metric/baseline recomputation and Streamlit interactions in both modes. Fixture sentiment checks concern controlled test data, not every natural-language review.

`verify_project.py` writes the actual run log and JSON summary under `outputs/verification/`. The checked-in CI workflow runs the same suite; it has not run on GitHub until the repository is uploaded. Software tests do not guarantee every model prediction is correct.

The verifier checks required structure, pinned package versions and imports, `pip check`, dataset/provenance/split consistency, both model sets, sample predictions, CLI inference without retraining, PDF presence/format, presentation OOXML integrity and the complete test suite. Any failed check returns a nonzero exit code. PDF page count and visual slide checks are recorded by the final artifact audit; the runtime file-format checks do not replace visual review. The READY verdict concerns the local technical package, not manual repository access or contribution approval.

The final audit also installed `requirements.txt` in a fresh isolated Python 3.11.9 environment and ran the complete verifier there. Streamlit's application and health endpoints returned HTTP 200. Full real-data retraining reproduced the saved metrics, model-comparison scores, split manifest and test predictions exactly. See `outputs/verification/clean_install.json`, `streamlit_startup.json`, `reproducibility_check.json`, `document_audit.json` and `security_and_package_audit.json` for the corresponding evidence.

## Files for evaluation

- `docs/ABSA_Hotel_Reviews_Final_Writeup.pdf`: two-page project summary.
- `docs/ABSA_Hotel_Reviews_Final_Presentation.pptx`: presentation for review.
- `DEMO_GUIDE.md`: exact commands, five pre-tested examples including errors, and viva discussion points.
- `outputs/real/metrics.csv`: final test metrics for validation-selected models.
- `outputs/real/validation_model_comparison.csv`: comparison used to choose models.
- `outputs/real/split_manifest.csv`: review hashes, hotel IDs and partitions.
- `outputs/real/test_predictions.csv`: every scored test prediction with true labels.
- `outputs/real/*_classification_report.json` and `*_confusion_matrix.csv`: class-level evidence.
- `outputs/real/error_examples.csv`: actual errors for discussion.
- `outputs/real/run_metadata.json`: dataset fingerprint, seed, split sizes, versions and model selection.
- `outputs/verification/test_results.txt`: complete test log.

## Custom data

Supply `review_text`, `overall_rating` and the seven aspect columns listed in `src/config.py`. Ratings must be integer 1-5 or blank; overall labels cannot be blank. Review IDs are generated if absent. Provide `hotel_id` for hotel-disjoint evaluation and at least five distinct hotels. At least 30 distinct reviews are required; each trained target needs at least five observed labels in each partition and two training classes.

```powershell
python run_project.py --data path/to/reviews.csv --out outputs/custom --models models/custom
python run_project.py --models models/custom --review "The staff were helpful."
```

Choose distinct custom directories to preserve the supplied reference results. Retrain if changing dependency versions; serialized scikit-learn models are tied to their training environment.

When `hotel_id` is supplied, every row must have a nonblank ID. Partial hotel IDs are rejected rather than silently falling back to a random review split. The included real-data split and metrics are unchanged by this guard.

## Limitations

Class imbalance can make accuracy misleading, especially for location. Some aspects have limited observed ratings. The full review is used for each model, so mixed sentiment and multi-aspect sentences can confuse the classifier. Implicit aspects and paraphrases may evade the keyword filter. Traditional unigram/bigram features have limited context and domain-specific vocabulary coverage. The corpus is historical, English-only and balanced by overall sentiment; it does not represent current hotel traffic. There are no real sleep-quality labels, and no food/breakfast model. The project is a university prototype, not a production-ready or perfectly accurate system. The demo guide explicitly records mistaken predictions.

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
