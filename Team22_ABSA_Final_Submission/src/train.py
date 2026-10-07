"""Model selection on validation data followed by isolated test evaluation."""
from __future__ import annotations
import hashlib
import json
import platform
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from .config import TARGET_COLUMNS, ASPECT_DISPLAY_NAMES
from .utils import rating_to_polarity, validate_dataset

POLARITIES = ["negative", "neutral", "positive"]


def build_models(seed=42):
    # Retain negations: common English stop-word lists remove "not".
    features = dict(ngram_range=(1, 2), min_df=1, max_features=12000, stop_words=None)
    return {
        "MultinomialNB_TFIDF": Pipeline([("features", TfidfVectorizer(**features)), ("model", MultinomialNB(alpha=0.5))]),
        "LinearSVC_Count_2gram": Pipeline([("features", CountVectorizer(**features)), ("model", LinearSVC(C=0.1, class_weight="balanced", max_iter=20000, random_state=seed))]),
        "LogReg_TFIDF": Pipeline([("features", TfidfVectorizer(**features)), ("model", LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000, random_state=seed))]),
    }


def make_splits(df, seed=42):
    """Approximately 60/20/20; whole hotels remain in a single partition."""
    if len(df) < 30:
        raise ValueError("At least 30 distinct reviews are required.")
    indices = np.arange(len(df))
    if "hotel_id" in df:
        if df.hotel_id.isna().any() or df.hotel_id.astype(str).str.strip().eq("").any():
            raise ValueError("hotel_id must be present on every review for a hotel-disjoint split.")
        groups = df.hotel_id.astype(str)
        if groups.nunique() < 5:
            raise ValueError("At least five hotels are required for a hotel-disjoint split.")
        dev, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(indices, groups=groups))
        tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1).split(dev, groups=groups.iloc[dev]))
        train, valid = dev[tr], dev[va]
    else:
        dev, test = train_test_split(indices, test_size=0.2, random_state=seed)
        train, valid = train_test_split(dev, test_size=0.25, random_state=seed + 1)
    return {"train": np.sort(train), "validation": np.sort(valid), "test": np.sort(test)}


def scores(y_true, y_pred):
    true_pol = [rating_to_polarity(v) for v in y_true]
    pred_pol = [rating_to_polarity(v) for v in y_pred]
    return {
        "star_accuracy": float(accuracy_score(y_true, y_pred)),
        "star_macro_f1": float(f1_score(y_true, y_pred, labels=[1, 2, 3, 4, 5], average="macro", zero_division=0)),
        "star_mae": float(mean_absolute_error(y_true, y_pred)),
        "polarity_accuracy": float(accuracy_score(true_pol, pred_pol)),
        "polarity_macro_f1": float(f1_score(true_pol, pred_pol, labels=POLARITIES, average="macro", zero_division=0)),
    }


def choose_model(rows):
    """This function never receives test scores or test labels."""
    return sorted(rows, key=lambda r: (-r["validation_polarity_macro_f1"], -r["validation_star_accuracy"], r["model"]))[0]["model"]


def wilson_interval(successes, count):
    z, p = 1.96, successes / count
    den = 1 + z*z/count
    centre = (p + z*z/(2*count))/den
    radius = z*np.sqrt((p*(1-p)+z*z/(4*count))/count)/den
    return [float(centre-radius), float(centre+radius)]


def train_and_evaluate(df, output_dir, model_dir, seed=42, dataset_name="user_supplied"):
    df = validate_dataset(df)
    output_path, model_path = Path(output_dir), Path(model_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)
    splits = make_splits(df, seed)
    manifest = df[[c for c in ["review_id", "hotel_id", "text_hash"] if c in df]].copy()
    for name, indices in splits.items():
        manifest.loc[indices, "split"] = name
    manifest.to_csv(output_path / "split_manifest.csv", index=False)
    fingerprint = hashlib.sha256(df[["review_id", "clean_review", *TARGET_COLUMNS]].to_csv(index=False).encode()).hexdigest()
    metadata = {
        "schema_version": 1, "dataset": dataset_name, "dataset_sha256": fingerprint,
        "seed": seed, "rows": len(df), "split_counts": {k: len(v) for k, v in splits.items()},
        "split_method": "hotel_disjoint" if "hotel_id" in manifest else "deduplicated_review_random",
        "selection": "validation polarity macro-F1; star accuracy tie-break; alphabetical final tie-break",
        "evaluation": "Refit on train+validation; evaluate test once. Saved models exclude test rows.",
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
        "targets": {},
    }
    selection_rows, results, prediction_rows, distributions = [], [], [], []
    fitted, test_predictions = {}, {}
    for target in TARGET_COLUMNS:
        labelled = {name: ids[df.loc[ids, target].notna().to_numpy()] for name, ids in splits.items()}
        for name, ids in labelled.items():
            for rating in range(1, 6):
                distributions.append({"target": target, "split": name, "rating": rating, "count": int((df.loc[ids, target] == rating).sum())})
        counts = {name: len(ids) for name, ids in labelled.items()}
        if any(v < 5 for v in counts.values()) or df.loc[labelled["train"], target].nunique() < 2:
            metadata["targets"][target] = {"status": "unavailable", "reason": "Insufficient observed labels; missing ratings are never invented.", "counts": counts}
            continue
        train, valid, test = (labelled[n] for n in ["train", "validation", "test"])
        candidates, target_rows = build_models(seed), []
        for name, pipeline in candidates.items():
            pipeline.fit(df.loc[train, "clean_review"], df.loc[train, target].astype(int))
            values = scores(df.loc[valid, target].astype(int), pipeline.predict(df.loc[valid, "clean_review"]))
            target_rows.append({"target": target, "model": name, **{f"validation_{k}": v for k, v in values.items()}})
        best_name = choose_model(target_rows)
        for row in target_rows:
            row["selected"] = row["model"] == best_name
        selection_rows.extend(target_rows)
        development = np.sort(np.concatenate([train, valid]))
        best = clone(candidates[best_name]).fit(df.loc[development, "clean_review"], df.loc[development, target].astype(int))
        pred = best.predict(df.loc[test, "clean_review"]).astype(int)
        truth = df.loc[test, target].astype(int).to_numpy()
        values = scores(truth, pred)
        dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(development), 1)), df.loc[development, target].astype(int))
        baseline = scores(truth, dummy.predict(np.zeros((len(test), 1))))
        true_pol, pred_pol = ([rating_to_polarity(v) for v in seq] for seq in [truth, pred])
        # Polarity has a different class distribution from stars: fit its own majority baseline.
        polarity_dummy = DummyClassifier(strategy="most_frequent").fit(
            np.zeros((len(development), 1)), [rating_to_polarity(v) for v in df.loc[development, target]])
        baseline_pol = polarity_dummy.predict(np.zeros((len(test), 1)))
        baseline["polarity_accuracy"] = float(accuracy_score(true_pol, baseline_pol))
        baseline["polarity_macro_f1"] = float(f1_score(true_pol, baseline_pol, labels=POLARITIES, average="macro", zero_division=0))
        lo, hi = wilson_interval(sum(a == b for a, b in zip(true_pol, pred_pol)), len(truth))
        results.append({"target": target, "target_display": ASPECT_DISPLAY_NAMES[target], "model": best_name,
                        "train_n": len(train), "validation_n": len(valid), "test_n": len(test),
                        **{f"test_{k}": v for k, v in values.items()}, **{f"baseline_{k}": v for k, v in baseline.items()},
                        "polarity_ci_low": lo, "polarity_ci_high": hi})
        fitted[target], test_predictions[target] = best, dict(zip(test.tolist(), pred.tolist()))
        filename = f"{target}_model.joblib"
        joblib.dump(best, model_path / filename)
        metadata["targets"][target] = {"status": "trained", "model": best_name, "file": filename, "counts": counts,
                                      "file_sha256": hashlib.sha256((model_path / filename).read_bytes()).hexdigest()}
        report = {"stars": classification_report(truth, pred, labels=[1, 2, 3, 4, 5], zero_division=0, output_dict=True),
                  "polarity": classification_report(true_pol, pred_pol, labels=POLARITIES, zero_division=0, output_dict=True)}
        (output_path / f"{target}_classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
        (output_path / f"{target}_classification_report.txt").write_text(
            "HELD-OUT TEST; model selected on validation\n\n" + classification_report(truth, pred, labels=[1, 2, 3, 4, 5], zero_division=0)
            + "\nPOLARITY\n" + classification_report(true_pol, pred_pol, labels=POLARITIES, zero_division=0), encoding="utf8")
        pd.DataFrame(confusion_matrix(truth, pred, labels=[1, 2, 3, 4, 5]), index=range(1, 6), columns=range(1, 6)).to_csv(output_path / f"{target}_confusion_matrix.csv", index_label="true_rating")
        for idx, actual, predicted in zip(test, truth, pred):
            prediction_rows.append({"review_id": df.loc[idx, "review_id"], "target": target, "actual_rating": int(actual),
                                    "predicted_rating": int(predicted), "actual_polarity": rating_to_polarity(actual),
                                    "predicted_polarity": rating_to_polarity(predicted), "review_text": df.loc[idx, "review_text"]})
        print(f"{target}: {best_name}; test n={len(test)}; polarity={values['polarity_accuracy']:.3f}; macro-F1={values['polarity_macro_f1']:.3f}", flush=True)
    if not results:
        raise ValueError("No target has enough labelled data to train and evaluate.")
    if "overall_rating" in fitted:
        for row in results:
            target = row["target"]
            ids = list(test_predictions[target])
            comparison = scores(df.loc[ids, target].astype(int), fitted["overall_rating"].predict(df.loc[ids, "clean_review"]))
            row["overall_transfer_polarity_accuracy"] = comparison["polarity_accuracy"]
    metrics = pd.DataFrame(results)
    metrics.to_csv(output_path / "metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(output_path / "validation_model_comparison.csv", index=False)
    pd.DataFrame(distributions).to_csv(output_path / "label_distribution.csv", index=False)
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(output_path / "test_predictions.csv", index=False)
    predictions.groupby("target", sort=False).head(3).to_csv(output_path / "sample_predictions.csv", index=False)
    predictions[predictions.actual_rating != predictions.predicted_rating].groupby("target", sort=False).head(5).to_csv(output_path / "error_examples.csv", index=False)
    for destination in [model_path / "manifest.json", output_path / "run_metadata.json"]:
        destination.write_text(json.dumps(metadata, indent=2), encoding="utf8")
    _create_plots(metrics, pd.DataFrame(selection_rows), output_path)
    return metrics


def _create_plots(metrics, validation, output):
    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    ax.barh(x - .18, metrics.test_polarity_accuracy, .35, label="Selected aspect model", color="#176b87")
    ax.barh(x + .18, metrics.baseline_polarity_accuracy, .35, label="Majority baseline", color="#adb5bd")
    ax.set(yticks=x, yticklabels=metrics.target_display, xlim=(0, 1.05), xlabel="Held-out polarity accuracy", title="Test performance and majority baseline")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "aspect_accuracy.png", dpi=180)
    plt.close(fig)
    overall = validation[validation.target == "overall_rating"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(overall.model, overall.validation_polarity_macro_f1, color="#176b87")
    ax.set(ylim=(0, 1), ylabel="Validation polarity macro-F1", title="Overall model selection (validation data only)")
    ax.tick_params(axis="x", labelrotation=12)
    fig.tight_layout()
    fig.savefig(output / "overall_model_comparison.png", dpi=180)
    plt.close(fig)
