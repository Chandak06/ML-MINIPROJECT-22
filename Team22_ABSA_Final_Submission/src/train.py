"""Model selection on validation data followed by isolated test evaluation."""
from __future__ import annotations
import hashlib
import json
import platform
import re
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
from .aspect_router import AspectTextTransformer, ROUTER_VERSION, route_review, evidence_for
from .confidence import attach_confidence, probability_estimates, select_uncertainty_threshold, is_uncertain
from .evaluation import hotel_bootstrap, mixed_label_mask

POLARITIES = ["negative", "neutral", "positive"]


def build_models(seed=42, target="overall_rating", strategy="full_review"):
    # Retain negations: common English stop-word lists remove "not".
    features = dict(ngram_range=(1, 2), min_df=1, max_features=12000, stop_words=None)
    models = {
        "MultinomialNB_TFIDF": Pipeline([("features", TfidfVectorizer(**features)), ("model", MultinomialNB(alpha=0.5))]),
        "LinearSVC_Count_2gram": Pipeline([("features", CountVectorizer(**features)), ("model", LinearSVC(C=0.1, class_weight="balanced", max_iter=20000, random_state=seed))]),
        "LogReg_TFIDF": Pipeline([("features", TfidfVectorizer(**features)), ("model", LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000, random_state=seed))]),
    }
    for pipeline in models.values():
        pipeline.steps.insert(0, ("text", AspectTextTransformer(target, strategy)))
    return models


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
        "within_one_star_accuracy": float((np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 1).mean()),
        "polarity_accuracy": float(accuracy_score(true_pol, pred_pol)),
        "polarity_macro_f1": float(f1_score(true_pol, pred_pol, labels=POLARITIES, average="macro", zero_division=0)),
    }


def choose_model(rows):
    """This function never receives test scores or test labels."""
    return choose_configuration(rows)["model"]


def choose_configuration(rows):
    return sorted(rows, key=lambda r: (-r["validation_polarity_macro_f1"], -r["validation_star_accuracy"],
                                       r.get("routing_strategy", "full_review") != "full_review", r["model"]))[0]


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
    split_manifest = df[[c for c in ["review_id", "hotel_id", "text_hash"] if c in df]].copy()
    for name, indices in splits.items():
        split_manifest.loc[indices, "split"] = name
    split_manifest.to_csv(output_path / "split_manifest.csv", index=False)
    fingerprint = hashlib.sha256(df[["review_id", "clean_review", *TARGET_COLUMNS]].to_csv(index=False).encode()).hexdigest()
    metadata = {
        "schema_version": 2, "dataset": dataset_name, "dataset_sha256": fingerprint,
        "seed": seed, "rows": len(df), "split_counts": {k: len(v) for k, v in splits.items()},
        "split_method": "hotel_disjoint" if "hotel_id" in df else "deduplicated_review_random",
        "selection": "validation polarity macro-F1; star accuracy; prefer full review on ties; model name",
        "evaluation": "Select every configuration and uncertainty threshold before refit and test. Refit on train+validation. Test rows excluded from every fit.",
        "mention_detection_version": ROUTER_VERSION,
        "confidence_definition": "Estimated probability of the base model's predicted polarity; native NB/LR probabilities are uncalibrated. SVM uses sigmoid OOF calibration.",
        "uncertainty_selection": "Maximize balanced accuracy of validation correctness detection; ties favor coverage. Threshold frozen before refit.",
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ["train.py", "utils.py", "aspect_router.py", "confidence.py", "evaluation.py"]},
        "targets": {},
    }
    selection_rows, jobs, calibration_rows, validation_predictions, validation_confidence_rows = [], {}, [], [], []

    def calibrate(model, ids, target, stage):
        groups = df.loc[ids, "hotel_id"].to_numpy() if "hotel_id" in df else None
        detail = attach_confidence(model, df.loc[ids, "review_text"], df.loc[ids, target].astype(int), groups, seed)
        for fold, (_, heldout) in enumerate(detail["folds"]):
            for local in heldout:
                idx = ids[local]
                calibration_rows.append({"target": target, "stage": stage, "fold": fold, "review_id": df.loc[idx, "review_id"],
                                         "hotel_id": df.loc[idx, "hotel_id"] if "hotel_id" in df else ""})
        return detail["method"]

    # PHASE A: train and validation only. No held-out predictions occur here.
    for target in TARGET_COLUMNS:
        labelled = {name: ids[df.loc[ids, target].notna().to_numpy()] for name, ids in splits.items()}
        counts = {name: len(ids) for name, ids in labelled.items()}
        if min(counts["train"], counts["validation"]) < 5 or df.loc[labelled["train"], target].nunique() < 2:
            metadata["targets"][target] = {"status": "unavailable", "reason": "Insufficient observed development labels; missing ratings are never invented.", "counts": counts}
            continue
        train, valid = labelled["train"], labelled["validation"]
        candidates, target_rows = {}, []
        for strategy in ["full_review", "aspect_text"]:
            for name, pipeline in build_models(seed, target, strategy).items():
                pipeline.fit(df.loc[train, "review_text"], df.loc[train, target].astype(int))
                predictions = pipeline.predict(df.loc[valid, "review_text"])
                values = scores(df.loc[valid, target].astype(int), predictions)
                row = {"target": target, "model": name, "routing_strategy": strategy, **{f"validation_{k}": v for k, v in values.items()}}
                target_rows.append(row)
                candidates[(strategy, name)] = pipeline
                for idx, pred in zip(valid, predictions):
                    validation_predictions.append({"target": target, "review_id": df.loc[idx, "review_id"], "model": name,
                                                   "routing_strategy": strategy, "actual_rating": int(df.loc[idx, target]), "predicted_rating": int(pred)})
        chosen = choose_configuration(target_rows)
        old = choose_configuration([r for r in target_rows if r["routing_strategy"] == "full_review"])
        for row in target_rows:
            row["selected"] = row is chosen
            row["selected_full_review_baseline"] = row is old
        selection_rows.extend(target_rows)
        model = candidates[(chosen["routing_strategy"], chosen["model"])]
        method = calibrate(model, train, target, "training_only")
        valid_pred = model.predict(df.loc[valid, "review_text"])
        confidence, _, _ = probability_estimates(model, df.loc[valid, "review_text"], valid_pred)
        correct = [rating_to_polarity(y) == rating_to_polarity(p) for y, p in zip(df.loc[valid, target], valid_pred)]
        threshold = (select_uncertainty_threshold(confidence, correct) if all(v is not None for v in confidence)
                     else {"threshold": None, "reason": "confidence_unavailable", "validation_n": len(valid)})
        for idx, prediction, conf, right in zip(valid, valid_pred, confidence, correct):
            validation_confidence_rows.append({"review_id": df.loc[idx, "review_id"], "target": target,
                                              "actual_rating": int(df.loc[idx, target]), "predicted_rating": int(prediction),
                                              "confidence": conf, "polarity_correct": right, "threshold": threshold["threshold"],
                                              "accepted": not is_uncertain(conf, threshold["threshold"])})
        metadata["targets"][target] = {
            "status": "trained", "target": target, "model": chosen["model"],
            "classifier": type(model.named_steps["model"]).__name__, "vectorizer": type(model.named_steps["features"]).__name__,
            "ngram_range": [1, 2], "routing_strategy": chosen["routing_strategy"], "mention_detection_version": ROUTER_VERSION,
            "selected_on": "validation_polarity_macro_f1", "validation_macro_f1": chosen["validation_polarity_macro_f1"],
            "confidence_method": method, "uncertainty": threshold, "counts": counts,
            "training_rows": counts["train"], "validation_rows": counts["validation"], "test_rows": counts["test"],
            "seed": seed, "full_review_baseline_model": old["model"],
        }
        jobs[target] = {"ids": labelled, "chosen": chosen, "old": old, "template": model,
                        "old_template": candidates[("full_review", old["model"])],
                        "validation_confidence": confidence, "validation_correct": correct}
        print(f"Selected {target}: {chosen['model']} / {chosen['routing_strategy']}; validation macro-F1={chosen['validation_polarity_macro_f1']:.4f}", flush=True)
    if not jobs:
        raise ValueError("No target has enough labelled development data.")
    # Frozen selection evidence is written before refitting or evaluating test.
    pd.DataFrame(selection_rows).to_csv(output_path / "validation_model_comparison.csv", index=False)
    pd.DataFrame(validation_predictions).to_csv(output_path / "validation_predictions.csv", index=False)
    pd.DataFrame(validation_confidence_rows).to_csv(output_path / "validation_confidence_predictions.csv", index=False)
    (output_path / "selection_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf8")

    # PHASE B: refit selected configurations and confidence models, development only.
    for target, job in jobs.items():
        dev = np.sort(np.concatenate([job["ids"]["train"], job["ids"]["validation"]]))
        model = clone(job["template"]).fit(df.loc[dev, "review_text"], df.loc[dev, target].astype(int))
        method = calibrate(model, dev, target, "development_refit")
        job["fitted"] = model
        job["old_fitted"] = (model if job["chosen"]["routing_strategy"] == "full_review" else
                             clone(job["old_template"]).fit(df.loc[dev, "review_text"], df.loc[dev, target].astype(int)))
        entry = metadata["targets"][target]
        entry["confidence_method"] = method
        entry["refit_rows"] = len(dev)
        entry["fit_review_ids_sha256"] = hashlib.sha256("\n".join(df.loc[dev, "review_id"]).encode()).hexdigest()
        filename = f"{target}_model.joblib"
        joblib.dump(model, model_path / filename)
        entry.update({"file": filename, "file_sha256": hashlib.sha256((model_path / filename).read_bytes()).hexdigest()})

    # PHASE C: final descriptive test evaluation. Never feed these values back.
    results, predictions, comparisons, distributions, per_class, routing_rows, confidence_rows, bootstrap_rows, mixed_rows = [], [], [], [], [], [], [], [], []
    for target in TARGET_COLUMNS:
        labelled = {name: ids[df.loc[ids, target].notna().to_numpy()] for name, ids in splits.items()}
        for partition, ids in labelled.items():
            for rating in range(1, 6):
                distributions.append({"target": target, "split": partition, "rating": rating, "count": int((df.loc[ids, target] == rating).sum())})
        if target not in jobs:
            continue
        job, entry = jobs[target], metadata["targets"][target]
        train, valid, test = (job["ids"][n] for n in ["train", "validation", "test"])
        if not len(test):
            entry["evaluation_status"] = "no_observed_test_labels"
            continue
        dev = np.sort(np.concatenate([train, valid]))
        truth = df.loc[test, target].astype(int).to_numpy()
        texts = df.loc[test, "review_text"]
        model, old_model = job["fitted"], job["old_fitted"]
        pred, old_pred = model.predict(texts).astype(int), old_model.predict(texts).astype(int)
        values, old_values = scores(truth, pred), scores(truth, old_pred)
        confidence, rating_confidence, _ = probability_estimates(model, texts, pred)
        true_pol = [rating_to_polarity(v) for v in truth]
        pred_pol = [rating_to_polarity(v) for v in pred]
        dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(dev), 1)), df.loc[dev, target].astype(int))
        baseline = scores(truth, dummy.predict(np.zeros((len(test), 1))))
        dev_polarities = [rating_to_polarity(v) for v in df.loc[dev, target]]
        polarity_dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(dev), 1)), dev_polarities)
        baseline_pol = polarity_dummy.predict(np.zeros((len(test), 1)))
        baseline["polarity_accuracy"] = float(accuracy_score(true_pol, baseline_pol))
        baseline["polarity_macro_f1"] = float(f1_score(true_pol, baseline_pol, labels=POLARITIES, average="macro", zero_division=0))
        lo, hi = wilson_interval(sum(a == b for a, b in zip(true_pol, pred_pol)), len(truth))
        metric = {"target": target, "target_display": ASPECT_DISPLAY_NAMES[target], "model": entry["model"], "routing_strategy": entry["routing_strategy"],
                  "train_n": len(train), "validation_n": len(valid), "test_n": len(test),
                  **{f"test_{k}": v for k, v in values.items()}, **{f"baseline_{k}": v for k, v in baseline.items()},
                  "polarity_ci_low": lo, "polarity_ci_high": hi}
        if "overall_rating" in jobs:
            transfer = scores(truth, jobs["overall_rating"]["fitted"].predict(texts))
            metric["overall_transfer_polarity_accuracy"] = transfer["polarity_accuracy"]
        results.append(metric)
        comparisons.append({"target": target, "old_model": job["old"]["model"], "new_model": entry["model"], "new_strategy": entry["routing_strategy"],
                            "old_validation_macro_f1": job["old"]["validation_polarity_macro_f1"], "new_validation_macro_f1": entry["validation_macro_f1"],
                            "test_n": len(test), **{f"old_test_{k}": v for k, v in old_values.items()}, **{f"new_test_{k}": v for k, v in values.items()},
                            "baseline_polarity_accuracy": baseline["polarity_accuracy"], "baseline_star_accuracy": baseline["star_accuracy"]})
        report = {"stars": classification_report(truth, pred, labels=[1, 2, 3, 4, 5], zero_division=0, output_dict=True),
                  "polarity": classification_report(true_pol, pred_pol, labels=POLARITIES, zero_division=0, output_dict=True)}
        (output_path / f"{target}_classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
        (output_path / f"{target}_classification_report.txt").write_text("HELD-OUT TEST; configuration selected on validation; all labelled rows scored before UI abstention\n\n" + classification_report(truth, pred, labels=[1, 2, 3, 4, 5], zero_division=0) + "\nPOLARITY\n" + classification_report(true_pol, pred_pol, labels=POLARITIES, zero_division=0), encoding="utf8")
        for kind, labels in [("stars", [1, 2, 3, 4, 5]), ("polarity", POLARITIES)]:
            for label in labels:
                per_class.append({"target": target, "label_type": kind, "class": label, **report[kind][str(label)]})
        pd.DataFrame(confusion_matrix(truth, pred, labels=[1, 2, 3, 4, 5]), index=range(1, 6), columns=range(1, 6)).to_csv(output_path / f"{target}_confusion_matrix.csv", index_label="true_rating")
        pd.DataFrame(confusion_matrix(true_pol, pred_pol, labels=POLARITIES), index=POLARITIES, columns=POLARITIES).to_csv(output_path / f"{target}_polarity_confusion_matrix.csv", index_label="true_polarity")
        for partition, ids in [("train", train), ("validation", valid), ("test", test)]:
            routed = [evidence_for(text, target, "aspect_text") for text in df.loc[ids, "review_text"]]
            routing_rows.append({"target": target, "split": partition, "labelled_n": len(ids), "mentioned_n": sum(r["mentioned"] for r in routed),
                                 "aspect_text_n": sum(r["evidence_source"] == "aspect_text" for r in routed),
                                 "whole_review_fallback_n": sum(r["evidence_source"] == "whole_review_fallback" for r in routed),
                                 "scope": "Rule coverage on observed ratings, not human-annotated mention accuracy"})
        threshold = entry["uncertainty"]["threshold"]
        accepted, emitted = [], []
        for j, (idx, actual, predicted) in enumerate(zip(test, truth, pred)):
            text = df.loc[idx, "review_text"]
            info = evidence_for(text, target, entry["routing_strategy"])
            known = model.named_steps["features"].transform(model.named_steps["text"].transform([text])).nnz > 0
            status = ("not_mentioned" if not info["mentioned"] else "unknown_vocabulary" if not known else
                      "uncertain" if is_uncertain(confidence[j], threshold) else "predicted")
            accepted.append(not is_uncertain(confidence[j], threshold))
            emitted.append(status == "predicted")
            predictions.append({"review_id": df.loc[idx, "review_id"], "hotel_id": df.loc[idx, "hotel_id"] if "hotel_id" in df else "",
                                "target": target, "actual_rating": int(actual), "predicted_rating": int(predicted), "actual_polarity": true_pol[j], "predicted_polarity": pred_pol[j],
                                "review_text": text, **info, "confidence": confidence[j], "rating_confidence": rating_confidence[j], "prediction_status": status,
                                "selected_model": entry["model"], "routing_strategy": entry["routing_strategy"], "old_predicted_rating": int(old_pred[j]),
                                "contains_negation": bool(re.search(r"\b(not|never|no)\b", df.loc[idx, "clean_review"])),
                                "contains_contrast_marker": bool(re.search(r"\b(but|although|however|though)\b", text, re.I)),
                                "minority_actual_polarity": true_pol[j] != polarity_dummy.classes_[int(np.argmax(polarity_dummy.class_prior_))]})
        for partition, conf, correct, accept, emitted_mask in [
            ("validation", job["validation_confidence"], job["validation_correct"], [not is_uncertain(c, threshold) for c in job["validation_confidence"]], None),
            ("test", confidence, [a == b for a, b in zip(true_pol, pred_pol)], accepted, emitted)]:
            correct, accept = np.asarray(correct), np.asarray(accept)
            confidence_rows.append({"target": target, "split": partition, "n": len(correct), "method": entry["confidence_method"], "threshold": threshold,
                                    "mean_confidence": float(np.mean(conf)) if all(c is not None for c in conf) else None,
                                    "correctness_brier_score": float(np.mean((np.asarray(conf) - correct) ** 2)) if all(c is not None for c in conf) else None,
                                    "all_row_polarity_accuracy": float(correct.mean()), "confidence_acceptance_coverage": float(accept.mean()),
                                    "accepted_polarity_accuracy": float(correct[accept].mean()) if accept.any() else None,
                                    "ui_prediction_coverage": None if emitted_mask is None else float(np.mean(emitted_mask)),
                                    "ui_predicted_polarity_accuracy": None if emitted_mask is None or not any(emitted_mask) else float(correct[emitted_mask].mean())})
        if "hotel_id" in df:
            bootstrap_rows.extend({"target": target, **row} for row in hotel_bootstrap(truth, pred, df.loc[test, "hotel_id"], old_pred, seed))
        mask = mixed_label_mask(df.loc[test], target, TARGET_COLUMNS)
        for system, labels in [("old_full_review", old_pred), ("new_validation_selected", pred)]:
            if mask.any():
                mixed_rows.append({"target": target, "system": system, "n": int(mask.sum()), **scores(truth[mask], labels[mask]), "subset": "observed_aspect_polarities_disagree"})
        print(f"Test {target}: polarity={values['polarity_accuracy']:.3f}; macro-F1={values['polarity_macro_f1']:.3f}; MAE={values['star_mae']:.3f}", flush=True)
    metrics, prediction_frame = pd.DataFrame(results), pd.DataFrame(predictions)
    metrics.to_csv(output_path / "metrics.csv", index=False)
    for name, rows in [("system_comparison.csv", comparisons), ("label_distribution.csv", distributions), ("per_class_metrics.csv", per_class),
                       ("aspect_routing_evaluation.csv", routing_rows), ("confidence_analysis.csv", confidence_rows),
                       ("hotel_bootstrap_metrics.csv", bootstrap_rows), ("mixed_aspect_evaluation.csv", mixed_rows)]:
        if rows:
            pd.DataFrame(rows).to_csv(output_path / name, index=False)
    pd.DataFrame(calibration_rows, columns=["target", "stage", "fold", "review_id", "hotel_id"]).to_csv(output_path / "calibration_folds.csv", index=False)
    prediction_frame.to_csv(output_path / "test_predictions.csv", index=False)
    prediction_frame.groupby("target", sort=False).head(3).to_csv(output_path / "sample_predictions.csv", index=False)
    prediction_frame[prediction_frame.actual_rating != prediction_frame.predicted_rating].to_csv(output_path / "error_examples.csv", index=False)
    # Deterministic examples from validation only; no invented mention ground truth.
    mention_examples = []
    for idx in splits["validation"][:10]:
        text = df.loc[idx, "review_text"]
        for target in TARGET_COLUMNS:
            mention_examples.append({"review_id": df.loc[idx, "review_id"], "split": "validation", "target": target, "original_review": text, **evidence_for(text, target, "aspect_text")})
    pd.DataFrame(mention_examples).to_csv(output_path / "mention_detection_examples.csv", index=False)
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
    overall = validation[(validation.target == "overall_rating") & (validation.routing_strategy == "full_review")]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(overall.model, overall.validation_polarity_macro_f1, color="#176b87")
    ax.set(ylim=(0, 1), ylabel="Validation polarity macro-F1", title="Overall model selection (validation data only)")
    ax.tick_params(axis="x", labelrotation=12)
    fig.tight_layout()
    fig.savefig(output / "overall_model_comparison.png", dpi=180)
    plt.close(fig)
