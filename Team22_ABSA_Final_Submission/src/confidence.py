"""Probability estimates and validation-only uncertainty thresholds."""
from __future__ import annotations
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from .utils import rating_to_polarity


def calibration_splits(y, groups=None, seed=42):
    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    folds = min(3, int(counts.min()))
    if groups is not None:
        groups = np.asarray(groups)
        folds = min(folds, len(np.unique(groups)))
    if folds < 2:
        return []
    splitter = (StratifiedGroupKFold(folds, shuffle=True, random_state=seed) if groups is not None
                else StratifiedKFold(folds, shuffle=True, random_state=seed))
    splits = list(splitter.split(np.zeros(len(y)), y, groups))
    if any(set(y[train]) != set(y) for train, _ in splits):
        return []
    return splits


def attach_confidence(model, X, y, groups=None, seed=42):
    """Fit SVM sigmoid calibration on out-of-fold margins, excluding outer test.

The base pipeline remains the rating predictor. Calibration supplies only
probabilities, so it cannot silently change the selected star predictions.
"""
    if hasattr(model, "predict_proba"):
        model.confidence_method_ = "native_probability_uncalibrated"
        return {"method": model.confidence_method_, "folds": []}
    splits = calibration_splits(y, groups, seed)
    if not splits:
        model.confidence_method_ = "unavailable_insufficient_calibration_data"
        return {"method": model.confidence_method_, "folds": []}
    calibrator = CalibratedClassifierCV(clone(model), method="sigmoid", cv=splits, ensemble=False, n_jobs=1)
    calibrator.fit(list(X), np.asarray(y))
    model.confidence_calibrator_ = calibrator
    model.confidence_method_ = "sigmoid_hotel_group_oof" if groups is not None else "sigmoid_stratified_oof"
    return {"method": model.confidence_method_, "folds": splits}


def probability_estimates(model, texts, predictions=None):
    texts = list(texts)
    predictions = model.predict(texts) if predictions is None else np.asarray(predictions)
    estimator = getattr(model, "confidence_calibrator_", model)
    if not hasattr(estimator, "predict_proba"):
        return [None] * len(texts), [None] * len(texts), None
    probabilities = estimator.predict_proba(texts)
    classes = np.asarray(estimator.classes_, dtype=int)
    class_polarities = np.asarray([rating_to_polarity(v) for v in classes])
    polarity_confidence, rating_confidence = [], []
    for row, prediction in zip(probabilities, predictions):
        polarity_confidence.append(float(row[class_polarities == rating_to_polarity(prediction)].sum()))
        rating_confidence.append(float(row[classes == prediction].sum()))
    return polarity_confidence, rating_confidence, probabilities


def select_uncertainty_threshold(confidence, correct):
    """Maximize balanced accuracy for detecting correct validation polarities.

Thresholds are observed validation probabilities (plus accept/reject-all
boundaries). Ties prefer more coverage. No arbitrary confidence percentage.
"""
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if len(confidence) != len(correct) or not len(correct):
        raise ValueError("Threshold selection needs paired validation outcomes.")
    if not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("Invalid probability estimates.")
    if correct.all():
        return {"threshold": 0.0, "selection_score": None, "reason": "all_validation_polarities_correct", "validation_n": len(correct)}
    if not correct.any():
        return {"threshold": float(np.nextafter(1., 2.)), "selection_score": None, "reason": "no_validation_polarities_correct", "validation_n": len(correct)}
    choices = []
    for threshold in np.r_[0., np.unique(confidence), np.nextafter(1., 2.)]:
        accept = confidence >= threshold
        balanced = .5 * (accept[correct].mean() + (~accept[~correct]).mean())
        choices.append((float(balanced), float(accept.mean()), -float(threshold)))
    score, coverage, minus_threshold = max(choices)
    return {"threshold": -minus_threshold, "selection_score": score, "validation_coverage": coverage,
            "validation_n": len(correct), "reason": "validation_correctness_balanced_accuracy"}


def is_uncertain(confidence, threshold):
    return confidence is not None and threshold is not None and confidence < threshold
