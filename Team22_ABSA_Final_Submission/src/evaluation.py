"""Descriptive held-out diagnostics; these functions never select a model."""
import numpy as np
from .utils import rating_to_polarity


def _statistics(truth, pred):
    true_pol = np.where(truth <= 2, 0, np.where(truth == 3, 1, 2))
    pred_pol = np.where(pred <= 2, 0, np.where(pred == 3, 1, 2))
    matrix = np.bincount(3 * true_pol + pred_pol, minlength=9).reshape(3, 3)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    f1 = np.divide(2 * matrix.diagonal(), denominator, out=np.zeros(3), where=denominator > 0)
    return {"star_accuracy": float((truth == pred).mean()),
            "within_one_star_accuracy": float((abs(truth - pred) <= 1).mean()),
            "star_mae": float(abs(truth - pred).mean()),
            "polarity_accuracy": float((true_pol == pred_pol).mean()),
            "polarity_macro_f1": float(f1.mean())}


def hotel_bootstrap(truth, pred, hotels, old_pred=None, seed=42, repetitions=500):
    """Percentile intervals resample whole hotels, preserving within-hotel rows.

These conditional intervals do not cover model fitting/selection uncertainty.
Paired old/new deltas use the identical sampled hotels.
"""
    truth, pred, hotels = np.asarray(truth, int), np.asarray(pred, int), np.asarray(hotels)
    if not len(truth):
        return []
    groups = [np.flatnonzero(hotels == hotel) for hotel in np.unique(hotels)]
    rng = np.random.default_rng(seed)
    point = _statistics(truth, pred)
    draws = {k: [] for k in point}
    deltas = {k: [] for k in point}
    old_pred = None if old_pred is None else np.asarray(old_pred, int)
    old_point = None if old_pred is None else _statistics(truth, old_pred)
    for _ in range(repetitions):
        indices = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        sample = _statistics(truth[indices], pred[indices])
        previous = None if old_pred is None else _statistics(truth[indices], old_pred[indices])
        for key in point:
            draws[key].append(sample[key])
            if previous is not None:
                deltas[key].append(sample[key] - previous[key])
    return [{"metric": key, "estimate": value, "ci_low": float(np.quantile(draws[key], .025)),
             "ci_high": float(np.quantile(draws[key], .975)), "hotels": len(groups), "reviews": len(truth),
             "bootstrap_repetitions": repetitions, "seed": seed,
             **({"old_estimate": old_point[key], "delta": value - old_point[key],
                 "delta_ci_low": float(np.quantile(deltas[key], .025)),
                 "delta_ci_high": float(np.quantile(deltas[key], .975))} if old_point else {})}
            for key, value in point.items()]


def mixed_label_mask(frame, target, targets):
    """A proxy subset: observed aspect polarities disagree within a review.

This describes rating disagreement, not a human-annotated contamination cause.
Overall is deliberately excluded from the aspect disagreement definition.
"""
    def mixed(row):
        values = [rating_to_polarity(row[t]) for t in targets if t != "overall_rating" and not np.isnan(row[t])]
        return len(set(values)) > 1
    return frame.apply(mixed, axis=1).to_numpy(bool)
