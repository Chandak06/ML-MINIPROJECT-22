"""Refresh fixed demonstration outputs from actual saved models after retraining.

These texts are diagnostics only. They never enter training, model selection,
calibration or threshold selection. No expected model ratings are hard-coded.
"""
import json
from .predict import ReviewPredictor

DIFFICULT_REVIEWS = [
    "The location was excellent and the staff were helpful, but the room was dirty.",
    "The room was spotless, but the receptionist was extremely rude.",
    "The hotel was expensive but perfectly located near the station.",
    "The staff were friendly.",
    "The room was not clean.",
    "The room was not bad.",
    "The location was terrible, but everything else about the hotel was fantastic.",
    "The room was filthy although the staff, location and food were excellent.",
    "The location was excellent, the staff were helpful and the hotel was great value for money, but the room was extremely dirty.",
]
DEMO_REVIEWS = [
    "The room was clean and spacious, but the staff was rude and the breakfast was disappointing.",
    "The staff were friendly and helpful. The room was dirty, but the location was close to the station.",
    "The room was noisy all night and I could not sleep.",
    "Beautiful view.",
    "The wifi was unusable and the conference room was dirty.",
]


def refresh_prediction_examples(root):
    predictor = ReviewPredictor(root / "models/real")
    baseline_path = root / "outputs/real/baseline_difficult_predictions.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf8")) if baseline_path.exists() else []
    old_by_review = {row["review"]: row["old_predictions"] for row in baseline}
    rows = [{"case": i + 1, "review": text, "old_predictions": old_by_review.get(text), "new_predictions": predictor.predict(text)}
            for i, text in enumerate(DIFFICULT_REVIEWS)]
    (root / "outputs/real/difficult_review_comparison.json").write_text(json.dumps({"scope": "All nine requested diagnostic sentences; qualitative examples, not an independent accuracy benchmark.", "examples": rows}, indent=2), encoding="utf8")
    (root / "outputs/demo_predictions.json").write_text(json.dumps({"scope": "Five fixed demonstration inputs; every output comes from saved real models, including mistakes and abstentions.",
                                                                   "examples": [{"review": text, "predictions": predictor.predict(text)} for text in DEMO_REVIEWS]}, indent=2), encoding="utf8")
