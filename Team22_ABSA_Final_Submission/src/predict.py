"""Inference using validated manifests; predictions never trigger training."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
import joblib
from .config import TARGET_COLUMNS, ASPECT_DISPLAY_NAMES
from .utils import clean_text, rating_to_polarity
from .aspect_router import route_review, evidence_for, ROUTER_VERSION
from .confidence import probability_estimates, is_uncertain

ROOT = Path(__file__).resolve().parents[1]
MENTIONS = {
    "service": r"\b(staff|service|reception|receptionist|concierge|manager|employees?|front desk)\b",
    "rooms": r"\b(rooms?|bedrooms?|furniture|balcony|bathrooms?|beds?)\b",
    "location": r"\b(location|located|near|nearby|central|centre|center|airport|station|walk|walking|area)\b",
    "value": r"\b(value|price|priced|expensive|cheap|money|cost|overpriced|affordable|worth)\b",
    "cleanliness": r"\b(clean|cleanliness|dirty|filthy|hygienic|hygiene|housekeeping|dust|stains?|spotless|mould|mold)\b",
    "sleep_quality": r"\b(sleep|slept|noisy|noise|quiet|mattress|soundproof)\b",
    "business_service": r"\b(business|wifi|wi fi|internet|conference|meeting|printing|work desk)\b",
}


class ReviewPredictor:
    def __init__(self, model_dir=ROOT / "models" / "real"):
        model_dir = Path(model_dir)
        manifest_path = model_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError("Model manifest missing. Run python run_project.py first.")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf8"))
        self.models = {}
        for target in TARGET_COLUMNS:
            entry = self.manifest["targets"][target]
            if entry["status"] != "trained":
                continue
            path = model_dir / entry["file"]
            if not path.exists():
                raise FileNotFoundError(f"Model missing: {path.name}. Retrain the selected dataset.")
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["file_sha256"]:
                raise ValueError(f"Model integrity check failed for {path.name}.")
            # Load only project-created files. Joblib is unsafe for untrusted model downloads.
            self.models[target] = joblib.load(path)
            if self.manifest.get("schema_version", 1) >= 2:
                text_step = self.models[target].named_steps.get("text")
                if (text_step is None or text_step.target != target or text_step.strategy != entry["routing_strategy"]
                        or entry["mention_detection_version"] != ROUTER_VERSION):
                    raise ValueError(f"Model/manifest routing mismatch for {target}.")
                actual_method = getattr(self.models[target], "confidence_method_", "unavailable")
                if actual_method != entry["confidence_method"]:
                    raise ValueError(f"Model/manifest confidence mismatch for {target}.")

    def predict(self, review_text: str, include_unmentioned=False):
        cleaned = clean_text(review_text)
        if not re.search(r"\w\w", cleaned):
            raise ValueError("Enter a review containing at least one word of two characters.")
        if len(review_text) > 20000:
            raise ValueError("Please keep the review under 20,000 characters.")
        routes = route_review(review_text)
        predictions = {}
        for target in TARGET_COLUMNS:
            entry = self.manifest["targets"][target]
            legacy = self.manifest.get("schema_version", 1) < 2
            mentioned = (target == "overall_rating" or bool(re.search(MENTIONS[target], cleaned))) if legacy else routes[target]["mentioned"]
            strategy = entry.get("routing_strategy", "full_review")
            info = evidence_for(review_text, target, strategy, routes)
            info["mentioned"] = bool(mentioned)
            status = "predicted"
            model = self.models.get(target)
            candidate, candidate_polarity, confidence, rating_confidence = None, None, None, None
            threshold = entry.get("uncertainty", {}).get("threshold")
            if target not in self.models:
                status = "unavailable"
            elif not mentioned and not include_unmentioned:
                status = "not_mentioned"
            else:
                inputs = [cleaned] if legacy else [review_text]
                feature_text = inputs if legacy else model.named_steps["text"].transform(inputs)
                if model.named_steps["features"].transform(feature_text).nnz == 0:
                    status = "unknown_vocabulary"
                else:
                    candidate = int(model.predict(inputs)[0])
                    candidate_polarity = rating_to_polarity(candidate)
                    conf, rating_conf, _ = probability_estimates(model, inputs, [candidate])
                    confidence, rating_confidence = conf[0], rating_conf[0]
                    if is_uncertain(confidence, threshold):
                        status = "uncertain"
            if status in {"unavailable", "not_mentioned"}:
                info["evidence"] = ""
                info["evidence_source"] = "none"
                info["fallback_reason"] = None
            elif not mentioned and include_unmentioned:
                info.update(evidence_source="whole_review_fallback", fallback_reason="explicit_unmentioned_estimate")
            rating = candidate if status == "predicted" else None
            predictions[ASPECT_DISPLAY_NAMES[target]] = {
                "rating": rating, "polarity": rating_to_polarity(rating) if rating is not None else None,
                "status": status, "prediction_status": status, **info,
                "confidence": confidence, "rating_confidence": rating_confidence,
                "confidence_method": entry.get("confidence_method", "legacy_native_probability" if model is not None and hasattr(model, "predict_proba") else "unavailable"),
                "confidence_threshold": threshold, "low_confidence": status == "uncertain",
                "candidate_rating": candidate, "candidate_polarity": candidate_polarity,
                "estimate": bool(not mentioned and include_unmentioned and model is not None),
                "selected_model": entry.get("model"), "routing_strategy": strategy,
            }
        return predictions


def predict_review(review_text, model_dir=ROOT / "models" / "real", include_unmentioned=False):
    return ReviewPredictor(model_dir).predict(review_text, include_unmentioned)
