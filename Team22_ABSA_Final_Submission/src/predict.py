"""Inference using validated manifests; predictions never trigger training."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
import joblib
from .config import TARGET_COLUMNS, ASPECT_DISPLAY_NAMES
from .utils import clean_text, rating_to_polarity

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

    def predict(self, review_text: str, include_unmentioned=False):
        cleaned = clean_text(review_text)
        if not re.search(r"\w\w", cleaned):
            raise ValueError("Enter a review containing at least one word of two characters.")
        if len(review_text) > 20000:
            raise ValueError("Please keep the review under 20,000 characters.")
        predictions = {}
        for target in TARGET_COLUMNS:
            mentioned = target == "overall_rating" or bool(re.search(MENTIONS[target], cleaned))
            status = "predicted"
            if target not in self.models:
                status = "unavailable"
            elif not mentioned and not include_unmentioned:
                status = "not_mentioned"
            elif self.models[target].named_steps["features"].transform([cleaned]).nnz == 0:
                status = "unknown_vocabulary"
            rating = int(self.models[target].predict([cleaned])[0]) if status == "predicted" else None
            predictions[ASPECT_DISPLAY_NAMES[target]] = {
                "rating": rating, "polarity": rating_to_polarity(rating) if rating is not None else None,
                "status": status, "mentioned": mentioned,
            }
        return predictions


def predict_review(review_text, model_dir=ROOT / "models" / "real", include_unmentioned=False):
    return ReviewPredictor(model_dir).predict(review_text, include_unmentioned)
