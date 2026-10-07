"""Input validation and deterministic preprocessing."""
from __future__ import annotations
import hashlib
import re
import unicodedata
from pathlib import Path
import pandas as pd
from .config import TARGET_COLUMNS


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Review text must be a string.")
    text = unicodedata.normalize("NFKC", text).lower().replace("\u2019", "'")
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"n't\b", " not", text)
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def rating_to_polarity(rating: int) -> str:
    value = float(rating)
    if not value.is_integer() or not 1 <= value <= 5:
        raise ValueError("Ratings must be integers from 1 to 5.")
    return "negative" if value <= 2 else "neutral" if value == 3 else "positive"


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ["review_text", *TARGET_COLUMNS] if c not in df]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.copy().reset_index(drop=True)
    if df.empty:
        raise ValueError("Dataset is empty.")
    if df.review_text.isna().any() or not df.review_text.map(lambda v: isinstance(v, str)).all():
        raise ValueError("Every review_text must be a non-empty string.")
    df["clean_review"] = df.review_text.map(clean_text)
    if not df.clean_review.str.contains(r"\w\w", regex=True).all():
        raise ValueError("Reviews must contain at least one word of two characters.")
    for target in TARGET_COLUMNS:
        values = pd.to_numeric(df[target], errors="coerce")
        invalid = df[target].notna() & (values.isna() | ~values.between(1, 5) | (values % 1 != 0))
        if invalid.any():
            raise ValueError(f"{target}: ratings must be 1-5 integers or blank.")
        df[target] = values
    if df.overall_rating.isna().any():
        raise ValueError("overall_rating cannot be missing.")
    if "review_id" not in df:
        df["review_id"] = [f"review-{i+1}" for i in range(len(df))]
    if df.review_id.isna().any() or df.review_id.astype(str).duplicated().any():
        raise ValueError("review_id values must be present and unique.")
    df["review_id"] = df.review_id.astype(str)
    if "hotel_id" in df:
        if df.hotel_id.isna().any() or df.hotel_id.astype(str).str.strip().eq("").any():
            raise ValueError("hotel_id must be present on every review when supplied; partial IDs would invalidate hotel-disjoint evaluation.")
        df["hotel_id"] = df.hotel_id.astype(str).str.strip()
    df["text_hash"] = df.clean_review.map(lambda s: hashlib.sha256(s.encode()).hexdigest())
    for _, group in df.groupby("text_hash"):
        if any(group[t].dropna().nunique() > 1 for t in TARGET_COLUMNS):
            raise ValueError("Duplicate review text has conflicting ratings.")
    return df.drop_duplicates("text_hash").reset_index(drop=True)


def load_dataset(path: str | Path) -> pd.DataFrame:
    return validate_dataset(pd.read_csv(path))
