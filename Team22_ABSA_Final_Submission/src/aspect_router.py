"""Stateless, inspectable English clause routing; no corpus labels or learned rules.

Rules are fixed before evaluation. This is a heuristic evidence extractor, not a
syntactic parser or an aspect-span model. Shared clauses can belong to several
aspects. Unresolved references and ambiguous mentions remain explicit limitations.
"""
from __future__ import annotations

import re
import unicodedata
from sklearn.base import BaseEstimator, TransformerMixin
from .config import ASPECT_COLUMNS
from .utils import clean_text

ROUTER_VERSION = "clauses-v1"
PATTERNS = {
    "service": r"\b(staff|service|reception(?:ists?)?|concierge|managers?|employees?|personnel|waiters?|waitresses?|porters?|front desk|check[ -]?(?:in|out)|helpful|unhelpful|rude|polite|impolite|friendly|unfriendly)\b",
    "rooms": r"\b(rooms?|bedrooms?|suites?|bathrooms?|furniture|balcon(?:y|ies)|beds?|air condition(?:ing|er)?|a[ ./-]?c|spacious|cramped|facilities)\b",
    "location": r"\b(locations?|located|neighbou?rhood|stations?|airports?|transport|attractions?|central|centre|center|nearby|distance|area|near|walking distance)\b",
    "value": r"\b(prices?|priced|pricing|costs?|expensive|inexpensive|cheap|value|money|tariffs?|overpriced|affordable|worth|charges?)\b",
    "cleanliness": r"\b(clean|cleaned|cleaning|cleanliness|unclean|dirty|filthy|spotless|hygiene|hygienic|unhygienic|stains?|stained|dust|dusty|housekeeping|linen|mould|mold|grimy)\b",
    "sleep_quality": r"\b(sleep|sleeping|slept|noisy|noise|quiet|mattress(?:es)?|soundproof)\b",
    "business_service": r"\b(wi[ -]?fi|internet|meetings?|conference|printing|printers?|business|work desk|working desk)\b",
}
_RE = {key: re.compile(pattern, re.I) for key, pattern in PATTERNS.items()}
_PREDICATE = re.compile(r"\b(is|are|was|were|be|been|felt|seemed|looked|had|has|have|cost|costs|smelled|worked|did|does|offered|provided)\b", re.I)
_CONTRAST = re.compile(r"\s*,?\s*\b(?:but|although|though|however|whereas|yet)\b\s*,?\s*", re.I)


def split_sentences(text):
    if not isinstance(text, str):
        raise ValueError("Review text must be a string.")
    text = unicodedata.normalize("NFKC", text).replace("\u2019", "'")
    # Protect the common A.C. spelling; retain punctuation in returned evidence.
    text = re.sub(r"\bA\.C\.(?=\s|$)", "AC", text, flags=re.I)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|[\r\n]+|;\s*", text) if s.strip()]


def detect_mentions(text):
    """Return strong lexical/phrase mentions; never predict sentiment from keywords."""
    normalized = clean_text(text)
    result = {target for target, pattern in _RE.items() if pattern.search(normalized)}
    # A front desk is reception, not automatically a business amenity. Room
    # service alone is not evidence about the physical room.
    room_check = re.sub(r"\broom service\b", "", normalized)
    if not _RE["rooms"].search(room_check):
        result.discard("rooms")
    return result


def _coordinated_clauses(text):
    """Split independent coordinated clauses, while keeping noun/adjective lists."""
    for match in re.finditer(r",\s*|\s+\band\b\s+", text, flags=re.I):
        left, right = text[:match.start()].strip(), text[match.end():].strip()
        if not left or not right:
            continue
        # Both sides need a predicate and an aspect. Thus 'staff, location and
        # food were excellent' stays intact, but two independent opinions split.
        if (_PREDICATE.search(clean_text(left)) and _PREDICATE.search(clean_text(right))
                and detect_mentions(left) and detect_mentions(right)):
            return _coordinated_clauses(left) + _coordinated_clauses(right)
    return [text.strip(" ,")]


def split_clauses(text):
    clauses = []
    for sentence in split_sentences(text):
        # Avoid splitting the paired construction 'not only ... but also ...'.
        parts = [sentence] if re.search(r"\bnot only\b.*\bbut also\b", sentence, re.I) else _CONTRAST.split(sentence)
        for part in parts:
            if part.strip(" ,"):
                clauses.extend(_coordinated_clauses(part))
    return [c for c in clauses if c]


def route_review(text):
    """Return original evidence, detection status and transparent fallback reasons."""
    clauses = split_clauses(text)
    evidence = {target: [] for target in ASPECT_COLUMNS}
    previous = set()
    for clause in clauses:
        mentions = detect_mentions(clause)
        # Restrained anaphora: only an explicit 'it/they' continuation and one
        # prior physical/service referent, never an unrelated positive clause.
        if re.match(r"^(?:it|they|this room|the room)\b", clause, re.I):
            referents = previous & {"rooms", "service", "business_service"}
            if len(referents) == 1:
                mentions |= referents
        for target in mentions:
            evidence[target].append(clause)
        previous = mentions
    routed = {}
    for target in ASPECT_COLUMNS:
        selected = list(dict.fromkeys(evidence[target]))
        ambiguous_desk = (target == "business_service" and not selected
                          and re.search(r"\bdesk\b", text, re.I)
                          and not re.search(r"\bfront\s+desk\b", text, re.I))
        routed[target] = {
            "mentioned": bool(selected) or bool(ambiguous_desk),
            "clauses": selected,
            "aspect_text": " ".join(selected),
            "routing_certain": bool(selected),
            "fallback_reason": "ambiguous_desk_reference" if ambiguous_desk else (None if selected else "no_explicit_aspect_evidence"),
        }
    routed["overall_rating"] = {"mentioned": True, "clauses": clauses, "aspect_text": text.strip(), "routing_certain": True, "fallback_reason": None}
    return routed


def evidence_for(text, target, strategy="aspect_text", routes=None):
    if strategy not in {"full_review", "aspect_text"}:
        raise ValueError(f"Unknown routing strategy: {strategy}")
    route = (routes or route_review(text))[target]
    if target == "overall_rating" or strategy == "full_review":
        used, source, reason = text.strip(), "whole_review_selected", "validation_selected_full_review"
    elif route["routing_certain"] and route["aspect_text"]:
        used, source, reason = route["aspect_text"], "aspect_text", None
    else:
        used, source, reason = text.strip(), "whole_review_fallback", route["fallback_reason"]
    return {"evidence": used, "evidence_source": source, "fallback_reason": reason,
            "routed_aspect_text": route["aspect_text"], "mentioned": route["mentioned"]}


class AspectTextTransformer(TransformerMixin, BaseEstimator):
    """Keep routing inside the serialized pipeline so training and inference agree."""
    def __init__(self, target="overall_rating", strategy="full_review"):
        self.target = target
        self.strategy = strategy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [clean_text(evidence_for(text, self.target, self.strategy)["evidence"]) for text in X]
