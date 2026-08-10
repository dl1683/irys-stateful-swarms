from __future__ import annotations

"""Lightweight corpus-level token rarity features; no external NLP required."""

import math
from collections import Counter
from collections.abc import Iterable

from .models import CompanyRecord
from .normalization import normalize_company_name


WEAK_WORDS = {
    "capital",
    "chemical",
    "energy",
    "global",
    "group",
    "holding",
    "holdings",
    "industry",
    "industries",
    "petrochemical",
    "solution",
    "technology",
}


def token_weights(records: Iterable[CompanyRecord]) -> dict[str, float]:
    token_sets = [set(normalize_company_name(record.name).tokens) for record in records]
    total = max(len(token_sets), 1)
    document_frequency = Counter(token for tokens in token_sets for token in tokens)
    weights: dict[str, float] = {}
    for token, frequency in document_frequency.items():
        weight = math.log((1 + total) / (1 + frequency)) + 1.0
        if token in WEAK_WORDS:
            weight *= 0.35
        weights[token] = weight
    return weights


def weighted_token_overlap(left: Iterable[str], right: Iterable[str], weights: dict[str, float]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    union = left_set | right_set
    shared = left_set & right_set
    numerator = sum(weights.get(token, 1.0) for token in shared)
    denominator = sum(weights.get(token, 1.0) for token in union)
    return numerator / denominator if denominator else 0.0
