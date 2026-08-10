from __future__ import annotations

"""Explainable pair scoring based on names, metadata, and explicit conflicts."""

from difflib import SequenceMatcher

from .models import CompanyCandidate, CompanyRecord, ResolverConfig
from .metadata import record_address, record_domain, record_registration
from .normalization import company_initials, normalize_company_name
from .token_stats import WEAK_WORDS, weighted_token_overlap


def _is_short_or_acronym(tokens: tuple[str, ...]) -> bool:
    return len(tokens) <= 1 or any(len(token) <= 3 and token.isalpha() for token in tokens)


def _token_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def score_candidate(
    left: CompanyRecord,
    right: CompanyRecord,
    config: ResolverConfig | None = None,
    weights: dict[str, float] | None = None,
) -> CompanyCandidate:
    config = config or ResolverConfig()
    weights = weights or {}
    left_forms = normalize_company_name(left.name)
    right_forms = normalize_company_name(right.name)
    evidence: list[str] = []
    conflicts: list[str] = []
    score = 0.0

    exact_core = bool(left_forms.core and left_forms.core == right_forms.core)
    sorted_match = bool(left_forms.sorted_tokens and left_forms.sorted_tokens == right_forms.sorted_tokens)
    if exact_core:
        score = max(score, 0.96)
        evidence.append("exact_core_name")
    if sorted_match and not exact_core:
        score = max(score, 0.9)
        evidence.append("sorted_token_match")

    fuzzy = SequenceMatcher(None, left_forms.core, right_forms.core).ratio() if left_forms.core and right_forms.core else 0.0
    overlap = _token_overlap(left_forms.tokens, right_forms.tokens)
    rarity_overlap = weighted_token_overlap(left_forms.tokens, right_forms.tokens, weights)
    shared = set(left_forms.tokens) & set(right_forms.tokens)
    distinctive_shared = sorted(token for token in shared if token not in WEAK_WORDS)
    if distinctive_shared:
        evidence.append("shared_distinctive_tokens:" + ",".join(distinctive_shared))
    if overlap:
        evidence.append(f"token_overlap:{overlap:.3f}")
    if rarity_overlap:
        evidence.append(f"rarity_weighted_overlap:{rarity_overlap:.3f}")
    if fuzzy:
        evidence.append(f"fuzzy_ratio:{fuzzy:.3f}")

    score = max(score, (0.45 * fuzzy) + (0.45 * overlap))
    if distinctive_shared:
        score += min(0.18, 0.09 * len(distinctive_shared))
    if distinctive_shared and any(token in WEAK_WORDS for token in set(left_forms.tokens) ^ set(right_forms.tokens)):
        score += 0.08
        evidence.append("related_industry_terms")
    if shared and shared <= WEAK_WORDS:
        score -= 0.25

    left_domain = record_domain(left)
    right_domain = record_domain(right)
    if left_domain and right_domain:
        if left_domain == right_domain:
            score += 0.28
            evidence.append("same_domain")
        else:
            conflicts.append("domain_conflict")
            score -= 0.18

    left_registration = record_registration(left)
    right_registration = record_registration(right)
    if left_registration and right_registration:
        if left_registration == right_registration:
            score += 0.35
            evidence.append("same_registration_number")
        else:
            conflicts.append("registration_number_conflict")
            score -= 0.45

    left_address = record_address(left)
    right_address = record_address(right)
    if left_address and right_address and left_address == right_address:
        score += 0.22
        evidence.append("same_address")

    same_domain = bool(left_domain and right_domain and left_domain == right_domain)
    same_registration = bool(left_registration and right_registration and left_registration == right_registration)
    same_address = bool(left_address and right_address and left_address == right_address)
    left_initials = company_initials(left_forms.tokens)
    right_initials = company_initials(right_forms.tokens)
    acronym_match = bool(left_initials and left_initials == right_initials and left_forms.core != right_forms.core)
    if acronym_match:
        evidence.append("acronym_match")
        score = max(score, config.review_threshold)
    short_guard = _is_short_or_acronym(left_forms.tokens) or _is_short_or_acronym(right_forms.tokens)
    score = max(0.0, min(1.0, score))

    review_evidence = bool(exact_core or sorted_match or distinctive_shared or same_domain or same_address or acronym_match or score >= config.review_threshold)
    if conflicts:
        tier = "review" if review_evidence or same_registration else "reject"
    elif same_registration:
        tier = "auto_merge"
    elif exact_core and score >= config.auto_merge_threshold:
        tier = "auto_merge"
    elif review_evidence:
        tier = "review"
    elif short_guard:
        tier = "reject"
    else:
        tier = "reject"

    return CompanyCandidate(
        left_record_id=min(left.record_id, right.record_id),
        right_record_id=max(left.record_id, right.record_id),
        score=score,
        tier=tier,
        evidence=tuple(dict.fromkeys(evidence)),
        conflicts=tuple(dict.fromkeys(conflicts)),
    )
