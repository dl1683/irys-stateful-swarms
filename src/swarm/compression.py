"""Deterministic blackboard compression for synthesis.

Addresses Open Research Question #3: "What's the best strategy for
selecting, ranking, or clustering entries to maximize information density
in the synthesis prompt without losing critical details?"

Multilingual, multi-domain, extensible compression that:

1. Scores entries using language-agnostic statistical and structural
   features — no hardcoded English or legal vocabulary.
2. Optionally defers ambiguous rankings to a ModelCaller (hybrid path).
3. Externalises all thresholds and optional term-lists via
   ``CompressionProfile``.

Zero LLM calls in the deterministic path. Pure scoring and selection.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from .blackboard import Blackboard
from .models import Entry
from .worker_dispatch import call_model


# ---------------------------------------------------------------------------
# CompressionProfile — all tunables externalised, no hardcoded domain logic
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompressionProfile:
    """Externalised configuration for compression scoring.

    No language- or domain-specific content is baked in.  ``type_weights``
    mirrors the system's own taxonomy (language-agnostic), while
    ``high_value_tags`` and ``content_boost_patterns`` default to *empty*
    and can be populated from a model pass or external config file.

    All statistical thresholds work across languages and domains.
    """

    # --- type taxonomy (system-defined, not language-specific) ---------------
    type_weights: dict[str, float] = field(default_factory=lambda: {
        "analysis": 1.0,
        "calculation": 0.95,
        "strategy": 0.80,
        "observation": 0.50,
        "gap": 0.60,
        "contradiction": 0.85,
    })
    default_type_weight: float = 0.30

    # Optional domain-specific tags — empty by default; populate from an
    # external config or a model pass, never hardcoded to one domain.
    high_value_tags: frozenset[str] = field(default_factory=frozenset)

    # Optional content-boost regex patterns ``(pattern_str, weight)``.
    # Empty by default.  Populate from model-generated domain config or
    # an external config file.
    content_boost_patterns: tuple[tuple[str, float], ...] = ()

    # --- composite weight vector (sums to 1.0) -------------------------------
    weight_type: float = 0.28
    weight_density: float = 0.25
    weight_tag: float = 0.12
    weight_cross_ref: float = 0.15
    weight_diversity: float = 0.05
    weight_confidence: float = 0.15

    # --- diversity thresholds -------------------------------------------------
    overrep_threshold: float = 0.40
    overrep_penalty_factor: float = 0.15
    underrep_threshold: float = 0.10
    underrep_bonus: float = 0.10
    underrep_min_total: int = 10

    # --- cross-reference tuning -----------------------------------------------
    cross_ref_per_link: float = 0.06
    cross_ref_max: float = 0.30

    # --- confidence -----------------------------------------------------------
    unknown_confidence: float = 0.50

    # --- hybrid / model-caller thresholds -------------------------------------
    low_confidence_threshold: float = 0.45
    max_model_candidates: int = 50

    @classmethod
    def from_blackboard(cls, blackboard: Blackboard) -> "CompressionProfile":
        """Infer a profile from the blackboard without LLM calls.

        Derives type weights from the actual distribution of entry types
        present — rare types get slightly higher weight — so the profile
        adapts to whatever domain the blackboard covers.
        """
        active = [e for e in blackboard.entries if e.status == "active"]
        if not active:
            return cls()

        type_counts: dict[str, int] = {}
        for e in active:
            type_counts[e.type] = type_counts.get(e.type, 0) + 1

        total = len(active)
        type_weights: dict[str, float] = {}
        for t, count in type_counts.items():
            freq = count / total
            # Rarer types → higher weight (inverse frequency)
            type_weights[t] = round(min(1.0, 0.5 + (1.0 - freq) * 0.5), 2)

        return cls(type_weights=type_weights)


# ---------------------------------------------------------------------------
# Language-agnostic feature extraction (Unicode-aware)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """NFKC normalise + casefold — never ``.lower()``."""
    return unicodedata.normalize("NFKC", text).casefold()


def _unicode_tokenize(text: str) -> list[str]:
    """Split on non-word boundaries respecting all Unicode scripts."""
    return [
        t
        for t in re.split(r"[^\w]+", unicodedata.normalize("NFKC", text), flags=re.UNICODE)
        if t
    ]


def _numeric_density(text: str) -> float:
    """Fraction of characters that are Unicode-numeric (any script).

    Numbers indicate data, measurements, quantities — a universal signal.
    Unicode general category ``N`` covers decimal digits (Nd), letter
    numbers (Nl), and other numbers (No) across all scripts.
    """
    if not text:
        return 0.0
    count = sum(1 for c in text if unicodedata.category(c).startswith("N"))
    # Scale so that ~5 % numeric content → ~1.0
    return min(1.0, count / max(len(text), 1) * 20)


def _structural_density(text: str) -> float:
    """Density of structural / organisational markers.

    Lists, enumerations, and structured content universally use punctuation
    and special characters for organisation, regardless of language.
    """
    if not text:
        return 0.0
    count = sum(
        1
        for c in text
        if unicodedata.category(c).startswith("P") or c in "•·▪◦●○◆◇►▸‣⁃"
    )
    return min(1.0, count / max(len(text), 1) * 15)


def _token_diversity(text: str) -> float:
    """Unique-token ratio (vocabulary richness).

    A higher ratio suggests more specific, less repetitive content.
    """
    tokens = _unicode_tokenize(text)
    if len(tokens) < 2:
        return 0.0
    unique = {_normalize(t) for t in tokens}
    return len(unique) / len(tokens)


def _avg_token_length(text: str) -> float:
    """Average token length — longer tokens often indicate specificity.

    Normalised to [0, 1] where 1 ≈ very long / specific tokens.
    """
    tokens = _unicode_tokenize(text)
    if not tokens:
        return 0.0
    avg = sum(len(t) for t in tokens) / len(tokens)
    return min(1.0, avg / 12.0)


def _line_count_feature(text: str) -> float:
    """Multi-line content often indicates structured analysis."""
    if not text:
        return 0.0
    lines = text.count("\n") + 1
    return min(1.0, lines / 10.0)


def content_density(text: str, profile: CompressionProfile) -> float:
    """Combined language-agnostic content density score in [0, 1].

    Uses only structural and statistical features — no vocabulary
    assumptions.  If ``profile.content_boost_patterns`` is populated
    (e.g. from a model-generated domain config), those regexes add
    an optional domain-specific boost.
    """
    if not text:
        return 0.0

    base = (
        0.30 * _numeric_density(text)
        + 0.25 * _structural_density(text)
        + 0.20 * _token_diversity(text)
        + 0.15 * _avg_token_length(text)
        + 0.10 * _line_count_feature(text)
    )

    # Optional domain-specific boost from externalised patterns
    boost = 0.0
    if profile.content_boost_patterns:
        for item in profile.content_boost_patterns:
            try:
                pattern_str, weight = item
                if re.search(pattern_str, text, re.UNICODE | re.IGNORECASE):
                    boost += weight
            except (re.error, ValueError, TypeError):
                pass  # skip malformed patterns/entries gracefully

    return min(1.0, base + boost)


# ---------------------------------------------------------------------------
# Entry scoring
# ---------------------------------------------------------------------------

@dataclass
class ScoredEntry:
    """An entry with its computed synthesis-relevance score."""

    entry: Entry
    type_score: float
    content_density: float
    tag_boost: float
    cross_ref_score: float
    source_diversity_bonus: float
    composite: float


def score_entry(
    entry: Entry,
    profile: CompressionProfile,
    cross_ref_count: int = 0,
) -> ScoredEntry:
    """Score a single entry for synthesis relevance.

    All features are language- and domain-agnostic; tunables come from
    *profile*, not from hardcoded constants.
    """
    # Type score
    type_score = profile.type_weights.get(entry.type, profile.default_type_weight)

    # Content density — statistical + structural features only
    density = content_density(entry.content or "", profile)

    # Tag boost (empty high_value_tags → no boost)
    tags = set(entry.tags or [])
    tag_boost = 0.2 if tags & profile.high_value_tags else 0.0

    # Cross-reference score
    cross_ref = min(profile.cross_ref_max, cross_ref_count * profile.cross_ref_per_link)

    # Source diversity is assigned in batch scoring (score_all_entries),
    # which has the per-document counts needed for the adjustment.
    source_diversity = 0.0

    # Confidence factor — treat 0 or missing as unknown
    raw_conf = getattr(entry, "confidence", None)
    conf = (
        raw_conf
        if (isinstance(raw_conf, (int, float)) and raw_conf > 0)
        else profile.unknown_confidence
    )
    conf_factor = min(1.0, conf)

    composite = (
        profile.weight_type * type_score
        + profile.weight_density * density
        + profile.weight_tag * tag_boost
        + profile.weight_cross_ref * cross_ref
        + profile.weight_diversity * source_diversity
        + profile.weight_confidence * conf_factor
    )

    return ScoredEntry(
        entry=entry,
        type_score=round(type_score, 4),
        content_density=round(density, 4),
        tag_boost=round(tag_boost, 4),
        cross_ref_score=round(cross_ref, 4),
        source_diversity_bonus=round(source_diversity, 4),
        composite=round(composite, 4),
    )


# ---------------------------------------------------------------------------
# Batch scoring with diversity adjustment
# ---------------------------------------------------------------------------

def _get_source_doc(entry: Entry) -> str:
    """Safely extract the source document name."""
    source = getattr(entry, "source", None)
    doc = getattr(source, "document", None) if source else None
    return doc or "cross_cutting"


def _get_cross_refs(entry: Entry) -> tuple[list[str], list[str]]:
    """Safely extract supports/contradicts lists."""
    supports = getattr(entry, "supports_entries", None) or []
    contradicts = getattr(entry, "contradicts_entries", None) or []
    return supports, contradicts


def score_all_entries(
    blackboard: Blackboard,
    profile: CompressionProfile | None = None,
) -> list[ScoredEntry]:
    """Score all active entries, adjusted for source-document diversity."""
    if profile is None:
        profile = CompressionProfile.from_blackboard(blackboard)

    active = [e for e in blackboard.entries if e.status == "active"]
    if not active:
        return []

    # Count entries per source document
    doc_counts: dict[str, int] = {}
    for e in active:
        doc = _get_source_doc(e)
        doc_counts[doc] = doc_counts.get(doc, 0) + 1

    total = len(active)

    # Cross-reference count per entry (how many other entries reference it)
    cross_ref_map: dict[str, int] = {}
    for e in active:
        supports, contradicts = _get_cross_refs(e)
        for ref_id in supports:
            cross_ref_map[ref_id] = cross_ref_map.get(ref_id, 0) + 1
        for ref_id in contradicts:
            cross_ref_map[ref_id] = cross_ref_map.get(ref_id, 0) + 1

    scored: list[ScoredEntry] = []
    for e in active:
        se = score_entry(e, profile, cross_ref_map.get(e.id, 0))

        # Diversity adjustment: penalise over-represented docs,
        # bonus for under-represented docs.
        doc = _get_source_doc(e)
        doc_frac = doc_counts.get(doc, 0) / total

        if doc_frac > profile.overrep_threshold:
            penalty = (doc_frac - profile.overrep_threshold) * profile.overrep_penalty_factor
            se.source_diversity_bonus = round(
                se.source_diversity_bonus - penalty, 4
            )
        elif doc_frac < profile.underrep_threshold and total > profile.underrep_min_total:
            se.source_diversity_bonus = round(
                se.source_diversity_bonus + profile.underrep_bonus, 4
            )

        # Recompute composite with updated diversity
        raw_conf = getattr(e, "confidence", None)
        conf = (
            raw_conf
            if (isinstance(raw_conf, (int, float)) and raw_conf > 0)
            else profile.unknown_confidence
        )
        se.composite = round(
            profile.weight_type * se.type_score
            + profile.weight_density * se.content_density
            + profile.weight_tag * se.tag_boost
            + profile.weight_cross_ref * se.cross_ref_score
            + profile.weight_diversity * se.source_diversity_bonus
            + profile.weight_confidence * min(1.0, conf),
            4,
        )
        scored.append(se)

    return scored


# ---------------------------------------------------------------------------
# Hybrid path — optional model adjudication
# ---------------------------------------------------------------------------

def _model_adjudicate(
    candidates: list[ScoredEntry],
    blackboard: Blackboard,
    caller: Any,
    profile: CompressionProfile,
) -> list[ScoredEntry]:
    """Re-score uncertain entries via a model for better ranking.

    The model receives the task context and entry previews, and returns
    relevance scores that are blended with the deterministic composite.
    Falls back to the original ranking on any error.
    """
    if not candidates:
        return candidates

    task_ctx = (blackboard.task_instruction or "")[:500]

    summaries: list[str] = []
    for i, se in enumerate(candidates):
        preview = (se.entry.content or "")[:300]
        summaries.append(f"[{i}] type={se.entry.type} | {preview}")

    prompt = (
        f"Task:\n{task_ctx}\n\n"
        f"Rate each entry's relevance (0-10) for the task above.\n"
        f"Return a JSON array: "
        f'[{{"index": 0, "score": 8}}, {{"index": 1, "score": 3}}]\n\n'
        + "\n".join(summaries)
    )

    try:
        payload, _ = call_model(caller, prompt, max_tokens=500)
    except Exception:
        return candidates  # graceful fallback

    if not isinstance(payload, list):
        return candidates

    # Apply model scores: blend 60 % deterministic + 40 % model
    for item in payload:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        model_score = item.get("score")
        if isinstance(idx, int) and isinstance(model_score, (int, float)):
            if 0 <= idx < len(candidates):
                model_norm = max(0.0, min(1.0, float(model_score) / 10.0))
                old = candidates[idx].composite
                candidates[idx].composite = round(0.6 * old + 0.4 * model_norm, 4)

    candidates.sort(key=lambda s: s.composite, reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Compression strategies
# ---------------------------------------------------------------------------

@dataclass
class CompressionResult:
    """Result of compressing a blackboard for synthesis."""

    selected: list[ScoredEntry]
    deferred: list[ScoredEntry]
    review_candidates: list[ScoredEntry]  # uncertain entries needing review
    total_scored: int
    target_count: int
    actual_count: int
    coverage_by_doc: dict[str, int]
    coverage_by_type: dict[str, int]
    tokens_saved_estimate: int


def compress_for_synthesis(
    blackboard: Blackboard,
    *,
    profile: CompressionProfile | None = None,
    caller: Any | None = None,
    target_count: int = 500,
    min_per_doc: int = 5,
    ensure_types: bool = True,
) -> CompressionResult:
    """Select the highest-value entries for synthesis, respecting diversity.

    Args:
        blackboard: the blackboard to compress.
        profile: scoring profile; ``None`` → inferred from blackboard.
        caller: optional ``ModelCaller`` for hybrid adjudication of
            uncertain entries.  When ``None``, uncertain entries are
            surfaced as ``review_candidates`` instead.
        target_count: max entries to select.
        min_per_doc: minimum entries to include per document (diversity floor).
        ensure_types: if True, guarantees at least one entry per type present.

    Returns:
        CompressionResult with selected/deferred entries and coverage stats.
    """
    if profile is None:
        profile = CompressionProfile.from_blackboard(blackboard)

    scored = score_all_entries(blackboard, profile)
    if not scored:
        return CompressionResult(
            selected=[],
            deferred=[],
            review_candidates=[],
            total_scored=0,
            target_count=target_count,
            actual_count=0,
            coverage_by_doc={},
            coverage_by_type={},
            tokens_saved_estimate=0,
        )

    # Sort by composite score descending
    scored.sort(key=lambda s: s.composite, reverse=True)

    # --- Hybrid path: adjudicate uncertain entries ---
    threshold = profile.low_confidence_threshold
    uncertain = [se for se in scored if se.composite < threshold]
    confident = [se for se in scored if se.composite >= threshold]

    review_candidates: list[ScoredEntry] = []

    if uncertain:
        if caller is not None:
            max_send = min(len(uncertain), profile.max_model_candidates)
            adjudicated = _model_adjudicate(
                uncertain[:max_send], blackboard, caller, profile
            )
            remaining = uncertain[max_send:]
            scored = sorted(
                confident + adjudicated + remaining,
                key=lambda s: s.composite,
                reverse=True,
            )
        else:
            # No caller → surface uncertain entries as review candidates.
            # They still participate in selection; the consumer is informed
            # that they need external validation.
            review_candidates = list(uncertain)

    # --- Selection pipeline ---
    cap = max(0, target_count)
    selected_ids: set[str] = set()
    selected: list[ScoredEntry] = []

    def _take(se: ScoredEntry) -> bool:
        if len(selected) >= cap or se.entry.id in selected_ids:
            return False
        selected.append(se)
        selected_ids.add(se.entry.id)
        return True

    # Phase 1: Diversity floor — round-robin by within-doc rank so one
    # dominant document cannot consume the whole cap before minor documents
    # get represented.
    doc_entries: dict[str, list[ScoredEntry]] = {}
    for se in scored:
        doc = _get_source_doc(se.entry)
        doc_entries.setdefault(doc, []).append(se)

    for rank in range(min_per_doc):
        if len(selected) >= cap:
            break
        for entries in doc_entries.values():
            if rank < len(entries):
                _take(entries[rank])
                if len(selected) >= cap:
                    break

    # Phase 2: Type diversity — ensure at least one entry per type
    if ensure_types:
        type_present = {se.entry.type for se in selected}
        missing_types = {se.entry.type for se in scored} - type_present
        for se in scored:
            if not missing_types or len(selected) >= cap:
                break
            if se.entry.type in missing_types and _take(se):
                missing_types.discard(se.entry.type)

    # Phase 3: Fill remaining slots by score
    for se in scored:
        if len(selected) >= cap:
            break
        _take(se)

    deferred = [se for se in scored if se.entry.id not in selected_ids]

    # Coverage stats
    coverage_doc: dict[str, int] = {}
    coverage_type: dict[str, int] = {}
    for se in selected:
        doc = _get_source_doc(se.entry)
        coverage_doc[doc] = coverage_doc.get(doc, 0) + 1
        coverage_type[se.entry.type] = coverage_type.get(se.entry.type, 0) + 1

    # Token estimate from actual deferred content (~4 chars per token)
    tokens_saved = sum(len(se.entry.content or "") for se in deferred) // 4

    return CompressionResult(
        selected=selected,
        deferred=deferred,
        review_candidates=review_candidates,
        total_scored=len(scored),
        target_count=target_count,
        actual_count=len(selected),
        coverage_by_doc=coverage_doc,
        coverage_by_type=coverage_type,
        tokens_saved_estimate=tokens_saved,
    )


# ---------------------------------------------------------------------------
# Ranked entry list for the curation prompt
# ---------------------------------------------------------------------------

def ranked_entries_for_curation(
    blackboard: Blackboard,
    max_entries: int = 500,
    profile: CompressionProfile | None = None,
    caller: Any | None = None,
) -> list[Entry]:
    """Return the top ``max_entries`` active entries ranked by synthesis
    relevance.

    This is a context-budget guard, NOT a replacement for curation.py's
    exhaustive per-document enumeration.  Use this only when the active set
    exceeds a hard context budget and some bounded loss is unavoidable.
    When everything fits, pass the full active set to curation.
    """
    result = compress_for_synthesis(
        blackboard,
        profile=profile,
        caller=caller,
        target_count=max_entries,
    )
    return [se.entry for se in result.selected]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def compression_report(result: CompressionResult) -> dict:
    """Human-readable compression report for diagnostics."""
    return {
        "total_scored": result.total_scored,
        "selected": result.actual_count,
        "deferred": len(result.deferred),
        "review_candidates": len(result.review_candidates),
        "target": result.target_count,
        "tokens_saved_estimate": result.tokens_saved_estimate,
        "coverage_by_doc": result.coverage_by_doc,
        "coverage_by_type": result.coverage_by_type,
        "top_10_scores": [
            {
                "id": se.entry.id,
                "type": se.entry.type,
                "composite": se.composite,
                "doc": _get_source_doc(se.entry),
            }
            for se in result.selected[:10]
        ],
    }
