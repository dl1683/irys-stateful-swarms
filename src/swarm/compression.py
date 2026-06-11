"""Deterministic blackboard compression for synthesis.

Addresses Open Research Question #3: "What's the best strategy for
selecting, ranking, or clustering entries to maximize information density
in the synthesis prompt without losing critical details?"

Current approach (curation.py): send ALL active entries grouped by document
to an LLM for curation. This module adds a deterministic pre-ranking and
diversity-aware selection step that runs BEFORE the LLM curation, ensuring
the LLM sees the highest-signal, most diverse subset when context windows
are tight.

Zero LLM calls. Pure deterministic scoring and selection.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from .blackboard import Blackboard
from .models import Entry


# ---------------------------------------------------------------------------
# Entry scoring
# ---------------------------------------------------------------------------

# Type weights: analysis and calculation carry more synthesis value than raw
# observations, which are abundant.
_TYPE_WEIGHTS = {
    "analysis": 1.0,
    "calculation": 0.95,
    "strategy": 0.80,
    "observation": 0.50,
    "gap": 0.60,
    "contradiction": 0.85,
}

# Tags that boost priority
_HIGH_VALUE_TAGS = {
    "entity_resolution", "debt_sensor", "state_conversion",
    "plan_coverage", "materiality:high",
}

_CONTENT_VALUE_PATTERNS = [
    (re.compile(r"\$\s*[\d,]+(?:\.\d+)?"), 0.15),       # dollar amounts
    (re.compile(r"\b\d+(?:\.\d+)?%"), 0.10),             # percentages
    (re.compile(r"\b(?:Section|Article|Clause)\s+\d"), 0.10),  # legal refs
    (re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"), 0.05),  # dates
    (re.compile(r"\b(?:shall|must|may not|prohibited)\b", re.I), 0.08),  # obligations
]


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
    source_doc_set: set[str],
    cross_ref_count: int = 0,
) -> ScoredEntry:
    """Score a single entry for synthesis relevance."""
    # Type score
    type_score = _TYPE_WEIGHTS.get(entry.type, 0.3)

    # Content density: presence of specific values
    content = entry.content or ""
    density = 0.0
    for pattern, weight in _CONTENT_VALUE_PATTERNS:
        if pattern.search(content):
            density += weight
    density = min(1.0, density)

    # Tag boost
    tags = set(entry.tags or [])
    tag_boost = 0.2 if tags & _HIGH_VALUE_TAGS else 0.0

    # Cross-reference score: entries that support or are supported by others
    # are more valuable (they're part of an evidence chain)
    ref_count = len(entry.supports_entries or []) + len(entry.contradicts_entries or [])
    cross_ref = min(0.3, ref_count * 0.05)

    # Source-diversity is assigned in batch scoring (score_all_entries), which
    # has the per-document counts needed to reward under-represented sources
    # and penalise over-represented ones. In isolation a single entry carries a
    # neutral 0.0 — the previous `doc in source_doc_set` check was always true
    # (source_doc_set is the set of all docs) and so added a constant to every
    # entry, contributing nothing to ranking.
    source_diversity = 0.0

    # Confidence factor
    conf_factor = min(1.0, entry.confidence) if entry.confidence > 0 else 0.5

    composite = (
        0.35 * type_score
        + 0.20 * density
        + 0.15 * tag_boost
        + 0.15 * cross_ref
        + 0.05 * source_diversity
        + 0.10 * conf_factor
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

def score_all_entries(blackboard: Blackboard) -> list[ScoredEntry]:
    """Score all active entries, adjusted for source-document diversity."""
    active = [e for e in blackboard.entries if e.status == "active"]
    if not active:
        return []

    # Count entries per source document
    doc_counts: dict[str, int] = {}
    for e in active:
        doc = e.source.document if e.source else "cross_cutting"
        doc_counts[doc or "cross_cutting"] = doc_counts.get(doc or "cross_cutting", 0) + 1

    total = len(active)
    doc_set = set(doc_counts.keys())

    # Cross-reference count per entry
    cross_ref_map: dict[str, int] = {}
    for e in active:
        for ref_id in (e.supports_entries or []):
            cross_ref_map[ref_id] = cross_ref_map.get(ref_id, 0) + 1
        for ref_id in (e.contradicts_entries or []):
            cross_ref_map[ref_id] = cross_ref_map.get(ref_id, 0) + 1

    scored: list[ScoredEntry] = []
    for e in active:
        se = score_entry(e, doc_set, cross_ref_map.get(e.id, 0))

        # Diversity adjustment: entries from overrepresented docs get penalized
        doc = e.source.document if e.source else "cross_cutting"
        doc_frac = doc_counts.get(doc or "cross_cutting", 0) / total
        if doc_frac > 0.4:
            # More than 40% of entries from one doc → slight penalty
            penalty = (doc_frac - 0.4) * 0.15
            se.source_diversity_bonus = max(0.0, se.source_diversity_bonus - penalty)
        elif doc_frac < 0.1 and total > 10:
            # Rare doc → bonus
            se.source_diversity_bonus += 0.10

        se.composite = round(
            0.35 * se.type_score
            + 0.20 * se.content_density
            + 0.15 * se.tag_boost
            + 0.15 * se.cross_ref_score
            + 0.05 * se.source_diversity_bonus
            + 0.10 * min(1.0, e.confidence if e.confidence > 0 else 0.5),
            4,
        )
        scored.append(se)

    return scored


# ---------------------------------------------------------------------------
# Compression strategies
# ---------------------------------------------------------------------------

@dataclass
class CompressionResult:
    """Result of compressing a blackboard for synthesis."""
    selected: list[ScoredEntry]       # entries to include in synthesis prompt
    deferred: list[ScoredEntry]       # entries excluded due to budget
    total_scored: int
    target_count: int
    actual_count: int
    coverage_by_doc: dict[str, int]   # how many entries selected per doc
    coverage_by_type: dict[str, int]  # how many entries selected per type
    tokens_saved_estimate: int        # rough estimate of tokens saved


def compress_for_synthesis(
    blackboard: Blackboard,
    *,
    target_count: int = 500,
    min_per_doc: int = 5,
    ensure_types: bool = True,
) -> CompressionResult:
    """Select the highest-value entries for synthesis, respecting diversity.

    Args:
        blackboard: the blackboard to compress.
        target_count: max entries to select.
        min_per_doc: minimum entries to include per document (diversity floor).
        ensure_types: if True, guarantees at least one entry per type present.

    Returns:
        CompressionResult with selected/deferred entries and coverage stats.
    """
    scored = score_all_entries(blackboard)
    if not scored:
        return CompressionResult(
            selected=[], deferred=[], total_scored=0,
            target_count=target_count, actual_count=0,
            coverage_by_doc={}, coverage_by_type={},
            tokens_saved_estimate=0,
        )

    # Sort by composite score descending
    scored.sort(key=lambda s: s.composite, reverse=True)

    # target_count is a HARD cap — the synthesis prompt has a finite context
    # budget, so no phase may push the selection past it. cap == 0 selects
    # nothing.
    cap = max(0, target_count)
    selected_ids: set[str] = set()
    selected: list[ScoredEntry] = []

    def _take(se: ScoredEntry) -> bool:
        if len(selected) >= cap or se.entry.id in selected_ids:
            return False
        selected.append(se)
        selected_ids.add(se.entry.id)
        return True

    # Phase 1: Diversity floor — up to min_per_doc per document, allocated
    # round-robin by within-doc rank so one dominant document cannot consume
    # the whole cap before minor documents get represented. (scored is already
    # sorted, so doc_entries lists are best-first.)
    doc_entries: dict[str, list[ScoredEntry]] = {}
    for se in scored:
        doc = se.entry.source.document if se.entry.source else "cross_cutting"
        doc_entries.setdefault(doc or "cross_cutting", []).append(se)

    for rank in range(min_per_doc):
        if len(selected) >= cap:
            break
        for entries in doc_entries.values():
            if rank < len(entries):
                _take(entries[rank])
                if len(selected) >= cap:
                    break

    # Phase 2: Type diversity — ensure at least one entry per type, within cap
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
        doc = se.entry.source.document if se.entry.source else "cross_cutting"
        coverage_doc[doc or "cross_cutting"] = coverage_doc.get(doc or "cross_cutting", 0) + 1
        coverage_type[se.entry.type] = coverage_type.get(se.entry.type, 0) + 1

    # Token estimate from the ACTUAL deferred content (~4 chars per token),
    # not a flat per-entry guess — entry sizes vary widely, so a constant
    # would misreport savings on short- or long-entry blackboards.
    tokens_saved = sum(len(se.entry.content or "") for se in deferred) // 4

    return CompressionResult(
        selected=selected,
        deferred=deferred,
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
) -> list[Entry]:
    """Return the top ``max_entries`` active entries ranked by synthesis
    relevance.

    This is a context-budget guard, NOT a replacement for curation.py's
    exhaustive per-document enumeration. curation deliberately tries to surface
    EVERY fact; pre-filtering defeats that. Use this only when the active set
    is larger than a hard context budget can hold and some bounded loss is
    unavoidable — feed the ranked subset to the LLM step instead of truncating
    arbitrarily. When everything fits, pass the full active set to curation.
    """
    result = compress_for_synthesis(blackboard, target_count=max_entries)
    return [se.entry for se in result.selected]


def compression_report(result: CompressionResult) -> dict:
    """Human-readable compression report for diagnostics."""
    return {
        "total_scored": result.total_scored,
        "selected": result.actual_count,
        "deferred": len(result.deferred),
        "target": result.target_count,
        "tokens_saved_estimate": result.tokens_saved_estimate,
        "coverage_by_doc": result.coverage_by_doc,
        "coverage_by_type": result.coverage_by_type,
        "top_10_scores": [
            {
                "id": se.entry.id,
                "type": se.entry.type,
                "composite": se.composite,
                "doc": se.entry.source.document if se.entry.source else "",
            }
            for se in result.selected[:10]
        ],
    }
