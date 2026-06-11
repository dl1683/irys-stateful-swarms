"""Deterministic cross-document entity resolution without LLM calls.

Addresses Open Research Question #1: near-duplicate entities extracted by
workers ("Zenith Petrochem" vs "Zenith Petrochemical", "Pinnacle Industrial
Solutions, Inc." vs "Pinnacle Industrial Solutions") that no worker flags.

This module scans the blackboard for observation entries containing entity
names, clusters near-duplicates using multi-signal string similarity, and
creates contradiction/mapping entries so downstream analysis workers can
reconcile them.

No LLM calls. Pure deterministic string processing.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .blackboard import Blackboard
from .models import Entry, EntrySource, WorkerRecord, gen_entry_id


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_SUFFIXES = {
    "inc", "inc.", "llc", "llc.", "ltd", "ltd.", "corp", "corp.",
    "co", "co.", "plc", "plc.", "gmbh", "gmbh.", "s.a.", "sa",
    "n.v.", "nv", "b.v.", "bv", "ag", "sas", "sarl",
    "limited", "corporation", "incorporated", "company",
    "partners", "partnership", "associates", "group", "holdings",
}


def _normalize_entity(text: str) -> str:
    """Collapse an entity name to a canonical comparison form."""
    s = text.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)          # strip punctuation
    s = re.sub(r"\s+", " ", s).strip()       # collapse whitespace
    words = s.split()
    # Drop legal-entity suffixes for matching
    words = [w for w in words if w not in _SUFFIXES]
    return " ".join(words)


def _depluralize(norm: str) -> str:
    """Collapse a trailing plural 's' on each token for plural-variant detection."""
    return " ".join(
        t[:-1] if t.endswith("s") and len(t) > 3 else t
        for t in norm.split()
    )


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------

def _trigrams(text: str) -> set[str]:
    words = text.split()
    joined = " ".join(words)
    if len(joined) < 3:
        return {joined}
    return {joined[i:i + 3] for i in range(len(joined) - 2)}


def _jaro_winkler(s1: str, s2: str) -> float:
    """Jaro-Winkler similarity (0..1). Fast, no deps."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    max_dist = max(len1, len2) // 2 - 1
    if max_dist < 0:
        max_dist = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (
        matches / len1
        + matches / len2
        + (matches - transpositions / 2) / matches
    ) / 3.0

    # Winkler prefix bonus
    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    return jaro + prefix * 0.1 * (1 - jaro)


def _levenshtein_ratio(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity (0..1)."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    # Use two-row DP (memory efficient)
    prev = list(range(len2 + 1))
    curr = [0] * (len2 + 1)
    for i in range(1, len1 + 1):
        curr[0] = i
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost, # substitution
            )
        prev, curr = curr, prev
    dist = prev[len2]
    return 1.0 - dist / max(len1, len2)


def _token_overlap(norm1: str, norm2: str) -> float:
    """Jaccard token overlap after normalization."""
    t1 = set(norm1.split())
    t2 = set(norm2.split())
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def _is_prefix_variant(norm1: str, norm2: str) -> bool:
    """True if one name is a prefix of the other (after suffix stripping).

    Catches both:
    - word-level: "northbrook capital markets" vs "northbrook capital markets llc"
    - char-level within final word: "zenith petrochem" vs "zenith petrochemical"
    """
    w1, w2 = norm1.split(), norm2.split()
    if not w1 or not w2:
        return False
    shorter, longer = (w1, w2) if len(w1) <= len(w2) else (w2, w1)
    # Word-level prefix (shorter is a prefix word-list of longer)
    if shorter == longer[:len(shorter)]:
        return True
    # Same length — check if all words except the last match and last is char-prefix
    if len(w1) == len(w2):
        for i in range(len(w1) - 1):
            if w1[i] != w2[i]:
                return False
        a, b = w1[-1], w2[-1]
        return a.startswith(b) or b.startswith(a)
    return False


# ---------------------------------------------------------------------------
# Composite score & threshold
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    entry_a: Entry
    entry_b: Entry
    raw_name_a: str
    raw_name_b: str
    norm_a: str
    norm_b: str
    jaro_winkler: float
    levenshtein: float
    trigram_jaccard: float
    token_overlap: float
    is_prefix: bool
    composite: float
    confidence: float


def _composite_score(jw: float, lev: float, tri: float, tok: float,
                     is_prefix: bool) -> float:
    """Weighted composite similarity. Prefix variants get a bonus."""
    base = 0.35 * jw + 0.25 * lev + 0.25 * tri + 0.15 * tok
    if is_prefix and base >= 0.5:
        base = min(1.0, base + 0.15)
    return round(base, 4)


# Composite-score thresholds. Fuzzy (non-exact) matches above MATCH_THRESHOLD
# are surfaced as *candidates* for review, not asserted as identical — pure
# string similarity cannot separate a true variant (Petrochem/Petrochemical)
# from distinct entities sharing a prefix (Petrochem/Petroleum).
MATCH_THRESHOLD = 0.72        # composite >= this → candidate same-entity match
HIGH_CONFIDENCE = 0.85        # composite >= this → strong candidate


# ---------------------------------------------------------------------------
# Entity extraction from blackboard entries
# ---------------------------------------------------------------------------

# Any sequence of capitalized words that looks like an organization name.
_ORG_LIKE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+"
    r"(?:[,\s]+(?:Inc|LLC|Ltd|Corp|Corporation|Company|Holdings|Group|Partners|Associates|Bank|Capital|Finance|Equipment|Industries|Petrochem(?:ical)?|Solutions))?)"
    r"(?:\.|,|;|\s|$)"
)

# Capitalized words that lead a sentence or label but are not part of the
# entity name ("For Zenith ...", "The Commitment Letter", "Notify ..."). These
# are stripped from the front (and a trailing bare "No"/"No.") of a captured
# name so canonical forms are clean.
_LEADING_NOISE = {
    "the", "a", "an", "for", "and", "all", "any", "each", "such", "this",
    "that", "these", "those", "to", "of", "in", "on", "by", "per", "as", "at",
    "or", "but", "if", "no", "see", "notify", "its", "their", "our", "we",
    "you", "it", "is", "are", "was", "were", "from", "with", "into", "between",
}

# Common legal/financial *defined terms* and generic concepts. A candidate
# whose every token is one of these is NOT a cross-document entity — it is a
# defined term or concept ("Term Loan", "Closing Date", "Administrative Agent",
# "Beneficial Ownership", "Exact Match"). This is the main precision guard: on
# real blackboards the org-like regex otherwise treats every Title-Cased
# defined term as an entity, flooding synthesis with false resolutions.
# Note: words that genuinely indicate organizations (bank, capital, group,
# holdings, trust, partners) are deliberately EXCLUDED so real names survive.
_DEFINED_TERMS = {
    "term", "loan", "loans", "closing", "date", "dates", "credit", "agreement",
    "agreements", "administrative", "agent", "agents", "business", "day", "days",
    "base", "rate", "rates", "eurocurrency", "revolving", "facility",
    "facilities", "asset", "sale", "prepayment", "commitment", "commitments",
    "letter", "letters", "beneficial", "beneficiary", "ownership", "exact",
    "match", "matches", "trade", "finance", "operations", "operation",
    "sanctions", "compliance", "officer", "collateral", "obligation",
    "obligations", "interest", "principal", "maturity", "default", "event",
    "events", "lender", "lenders", "borrower", "borrowers", "guarantor",
    "guarantors", "security", "payment", "payments", "fee", "fees", "notice",
    "notices", "party", "parties", "section", "sections", "article", "articles",
    "schedule", "schedules", "exhibit", "exhibits", "annex", "appendix",
    "definition", "definitions", "representation", "representations", "warranty",
    "warranties", "covenant", "covenants", "condition", "conditions",
    "exclusion", "exclusions", "endorsement", "amendment", "amendments",
    "effective", "applicable", "issuer", "firm", "trustee", "obligor", "seller",
    "buyer", "purchaser", "supplier", "vendor", "counterparty",
    # Common defined-term / concept words seen flooding real blackboards.
    "material", "adverse", "effect", "governmental", "authority",
    "authorization", "authorizations", "subsidiary", "subsidiaries",
    "restricted", "unrestricted", "possible", "potential", "permitted",
    "specified", "designated", "relevant",
}


def _clean_entity_name(raw: str) -> str:
    """Strip leading sentence/label noise and a trailing bare 'No' from a name."""
    words = raw.strip().rstrip(".,;:").split()
    while words and words[0].lower().strip(".,;:") in _LEADING_NOISE:
        words.pop(0)
    while words and words[-1].lower().strip(".") == "no":
        words.pop()
    return " ".join(words)


def _looks_like_entity(norm: str) -> bool:
    """False for empty/too-short names and names made entirely of defined terms."""
    tokens = norm.split()
    if not tokens or len(norm) < 3:
        return False
    # All tokens are generic defined-term/concept words → not an entity.
    return not all(t in _DEFINED_TERMS for t in tokens)


def _extract_entities_from_entry(entry: Entry) -> list[tuple[str, str]]:
    """Extract (raw_entity_name, source_document) from an entry.

    Leading sentence/label words are stripped and pure defined-term/concept
    names are dropped, so the resolver sees actual entity names rather than
    Title-Cased legal boilerplate.
    """
    if not entry.content:
        return []
    doc = entry.source.document if entry.source else ""
    entities = []
    seen = set()
    for m in _ORG_LIKE.finditer(entry.content):
        name = _clean_entity_name(m.group(1))
        if len(name) < 5 or len(name) > 80:
            continue
        if not _looks_like_entity(_normalize_entity(name)):
            continue
        key = name.lower()
        if key not in seen:
            seen.add(key)
            entities.append((name, doc))
    return entities


# ---------------------------------------------------------------------------
# Core resolution
# ---------------------------------------------------------------------------

@dataclass
class ResolutionResult:
    clusters: list[list[dict]]      # each cluster = list of {entry_id, raw_name, norm_name, doc}
    match_pairs: list[MatchResult]  # all pairwise matches above threshold
    entries_created: list[Entry]    # mapping/contradiction entries added
    tokens_used: int                # always 0 — no LLM calls


def resolve_entities(
    blackboard: Blackboard,
    *,
    threshold: float = MATCH_THRESHOLD,
    max_entries: int = 500,
) -> ResolutionResult:
    """Scan the blackboard for near-duplicate entity names.

    Creates mapping entries on the blackboard so downstream workers can
    reconcile findings across documents.

    Returns ResolutionResult with clusters, match pairs, and created entries.
    """
    # 1. Gather entity occurrences from active observation entries
    active = [e for e in blackboard.entries
              if e.status == "active" and e.type in ("observation", "analysis")]

    occurrences: list[dict] = []  # {entry_id, raw, norm, doc}
    seen_norm: dict[str, list[dict]] = {}  # norm -> [occurrence dicts]

    for entry in active[:max_entries]:
        for raw_name, doc in _extract_entities_from_entry(entry):
            norm = _normalize_entity(raw_name)
            if len(norm) < 3:
                continue
            occ = {
                "entry_id": entry.id,
                "raw": raw_name,
                "norm": norm,
                "doc": doc,
                "content_snippet": entry.content[:200],
            }
            occurrences.append(occ)
            seen_norm.setdefault(norm, []).append(occ)

    if len(occurrences) < 2:
        return ResolutionResult([], [], [], 0)

    # 2. Fuzzy matching between distinct normalized names. Exact-normalized
    # duplicates are handled by the unified clusterer below (a single norm with
    # >1 distinct surface form is an exact cluster), so they are NOT collected
    # separately — doing both previously double-created resolution entries.
    unique_norms: list[tuple[str, list[dict]]] = []
    for norm, occs in seen_norm.items():
        unique_norms.append((norm, occs))

    match_pairs: list[MatchResult] = []
    n = len(unique_norms)
    for i in range(n):
        norm_i, occs_i = unique_norms[i]
        for j in range(i + 1, n):
            norm_j, occs_j = unique_norms[j]

            # Quick pre-filter: share at least one token
            tokens_i = set(norm_i.split())
            tokens_j = set(norm_j.split())
            if not (tokens_i & tokens_j):
                continue

            # Skip pure singular/plural variants of the same term — the same
            # defined term written two ways, not a cross-document entity variant.
            if _depluralize(norm_i) == _depluralize(norm_j):
                continue

            jw = _jaro_winkler(norm_i, norm_j)
            if jw < 0.5:
                continue  # early exit

            lev = _levenshtein_ratio(norm_i, norm_j)
            tri_set_i = _trigrams(norm_i)
            tri_set_j = _trigrams(norm_j)
            tri_jacc = (len(tri_set_i & tri_set_j) /
                        len(tri_set_i | tri_set_j)) if (tri_set_i | tri_set_j) else 0.0
            tok = _token_overlap(norm_i, norm_j)
            is_pre = _is_prefix_variant(norm_i, norm_j)

            composite = _composite_score(jw, lev, tri_jacc, tok, is_pre)
            if composite < threshold:
                continue

            # Use the first occurrence from each group as representative
            occ_a = occs_i[0]
            occ_b = occs_j[0]
            entry_a = blackboard.find_entry(occ_a["entry_id"])
            entry_b = blackboard.find_entry(occ_b["entry_id"])
            if not entry_a or not entry_b:
                continue

            confidence = min(0.98, composite)
            if composite >= HIGH_CONFIDENCE:
                confidence = min(0.98, composite + 0.05)

            match_pairs.append(MatchResult(
                entry_a=entry_a,
                entry_b=entry_b,
                raw_name_a=occ_a["raw"],
                raw_name_b=occ_b["raw"],
                norm_a=norm_i,
                norm_b=norm_j,
                jaro_winkler=jw,
                levenshtein=lev,
                trigram_jaccard=tri_jacc,
                token_overlap=tok,
                is_prefix=is_pre,
                composite=composite,
                confidence=confidence,
            ))

    # 3. Cluster (union-find over norms) — unifies exact + fuzzy and dedupes,
    #    so each distinct entity set yields exactly one resolution entry.
    clusters = _cluster_matches(match_pairs, seen_norm)

    # 4. Create one blackboard entry per cluster.
    created = _create_resolution_entries(blackboard, clusters, match_pairs)

    return ResolutionResult(
        clusters=clusters,
        match_pairs=match_pairs,
        entries_created=created,
        tokens_used=0,
    )


# ---------------------------------------------------------------------------
# Clustering (union-find)
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self):
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: str, y: str):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self._parent[rx] = ry


def _cluster_matches(
    match_pairs: list[MatchResult],
    seen_norm: dict[str, list[dict]],
) -> list[list[dict]]:
    """Group entity occurrences into clusters with union-find over norms.

    Unifies two cases into one deduped pass:
    - exact clusters: one normalized form with >1 distinct surface form
      (e.g. "Pinnacle Industrial Solutions, Inc." vs "...Solutions");
    - fuzzy clusters: distinct norms joined by an above-threshold match pair.

    A cluster is returned only if it holds more than one distinct surface form —
    repeating the identical string is not a resolution.
    """
    uf = _UnionFind()
    for norm in seen_norm:
        uf.find(norm)  # ensure every norm is a node, even with no match pair
    for pair in match_pairs:
        uf.union(pair.norm_a, pair.norm_b)

    groups: dict[str, list[dict]] = {}
    for norm, occs in seen_norm.items():
        root = uf.find(norm)
        bucket = groups.setdefault(root, [])
        for occ in occs:
            if not any(o["entry_id"] == occ["entry_id"] and o["raw"] == occ["raw"]
                       for o in bucket):
                bucket.append(occ)

    return [
        members for members in groups.values()
        if len({m["raw"].strip().lower() for m in members}) > 1
    ]


# ---------------------------------------------------------------------------
# Blackboard entry creation
# ---------------------------------------------------------------------------

def _canonical_name(cluster: list[dict]) -> str:
    """Most frequent surface form in a cluster, tie-broken by length."""
    freq = Counter(o["raw"].strip() for o in cluster if o["raw"].strip())
    if not freq:
        return ""
    return max(freq.items(), key=lambda kv: (kv[1], len(kv[0])))[0]


def _create_resolution_entries(
    blackboard: Blackboard,
    clusters: list[list[dict]],
    match_pairs: list[MatchResult],
) -> list[Entry]:
    """Add one entity-resolution entry per cluster.

    Exact-normalized clusters (same entity differing only by punctuation or a
    legal suffix) are asserted with high confidence. Fuzzy clusters are framed
    as CANDIDATES for review with calibrated, lower confidence — deterministic
    similarity cannot guarantee similar-but-distinct names are one entity.
    """
    created: list[Entry] = []
    worker = WorkerRecord(
        worker_id="entity_resolution",
        description="deterministic_entity_clustering",
        iteration=blackboard.iteration,
    )

    for cluster in clusters:
        canonical = _canonical_name(cluster)
        variants = sorted(
            {o["raw"].strip() for o in cluster if o["raw"].strip()} - {canonical}
        )
        if not variants:
            continue

        distinct_norms = {c["norm"] for c in cluster}
        is_exact = len(distinct_norms) == 1

        best_composite = 0.0
        for pair in match_pairs:
            if pair.norm_a in distinct_norms and pair.norm_b in distinct_norms:
                best_composite = max(best_composite, pair.composite)

        variant_lines = []
        for v in variants:
            v_docs = sorted({o["doc"] for o in cluster
                             if o["raw"].strip() == v and o["doc"]})
            doc_tag = f" (from {', '.join(v_docs)})" if v_docs else ""
            variant_lines.append(f'- "{v}"{doc_tag}')

        if is_exact:
            best_composite = max(best_composite, 0.95)
            confidence = 0.9
            header = (
                f'ENTITY RESOLUTION: "{canonical}" appears under '
                f"{len(variants) + 1} surface form(s) (identical after "
                "normalization):"
            )
            note = ("These are the same entity written differently and should "
                    "be cross-referenced in all analysis.")
            kind = "exact"
        else:
            confidence = round(min(0.7, best_composite or 0.6), 2)
            header = (
                f'ENTITY RESOLUTION (CANDIDATE): "{canonical}" may be the same '
                f"entity as {len(variants)} similar name(s):"
            )
            note = ("Similar but NOT identical after normalization — review "
                    "before merging. String similarity cannot distinguish a "
                    "true variant (Petrochem/Petrochemical) from distinct "
                    "entities that merely share a prefix (Petrochem/Petroleum).")
            kind = "candidate"

        content = (
            header + "\n" + "\n".join(variant_lines)
            + f'\nCanonical form: "{canonical}"\n' + note
        )

        entry = Entry(
            id=gen_entry_id(),
            type="analysis",
            content=content,
            source=EntrySource(
                document="; ".join(sorted({c["doc"] for c in cluster if c["doc"]})),
                section="cross_document",
                evidence=f"Entity resolution ({kind}): {len(variants) + 1} forms "
                         f"(composite={best_composite:.3f})",
            ),
            created_by=worker,
            confidence=confidence,
            tags=[
                "entity_resolution",
                kind,
                f"variant_count:{len(variants) + 1}",
                f"composite:{best_composite:.3f}",
            ],
            status="active",
        )
        blackboard.add_entry(entry)
        created.append(entry)

    return created


# ---------------------------------------------------------------------------
# Convenience: run as a post-extraction maintenance step
# ---------------------------------------------------------------------------

def run_entity_resolution(blackboard: Blackboard, **kwargs) -> dict:
    """Run entity resolution and return a summary report.

    Intended to be called after extraction phases complete, before synthesis.
    Zero LLM tokens.
    """
    result = resolve_entities(blackboard, **kwargs)
    return {
        "schema_version": 1,
        "clusters_found": len(result.clusters),
        "match_pairs": len(result.match_pairs),
        "entries_created": len(result.entries_created),
        "tokens_used": 0,
        "clusters": [
            {
                "canonical": _canonical_name(c),
                "variants": [
                    {"raw": m["raw"], "doc": m["doc"], "entry_id": m["entry_id"]}
                    for m in c
                ],
            }
            for c in result.clusters
        ],
        "matches": [
            {
                "name_a": m.raw_name_a,
                "name_b": m.raw_name_b,
                "composite": m.composite,
                "jaro_winkler": round(m.jaro_winkler, 4),
                "levenshtein": round(m.levenshtein, 4),
                "trigram_jaccard": round(m.trigram_jaccard, 4),
                "token_overlap": round(m.token_overlap, 4),
                "is_prefix": m.is_prefix,
            }
            for m in result.match_pairs
        ],
    }
