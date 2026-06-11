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


# Thresholds — tuned to minimize false positives on the repo's examples
MATCH_THRESHOLD = 0.72        # composite >= this → likely same entity
HIGH_CONFIDENCE = 0.85        # composite >= this → almost certain


# ---------------------------------------------------------------------------
# Entity extraction from blackboard entries
# ---------------------------------------------------------------------------

# Heuristic patterns for entity-like substrings in observation content
_ENTITY_PATTERN = re.compile(
    r"(?:"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?::\s|,|\s—|\s–|\s\()"  # capitalized multi-word before punctuation
    r"|"
    r"(?:^|\.\s+)([A-Z][\w\s,]{5,50}?)(?:\s+(?:is|has|holds|commits|provides|located|incorporated|organized))"
    r")",
    re.MULTILINE,
)

# Broader: any sequence of capitalized words that looks like an organization
_ORG_LIKE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+"
    r"(?:[,\s]+(?:Inc|LLC|Ltd|Corp|Corporation|Company|Holdings|Group|Partners|Associates|Bank|Capital|Finance|Equipment|Industries|Petrochem(?:ical)?|Solutions))?)"
    r"(?:\.|,|;|\s|$)"
)


def _extract_entities_from_entry(entry: Entry) -> list[tuple[str, str]]:
    """Extract (raw_entity_name, source_document) from an entry."""
    if not entry.content:
        return []
    doc = entry.source.document if entry.source else ""
    entities = []
    seen = set()
    for m in _ORG_LIKE.finditer(entry.content):
        name = m.group(1).strip().rstrip(".,;:")
        if len(name) < 5 or len(name) > 80:
            continue
        # Skip generic words
        lower = name.lower()
        if lower in {"the company", "the firm", "the bank", "the issuer",
                      "the borrower", "the lender", "the agent", "section",
                      "article", "schedule", "exhibit"}:
            continue
        key = lower
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

    # 2. Deduplicate exact-normalized matches (same entity, different entries)
    exact_clusters: list[list[dict]] = []
    for norm, occs in seen_norm.items():
        if len(occs) >= 2:
            # Different entries or different docs mentioning same entity
            exact_clusters.append(occs)

    # 3. Fuzzy matching between distinct normalized names
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

    # 4. Cluster matches (union-find)
    clusters = _cluster_matches(match_pairs, seen_norm)

    # 5. Create blackboard entries for each cluster (fuzzy + exact)
    all_clusters = clusters + exact_clusters
    created = _create_resolution_entries(blackboard, all_clusters, match_pairs)

    return ResolutionResult(
        clusters=all_clusters,
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
    """Group matched entities into clusters using union-find."""
    uf = _UnionFind()
    for pair in match_pairs:
        uf.union(pair.norm_a, pair.norm_b)

    groups: dict[str, list[dict]] = {}
    for norm in seen_norm:
        root = uf.find(norm)
        groups.setdefault(root, [])
        for occ in seen_norm[norm]:
            # Avoid duplicates
            if not any(o["entry_id"] == occ["entry_id"] and o["raw"] == occ["raw"]
                       for o in groups[root]):
                groups[root].append(occ)

    # Only return clusters with >1 distinct normalized form
    return [members for members in groups.values()
            if len({m["norm"] for m in members}) > 1]


# ---------------------------------------------------------------------------
# Blackboard entry creation
# ---------------------------------------------------------------------------

def _create_resolution_entries(
    blackboard: Blackboard,
    clusters: list[list[dict]],
    match_pairs: list[MatchResult],
) -> list[Entry]:
    """Add entity-resolution mapping entries to the blackboard."""
    created: list[Entry] = []
    worker = WorkerRecord(
        worker_id="entity_resolution",
        description="deterministic_entity_clustering",
        iteration=blackboard.iteration,
    )

    for cluster in clusters:
        if len(cluster) < 2:
            continue

        # Pick the longest normalized name as canonical
        canonical = max(cluster, key=lambda c: len(c["norm"]))
        variants = [c for c in cluster if c["raw"] != canonical["raw"]]

        if not variants:
            continue

        # Build a content string listing all name variants and their sources
        variant_lines = []
        for v in variants:
            doc_tag = f" (from {v['doc']})" if v["doc"] else ""
            variant_lines.append(f"- \"{v['raw']}\"{doc_tag}")

        content = (
            f"ENTITY RESOLUTION: \"{canonical['raw']}\" appears under "
            f"{len(cluster)} name variant(s) across documents:\n"
            + "\n".join(variant_lines)
            + f"\nCanonical form: \"{canonical['raw']}\""
            + "\nThese likely refer to the same entity and should be "
            "cross-referenced in all analysis."
        )

        # Find the best match pair for this cluster to get composite score
        best_composite = 0.0
        for pair in match_pairs:
            cluster_norms = {c["norm"] for c in cluster}
            if pair.norm_a in cluster_norms and pair.norm_b in cluster_norms:
                best_composite = max(best_composite, pair.composite)

        # Exact-normalization clusters (all norms identical) get high confidence
        distinct_norms = {c["norm"] for c in cluster}
        if len(distinct_norms) == 1:
            best_composite = max(best_composite, 0.92)

        entry = Entry(
            id=gen_entry_id(),
            type="analysis",
            content=content,
            source=EntrySource(
                document="; ".join(sorted({c["doc"] for c in cluster if c["doc"]})),
                section="cross_document",
                evidence=f"Entity resolution: {len(cluster)} variants detected "
                         f"(composite={best_composite:.3f})",
            ),
            created_by=worker,
            confidence=min(0.95, best_composite + 0.05),
            tags=[
                "entity_resolution",
                f"variant_count:{len(cluster)}",
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
                "canonical": max(c, key=lambda x: len(x["norm"]))["raw"],
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
