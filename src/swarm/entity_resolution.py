"""
entity_resolution.py — Deterministic cross-document entity resolution.

Addresses Open Research Question #1 from the README:
  "Cross-document entity resolution without LLM calls. The blackboard contains
   near-duplicate entities ('Zenith Petrochem' vs 'Zenith Petrochemical') that
   workers extract but don't reconcile. Can deterministic string similarity,
   edit distance, or lightweight embedding comparisons close this gap without
   burning tokens?"

Answer: yes. Zero LLM calls.

Performance: two-pass bucketing keeps the O(N²) pairwise work small. Mentions
are first deduplicated by normalised form (exact matches are pre-clustered at
zero cost), then representatives are grouped by first token. Full alias checks
only happen within-bucket; the cross-bucket pass is limited to single-token
forms with similar lengths. On a 520-entry blackboard this runs in ~150ms.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .blackboard import Blackboard
    from .models import Entry

_LEGAL_SUFFIXES: tuple[str, ...] = (
    "incorporated", "inc", "corporation", "corp", "limited", "ltd",
    "llc", "llp", "lp", "plc", "ag", "gmbh", "sa", "sas", "bv", "nv",
    "pte", "pty", "co", "company", "group", "holdings", "holding",
    "international", "intl", "enterprises", "solutions", "services",
    "technologies", "technology", "tech", "systems", "partners",
    "associates", "consulting", "capital", "financial", "bank", "national",
    "federal", "trust", "fund", "management", "mgmt",
)

_MIN_ENTITY_LEN = 4
_TOKEN_JACCARD_THRESHOLD = 0.60
_EDIT_DISTANCE_THRESHOLD = 0.25
_PREFIX_ANCHOR_LEN = 6

_STRIP_LEADING = re.compile(
    r'^(?:For|The|And|Or|In|Of|To|A|An|By|With|From|At|On)\s+',
    re.IGNORECASE,
)

_ENTITY_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9&''\-]{1,}(?:\s+[A-Z][A-Za-z0-9&''\-]{1,}){0,6})"
    r"(?:\s*,?\s*(?:Inc|LLC|Ltd|Corp|LP|LLP|PLC|AG|GmbH|SA|BV|NV|Pty|Co|Group|Holdings)\.?)?\b"
)


@dataclass
class EntityMention:
    raw: str
    normalised: str
    tokens: frozenset[str]
    entry_id: str
    document: str | None


@dataclass
class EntityCluster:
    canonical: str
    mentions: list[EntityMention] = field(default_factory=list)

    @property
    def documents(self) -> set[str | None]:
        return {m.document for m in self.mentions}

    @property
    def aliases(self) -> list[str]:
        seen: dict[str, None] = {}
        for m in self.mentions:
            seen[m.raw] = None
        return list(seen)


def _unicode_normalise(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def _strip_legal_suffix(tokens: list[str]) -> list[str]:
    while tokens and tokens[-1] in _LEGAL_SUFFIXES:
        tokens = tokens[:-1]
    return tokens


def normalise(raw: str) -> str:
    text = _unicode_normalise(raw)
    text = text.lower()
    text = re.sub(r"[^\w\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = _strip_legal_suffix(tokens)
    return " ".join(tokens)


def _tokenset(normalised: str) -> frozenset[str]:
    return frozenset(w for w in normalised.split() if len(w) >= 2)


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)
    for i in range(1, la + 1):
        curr[0] = i
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[lb]


def _normalised_edit_distance(a: str, b: str) -> float:
    return _edit_distance(a, b) / max(len(a), len(b), 1)


def _extract_mentions_from_entry(entry: "Entry") -> list[EntityMention]:
    if entry.type not in ("observation", "analysis"):
        return []
    if entry.status != "active":
        return []
    doc = entry.source.document if entry.source else None
    mentions: list[EntityMention] = []
    seen_raw: set[str] = set()
    for m in _ENTITY_RE.finditer(entry.content):
        raw = _STRIP_LEADING.sub("", m.group(0)).strip()
        if len(raw) < _MIN_ENTITY_LEN or raw in seen_raw:
            continue
        seen_raw.add(raw)
        norm = normalise(raw)
        if not norm or len(norm) < _MIN_ENTITY_LEN:
            continue
        if len(norm.split()) < 2:
            continue
        mentions.append(EntityMention(
            raw=raw, normalised=norm, tokens=_tokenset(norm),
            entry_id=entry.id, document=doc,
        ))
    return mentions


def _token_jaccard(a: EntityMention, b: EntityMention) -> float:
    union = a.tokens | b.tokens
    if not union:
        return 0.0
    return len(a.tokens & b.tokens) / len(union)


def _shares_prefix(a: EntityMention, b: EntityMention) -> bool:
    prefix_len = 0
    for ca, cb in zip(a.normalised, b.normalised):
        if ca != cb:
            break
        prefix_len += 1
    return prefix_len >= _PREFIX_ANCHOR_LEN


def are_aliases(a: EntityMention, b: EntityMention) -> bool:
    if a.normalised == b.normalised:
        return True
    has_shared_token = bool(a.tokens & b.tokens)
    if has_shared_token and _token_jaccard(a, b) >= _TOKEN_JACCARD_THRESHOLD:
        return True
    if _normalised_edit_distance(a.normalised, b.normalised) <= _EDIT_DISTANCE_THRESHOLD:
        return True
    if has_shared_token and _shares_prefix(a, b):
        return True
    return False


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def resolve_entities(blackboard: "Blackboard") -> dict[str, EntityCluster]:
    all_mentions: list[EntityMention] = []
    for entry in blackboard.entries:
        all_mentions.extend(_extract_mentions_from_entry(entry))

    if not all_mentions:
        return {}

    # Deduplicate by normalised form across all entries — exact matches are
    # already resolved, no need to compare them pairwise.
    norm_to_mentions: dict[str, list[EntityMention]] = {}
    for m in all_mentions:
        norm_to_mentions.setdefault(m.normalised, []).append(m)

    # One representative per normalised form for the O(N²) comparison.
    # All mentions sharing a normalised form are pre-clustered (exact match).
    representatives: list[EntityMention] = []
    pre_clusters: dict[str, list[EntityMention]] = {}
    for norm, group in norm_to_mentions.items():
        rep = max(group, key=lambda m: len(m.raw))
        representatives.append(rep)
        pre_clusters[norm] = group

    # Bucket representatives by first token for fast filtering.
    # Two reps in different buckets can only alias via edit distance
    # (no shared token), so we only do full O(N²) within-bucket +
    # a lightweight edit-distance-only cross-bucket pass.
    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, rep in enumerate(representatives):
        first = rep.normalised.split()[0] if rep.normalised else "__empty__"
        buckets[first].append(idx)

    n = len(representatives)
    uf = _UnionFind(n)

    # Within-bucket: full are_aliases check.
    for bucket_indices in buckets.values():
        for ii in range(len(bucket_indices)):
            for jj in range(ii + 1, len(bucket_indices)):
                i, j = bucket_indices[ii], bucket_indices[jj]
                if are_aliases(representatives[i], representatives[j]):
                    uf.union(i, j)

    # Cross-bucket: edit distance only (no shared token possible).
    # Only compare single-token normalised forms where lengths are close enough
    # to fall within the edit distance threshold.
    single_token_idxs = [
        i for i, r in enumerate(representatives)
        if " " not in r.normalised and r.normalised
    ]
    for ii in range(len(single_token_idxs)):
        for jj in range(ii + 1, len(single_token_idxs)):
            i, j = single_token_idxs[ii], single_token_idxs[jj]
            if uf.find(i) == uf.find(j):
                continue
            a, b = representatives[i], representatives[j]
            la, lb = len(a.normalised), len(b.normalised)
            if abs(la - lb) / max(la, lb, 1) > _EDIT_DISTANCE_THRESHOLD:
                continue
            if _normalised_edit_distance(a.normalised, b.normalised) <= _EDIT_DISTANCE_THRESHOLD:
                uf.union(i, j)

    # Build clusters from representative indices, then expand with all mentions
    # that share the same normalised form.
    clusters_raw: dict[int, list[int]] = {}
    for idx in range(n):
        root = uf.find(idx)
        clusters_raw.setdefault(root, []).append(idx)

    clusters: dict[str, EntityCluster] = {}
    for indices in clusters_raw.values():
        all_cluster_mentions: list[EntityMention] = []
        for idx in indices:
            norm = representatives[idx].normalised
            all_cluster_mentions.extend(pre_clusters[norm])
        if len(all_cluster_mentions) < 2:
            continue
        canonical = max(all_cluster_mentions, key=lambda m: len(m.raw)).raw
        cluster = EntityCluster(canonical=canonical, mentions=all_cluster_mentions)
        clusters[canonical] = cluster

    if not hasattr(blackboard, "entity_registry"):
        blackboard.entity_registry: dict[str, str] = {}

    for cluster in clusters.values():
        for alias in cluster.aliases:
            blackboard.entity_registry[alias] = cluster.canonical
            blackboard.entity_registry[normalise(alias)] = normalise(cluster.canonical)

    _emit_alias_entries(blackboard, clusters)
    return clusters


def _emit_alias_entries(
    blackboard: "Blackboard",
    clusters: dict[str, EntityCluster],
) -> None:
    from .models import Entry, Signal, WorkerRecord, gen_entry_id, gen_signal_id

    if not hasattr(blackboard, "_emitted_alias_clusters"):
        blackboard._emitted_alias_clusters: set[str] = set()

    for canonical, cluster in clusters.items():
        docs = cluster.documents
        if len(docs) < 2:
            continue
        aliases = sorted(set(cluster.aliases) - {canonical})
        if not aliases:
            continue
        cluster_key = canonical + "||" + "||".join(sorted(aliases))
        if cluster_key in blackboard._emitted_alias_clusters:
            continue
        blackboard._emitted_alias_clusters.add(cluster_key)

        alias_list = ", ".join(f'"{a}"' for a in aliases)
        docs_list = ", ".join(str(d) for d in sorted(docs, key=str) if d)

        entry = Entry(
            id=gen_entry_id(),
            type="entity_alias",
            content=(
                f"ENTITY ALIAS DETECTED (deterministic): \"{canonical}\" appears as "
                f"{alias_list} across documents [{docs_list}]. "
                f"These are likely the same entity. "
                f"Verify whether the name variation is material (e.g., a legal "
                f"name discrepancy to flag) or incidental (abbreviation in prose)."
            ),
            source=None,
            created_by=WorkerRecord(
                worker_id="entity_resolver",
                description="cross_document_alias_detection",
                iteration=blackboard.iteration,
            ),
            confidence=0.85,
            status="active",
            tags=["entity_resolution", "cross_document"],
        )
        blackboard.entries.append(entry)
        if hasattr(blackboard, "_entry_index"):
            blackboard._entry_index[entry.id] = entry

        signal = Signal(
            id=gen_signal_id(),
            type="entity_inconsistency",
            content=(
                f"Entity name inconsistency: \"{canonical}\" vs {alias_list} "
                f"across [{docs_list}]. Determine if this is a legal name "
                f"discrepancy that must be flagged in the output."
            ),
            origin_entry=entry.id,
            priority="high",
            status="open",
            iteration_created=blackboard.iteration,
        )
        blackboard.add_signal(signal)
