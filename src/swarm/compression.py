"""Token-aware blackboard compression for synthesis.

Selects, ranks, and clusters blackboard entries to maximize information
density within finite context windows — replacing the current simple
head-truncation approach.

Env gating:
  SWARM_ENABLE_COMPRESSION=1 — enables the full compression pipeline
  SWARM_COMPRESSION_TOKEN_BUDGET=24000 — max chars for compressed output
  SWARM_COMPRESSION_CLUSTER_THRESHOLD=0.60 — Jaccard threshold for clustering
  SWARM_COMPRESSION_MMR_LAMBDA=0.50 — diversity-relevance tradeoff (0=pure relevance, 1=pure diversity)
  SWARM_COMPRESSION_MAX_ENTRIES=500 — cap on total entries to consider

Design:
  1. Clustering: Group entries by source document proximity, content similarity
     (Jaccard over trigrams), and signal address patterns.
  2. Maximum coverage selection: Within each cluster, greedily select entries
     that maximize coverage of open signals and completeness criteria, subject
     to a token budget.
  3. Diversity-aware ranking: Apply MMR within each cluster to prevent
     near-identical entries from crowding out different topics.
  4. Token budget planner: Allocate budgets across sections proportionally to
     signal density weighted by materiality.
  5. Report: Write compression_report.json with selection decisions.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .blackboard import Blackboard
from .models import Entry


# --- Env gating ---

_BUDGET_KEY = "SWARM_COMPRESSION_TOKEN_BUDGET"
_CLUSTER_THRESHOLD_KEY = "SWARM_COMPRESSION_CLUSTER_THRESHOLD"
_MMR_LAMBDA_KEY = "SWARM_COMPRESSION_MMR_LAMBDA"
_MAX_ENTRIES_KEY = "SWARM_COMPRESSION_MAX_ENTRIES"

# Factor to convert "chars" to approximate token count
_CHARS_PER_TOKEN = 4.0


def compression_enabled() -> bool:
    return _env_on("SWARM_ENABLE_COMPRESSION")


def _token_budget() -> int:
    raw = os.getenv(_BUDGET_KEY, "24000").strip()
    try:
        return max(1000, int(raw))
    except (ValueError, TypeError):
        return 24000


def _cluster_threshold() -> float:
    raw = os.getenv(_CLUSTER_THRESHOLD_KEY, "0.60").strip()
    try:
        return max(0.0, min(1.0, float(raw)))
    except (ValueError, TypeError):
        return 0.60


def _mmr_lambda() -> float:
    raw = os.getenv(_MMR_LAMBDA_KEY, "0.50").strip()
    try:
        return max(0.0, min(1.0, float(raw)))
    except (ValueError, TypeError):
        return 0.50


def _max_entries() -> int:
    raw = os.getenv(_MAX_ENTRIES_KEY, "500").strip()
    try:
        return max(10, int(raw))
    except (ValueError, TypeError):
        return 500


# --- Content similarity (Jaccard trigram, consistent with codebase) ---


def _jaccard_trigram(a: str, b: str) -> float:
    """Jaccard similarity over character trigrams of two strings."""
    def trigrams(s: str) -> set[str]:
        clean = re.sub(r"[^a-z0-9\s]", "", s.lower())
        tokens = re.sub(r"\s+", " ", clean).strip().split()
        text = " ".join(tokens)
        if len(text) < 3:
            return {text}
        return {text[i:i+3] for i in range(len(text) - 2)}
    set_a = trigrams(a)
    set_b = trigrams(b)
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    return len(intersection) / max(len(set_a | set_b), 1)


def _entry_content_similarity(e1: Entry, e2: Entry) -> float:
    """Jaccard similarity between two entries' content."""
    return _jaccard_trigram(e1.content, e2.content)


# --- Entry content length ---


def _entry_chars(entry: Entry) -> int:
    """Approximate length of an entry in characters (including metadata overhead)."""
    base = len(entry.content)
    if entry.source and entry.source.evidence:
        base += len(entry.source.evidence)
    return base + 80  # overhead for ID, type, source fields


# --- Clustering ---


def _cluster_entries(
    entries: list[Entry],
    threshold: float,
) -> list[dict[str, Any]]:
    """Cluster entries by content similarity and source proximity.

    Returns a list of clusters, each with:
      - id: cluster identifier
      - label: dominant topic/source
      - entry_ids: ordered list of entry IDs in the cluster
      - signal_density: ratio of entries addressing open signals
      - materiality_weight: highest materiality in cluster
    """
    n = len(entries)
    if n == 0:
        return []
    if n == 1:
        return [{
            "id": "cluster_001",
            "label": entries[0].source.document if entries[0].source and entries[0].source.document else "only_entry",
            "entry_ids": [entries[0].id],
            "entry_count": 1,
            "signal_density": 0.0,
            "materiality_weight": 1.0,
        }]

    # Build similarity matrix
    adj: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            # Source proximity boost
            source_boost = 0.1
            if entries[i].source and entries[j].source:
                if entries[i].source.document == entries[j].source.document:
                    source_boost = 0.2
                    if (entries[i].source.section
                            and entries[j].source.section
                            and entries[i].source.section == entries[j].source.section):
                        source_boost = 0.3

            sim = _entry_content_similarity(entries[i], entries[j]) + source_boost
            if sim >= threshold:
                adj[i].add(j)
                adj[j].add(i)

    # Find connected components (clusters)
    visited = [False] * n
    clusters: list[set[int]] = []
    for i in range(n):
        if not visited[i]:
            component = set()
            stack = [i]
            while stack:
                idx = stack.pop()
                if visited[idx]:
                    continue
                visited[idx] = True
                component.add(idx)
                for neighbor in adj[idx]:
                    if not visited[neighbor]:
                        stack.append(neighbor)
            clusters.append(component)

    # Build cluster metadata
    result = []
    for idx, component in enumerate(clusters, 1):
        cluster_entries = [entries[i] for i in component]
        # Sort by confidence descending within cluster
        cluster_entries.sort(key=lambda e: (-e.confidence, e.id))

        # Count entries that address signals
        signal_addressing = sum(
            1 for e in cluster_entries
            if e.addresses_signals
        )
        signal_density = signal_addressing / max(len(cluster_entries), 1)

        # Determine label from the most common source document
        doc_counter = Counter()
        for e in cluster_entries:
            if e.source and e.source.document:
                doc_counter[e.source.document] += 1
        label = doc_counter.most_common(1)[0][0] if doc_counter else "cross_cutting"

        # Materiality: use highest from importance of addressed signals
        materiality_weight = 1.0 + signal_density  # boost by signal density

        result.append({
            "id": f"cluster_{idx:03d}",
            "label": label,
            "entry_ids": [e.id for e in cluster_entries],
            "entry_count": len(cluster_entries),
            "signal_density": round(signal_density, 4),
            "materiality_weight": round(materiality_weight, 2),
        })

    # Sort clusters by materiality_weight descending
    result.sort(key=lambda c: -c["materiality_weight"])
    return result


# --- Maximum coverage selection ---


def _signal_coverage_score(
    entries: list[Entry],
    signal_ids: set[str],
) -> float:
    """Score how well a set of entries covers a set of signal IDs."""
    if not signal_ids:
        return 0.0
    covered = set()
    for e in entries:
        covered.update(e.addresses_signals)
    return len(covered & signal_ids) / len(signal_ids)


def _greedy_select(
    candidates: list[Entry],
    budget_chars: int,
    signal_ids: set[str],
) -> list[Entry]:
    """Greedily select entries to maximize signal coverage within budget.

    Each iteration picks the entry with the highest marginal gain
    (new signals covered per character cost).
    """
    if not candidates or budget_chars <= 0:
        return []

    selected: list[Entry] = []
    selected_ids: set[str] = set()
    covered_signals: set[str] = set()
    remaining = list(candidates)
    used_chars = 0

    while remaining and used_chars < budget_chars:
        best_idx = -1
        best_marginal = -1.0

        for i, entry in enumerate(remaining):
            if entry.id in selected_ids:
                continue
            cost = _entry_chars(entry)
            if used_chars + cost > budget_chars:
                continue

            new_signals = set(entry.addresses_signals) - covered_signals
            marginal_gain = len(new_signals) / max(cost, 1)

            # Boost for high confidence
            if entry.confidence >= 0.8:
                marginal_gain *= 1.2
            # Boost for analytical types
            if entry.type in ("analysis", "calculation", "strategy"):
                marginal_gain *= 1.3

            if marginal_gain > best_marginal:
                best_marginal = marginal_gain
                best_idx = i

        if best_idx == -1:
            break

        entry = remaining.pop(best_idx)
        selected.append(entry)
        selected_ids.add(entry.id)
        used_chars += _entry_chars(entry)
        covered_signals.update(entry.addresses_signals)

        # If we've covered all signals, stop selecting
        if signal_ids and signal_ids <= covered_signals:
            break

    return selected


# --- MMR diversity ranking ---


def _mmr_rank(
    entries: list[Entry],
    lambda_param: float,
) -> list[Entry]:
    """Maximal Marginal Relevance ranking for diversity.

    Picks entries one by one, balancing relevance (confidence) against
    similarity to already-selected entries.
    """
    if not entries or len(entries) <= 1:
        return entries

    # Start with the highest-confidence entry
    ranked: list[Entry] = [entries[0]]
    remaining = list(entries[1:])

    while remaining and len(ranked) < len(entries):
        best_idx = -1
        best_score = -float("inf")

        for i, candidate in enumerate(remaining):
            # Relevance: normalize confidence to [0, 1]
            relevance = candidate.confidence

            # Diversity: max similarity to any already-ranked entry
            max_sim = max(
                _entry_content_similarity(candidate, chosen)
                for chosen in ranked
            ) if ranked else 0.0

            # MMR score
            score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx >= 0:
            ranked.append(remaining.pop(best_idx))

    return ranked


# --- Token budget planner ---


def _plan_token_budgets(
    clusters: list[dict],
    total_budget: int,
    open_signal_count: int,
) -> dict[str, int]:
    """Allocate token budgets across clusters proportionally.

    Allocation weights:
      signal_density * materiality_weight

    Clusters with more signal activity get larger budgets.
    """
    if not clusters:
        return {}

    weights: dict[str, float] = {}
    total_weight = 0.0
    for cluster in clusters:
        # Base weight: proportional to entry count
        base = cluster["entry_count"]
        # Boost: signal density * materiality
        boost = cluster["signal_density"] * cluster["materiality_weight"]
        weight = base * (1.0 + boost)
        weights[cluster["id"]] = weight
        total_weight += weight

    if total_weight == 0:
        # Equal distribution
        per_cluster = total_budget // max(len(clusters), 1)
        return {c["id"]: per_cluster for c in clusters}

    budgets: dict[str, int] = {}
    allocated = 0
    for cluster in clusters:
        raw = int(total_budget * weights[cluster["id"]] / total_weight)
        # Minimum per cluster: at least 10% of total budget or 500, whichever smaller
        min_cluster = min(max(50, total_budget // 10), 500)
        raw = max(min_cluster, raw)
        # Round down to nearest 100 for budgets over 1000
        if raw >= 1000:
            raw = (raw // 100) * 100
        budgets[cluster["id"]] = raw
        allocated += raw

    # Distribute any remainder to the largest cluster
    if allocated < total_budget and clusters:
        remainder = total_budget - allocated
        biggest = max(clusters, key=lambda c: c["entry_count"])
        budgets[biggest["id"]] = budgets.get(biggest["id"], 0) + remainder

    return budgets


# --- Main compression pipeline ---


def compress_blackboard(
    blackboard: Blackboard,
    must_include: list[dict] | None = None,
) -> dict[str, Any]:
    """Run the full compression pipeline on a blackboard.

    Args:
        blackboard: The blackboard to compress.
        must_include: Optional pre-existing must_include items from curation.

    Returns:
        A dict with:
          - enabled: whether compression is active
          - clusters: list of clusters found
          - selected_entries: IDs of entries selected for synthesis
          - budgets: token budget per cluster
          - signal_coverage: fraction of open signals addressed
          - warnings: list of diagnostic warnings
    """
    if not compression_enabled():
        return {"enabled": False, "selected_ids": [], "clusters": [], "budgets": {}}

    threshold = _cluster_threshold()
    budget = _token_budget()
    mmr_lambda = _mmr_lambda()
    max_entries = _max_entries()

    # Gather active entries (capped)
    active = [e for e in blackboard.entries if e.status == "active"]
    if len(active) > max_entries:
        # Prioritize: analytical types first, then by confidence
        def priority(e: Entry) -> tuple[int, float]:
            type_order = {"analysis": 4, "calculation": 4, "strategy": 3, "contradiction": 2, "observation": 1, "gap": 0}
            return (type_order.get(e.type, 0), e.confidence)
        active.sort(key=priority, reverse=True)
        active = active[:max_entries]

    # Step 1: Cluster
    clusters = _cluster_entries(active, threshold)
    if not clusters:
        return {
            "enabled": True,
            "converged": True,
            "clusters": [],
            "selected_ids": [],
            "budgets": {},
            "signal_coverage": 0.0,
            "warnings": ["No clusters formed from active entries."],
        }

    # Gather open signal IDs
    signal_ids = set[str]()
    if must_include:
        for item in must_include:
            if isinstance(item, dict):
                eids = item.get("entry_ids", []) or [item.get("entry_id", "")]
                for eid in eids if isinstance(eids, list) else [eids]:
                    if eid and eid not in signal_ids:
                        signal_ids.add(eid)
    # Also get from actual open signals
    for s in blackboard.signals:
        if s.status == "open":
            signal_ids.add(s.id)

    # Step 2: Plan token budgets per cluster
    budgets = _plan_token_budgets(clusters, budget, len(signal_ids))

    # Step 3: For each cluster, run MMR then greedy selection within its budget
    by_cluster: dict[str, list[Entry]] = {}
    for e in active:
        # Find which cluster this entry belongs to
        for cluster in clusters:
            if e.id in cluster["entry_ids"]:
                by_cluster.setdefault(cluster["id"], []).append(e)
                break

    selected_entries: list[Entry] = []
    selected_ids: set[str] = set()
    total_used = 0

    for cluster in clusters:
        cid = cluster["id"]
        cluster_budget = budgets.get(cid, 0)
        entries_in_cluster = by_cluster.get(cid, [])

        if not entries_in_cluster:
            continue

        # MMR ranking for diversity
        ranked = _mmr_rank(entries_in_cluster, mmr_lambda)

        # Greedy selection within budget
        selected = _greedy_select(ranked, cluster_budget, signal_ids)

        for entry in selected:
            if entry.id not in selected_ids:
                selected_entries.append(entry)
                selected_ids.add(entry.id)
                total_used += _entry_chars(entry)

    # Sort selected by original iteration order for coherence
    selected_entries.sort(key=lambda e: (e.created_by.iteration, e.id))

    # Compute signal coverage
    covered_signals = set()
    for e in selected_entries:
        covered_signals.update(e.addresses_signals)
    signal_coverage = len(covered_signals & signal_ids) / max(len(signal_ids), 1)

    # Warnings
    warnings: list[str] = []
    if len(selected_entries) < 10 and len(active) > 50:
        warnings.append(
            "HIGH COMPRESSION RATIO: Selected only "
            f"{len(selected_entries)}/{len(active)} entries. "
            "Consider increasing SWARM_COMPRESSION_TOKEN_BUDGET."
        )
    if signal_coverage < 0.5 and signal_ids:
        warnings.append(
            f"LOW SIGNAL COVERAGE: Only {signal_coverage:.0%} of "
            f"signals covered by selected entries."
        )
    if total_used > budget * 1.1:
        warnings.append(
            f"BUDGET OVERFLOW: Used {total_used}/{budget} chars "
            f"({total_used/budget:.0%})."
        )

    result = {
        "enabled": True,
        "converged": True,
        "clusters": clusters,
        "selected_ids": list(selected_ids),
        "selected_count": len(selected_entries),
        "total_entries": len(active),
        "budgets": budgets,
        "chars_used": total_used,
        "chars_budget": budget,
        "signal_coverage": round(signal_coverage, 4),
        "warnings": warnings,
        "selection_details": [
            {
                "id": e.id,
                "type": e.type,
                "confidence": e.confidence,
                "cluster": next(
                    (c["id"] for c in clusters if e.id in c["entry_ids"]),
                    "unclustered",
                ),
                "chars": _entry_chars(e),
            }
            for e in selected_entries
        ],
    }

    _write_report(blackboard.output_dir, result)

    return result


def compress_to_entry_ids(
    blackboard: Blackboard,
    must_include: list[dict] | None = None,
) -> list[str]:
    """Convenience wrapper: returns just the selected entry IDs.

    For direct use in synthesis pipeline.
    """
    result = compress_blackboard(blackboard, must_include)
    return result.get("selected_ids", [])


def _write_report(output_dir: str, report: dict) -> None:
    if not output_dir:
        return
    swarm_dir = Path(output_dir) / "swarm"
    swarm_dir.mkdir(parents=True, exist_ok=True)
    (swarm_dir / "compression_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )


def _env_on(key: str) -> bool:
    return os.getenv(key, "").strip().lower() in ("1", "true", "yes", "on")
