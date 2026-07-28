"""Entropy-based convergence detection for the blackboard.

Provides information-theoretic, graph-structural, and confidence-distribution
signals to determine when the blackboard has stabilized, replacing the
current fixed-iteration approach.

Env gating:
  SWARM_ENABLE_ENTROPY_CONVERGENCE=1 — enables the full entropy convergence pipeline
  SWARM_ENTROPY_CONVERGENCE_LOOKBACK=3 — iterations to look back for plateau detection
  SWARM_ENTROPY_CONVERGENCE_INFO_THRESHOLD=0.15 — min info gain ratio to block convergence
  SWARM_ENTROPY_CONVERGENCE_GRAPH_THRESHOLD=0.02 — max graph density delta to block convergence
  SWARM_ENTROPY_CONVERGENCE_CONFIDENCE_THRESHOLD=0.05 — max confidence mean delta to block

Design:
  1. Information-theoretic signal: tracks new entries, signal resolutions, and
     unique entity additions per iteration. Converges when info gain plateaus.
  2. Graph-structural signal: builds an entry relationship graph from
     supports/contradicts/supersedes edges. Converges when graph density
     stabilizes.
  3. Confidence-distribution signal: monitors shifts in the confidence
     distribution (mean, variance, skew). Converges when distribution stabilizes
     and flags the "uniform 0.9+" problem.
  4. Hybrid verdict: weighted combination of all three signals.
  5. Metrics written to convergence_metrics.json per iteration.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

from .blackboard import Blackboard
from .models import Entry


# --- Env gating ---

_LOOKBACK_KEY = "SWARM_ENTROPY_CONVERGENCE_LOOKBACK"
_INFO_THRESHOLD_KEY = "SWARM_ENTROPY_CONVERGENCE_INFO_THRESHOLD"
_GRAPH_THRESHOLD_KEY = "SWARM_ENTROPY_CONVERGENCE_GRAPH_THRESHOLD"
_CONFIDENCE_THRESHOLD_KEY = "SWARM_ENTROPY_CONVERGENCE_CONFIDENCE_THRESHOLD"

# Weights for the hybrid verdict
_INFO_WEIGHT = 0.4
_GRAPH_WEIGHT = 0.3
_CONFIDENCE_WEIGHT = 0.3


def entropy_convergence_enabled() -> bool:
    return _env_on("SWARM_ENABLE_ENTROPY_CONVERGENCE")


def _lookback() -> int:
    raw = os.getenv(_LOOKBACK_KEY, "3").strip()
    try:
        return max(2, int(raw))
    except (ValueError, TypeError):
        return 3


def _info_threshold() -> float:
    raw = os.getenv(_INFO_THRESHOLD_KEY, "0.15").strip()
    try:
        return max(0.0, float(raw))
    except (ValueError, TypeError):
        return 0.15


def _graph_threshold() -> float:
    raw = os.getenv(_GRAPH_THRESHOLD_KEY, "0.02").strip()
    try:
        return max(0.0, float(raw))
    except (ValueError, TypeError):
        return 0.02


def _confidence_threshold() -> float:
    raw = os.getenv(_CONFIDENCE_THRESHOLD_KEY, "0.05").strip()
    try:
        return max(0.0, float(raw))
    except (ValueError, TypeError):
        return 0.05


# --- Signal 1: Information-theoretic ---


def _compute_info_gain(bb: Blackboard, *, lookback: int) -> dict[str, Any]:
    """Compute the information gain signal.

    Metrics:
    - new_entry_count: new entries added this iteration
    - new_entry_ratio: new / total active entries
    - signal_resolution_count: signals addressed this iteration
    - signal_resolution_ratio: addressed / open signals
    - unique_entity_gain: new unique entity names this iteration
    - info_gain_score: weighted composite of the above
    """
    active = [e for e in bb.entries if e.status == "active"]
    total = len(active)

    # Count entries by iteration
    iter_counts = Counter(e.created_by.iteration for e in active)
    current_iter_entries = iter_counts.get(bb.iteration, 0)
    prev_iter_entries = sum(
        iter_counts.get(i, 0) for i in range(max(0, bb.iteration - lookback), bb.iteration)
    )

    # Signal resolution
    open_before = sum(
        1 for s in bb.signals if s.status == "open"
        and s.iteration_created <= bb.iteration
    )
    addressed_this_iter = sum(
        1 for s in bb.signals
        if s.status == "addressed"
        and s.iteration_created <= bb.iteration
        and not hasattr(s, '_prev_status')  # approximate: any addressed signal
    )
    # Better: count signals addressed by entries in this iteration
    addressed_ids = set()
    for e in active:
        if e.created_by.iteration == bb.iteration:
            addressed_ids.update(e.addresses_signals)
    signal_resolution_count = len(
        [s for s in bb.signals if s.id in addressed_ids and s.status == "addressed"]
    )

    # Unique entities: crude heuristic — count distinct capitalized phrases
    import re
    current_entities = set()
    for e in active:
        if e.created_by.iteration == bb.iteration:
            for m in re.finditer(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}", e.content):
                current_entities.add(m.group())
    prev_entities = set()
    for e in active:
        if e.created_by.iteration < bb.iteration:
            for m in re.finditer(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}", e.content):
                prev_entities.add(m.group())
    new_entity_count = len(current_entities - prev_entities)
    entity_gain_ratio = new_entity_count / max(len(current_entities | prev_entities), 1)

    # Info gain score
    new_entry_ratio = current_iter_entries / max(total, 1)
    signal_resolution_ratio = signal_resolution_count / max(open_before, 1)
    prev_entry_ratio = prev_iter_entries / max(total * lookback, 1)

    # Score: higher = more information gain (less convergence)
    new_info = new_entry_ratio + signal_resolution_ratio + entity_gain_ratio
    old_info = prev_entry_ratio if prev_entry_ratio > 0 else new_info
    info_gain = (new_info - old_info) / max(old_info, 0.01)
    # Normalize to [-1, 1] range roughly
    info_gain_score = max(-1.0, min(1.0, info_gain))

    return {
        "new_entry_count": current_iter_entries,
        "new_entry_ratio": round(new_entry_ratio, 4),
        "signal_resolution_count": signal_resolution_count,
        "signal_resolution_ratio": round(signal_resolution_ratio, 4),
        "new_unique_entities": new_entity_count,
        "entity_gain_ratio": round(entity_gain_ratio, 4),
        "info_gain_score": round(info_gain_score, 4),
        "converged": info_gain_score < _info_threshold(),
    }


# --- Signal 2: Graph-structural ---


def _build_entry_graph(entries: list[Entry]) -> dict[str, Any]:
    """Build a relationship graph from entry supports/contradicts/supersedes edges.

    Returns graph metrics:
    - node_count, edge_count
    - density (edges / possible edges)
    - clustering coefficient (fraction of triads with all three edges)
    - largest_component_size
    """
    active = [e for e in entries if e.status == "active"]
    node_ids = set(e.id for e in active)
    edges: set[tuple[str, str]] = set()

    for e in active:
        for target_id in e.supports_entries:
            if target_id in node_ids:
                edges.add(tuple(sorted((e.id, target_id))))
        for target_id in e.contradicts_entries:
            if target_id in node_ids:
                edges.add(tuple(sorted((e.id, target_id))))
        for target_id in e.supersedes_entries:
            if target_id in node_ids:
                edges.add(tuple(sorted((e.id, target_id))))

    n = len(node_ids)
    possible_edges = n * (n - 1) / 2 if n > 1 else 1
    density = len(edges) / possible_edges

    # Simple clustering coefficient: fraction of nodes with degree >= 2
    # that form triangles
    degree: dict[str, set[str]] = {nid: set() for nid in node_ids}
    for a, b in edges:
        degree[a].add(b)
        degree[b].add(a)

    triangles = 0
    triples = 0
    for a, b in edges:
        for c in degree[a] & degree[b]:
            if c != a and c != b:
                key = tuple(sorted((a, b, c)))
                triangles += 1 / 3  # each triangle counted 3 times
                break
    for nid in node_ids:
        d = len(degree[nid])
        if d >= 2:
            triples += d * (d - 1) / 2
    clustering = triangles / max(triples, 1)

    # Largest connected component
    visited: set[str] = set()
    largest = 0
    for nid in node_ids:
        if nid in visited:
            continue
        stack = [nid]
        component = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            for neighbor in degree.get(current, set()):
                if neighbor not in visited:
                    stack.append(neighbor)
        largest = max(largest, len(component))

    return {
        "node_count": n,
        "edge_count": len(edges),
        "density": round(density, 6),
        "clustering_coefficient": round(clustering, 4),
        "largest_component_ratio": round(largest / max(n, 1), 4),
    }


def _compute_graph_change(
    graph_now: dict[str, Any],
    graph_prev: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compute change in graph metrics between iterations."""
    if graph_prev is None:
        return {
            "density_delta": 0.0,
            "clustering_delta": 0.0,
            "converged": True,
        }
    density_delta = abs(graph_now["density"] - graph_prev["density"])
    clustering_delta = abs(
        graph_now["clustering_coefficient"] - graph_prev["clustering_coefficient"]
    )
    return {
        "density_delta": round(density_delta, 6),
        "clustering_delta": round(clustering_delta, 4),
        "converged": (
            density_delta < _graph_threshold()
            and clustering_delta < _graph_threshold() * 2
        ),
    }


# --- Signal 3: Confidence distribution ---


def _compute_confidence_stats(entries: list[Entry]) -> dict[str, float]:
    """Compute statistics of the confidence distribution."""
    active = [e for e in entries if e.status == "active"]
    confidences = [e.confidence for e in active if e.confidence is not None]

    if not confidences:
        return {"mean": 0.0, "variance": 0.0, "skewness": 0.0, "uniform_flag": True}

    n = len(confidences)
    mean = sum(confidences) / n
    variance = sum((c - mean) ** 2 for c in confidences) / n
    std = math.sqrt(variance)
    skewness = (
        sum((c - mean) ** 3 for c in confidences) / n / max(std ** 3, 0.001)
        if std > 0 else 0.0
    )

    # Detect "uniform 0.9+" problem: >95% of entries have confidence >= 0.9
    high_conf = sum(1 for c in confidences if c >= 0.9)
    uniform_flag = high_conf / n > 0.95

    return {
        "mean": round(mean, 4),
        "variance": round(variance, 6),
        "std_dev": round(std, 4),
        "skewness": round(skewness, 4),
        "high_confidence_ratio": round(high_conf / n, 4),
        "uniform_flag": uniform_flag,
    }


def _compute_confidence_change(
    stats_now: dict[str, float],
    stats_prev: dict[str, float] | None,
) -> dict[str, Any]:
    """Compute change in confidence distribution."""
    if stats_prev is None:
        return {"mean_delta": 0.0, "converged": True}
    mean_delta = abs(stats_now["mean"] - stats_prev["mean"])
    return {
        "mean_delta": round(mean_delta, 4),
        "uniform_flag": stats_now.get("uniform_flag", False),
        "converged": mean_delta < _confidence_threshold(),
    }


# --- History tracking ---


class ConvergenceHistory:
    """Tracks convergence metrics across iterations.

    Stored on the blackboard as a side-band dict so metrics survive
    across the main loop.
    """

    def __init__(self) -> None:
        self.info_gain_history: list[dict] = []
        self.graph_history: list[dict] = []
        self.confidence_history: list[dict] = []
        self.last_graph: dict | None = None
        self.last_confidence: dict | None = None

    def record(
        self,
        info_gain: dict,
        graph: dict,
        confidence: dict,
    ) -> None:
        self.info_gain_history.append(info_gain)
        self.graph_history.append(graph)
        self.confidence_history.append(confidence)
        self.last_graph = graph
        self.last_confidence = confidence

    def to_dict(self) -> dict:
        return {
            "info_gain_history": self.info_gain_history,
            "graph_history": self.graph_history,
            "confidence_history": self.confidence_history,
        }


_history_attr = "_convergence_history"


def _get_history(bb: Blackboard) -> ConvergenceHistory:
    if not hasattr(bb, _history_attr) or getattr(bb, _history_attr) is None:
        setattr(bb, _history_attr, ConvergenceHistory())
    return getattr(bb, _history_attr)


# --- Main entry point ---


def compute_convergence(bb: Blackboard) -> dict[str, Any]:
    """Compute convergence signals for a blackboard.

    Returns a verdict dict with per-signal scores and a hybrid converged flag.
    """
    if not entropy_convergence_enabled():
        return {
            "enabled": False,
            "converged": False,
            "signals": {},
        }

    lookback = _lookback()
    active = [e for e in bb.entries if e.status == "active"]
    history = _get_history(bb)

    # Signal 1: Information gain
    info_gain = _compute_info_gain(bb, lookback=lookback)

    # Signal 2: Graph-structural
    graph = _build_entry_graph(active)
    graph_change = _compute_graph_change(graph, history.last_graph)

    # Signal 3: Confidence distribution
    conf_stats = _compute_confidence_stats(active)
    conf_change = _compute_confidence_change(conf_stats, history.last_confidence)

    # Record history
    history.record(info_gain, graph, conf_stats)

    # Hybrid verdict
    info_converged = info_gain.get("converged", False)
    graph_converged = graph_change.get("converged", False)
    conf_converged = conf_change.get("converged", False)

    # Weighted score (0 = fully converged, 1 = no convergence)
    info_score = 0.0 if info_converged else min(1.0, abs(info_gain.get("info_gain_score", 0)))
    graph_score = 0.0 if graph_converged else 0.3
    conf_score = 0.0 if conf_converged else 0.3

    hybrid_score = (
        _INFO_WEIGHT * info_score
        + _GRAPH_WEIGHT * graph_score
        + _CONFIDENCE_WEIGHT * conf_score
    )

    # Converge if hybrid score is low AND all three signals agree OR
    # two signals strongly agree
    signal_count = sum([info_converged, graph_converged, conf_converged])
    converged = signal_count >= 2 and hybrid_score < 0.3

    result = {
        "enabled": True,
        "iteration": bb.iteration,
        "converged": converged,
        "hybrid_score": round(hybrid_score, 4),
        "signal_agreement": signal_count,
        "signals": {
            "info_gain": {
                "converged": info_converged,
                "score": round(info_score, 4),
                "details": info_gain,
            },
            "graph_structural": {
                "converged": graph_converged,
                "score": round(graph_score, 4),
                "details": graph,
                "change": graph_change,
            },
            "confidence_distribution": {
                "converged": conf_converged,
                "score": round(conf_score, 4),
                "details": conf_stats,
                "change": conf_change,
            },
        },
        "warnings": [],
    }

    # Add warnings for known failure modes
    if conf_stats.get("uniform_flag"):
        result["warnings"].append(
            "CONFIDENCE COLLAPSE: >95% of entries have confidence >= 0.9. "
            "The confidence field carries no information — disputed detection "
            "will never trigger."
        )
    if graph.get("edge_count", 0) == 0 and len(active) >= 3:
        result["warnings"].append(
            "DEAD GRAPH: Zero edges in entry relationship graph. "
            "supports_entries, contradicts_entries, and supersedes_entries "
            "are all empty — the knowledge graph infrastructure is unused."
        )
    if info_gain.get("signal_resolution_count", 0) == 0 and bb.iteration > 3:
        result["warnings"].append(
            "SIGNAL STALL: No signals were resolved this iteration. "
            "If this persists, signals are accumulating without resolution."
        )

    # Write report
    _write_metrics(bb.output_dir, result, history)

    return result


def _write_metrics(
    output_dir: str,
    verdict: dict,
    history: ConvergenceHistory,
) -> None:
    if not output_dir:
        return
    swarm_dir = Path(output_dir) / "swarm"
    swarm_dir.mkdir(parents=True, exist_ok=True)
    (swarm_dir / "convergence_metrics.json").write_text(
        json.dumps(verdict, indent=2),
        encoding="utf-8",
    )
    (swarm_dir / "convergence_history.json").write_text(
        json.dumps(history.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )


def _env_on(key: str) -> bool:
    return os.getenv(key, "").strip().lower() in ("1", "true", "yes", "on")
