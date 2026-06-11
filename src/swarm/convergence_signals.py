"""Deterministic convergence signals for the blackboard.

Addresses Open Research Question #2: "Is there a better signal for
'the blackboard has stabilized' — information-theoretic, graph-structural,
or confidence-distribution-based?"

Current approach (convergence.py): fixed iteration count + binary LLM
adversarial check. This module adds quantitative, zero-cost signals that
the orchestrator and convergence checker can use to detect stabilization
WITHOUT burning LLM tokens on every iteration.

All signals are deterministic. Zero LLM calls.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .blackboard import Blackboard
from .models import Entry


# ---------------------------------------------------------------------------
# Signal snapshot (one per iteration)
# ---------------------------------------------------------------------------

@dataclass
class IterationSnapshot:
    """Lightweight snapshot of blackboard state at one iteration."""
    iteration: int
    total_entries: int
    active_entries: int
    entries_by_type: dict[str, int]
    open_signals: int
    addressed_signals: int
    mean_confidence: float
    std_confidence: float
    documents_with_entries: int
    total_documents: int
    analysis_entries: int
    calculation_entries: int
    gap_entries: int


def take_snapshot(blackboard: Blackboard) -> IterationSnapshot:
    """Capture deterministic metrics from current blackboard state."""
    active = [e for e in blackboard.entries if e.status == "active"]
    confidences = [e.confidence for e in active if e.confidence > 0]

    type_counts: dict[str, int] = {}
    for e in active:
        type_counts[e.type] = type_counts.get(e.type, 0) + 1

    docs_with_entries = set()
    for e in active:
        if e.source and e.source.document:
            docs_with_entries.add(e.source.document)

    # Total documents "in play" = registered documents UNION any document a
    # live entry cites. Counting only blackboard.documents understates the
    # denominator when entries reference docs that were never registered, which
    # let cross_doc_coverage exceed 1.0. The union guarantees
    # documents_with_entries <= total_documents.
    all_docs = {d.name for d in blackboard.documents if d.name} | docs_with_entries

    open_sigs = sum(1 for s in blackboard.signals if s.status == "open")
    addressed_sigs = sum(1 for s in blackboard.signals if s.status == "addressed")

    mean_c = sum(confidences) / len(confidences) if confidences else 0.0
    std_c = (
        math.sqrt(sum((c - mean_c) ** 2 for c in confidences) / len(confidences))
        if len(confidences) > 1 else 0.0
    )

    return IterationSnapshot(
        iteration=blackboard.iteration,
        total_entries=len(blackboard.entries),
        active_entries=len(active),
        entries_by_type=type_counts,
        open_signals=open_sigs,
        addressed_signals=addressed_sigs,
        mean_confidence=round(mean_c, 4),
        std_confidence=round(std_c, 4),
        documents_with_entries=len(docs_with_entries),
        total_documents=len(all_docs),
        analysis_entries=type_counts.get("analysis", 0),
        calculation_entries=type_counts.get("calculation", 0),
        gap_entries=type_counts.get("gap", 0),
    )


# ---------------------------------------------------------------------------
# Convergence scorer
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceScore:
    """Composite convergence assessment from deterministic signals."""
    overall: float                   # 0..1, higher = more converged
    information_gain_rate: float     # entries added in recent window
    gain_deceleration: float         # how much gain has slowed (0=steady, 1=stopped)
    signal_resolution_rate: float    # fraction of signals addressed
    confidence_stability: float      # how stable mean confidence is (0=volatile, 1=stable)
    type_balance: float              # diversity of entry types (0=one type, 1=balanced)
    cross_doc_coverage: float        # fraction of docs with entries
    recommendation: str              # "continue" | "evaluate" | "stop"
    signals_detail: dict[str, float] = field(default_factory=dict)


# Minimum iterations before convergence is even considered
MIN_ITERATIONS_FOR_CONVERGENCE = 2

# Window size for rate-of-change calculations
RATE_WINDOW = 3

# Window for the information-gain rate. Deliberately short (2) so a genuine
# plateau is detected promptly: deceleration compares the gain over the last
# GAIN_RATE_WINDOW iterations against the peak windowed gain. A larger window
# let a single early extraction spike contaminate the "current" rate and made
# a flat blackboard read as still gaining.
GAIN_RATE_WINDOW = 2

# Thresholds
GAIN_STOP_THRESHOLD = 0.05        # <5% of peak gain = stopped gaining
CONFIDENCE_STABILITY_THRESHOLD = 0.02  # std dev change < 0.02 = stable
SIGNAL_RESOLUTION_TARGET = 0.70   # 70% signals addressed = good
TYPE_BALANCE_MIN_ENTRIES = 10     # need at least this many entries to assess balance
CROSS_DOC_COVERAGE_TARGET = 0.80  # 80% of docs should have entries


def compute_convergence_score(
    snapshots: list[IterationSnapshot],
) -> ConvergenceScore:
    """Compute convergence score from a history of iteration snapshots.

    Args:
        snapshots: chronological list of IterationSnapshot, one per iteration.

    Returns:
        ConvergenceScore with component scores and overall recommendation.
    """
    if len(snapshots) < MIN_ITERATIONS_FOR_CONVERGENCE:
        return ConvergenceScore(
            overall=0.0,
            information_gain_rate=0.0,
            gain_deceleration=0.0,
            signal_resolution_rate=0.0,
            confidence_stability=0.0,
            type_balance=0.0,
            cross_doc_coverage=0.0,
            recommendation="continue",
            signals_detail={"reason": "too_few_iterations"},
        )

    latest = snapshots[-1]

    # 1. Information gain rate
    gain_rate, gain_decel = _compute_gain_metrics(snapshots)

    # 2. Signal resolution rate
    total_signals = latest.open_signals + latest.addressed_signals
    signal_rate = (
        latest.addressed_signals / total_signals
        if total_signals > 0 else 1.0
    )

    # 3. Confidence stability
    conf_stability = _compute_confidence_stability(snapshots)

    # 4. Type balance
    type_balance = _compute_type_balance(latest)

    # 5. Cross-document coverage (clamped: a hand-built snapshot can carry
    # documents_with_entries > total_documents, which must not score > 1.0).
    cross_doc = (
        min(1.0, latest.documents_with_entries / latest.total_documents)
        if latest.total_documents > 0 else 1.0
    )

    # Composite score (weighted)
    overall = (
        0.30 * (1.0 - gain_decel)       # slowing gain is GOOD for convergence
        + 0.25 * signal_rate             # resolved signals is GOOD
        + 0.20 * conf_stability          # stable confidence is GOOD
        + 0.15 * type_balance            # diverse types is GOOD
        + 0.10 * cross_doc               # broad coverage is GOOD
    )

    # Recommendation
    if overall >= 0.75 and gain_decel >= 0.8 and signal_rate >= SIGNAL_RESOLUTION_TARGET:
        recommendation = "stop"
    elif overall >= 0.55 and gain_decel >= 0.6:
        recommendation = "evaluate"
    else:
        recommendation = "continue"

    return ConvergenceScore(
        overall=round(overall, 4),
        information_gain_rate=round(gain_rate, 2),
        gain_deceleration=round(gain_decel, 4),
        signal_resolution_rate=round(signal_rate, 4),
        confidence_stability=round(conf_stability, 4),
        type_balance=round(type_balance, 4),
        cross_doc_coverage=round(cross_doc, 4),
        recommendation=recommendation,
        signals_detail={
            "total_signals": float(total_signals),
            "open_signals": float(latest.open_signals),
            "addressed_signals": float(latest.addressed_signals),
            "mean_confidence": latest.mean_confidence,
            "std_confidence": latest.std_confidence,
            "active_entries": float(latest.active_entries),
            "analysis_entries": float(latest.analysis_entries),
            "calculation_entries": float(latest.calculation_entries),
            "gap_entries": float(latest.gap_entries),
            "documents_with_entries": float(latest.documents_with_entries),
            "total_documents": float(latest.total_documents),
        },
    )


# ---------------------------------------------------------------------------
# Component calculations
# ---------------------------------------------------------------------------

def _compute_gain_metrics(
    snapshots: list[IterationSnapshot],
) -> tuple[float, float]:
    """Compute information gain rate and deceleration.

    Returns:
        (current_gain_rate, deceleration)
        deceleration: 0 = still gaining fast, 1 = completely stopped.

    Both the peak and the current rate are GAIN_RATE_WINDOW-wide moving
    averages, each divided by the number of elements actually in the window
    (never a fixed divisor), so a partial early window is not understated and
    the two are directly comparable.
    """
    if len(snapshots) < 2:
        return 0.0, 0.0

    # Gains = net active entries added per iteration (clamped at 0).
    gains = [
        max(0, snapshots[i].active_entries - snapshots[i - 1].active_entries)
        for i in range(1, len(snapshots))
    ]
    if not gains:
        return 0.0, 1.0

    w = min(GAIN_RATE_WINDOW, len(gains))

    def _window_avg(seq: list[int]) -> float:
        return sum(seq) / len(seq) if seq else 0.0

    # Peak windowed gain over the whole history (actual-length division so an
    # early partial window is not divided by a larger fixed window size).
    peak_gain = max(
        _window_avg(gains[max(0, i - w + 1):i + 1])
        for i in range(len(gains))
    )

    # Current rate = average gain over the last w iterations only, so stale
    # early spikes do not mask a present-day plateau.
    current_rate = _window_avg(gains[-w:])

    deceleration = (
        1.0 if peak_gain <= 0
        else max(0.0, 1.0 - (current_rate / peak_gain))
    )
    return current_rate, deceleration


def _compute_confidence_stability(snapshots: list[IterationSnapshot]) -> float:
    """How stable is the mean confidence? Returns 0..1 (1=stable)."""
    if len(snapshots) < 2:
        return 0.0

    window = min(RATE_WINDOW, len(snapshots))
    recent = snapshots[-window:]
    means = [s.mean_confidence for s in recent]
    stds = [s.std_confidence for s in recent]

    # Variance of mean confidence across recent iterations
    if len(means) < 2:
        return 0.5
    mean_of_means = sum(means) / len(means)
    var_of_means = sum((m - mean_of_means) ** 2 for m in means) / len(means)

    # Low variance = stable
    # Normalize: if var < 0.0001 → 1.0, if var > 0.01 → 0.0
    stability = max(0.0, 1.0 - (var_of_means / 0.01))
    return min(1.0, stability)


def _compute_type_balance(snapshot: IterationSnapshot) -> float:
    """How balanced are entry types? Returns 0..1 (1=perfectly balanced)."""
    counts = snapshot.entries_by_type
    if not counts or snapshot.active_entries < TYPE_BALANCE_MIN_ENTRIES:
        return 0.0

    # Shannon entropy over type distribution, normalized to [0, 1]
    total = sum(counts.values())
    if total == 0:
        return 0.0

    probs = [c / total for c in counts.values() if c > 0]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0

    return min(1.0, entropy / max_entropy) if max_entropy > 0 else 0.0


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------

def should_force_converge(
    snapshots: list[IterationSnapshot],
    *,
    budget_pct: float = 0.0,
    iteration: int = 0,
    max_iterations: int = 15,
) -> tuple[bool, str]:
    """Deterministic decision: should we force convergence?

    Returns (should_converge, reason).
    This is a PRE-CHECK before the LLM adversarial convergence check,
    designed to save tokens when convergence is obvious.
    """
    if len(snapshots) < MIN_ITERATIONS_FOR_CONVERGENCE:
        return False, "too_few_iterations"

    score = compute_convergence_score(snapshots)

    # Budget emergency: if >=90% used, always converge. Checked BEFORE the
    # lower pressure threshold so a decent score at >=90% still reports the
    # emergency reason instead of being shadowed by the 75% branch.
    if budget_pct >= 90:
        return True, f"budget_emergency({budget_pct:.0f}%)"

    # Budget pressure: if >=75% used, converge if the score is decent.
    if budget_pct >= 75 and score.overall >= 0.5:
        return True, f"budget_pressure({budget_pct:.0f}%)_score({score.overall:.2f})"

    # Strong convergence signal: all components good
    if (score.gain_deceleration >= 0.85
            and score.signal_resolution_rate >= 0.80
            and score.confidence_stability >= 0.80):
        return True, f"strong_convergence(score={score.overall:.2f})"

    # Max iterations approaching
    if iteration >= max_iterations - 1:
        return True, f"max_iterations({iteration}/{max_iterations})"

    return False, f"not_ready(score={score.overall:.2f},rec={score.recommendation})"


def convergence_report(snapshots: list[IterationSnapshot]) -> dict:
    """Generate a human-readable convergence report for diagnostics."""
    score = compute_convergence_score(snapshots)
    return {
        "iterations_analyzed": len(snapshots),
        "overall_score": score.overall,
        "recommendation": score.recommendation,
        "components": {
            "information_gain": {
                "rate": score.information_gain_rate,
                "deceleration": score.gain_deceleration,
                "interpretation": (
                    "gaining" if score.gain_deceleration < 0.3
                    else "slowing" if score.gain_deceleration < 0.7
                    else "plateaued"
                ),
            },
            "signal_resolution": {
                "rate": score.signal_resolution_rate,
                "open": int(score.signals_detail.get("open_signals", 0)),
                "addressed": int(score.signals_detail.get("addressed_signals", 0)),
            },
            "confidence_stability": {
                "score": score.confidence_stability,
                "mean": score.signals_detail.get("mean_confidence", 0),
                "std": score.signals_detail.get("std_confidence", 0),
            },
            "type_balance": {
                "score": score.type_balance,
                "observations": int(
                    score.signals_detail.get("active_entries", 0)
                    - score.signals_detail.get("analysis_entries", 0)
                    - score.signals_detail.get("calculation_entries", 0)
                    - score.signals_detail.get("gap_entries", 0)
                ),
                "analyses": int(score.signals_detail.get("analysis_entries", 0)),
                "calculations": int(score.signals_detail.get("calculation_entries", 0)),
                "gaps": int(score.signals_detail.get("gap_entries", 0)),
            },
            "cross_document_coverage": {
                "score": score.cross_doc_coverage,
                "docs_with_entries": int(
                    score.signals_detail.get("documents_with_entries", 0)
                ),
                "total_docs": int(score.signals_detail.get("total_documents", 0)),
            },
        },
    }
