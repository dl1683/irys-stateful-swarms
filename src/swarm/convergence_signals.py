"""Deterministic convergence signals for the blackboard.

Addresses Open Research Question #2: "Is there a better signal for
'the blackboard has stabilized' — information-theoretic, graph-structural,
or confidence-distribution-based?"

Current approach (convergence.py): fixed iteration count + binary LLM
adversarial check.  This module adds quantitative, zero-cost signals that
the orchestrator and convergence checker can use to detect stabilization
WITHOUT burning LLM tokens on every iteration.

All core signals are deterministic.  An optional ``caller`` parameter
enables a hybrid path where borderline cases are adjudicated by a model.

Design principles
-----------------
* **Zero hardcoded vocabulary** — no English/legal term lists baked in.
* **Configurable thresholds** — every numeric gate lives in
  :class:`ConvergenceConfig` (frozen dataclass, safe defaults).
* **Domain-agnostic** — entry-type taxonomy comes from the task, not from
  constants.  ``type_aliases_mapping`` adapts to any domain.
* **Multilingual by construction** — all text ops use NFKC + casefold.
* **Hybrid** — deterministic fast-path; optional model adjudication for
  ambiguous scores; graceful degradation when no model is available.
"""
from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from .blackboard import Blackboard
from .models import Entry

if TYPE_CHECKING:  # pragma: no cover — import-time only
    from .worker_dispatch import call_model as _call_model_fn


# ---------------------------------------------------------------------------
# Model-caller protocol (avoids hard import-time dependency)
# ---------------------------------------------------------------------------

class ModelCaller(Protocol):
    """Minimal protocol for an LLM caller."""
    def __call__(self, prompt: str, *, max_tokens: int = 512) -> Any: ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConvergenceConfig:
    """All tunable thresholds and weights for convergence detection.

    Every field has a safe default inferred from typical swarm behaviour.
    Override any field to adapt to a different domain, language, iteration
    budget, or quality bar.
    """

    # --- iteration / window geometry ---
    min_iterations_for_convergence: int = 2
    rate_window: int = 3
    gain_rate_window: int = 2

    # --- information-gain thresholds ---
    gain_stop_threshold: float = 0.05

    # --- confidence stability ---
    confidence_stability_threshold: float = 0.02
    variance_normalization: float = 0.01

    # --- signal resolution ---
    signal_resolution_target: float = 0.70

    # --- type balance ---
    type_balance_min_entries: int = 10

    # --- cross-document coverage ---
    cross_doc_coverage_target: float = 0.80

    # --- composite weights (must sum to 1.0) ---
    weight_gain_decel: float = 0.30
    weight_signal_rate: float = 0.25
    weight_conf_stability: float = 0.20
    weight_type_balance: float = 0.15
    weight_cross_doc: float = 0.10

    # --- recommendation thresholds ---
    stop_overall_min: float = 0.75
    stop_gain_decel_min: float = 0.80
    stop_signal_rate_min: float = 0.70
    eval_overall_min: float = 0.55
    eval_gain_decel_min: float = 0.60
    # minimum active entries before "stop" is permitted (a near-empty board is never converged)
    min_entries_for_stop: int = 1
    # max recent entry-count drop tolerated before "stop" (guards against oscillation/churn)
    max_entry_drop_for_stop: int = 1

    # --- should_force_converge thresholds ---
    force_gain_decel: float = 0.85
    force_signal_rate: float = 0.80
    force_conf_stability: float = 0.80
    force_overall_min: float = 0.50
    budget_emergency_pct: float = 90.0
    budget_pressure_pct: float = 75.0

    # --- type alias mapping (domain-specific entry-type names → human label) ---
    # Example for legal domain:
    #   {"analysis": "analysis", "calculation": "calculation", "gap": "gap"}
    # Example for German legal domain:
    #   {"Analyse": "Analyse", "Berechnung": "Berechnung", "Lücke": "Lücke"}
    # Empty mapping → no type-specific counts.
    type_aliases_mapping: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Data classes
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


@dataclass
class IterationSnapshot:
    """Lightweight snapshot of blackboard state at one iteration.

    All fields carry safe defaults so that a partial snapshot is legal.
    """
    iteration: int = 0
    total_entries: int = 0
    active_entries: int = 0
    entries_by_type: dict[str, int] = field(default_factory=dict)
    open_signals: int = 0
    addressed_signals: int = 0
    mean_confidence: float = 0.0
    std_confidence: float = 0.0
    documents_with_entries: int = 0
    total_documents: int = 0
    # Domain-specific type counts, keyed by human label from config
    type_aliases: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Snapshot capture
# ---------------------------------------------------------------------------

def _nfkc_fold(s: str) -> str:
    """NFKC-normalize then casefold — the only text normalisation allowed."""
    return unicodedata.normalize("NFKC", s).casefold()


def take_snapshot(
    blackboard: Blackboard,
    *,
    config: ConvergenceConfig | None = None,
) -> IterationSnapshot:
    """Capture deterministic metrics from current blackboard state.

    ``config`` is optional; when *None* a default ``ConvergenceConfig``
    is used (all thresholds at safe defaults).
    """
    if config is None:
        config = ConvergenceConfig()

    active = [e for e in blackboard.entries if e.status == "active"]
    confidences = [e.confidence for e in active if e.confidence > 0]

    type_counts: dict[str, int] = {}
    for e in active:
        type_counts[e.type] = type_counts.get(e.type, 0) + 1

    # --- domain-specific type aliases (configurable) ---
    alias_mapping = config.type_aliases_mapping
    if alias_mapping:
        # Build a lookup from normalised entry-type → original type string
        norm_to_raw: dict[str, str] = {}
        for raw in type_counts:
            norm_to_raw[_nfkc_fold(raw)] = raw

        type_aliases: dict[str, int] = {}
        for alias_key, alias_label in alias_mapping.items():
            norm_key = _nfkc_fold(alias_key)
            matched_raw = norm_to_raw.get(norm_key)
            type_aliases[alias_label] = type_counts.get(matched_raw, 0) if matched_raw else 0
    else:
        type_aliases = {}

    docs_with_entries: set[str] = set()
    for e in active:
        if e.source and e.source.document:
            docs_with_entries.add(e.source.document)

    # Total documents "in play" = registered documents UNION any document a
    # live entry cites.  Counting only blackboard.documents understates the
    # denominator when entries reference docs that were never registered,
    # which let cross_doc_coverage exceed 1.0.  The union guarantees
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
        iteration=getattr(blackboard, "iteration", 0),
        total_entries=len(blackboard.entries),
        active_entries=len(active),
        entries_by_type=type_counts,
        open_signals=open_sigs,
        addressed_signals=addressed_sigs,
        mean_confidence=round(mean_c, 4),
        std_confidence=round(std_c, 4),
        documents_with_entries=len(docs_with_entries),
        total_documents=len(all_docs),
        type_aliases=type_aliases,
    )


# ---------------------------------------------------------------------------
# Convergence scorer
# ---------------------------------------------------------------------------

def compute_convergence_score(
    snapshots: list[IterationSnapshot],
    *,
    config: ConvergenceConfig | None = None,
    caller: ModelCaller | None = None,
) -> ConvergenceScore:
    """Compute convergence score from a history of iteration snapshots.

    Args:
        snapshots:  chronological list, one per iteration.
        config:     optional thresholds/weights override.
        caller:     optional LLM caller for hybrid adjudication of
                    borderline scores.  When *None* the deterministic
                    result is returned as-is.

    Returns:
        ConvergenceScore with component scores and overall recommendation.
    """
    if config is None:
        config = ConvergenceConfig()

    if len(snapshots) < config.min_iterations_for_convergence:
        rec = "continue"
        if caller is not None:
            rec = _adjudicate(
                caller, 0.0, 0.0, 0.0, 0.0, 0.0, rec, config,
            )
        return ConvergenceScore(
            overall=0.0,
            information_gain_rate=0.0,
            gain_deceleration=0.0,
            signal_resolution_rate=0.0,
            confidence_stability=0.0,
            type_balance=0.0,
            cross_doc_coverage=0.0,
            recommendation=rec,
            signals_detail={"reason": "too_few_iterations"},
        )

    latest = snapshots[-1]

    # 1. Information gain
    gain_rate, gain_decel = _compute_gain_metrics(snapshots, config)

    # 2. Signal resolution
    total_signals = latest.open_signals + latest.addressed_signals
    signal_rate = (
        latest.addressed_signals / total_signals
        if total_signals > 0 else 1.0
    )

    # 3. Confidence stability
    conf_stability = _compute_confidence_stability(snapshots, config)

    # 4. Type balance
    type_balance = _compute_type_balance(latest, config)

    # 5. Cross-document coverage (clamped: a hand-built snapshot can carry
    #    documents_with_entries > total_documents).
    cross_doc = (
        min(1.0, latest.documents_with_entries / latest.total_documents)
        if latest.total_documents > 0 else 1.0
    )

    # Composite score (weighted)
    overall = (
        config.weight_gain_decel * gain_decel   # high deceleration (gain stopped) = converged
        + config.weight_signal_rate * signal_rate
        + config.weight_conf_stability * conf_stability
        + config.weight_type_balance * type_balance
        + config.weight_cross_doc * cross_doc
    )

    # Recommendation
    if (overall >= config.stop_overall_min
            and gain_decel >= config.stop_gain_decel_min
            and signal_rate >= config.stop_signal_rate_min):
        recommendation = "stop"
    elif overall >= config.eval_overall_min and gain_decel >= config.eval_gain_decel_min:
        recommendation = "evaluate"
    else:
        recommendation = "continue"

    # Optional hybrid adjudication — only consult the model for borderline cases; a confident
    # deterministic "stop" is honoured as-is (saves tokens).
    if caller is not None and recommendation != "stop":
        recommendation = _adjudicate(
            caller,
            overall,
            gain_decel,
            signal_rate,
            conf_stability,
            cross_doc,
            recommendation,
            config,
        )

    # Churn guard: a converged board does not shed entries. Meaningful recent entry-count
    # drops (oscillation/churn) mean it is not settled, even if the tail looks decelerated.
    _recent_counts = [s.active_entries for s in snapshots[-(config.rate_window + 1):]]
    _max_drop = max((_recent_counts[i - 1] - _recent_counts[i]
                     for i in range(1, len(_recent_counts))), default=0)
    if recommendation == "stop" and _max_drop > config.max_entry_drop_for_stop:
        recommendation = "evaluate"

    # Substance guard: a near-empty blackboard can never be "converged".
    if recommendation == "stop" and latest.active_entries < config.min_entries_for_stop:
        recommendation = "continue"

    return ConvergenceScore(
        overall=round(overall, 4),
        information_gain_rate=round(gain_rate, 4),
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
            "type_aliases": latest.type_aliases,
            "documents_with_entries": float(latest.documents_with_entries),
            "total_documents": float(latest.total_documents),
        },
    )


# ---------------------------------------------------------------------------
# Component calculations
# ---------------------------------------------------------------------------

def _compute_gain_metrics(
    snapshots: list[IterationSnapshot],
    config: ConvergenceConfig,
) -> tuple[float, float]:
    """Compute information gain rate and deceleration.

    Returns (current_gain_rate, deceleration).
    deceleration: 0 = still gaining fast, 1 = completely stopped.
    """
    if len(snapshots) < 2:
        return 0.0, 0.0

    gains = [
        max(0, snapshots[i].active_entries - snapshots[i - 1].active_entries)
        for i in range(1, len(snapshots))
    ]
    if not gains:
        return 0.0, 1.0

    w = min(config.gain_rate_window, len(gains))

    def _window_avg(seq: list[int]) -> float:
        return sum(seq) / len(seq) if seq else 0.0

    peak_gain = max(
        _window_avg(gains[max(0, i - w + 1):i + 1])
        for i in range(len(gains))
    )
    current_rate = _window_avg(gains[-w:])

    deceleration = (
        1.0 if peak_gain <= 0
        else max(0.0, 1.0 - (current_rate / peak_gain))
    )
    return current_rate, deceleration


def _compute_confidence_stability(
    snapshots: list[IterationSnapshot],
    config: ConvergenceConfig,
) -> float:
    """How stable is the mean confidence?  Returns 0..1 (1=stable)."""
    if len(snapshots) < 2:
        return 0.0

    window = min(config.rate_window, len(snapshots))
    recent = snapshots[-window:]
    means = [s.mean_confidence for s in recent]

    if len(means) < 2:
        return 0.5
    mean_of_means = sum(means) / len(means)
    var_of_means = sum((m - mean_of_means) ** 2 for m in means) / len(means)

    # Low variance → stable.  Normalize against config threshold.
    stability = max(0.0, 1.0 - (var_of_means / config.variance_normalization))
    return min(1.0, stability)


def _compute_type_balance(
    snapshot: IterationSnapshot,
    config: ConvergenceConfig,
) -> float:
    """How balanced are entry types?  Returns 0..1 (1=perfectly balanced).

    Uses Shannon entropy normalised to [0, 1].
    """
    counts = snapshot.entries_by_type
    if not counts or snapshot.active_entries < config.type_balance_min_entries:
        return 0.0

    total = sum(counts.values())
    if total == 0:
        return 0.0

    probs = [c / total for c in counts.values() if c > 0]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0

    if max_entropy <= 0:
        return 0.0
    if len(counts) < 2:
        return 0.0

    return min(1.0, entropy / max_entropy)


def _cross_doc_coverage(latest: IterationSnapshot) -> float:
    """Fraction of documents with entries, clamped to [0, 1]."""
    if latest.total_documents <= 0:
        return 1.0
    return min(1.0, latest.documents_with_entries / latest.total_documents)


# ---------------------------------------------------------------------------
# Hybrid adjudication
# ---------------------------------------------------------------------------

def _adjudicate(
    caller: ModelCaller,
    overall: float,
    gain_decel: float,
    signal_rate: float,
    conf_stability: float,
    cross_doc: float,
    default_rec: str,
    config: ConvergenceConfig,
) -> str:
    """Ask the model to adjudicate a borderline convergence decision.

    Only called when the deterministic recommendation is *not* a confident
    "stop" — i.e. the score is in the evaluate or continue zone.
    """
    prompt = (
        "You are a convergence judge for a multi-agent document-analysis "
        "swarm. Based on the following deterministic metrics, decide whether "
        "the blackboard has converged.\n\n"
        f"Overall score: {overall:.4f}\n"
        f"Gain deceleration: {gain_decel:.4f} (1 = stopped gaining)\n"
        f"Signal resolution rate: {signal_rate:.4f}\n"
        f"Confidence stability: {conf_stability:.4f}\n"
        f"Cross-document coverage: {cross_doc:.4f}\n"
        f"Default recommendation: {default_rec}\n\n"
        "Reply with JSON: {\"recommendation\": \"stop\"|\"evaluate\"|\"continue\"}"
    )
    try:
        # ModelCaller is a direct callable (see the protocol above); invoke it per that
        # contract. Accept either a (payload, tokens) tuple or a bare payload dict.
        result = caller(prompt, max_tokens=64)
        payload = result[0] if isinstance(result, tuple) else result
        if isinstance(payload, dict) and "recommendation" in payload:
            rec = str(payload["recommendation"]).strip().lower()
            if rec in ("stop", "evaluate", "continue"):
                return rec
    except Exception:
        pass
    return default_rec


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------

def should_force_converge(
    snapshots: list[IterationSnapshot],
    *,
    budget_pct: float = 0.0,
    iteration: int = 0,
    max_iterations: int = 15,
    config: ConvergenceConfig | None = None,
    caller: ModelCaller | None = None,
) -> tuple[bool, str]:
    """Deterministic decision: should we force convergence?

    Returns (should_converge, reason).

    This is a PRE-CHECK before the LLM adversarial convergence check,
    designed to save tokens when convergence is obvious — or when the
    optional ``caller`` confirms an ambiguous case.
    """
    if config is None:
        config = ConvergenceConfig()

    if len(snapshots) < config.min_iterations_for_convergence:
        return False, "too_few_iterations"

    score = compute_convergence_score(snapshots, config=config, caller=caller)

    # Budget emergency
    if budget_pct >= config.budget_emergency_pct:
        return True, f"budget_emergency({budget_pct:.0f}%)"

    # Budget pressure
    if budget_pct >= config.budget_pressure_pct and score.overall >= config.force_overall_min:
        return True, f"budget_pressure({budget_pct:.0f}%)_score({score.overall:.2f})"

    # Strong convergence signal
    if (score.gain_deceleration >= config.force_gain_decel
            and score.signal_resolution_rate >= config.force_signal_rate
            and score.confidence_stability >= config.force_conf_stability):
        return True, f"strong_convergence(score={score.overall:.2f})"

    # Max iterations approaching
    if iteration >= max_iterations - 1:
        return True, f"max_iterations({iteration}/{max_iterations})"

    # Hybrid: if the (possibly model-adjudicated) recommendation is "stop",
    # honour it even though the deterministic thresholds didn't fire.
    if score.recommendation == "stop":
        return True, f"hybrid_stop(score={score.overall:.2f})"

    return False, (
        f"not_ready(score={score.overall:.2f},"
        f"rec={score.recommendation})"
    )


def convergence_report(snapshots: list[IterationSnapshot]) -> dict:
    """Generate a human-readable convergence report for diagnostics."""
    score = compute_convergence_score(snapshots)
    details = score.signals_detail

    # Pull type aliases from details (always present when score is computed)
    ta: dict[str, int] = details.get("type_aliases", {})

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
                "open": int(details.get("open_signals", 0)),
                "addressed": int(details.get("addressed_signals", 0)),
            },
            "confidence_stability": {
                "score": score.confidence_stability,
                "mean": details.get("mean_confidence", 0),
                "std": details.get("std_confidence", 0),
            },
            "type_balance": {
                "score": score.type_balance,
                "observations": int(
                    details.get("active_entries", 0) - sum(ta.values())
                ),
                "type_aliases": ta,
            },
            "cross_document_coverage": {
                "score": score.cross_doc_coverage,
                "docs_with_entries": int(
                    details.get("documents_with_entries", 0)
                ),
                "total_docs": int(details.get("total_documents", 0)),
            },
        },
    }
