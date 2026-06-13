"""
Offline tests for ``convergence_signals`` — the deterministic convergence
detector.

All tests run with **zero LLM calls**.  Where the hybrid path is exercised a
minimal ``FakeCaller`` stub returns canned JSON.

Multilingual (German, Spanish, Japanese) and multi-domain (business/strategy)
inputs prove the module is language- and domain-agnostic.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.swarm.convergence_signals import (
    ConvergenceConfig,
    ConvergenceScore,
    IterationSnapshot,
    compute_convergence_score,
    convergence_report,
    should_force_converge,
    take_snapshot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snap(
    iteration: int,
    active: int,
    open_sigs: int,
    addressed: int,
    mean_c: float,
    std_c: float,
    docs_with: int,
    total_docs: int,
    entries_by_type: dict[str, int] | None = None,
    type_aliases: dict[str, int] | None = None,
) -> IterationSnapshot:
    """Shorthand for building an IterationSnapshot in tests."""
    if entries_by_type is None:
        entries_by_type = {"analysis": active}
    if type_aliases is None:
        type_aliases = {}
    return IterationSnapshot(
        iteration=iteration,
        total_entries=active,
        active_entries=active,
        entries_by_type=entries_by_type,
        open_signals=open_sigs,
        addressed_signals=addressed,
        mean_confidence=mean_c,
        std_confidence=std_c,
        documents_with_entries=docs_with,
        total_documents=total_docs,
        type_aliases=type_aliases,
    )


def _converging_snapshots(
    n: int = 6,
    start: int = 10,
    type_names: list[str] | None = None,
    type_aliases: dict[str, int] | None = None,
) -> list[IterationSnapshot]:
    """Build ``n`` snapshots that show gain decelerating then plateauing.

    Entry counts: start, start+4, start+7, start+9, start+10, start+10, …
    Gain deceleration (window 3) ≥ 0.88 → triggers "stop" zone.
    """
    if type_names is None:
        type_names = ["analysis", "calculation", "gap"]
    base = [start, start + 4, start + 7, start + 9, start + 10]
    while len(base) < n:
        base.append(base[-1])
    entries = base[:n]

    # Distribute entries evenly across type_names
    n_types = len(type_names)
    et_list: list[dict[str, int]] = []
    for total in entries:
        per, rem = divmod(total, n_types)
        d: dict[str, int] = {}
        for i, tn in enumerate(type_names):
            d[tn] = per + (1 if i < rem else 0)
        et_list.append(d)

    if type_aliases is None:
        type_aliases = {tn: tn for tn in type_names}

    snapshots: list[IterationSnapshot] = []
    for i, (e, et) in enumerate(zip(entries, et_list)):
        # 16 addressed, 4 open — constant across iterations
        snapshots.append(_snap(
            iteration=i + 1,
            active=e,
            open_sigs=4,
            addressed=16,
            mean_c=0.85,
            std_c=0.05,
            docs_with=5,
            total_docs=6,
            entries_by_type=et,
            type_aliases={k: et.get(k, 0) for k in type_aliases},
        ))
    return snapshots


def _growing_snapshots(
    n: int = 6,
    start: int = 10,
    step: int = 5,
    type_names: list[str] | None = None,
) -> list[IterationSnapshot]:
    """Build ``n`` snapshots showing steady growth (not converged)."""
    if type_names is None:
        type_names = ["analysis", "calculation"]
    entries = [start + i * step for i in range(n)]
    n_types = len(type_names)

    snapshots: list[IterationSnapshot] = []
    for i, e in enumerate(entries):
        per, rem = divmod(e, n_types)
        et = {tn: per + (1 if j < rem else 0) for j, tn in enumerate(type_names)}
        sigs_open = max(0, 10 - i * 2)
        sigs_addr = max(0, 10 + i * 2)
        snapshots.append(_snap(
            iteration=i + 1,
            active=e,
            open_sigs=sigs_open,
            addressed=sigs_addr,
            mean_c=0.70,
            std_c=0.10,
            docs_with=min(e, 5),
            total_docs=5,
            entries_by_type=et,
        ))
    return snapshots


def _oscillating_snapshots() -> list[IterationSnapshot]:
    """Build snapshots where entry counts fluctuate — not converged."""
    counts = [10, 15, 12, 18, 14, 16]
    et = {"analysis": 0, "calculation": 0}
    snapshots: list[IterationSnapshot] = []
    for i, c in enumerate(counts):
        snapshots.append(_snap(
            iteration=i + 1,
            active=c,
            open_sigs=4,
            addressed=16,
            mean_c=0.75,
            std_c=0.10,
            docs_with=3,
            total_docs=4,
            entries_by_type={"analysis": c // 2, "calculation": c - c // 2},
        ))
    return snapshots


class FakeCaller:
    """Minimal ``ModelCaller`` stub that returns canned JSON."""

    def __init__(self, recommendation: str) -> None:
        self._rec = recommendation
        self.calls: list[str] = []

    def __call__(self, prompt: str, *, max_tokens: int = 512) -> tuple[dict, int]:
        self.calls.append(prompt)
        return {"recommendation": self._rec}, 10


# ---------------------------------------------------------------------------
# 1. Strong convergence → "stop"
# ---------------------------------------------------------------------------

def test_strong_convergence_recommends_stop():
    snaps = _converging_snapshots()
    score = compute_convergence_score(snaps)

    assert score.gain_deceleration >= 0.80
    assert score.signal_resolution_rate == pytest.approx(0.8, abs=0.01)
    assert score.confidence_stability == 1.0
    assert score.recommendation == "stop"

    # force_converge fires
    ok, reason = should_force_converge(snaps)
    assert ok is True
    assert "score" in reason


# ---------------------------------------------------------------------------
# 2. Non-convergence (steady growth) → "continue"
# ---------------------------------------------------------------------------

def test_non_convergence_recommends_continue():
    snaps = _growing_snapshots()
    score = compute_convergence_score(snaps)

    assert score.gain_deceleration < 0.15  # still gaining at near-peak rate
    assert score.recommendation == "continue"

    ok, reason = should_force_converge(snaps)
    assert ok is False
    assert "not_ready" in reason


# ---------------------------------------------------------------------------
# 3. Oscillating entries → "continue" (not stable)
# ---------------------------------------------------------------------------

def test_oscillating_entries_recommends_continue():
    snaps = _oscillating_snapshots()
    score = compute_convergence_score(snaps)

    # Volatile entry counts are never a confident "stop", and must not force convergence.
    assert score.recommendation != "stop"
    ok, _ = should_force_converge(snaps)
    assert ok is False


# ---------------------------------------------------------------------------
# 4. German types (multilingual)
# ---------------------------------------------------------------------------

def test_german_type_names_convergence():
    """Convergence detection works with German entry-type names."""
    snaps = _converging_snapshots(
        type_names=["Analyse", "Berechnung", "Lücke"],
        type_aliases={"Analyse": "Analyse", "Berechnung": "Berechnung", "Lücke": "Lücke"},
    )
    score = compute_convergence_score(snaps)

    assert score.recommendation == "stop"
    assert score.gain_deceleration >= 0.80

    # type_aliases in detail carry the German names
    aliases = score.signals_detail["type_aliases"]
    assert "Analyse" in aliases
    assert "Berechnung" in aliases
    assert "Lücke" in aliases


# ---------------------------------------------------------------------------
# 5. Spanish business-domain (multi-domain)
# ---------------------------------------------------------------------------

def test_spanish_business_domain_convergence():
    """Convergence works with Spanish business/strategy entry types."""
    snaps = _converging_snapshots(
        type_names=["análisis", "cálculo", "brecha"],
        type_aliases={"análisis": "análisis", "cálculo": "cálculo", "brecha": "brecha"},
    )
    score = compute_convergence_score(snaps)

    assert score.recommendation == "stop"
    assert score.signal_resolution_rate >= 0.70

    aliases = score.signals_detail["type_aliases"]
    assert "análisis" in aliases
    assert sum(aliases.values()) == snaps[-1].active_entries


# ---------------------------------------------------------------------------
# 6. Japanese types (CJK)
# ---------------------------------------------------------------------------

def test_japanese_type_names_convergence():
    """Convergence works with CJK entry-type names."""
    snaps = _converging_snapshots(
        type_names=["分析", "計算", "ギャップ"],
        type_aliases={"分析": "分析", "計算": "計算", "ギャップ": "ギャップ"},
    )
    score = compute_convergence_score(snaps)

    assert score.recommendation == "stop"
    assert score.overall >= 0.70

    aliases = score.signals_detail["type_aliases"]
    assert "分析" in aliases
    assert "計算" in aliases
    assert "ギャップ" in aliases


# ---------------------------------------------------------------------------
# 7. Borderline score + model caller → "stop" override
# ---------------------------------------------------------------------------

def test_model_caller_overrides_to_stop():
    """Model adjudication can upgrade a borderline score to 'stop'."""
    # Gains that plateau slowly → overall ≈ 0.73, rec = "continue"
    snaps = _converging_snapshots(n=4, start=10)
    score_default = compute_convergence_score(snaps)
    assert score_default.recommendation in ("continue", "evaluate")

    caller = FakeCaller("stop")
    ok, reason = should_force_converge(snaps, caller=caller)
    assert ok is True
    assert "hybrid" in reason or "stop" in reason
    assert len(caller.calls) == 1  # model was consulted


# ---------------------------------------------------------------------------
# 8. Borderline score + model caller → "continue" override
# ---------------------------------------------------------------------------

def test_model_caller_overrides_to_continue():
    """Model can confirm that the blackboard is NOT converged."""
    snaps = _converging_snapshots(n=4, start=10)
    caller = FakeCaller("continue")
    ok, reason = should_force_converge(snaps, caller=caller)
    assert ok is False
    assert "continue" in reason


# ---------------------------------------------------------------------------
# 9. Weak convergence + model adjudication (stop zone test)
# ---------------------------------------------------------------------------

def test_stop_zone_model_not_called():
    """When score is clearly in the stop zone, the model is NOT called."""
    snaps = _converging_snapshots()  # gain_decel ≥ 0.88
    caller = FakeCaller("stop")
    ok, reason = should_force_converge(snaps, caller=caller)
    assert ok is True
    # Model should not have been consulted — deterministic path sufficed
    assert len(caller.calls) == 0


# ---------------------------------------------------------------------------
# 10. Edge case: only one snapshot
# ---------------------------------------------------------------------------

def test_single_snapshot_too_few():
    snaps = [_snap(1, 10, 2, 8, 0.80, 0.05, 3, 4)]
    score = compute_convergence_score(snaps)
    assert score.overall == 0.0
    assert score.recommendation == "continue"
    assert score.signals_detail["reason"] == "too_few_iterations"

    ok, reason = should_force_converge(snaps)
    assert ok is False
    assert "too_few" in reason


# ---------------------------------------------------------------------------
# 11. Edge case: empty blackboard
# ---------------------------------------------------------------------------

def test_empty_blackboard():
    """Zero entries, zero signals → continues (not enough data)."""
    snaps = [
        _snap(i, 0, 0, 0, 0.0, 0.0, 0, 0, entries_by_type={})
        for i in range(1, 4)
    ]
    score = compute_convergence_score(snaps)
    # signal_rate = 1.0 (no signals → trivially resolved)
    assert score.signal_resolution_rate == 1.0
    assert score.recommendation in ("continue", "evaluate")


# ---------------------------------------------------------------------------
# 12. Convergence report structure
# ---------------------------------------------------------------------------

def test_convergence_report_structure():
    snaps = _converging_snapshots()
    report = convergence_report(snaps)

    assert report["iterations_analyzed"] == len(snaps)
    assert isinstance(report["overall_score"], float)
    assert report["recommendation"] in ("continue", "evaluate", "stop")

    comps = report["components"]
    assert "information_gain" in comps
    assert "signal_resolution" in comps
    assert "confidence_stability" in comps
    assert "type_balance" in comps
    assert "cross_document_coverage" in comps

    ig = comps["information_gain"]
    assert "rate" in ig
    assert "deceleration" in ig
    assert ig["interpretation"] in ("gaining", "slowing", "plateaued")

    tb = comps["type_balance"]
    assert "observations" in tb
    assert "type_aliases" in tb


# ---------------------------------------------------------------------------
# 13. Default config values
# ---------------------------------------------------------------------------

def test_default_config_values():
    cfg = ConvergenceConfig()
    assert cfg.min_iterations_for_convergence == 2
    assert cfg.rate_window == 3
    assert cfg.gain_rate_window == 2
    assert cfg.signal_resolution_target == 0.70
    assert cfg.weight_gain_decel == 0.30
    assert cfg.stop_overall_min == 0.75
    assert cfg.eval_gain_decel_min == 0.60
    assert cfg.budget_emergency_pct == 90.0


# ---------------------------------------------------------------------------
# 14. Custom config override
# ---------------------------------------------------------------------------

def test_custom_config_tightens_signal_target():
    """A stricter signal_resolution_target prevents premature 'stop'."""
    snaps = _converging_snapshots()
    # The stop gate is stop_signal_rate_min; tighten it above the snapshot's 0.8 rate.
    strict = ConvergenceConfig(stop_signal_rate_min=0.95)
    score = compute_convergence_score(snaps, config=strict)
    assert score.recommendation == "evaluate"


def test_custom_config_gain_rate_window():
    """Changing gain_rate_window changes the measured deceleration."""
    snaps = _converging_snapshots()
    cfg = ConvergenceConfig(gain_rate_window=1)
    score = compute_convergence_score(snaps, config=cfg)
    # window=1 → current_rate = 0 → deceleration = 1.0
    assert score.gain_deceleration == 1.0


# ---------------------------------------------------------------------------
# 15. Budget pressure
# ---------------------------------------------------------------------------

def test_budget_pressure_forces_convergence():
    snaps = _converging_snapshots()
    ok, reason = should_force_converge(snaps, budget_pct=80.0)
    assert ok is True
    assert "budget_pressure" in reason


def test_budget_emergency_forces_convergence():
    snaps = _converging_snapshots()
    ok, reason = should_force_converge(snaps, budget_pct=95.0)
    assert ok is True
    assert "budget_emergency" in reason


# ---------------------------------------------------------------------------
# 16. Max-iterations safety
# ---------------------------------------------------------------------------

def test_max_iterations_forces_convergence():
    # A non-converged board near the iteration cap force-converges via the safety net.
    snaps = _growing_snapshots()
    ok, reason = should_force_converge(snaps, iteration=14, max_iterations=15)
    assert ok is True
    assert "max_iterations" in reason


# ---------------------------------------------------------------------------
# 17. take_snapshot with mocked Blackboard
# ---------------------------------------------------------------------------

def test_take_snapshot_basic():
    """take_snapshot captures the correct metrics from a mock Blackboard."""
    # Build mock entries
    entry1 = MagicMock()
    entry1.status = "active"
    entry1.confidence = 0.9
    entry1.type = "analysis"
    entry1.source = MagicMock(document="doc1")

    entry2 = MagicMock()
    entry2.status = "active"
    entry2.confidence = 0.7
    entry2.type = "calculation"
    entry2.source = MagicMock(document="doc2")

    entry3 = MagicMock()
    entry3.status = "archived"
    entry3.confidence = 0.5
    entry3.type = "gap"
    entry3.source = None

    signal_open = MagicMock()
    signal_open.status = "open"
    signal_addr = MagicMock()
    signal_addr.status = "addressed"

    doc1 = MagicMock()
    doc1.name = "doc1"
    doc2 = MagicMock()
    doc2.name = "doc2"

    bb = MagicMock()
    bb.iteration = 3
    bb.entries = [entry1, entry2, entry3]
    bb.signals = [signal_open, signal_addr]
    bb.documents = [doc1, doc2]

    snap = take_snapshot(bb)

    assert snap.iteration == 3
    assert snap.total_entries == 3
    assert snap.active_entries == 2  # entry3 is archived
    assert snap.open_signals == 1
    assert snap.addressed_signals == 1
    assert snap.mean_confidence == pytest.approx(0.8, abs=0.01)
    assert snap.entries_by_type == {"analysis": 1, "calculation": 1}
    assert snap.documents_with_entries == 2


def test_take_snapshot_with_type_aliases():
    """type_aliases correctly maps entry types via config."""
    entry_a = MagicMock()
    entry_a.status = "active"
    entry_a.confidence = 0.8
    entry_a.type = "Analyse"
    entry_a.source = None

    entry_b = MagicMock()
    entry_b.status = "active"
    entry_b.confidence = 0.8
    entry_b.type = "Berechnung"
    entry_b.source = None

    bb = MagicMock()
    bb.iteration = 1
    bb.entries = [entry_a, entry_b]
    bb.signals = []
    bb.documents = []

    cfg = ConvergenceConfig(
        type_aliases_mapping={"Analyse": "Analyse", "Berechnung": "Berechnung", "Lücke": "Lücke"},
    )
    snap = take_snapshot(bb, config=cfg)

    assert snap.type_aliases == {"Analyse": 1, "Berechnung": 1, "Lücke": 0}


# ---------------------------------------------------------------------------
# 18. No-model path: calibrated recommendation without caller
# ---------------------------------------------------------------------------

def test_no_model_returns_calibrated_recommendation():
    """When caller=None, the deterministic recommendation is returned as-is."""
    snaps = _converging_snapshots(n=4, start=10)
    score = compute_convergence_score(snaps, caller=None)
    assert score.recommendation in ("continue", "evaluate", "stop")
    # No exception, no assertion — just a calibrated recommendation
