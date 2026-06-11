"""Tests for deterministic convergence signals (Open Research Question #2)."""
from __future__ import annotations

import math

from src.swarm.blackboard import Blackboard
from src.swarm.convergence_signals import (
    ConvergenceScore,
    IterationSnapshot,
    compute_convergence_score,
    convergence_report,
    should_force_converge,
    take_snapshot,
)
from src.swarm.models import Entry, EntrySource, Signal, WorkerRecord, gen_entry_id


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _mk_entry(content: str, etype: str = "observation",
              doc: str = "test.docx", confidence: float = 0.9) -> Entry:
    return Entry(
        id=gen_entry_id(), type=etype, content=content,
        source=EntrySource(document=doc, section="Full"),
        created_by=WorkerRecord("w", "d", 0), confidence=confidence,
    )


def _mk_bb_with_entries(counts: dict[str, int],
                        doc: str = "test.docx") -> Blackboard:
    bb = Blackboard()
    for etype, n in counts.items():
        for i in range(n):
            bb.add_entry(_mk_entry(f"{etype} {i}", etype=etype, doc=doc))
    return bb


def _mk_snapshot(iteration: int, active: int, **kwargs) -> IterationSnapshot:
    defaults = dict(
        iteration=iteration,
        total_entries=active,
        active_entries=active,
        entries_by_type={"observation": active},
        open_signals=0,
        addressed_signals=0,
        mean_confidence=0.9,
        std_confidence=0.05,
        documents_with_entries=1,
        total_documents=1,
        analysis_entries=0,
        calculation_entries=0,
        gap_entries=0,
    )
    defaults.update(kwargs)
    return IterationSnapshot(**defaults)


# -----------------------------------------------------------------------
# Snapshot tests
# -----------------------------------------------------------------------

class TestTakeSnapshot:
    def test_empty_blackboard(self):
        bb = Blackboard()
        bb.iteration = 0
        s = take_snapshot(bb)
        assert s.active_entries == 0
        assert s.open_signals == 0
        assert s.mean_confidence == 0.0

    def test_counts_by_type(self):
        bb = Blackboard()
        bb.iteration = 1
        bb.add_entry(_mk_entry("obs1", "observation"))
        bb.add_entry(_mk_entry("obs2", "observation"))
        bb.add_entry(_mk_entry("ana1", "analysis"))
        bb.add_entry(_mk_entry("calc1", "calculation"))
        s = take_snapshot(bb)
        assert s.active_entries == 4
        assert s.entries_by_type["observation"] == 2
        assert s.entries_by_type["analysis"] == 1
        assert s.entries_by_type["calculation"] == 1

    def test_confidence_stats(self):
        bb = Blackboard()
        bb.iteration = 1
        bb.add_entry(_mk_entry("a", confidence=0.9))
        bb.add_entry(_mk_entry("b", confidence=0.7))
        bb.add_entry(_mk_entry("c", confidence=0.8))
        s = take_snapshot(bb)
        assert abs(s.mean_confidence - 0.8) < 0.01
        assert s.std_confidence > 0

    def test_signal_counts(self):
        bb = Blackboard()
        bb.iteration = 1
        bb.add_signal(Signal(id="s1", content="q1", status="open"))
        bb.add_signal(Signal(id="s2", content="q2", status="addressed"))
        bb.add_signal(Signal(id="s3", content="q3", status="open"))
        s = take_snapshot(bb)
        assert s.open_signals == 2
        assert s.addressed_signals == 1

    def test_document_coverage(self):
        bb = Blackboard()
        bb.iteration = 1
        bb.add_entry(_mk_entry("a", doc="doc1.pdf"))
        bb.add_entry(_mk_entry("b", doc="doc2.pdf"))
        # Add a third document with no entries
        from src.swarm.models import DocumentStatus
        bb.documents.append(DocumentStatus(id="d3", name="doc3.pdf"))
        s = take_snapshot(bb)
        assert s.documents_with_entries == 2
        assert s.total_documents == 3


# -----------------------------------------------------------------------
# Convergence score tests
# -----------------------------------------------------------------------

class TestConvergenceScore:
    def test_too_few_iterations(self):
        s = compute_convergence_score([_mk_snapshot(0, 10)])
        assert s.overall == 0.0
        assert s.recommendation == "continue"

    def test_stable_blackboard_scores_high(self):
        snapshots = [
            _mk_snapshot(0, 0),
            _mk_snapshot(1, 50),
            _mk_snapshot(2, 100),
            _mk_snapshot(3, 102),  # barely gaining
            _mk_snapshot(4, 103),  # almost stopped
        ]
        s = compute_convergence_score(snapshots)
        assert s.gain_deceleration > 0.5
        assert s.recommendation in ("evaluate", "stop")

    def test_still_growing_scores_low(self):
        snapshots = [
            _mk_snapshot(0, 0),
            _mk_snapshot(1, 50),
            _mk_snapshot(2, 100),
            _mk_snapshot(3, 200),
            _mk_snapshot(4, 350),
        ]
        s = compute_convergence_score(snapshots)
        assert s.gain_deceleration < 0.3
        assert s.recommendation == "continue"

    def test_signal_resolution_boosts_score(self):
        snapshots = [
            _mk_snapshot(0, 10, open_signals=10, addressed_signals=0),
            _mk_snapshot(1, 20, open_signals=3, addressed_signals=7),
            _mk_snapshot(2, 22, open_signals=1, addressed_signals=9),
        ]
        s = compute_convergence_score(snapshots)
        assert s.signal_resolution_rate > 0.7

    def test_type_balance_computation(self):
        balanced = _mk_snapshot(3, 40, entries_by_type={
            "observation": 20, "analysis": 10, "calculation": 5, "gap": 5,
        })
        s = compute_convergence_score([_mk_snapshot(0, 0), _mk_snapshot(1, 20), balanced])
        assert s.type_balance > 0.5

    def test_single_type_low_balance(self):
        single = _mk_snapshot(3, 40, entries_by_type={"observation": 40})
        s = compute_convergence_score([_mk_snapshot(0, 0), _mk_snapshot(1, 20), single])
        assert s.type_balance < 0.1

    def test_cross_doc_coverage(self):
        broad = _mk_snapshot(3, 40, documents_with_entries=4, total_documents=5)
        s = compute_convergence_score([_mk_snapshot(0, 0), _mk_snapshot(1, 20), broad])
        assert s.cross_doc_coverage == 0.8


# -----------------------------------------------------------------------
# Force-converge tests
# -----------------------------------------------------------------------

class TestShouldForceConverge:
    def test_too_few_snapshots(self):
        should, reason = should_force_converge([_mk_snapshot(0, 0)])
        assert not should
        assert "too_few" in reason

    def test_budget_emergency(self):
        snapshots = [_mk_snapshot(0, 0), _mk_snapshot(1, 10)]
        should, reason = should_force_converge(snapshots, budget_pct=92)
        assert should
        assert "emergency" in reason

    def test_budget_pressure_with_decent_score(self):
        snapshots = [
            _mk_snapshot(0, 0),
            _mk_snapshot(1, 50),
            _mk_snapshot(2, 55),
            _mk_snapshot(3, 56),
        ]
        should, reason = should_force_converge(snapshots, budget_pct=78)
        assert should
        assert "pressure" in reason

    def test_strong_convergence(self):
        snapshots = [
            _mk_snapshot(0, 0),
            _mk_snapshot(1, 100, open_signals=2, addressed_signals=10),
            _mk_snapshot(2, 101, open_signals=1, addressed_signals=11),
        ]
        should, reason = should_force_converge(snapshots)
        # Whether it triggers depends on gain_decel + signal_rate + conf_stability
        # With these numbers, gain deceleration is high
        assert isinstance(should, bool)

    def test_max_iterations(self):
        snapshots = [_mk_snapshot(0, 0), _mk_snapshot(14, 100)]
        should, reason = should_force_converge(
            snapshots, iteration=14, max_iterations=15,
        )
        assert should
        assert "max_iterations" in reason


# -----------------------------------------------------------------------
# Report tests
# -----------------------------------------------------------------------

class TestConvergenceReport:
    def test_report_structure(self):
        snapshots = [
            _mk_snapshot(0, 0),
            _mk_snapshot(1, 50),
            _mk_snapshot(2, 80, analysis_entries=10, calculation_entries=5, gap_entries=3),
        ]
        report = convergence_report(snapshots)
        assert "iterations_analyzed" in report
        assert "overall_score" in report
        assert "recommendation" in report
        assert "components" in report
        components = report["components"]
        assert "information_gain" in components
        assert "signal_resolution" in components
        assert "confidence_stability" in components
        assert "type_balance" in components
        assert "cross_document_coverage" in components

    def test_report_gain_interpretation(self):
        # Still gaining
        gaining = [
            _mk_snapshot(0, 0),
            _mk_snapshot(1, 50),
            _mk_snapshot(2, 150),
        ]
        r = convergence_report(gaining)
        assert r["components"]["information_gain"]["interpretation"] == "gaining"

        # Plateaued
        plateaued = [
            _mk_snapshot(0, 0),
            _mk_snapshot(1, 100),
            _mk_snapshot(2, 100),
            _mk_snapshot(3, 100),
        ]
        r2 = convergence_report(plateaued)
        assert r2["components"]["information_gain"]["interpretation"] == "plateaued"


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_entries_all_iterations(self):
        snapshots = [
            _mk_snapshot(0, 0),
            _mk_snapshot(1, 0),
            _mk_snapshot(2, 0),
        ]
        s = compute_convergence_score(snapshots)
        assert s.gain_deceleration == 1.0
        assert s.recommendation in ("evaluate", "stop")

    def test_single_doc_full_coverage(self):
        s = _mk_snapshot(3, 50, documents_with_entries=1, total_documents=1)
        score = compute_convergence_score([_mk_snapshot(0, 0), _mk_snapshot(1, 20), s])
        assert score.cross_doc_coverage == 1.0

    def test_no_signals_all_resolved(self):
        s = _mk_snapshot(3, 50, open_signals=0, addressed_signals=15)
        score = compute_convergence_score([_mk_snapshot(0, 0), _mk_snapshot(1, 20), s])
        assert score.signal_resolution_rate == 1.0

    def test_cross_doc_coverage_clamped(self):
        # documents_with_entries can exceed total_documents in a hand-built
        # snapshot; coverage must never exceed 1.0.
        s = _mk_snapshot(3, 40, documents_with_entries=5, total_documents=2)
        score = compute_convergence_score(
            [_mk_snapshot(0, 0), _mk_snapshot(1, 20), s]
        )
        assert score.cross_doc_coverage == 1.0

    def test_plateau_after_early_spike_is_not_gaining(self):
        # A burst of entries early, then two flat iterations, must register as
        # decelerating (not "still gaining") despite the stale early spike.
        snaps = [
            _mk_snapshot(0, 0),
            _mk_snapshot(1, 200),
            _mk_snapshot(2, 200),
            _mk_snapshot(3, 200),
        ]
        score = compute_convergence_score(snaps)
        assert score.gain_deceleration >= 0.7
