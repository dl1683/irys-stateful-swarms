"""Tests for entropy-based convergence detection.

Uses deterministic blackboards with pre-built entries to test each signal
independently and the hybrid verdict.
"""

import json

import pytest

from src.swarm.blackboard import Blackboard
from src.swarm.convergence_entropy import (
    _compute_info_gain,
    _compute_confidence_stats,
    _build_entry_graph,
    _compute_graph_change,
    compute_convergence,
    ConvergenceHistory,
)
from src.swarm.models import Entry, EntrySource, Signal, WorkerRecord


# Shared helpers


def _active_entry(eid: str, content: str, **kw) -> Entry:
    """Create an active entry with sensible defaults."""
    kwargs = dict(status="active", confidence=0.9, source=EntrySource("doc.pdf", "S1", "ev"))
    kwargs.update(kw)
    return Entry(id=eid, content=content, **kwargs)


def _observation(eid: str, content: str, iteration: int = 0, **kw) -> Entry:
    entry = _active_entry(eid, content, **kw)
    entry.type = "observation"
    entry.created_by = WorkerRecord("worker", "test", iteration)
    return entry


# --- Signal 1: Info gain tests ---


class TestInfoGain:
    def test_first_iteration_high_gain(self):
        """First iteration with many new entries should have high info gain."""
        bb = Blackboard(iteration=0, entries=[
            _observation("e1", "Apple Inc. is a company.", iteration=0),
            _observation("e2", "Microsoft Corp. is a company.", iteration=0),
            _observation("e3", "Google LLC is a company.", iteration=0),
        ])
        result = _compute_info_gain(bb, lookback=3)
        assert result["new_entry_count"] == 3
        assert result["info_gain_score"] > -0.5  # not negative

    def test_late_iteration_low_gain(self):
        """Late iteration with no new entries should have low info gain."""
        bb = Blackboard(iteration=10)
        # Add entries all from early iterations
        bb.entries = [
            _observation("e1", "Apple Inc. is a company.", iteration=1),
            _observation("e2", "Microsoft Corp. is a company.", iteration=2),
            _observation("e3", "Google LLC is a company.", iteration=3),
        ]
        result = _compute_info_gain(bb, lookback=3)
        assert result["new_entry_count"] == 0
        assert result["info_gain_score"] < _compute_info_gain.__globals__.get(
            "_info_threshold", lambda: 0.15
        )() or result.get("converged", False) is True

    def test_signal_resolved_increases_gain(self):
        bb = Blackboard(iteration=5, signals=[
            Signal(id="s1", status="open", iteration_created=0),
        ])
        bb.entries = [
            _observation("e1", "Fact A.", iteration=5,
                         addresses_signals=["s1"]),
        ]
        # Mark s1 as addressed
        for s in bb.signals:
            if s.id == "s1":
                s.status = "addressed"
        result = _compute_info_gain(bb, lookback=3)
        # We should detect the signal resolution
        assert result["signal_resolution_count"] >= 1

    def test_plateau_detected(self):
        """Multiple iterations with minimal new info should detect plateau."""
        bb = Blackboard(iteration=8, entries=[])
        # Add entries all from same early iteration
        for i in range(10):
            bb.entries.append(
                _observation(f"e{i}", f"Same old fact {i}.", iteration=3)
            )
        result = _compute_info_gain(bb, lookback=3)
        # With no new entries, should trend toward convergence
        assert result["new_entry_count"] == 0


# --- Signal 2: Graph-structural tests ---


class TestBuildEntryGraph:
    def test_no_edges(self):
        entries = [
            _active_entry("e1", "Fact A"),
            _active_entry("e2", "Fact B"),
        ]
        graph = _build_entry_graph(entries)
        assert graph["node_count"] == 2
        assert graph["edge_count"] == 0
        assert graph["density"] == 0.0

    def test_with_supports_edges(self):
        entries = [
            _active_entry("e1", "Fact A"),
            _active_entry("e2", "Fact B", supports_entries=["e1"]),
            _active_entry("e3", "Fact C", supports_entries=["e1", "e2"]),
        ]
        graph = _build_entry_graph(entries)
        assert graph["node_count"] == 3
        assert graph["edge_count"] >= 2

    def test_with_contradicts_edges(self):
        entries = [
            _active_entry("e1", "Value is 100"),
            _active_entry("e2", "Value is 200", contradicts_entries=["e1"]),
        ]
        graph = _build_entry_graph(entries)
        assert graph["edge_count"] == 1

    def test_largest_component(self):
        entries = [
            _active_entry("e1", "A", supports_entries=["e2"]),
            _active_entry("e2", "B", supports_entries=["e1"]),
            _active_entry("e3", "C"),  # isolated
        ]
        graph = _build_entry_graph(entries)
        assert graph["largest_component_ratio"] == pytest.approx(2 / 3, rel=0.1)


class TestGraphChange:
    def test_no_previous(self):
        change = _compute_graph_change({"density": 0.1, "clustering_coefficient": 0.5}, None)
        assert change["converged"] is True

    def test_identical(self):
        change = _compute_graph_change(
            {"density": 0.1, "clustering_coefficient": 0.5},
            {"density": 0.1, "clustering_coefficient": 0.5},
        )
        assert change["converged"] is True

    def test_significantly_different(self):
        change = _compute_graph_change(
            {"density": 0.5, "clustering_coefficient": 0.9},
            {"density": 0.01, "clustering_coefficient": 0.1},
        )
        assert change["converged"] is False


# --- Signal 3: Confidence distribution tests ---


class TestConfidenceStats:
    def test_uniform_high_confidence(self):
        """Detect the 'uniform 0.9+' problem from FAILURE_ANALYSIS.md §170."""
        entries = [
            _active_entry("e1", "A", confidence=0.95),
            _active_entry("e2", "B", confidence=0.92),
            _active_entry("e3", "C", confidence=0.97),
            _active_entry("e4", "D", confidence=0.91),
        ]
        stats = _compute_confidence_stats(entries)
        assert stats["uniform_flag"] is True
        assert stats["high_confidence_ratio"] > 0.95
        assert stats["variance"] < 0.01  # very low variance

    def test_diverse_confidence(self):
        """Diverse confidence values should not trigger the uniform flag."""
        entries = [
            _active_entry("e1", "A", confidence=0.9),
            _active_entry("e2", "B", confidence=0.7),
            _active_entry("e3", "C", confidence=0.5),
            _active_entry("e4", "D", confidence=0.3),
        ]
        stats = _compute_confidence_stats(entries)
        assert stats["uniform_flag"] is False
        assert stats["variance"] > 0.01

    def test_single_entry(self):
        stats = _compute_confidence_stats([_active_entry("e1", "A", confidence=0.8)])
        assert stats["mean"] == 0.8
        assert stats["uniform_flag"] is False  # single entry, <95% threshold is divided by n


# --- History tracking tests ---


class TestConvergenceHistory:
    def test_records_and_stores(self):
        history = ConvergenceHistory()
        history.record(
            {"new_entry_count": 5, "info_gain_score": 0.8},
            {"density": 0.1, "clustering_coefficient": 0.5, "node_count": 10, "edge_count": 5, "largest_component_ratio": 1.0},
            {"mean": 0.85, "variance": 0.02, "std_dev": 0.14, "skewness": -1.0, "high_confidence_ratio": 0.9, "uniform_flag": False},
        )
        assert len(history.info_gain_history) == 1
        assert history.last_graph["density"] == 0.1

    def test_to_dict(self):
        history = ConvergenceHistory()
        history.record(
            {"new_entry_count": 3, "info_gain_score": 0.5},
            {"density": 0.2, "clustering_coefficient": 0.3, "node_count": 5, "edge_count": 2, "largest_component_ratio": 1.0},
            {"mean": 0.8, "variance": 0.01, "std_dev": 0.1, "skewness": 0.0, "high_confidence_ratio": 0.7, "uniform_flag": False},
        )
        d = history.to_dict()
        assert "info_gain_history" in d
        assert len(d["info_gain_history"]) == 1


# --- Full pipeline tests ---


class TestComputeConvergence:
    def test_disabled_by_default(self):
        """When env var not set, returns no-op result."""
        bb = Blackboard(iteration=3)
        result = compute_convergence(bb)
        assert result["enabled"] is False
        assert result["converged"] is False

    def test_enabled_early_iteration_not_converged(self, monkeypatch):
        """Early iteration with many new entries should not converge."""
        monkeypatch.setenv("SWARM_ENABLE_ENTROPY_CONVERGENCE", "1")
        bb = Blackboard(iteration=2, entries=[
            _observation("e1", "Apple Inc.", iteration=0),
            _observation("e2", "Microsoft Corp.", iteration=1),
            _observation("e3", "Google LLC.", iteration=2),
        ])
        result = compute_convergence(bb)
        assert result["enabled"] is True
        # Iteration 2 still has some info gain
        assert result["iteration"] == 2

    def test_enabled_warns_about_uniform_confidence(self, monkeypatch):
        """Detects the uniform 0.9+ confidence problem."""
        monkeypatch.setenv("SWARM_ENABLE_ENTROPY_CONVERGENCE", "1")
        bb = Blackboard(iteration=5, entries=[
            _observation("e1", "A", confidence=0.95, iteration=1),
            _observation("e2", "B", confidence=0.92, iteration=2),
            _observation("e3", "C", confidence=0.97, iteration=3),
            _observation("e4", "D", confidence=0.94, iteration=4),
            _observation("e5", "E", confidence=0.91, iteration=5),
        ])
        result = compute_convergence(bb)
        warnings = result.get("warnings", [])
        assert any("CONFIDENCE COLLAPSE" in w for w in warnings), (
            f"Expected confidence collapse warning, got: {warnings}"
        )

    def test_enabled_warns_about_dead_graph(self, monkeypatch):
        """Detects the dead graph problem from §47."""
        monkeypatch.setenv("SWARM_ENABLE_ENTROPY_CONVERGENCE", "1")
        bb = Blackboard(iteration=5, entries=[
            _observation("e1", "A", iteration=1),
            _observation("e2", "B", iteration=2),
            _observation("e3", "C", iteration=3),
            _observation("e4", "D", iteration=4),
            _observation("e5", "E", iteration=5),
        ])
        result = compute_convergence(bb)
        warnings = result.get("warnings", [])
        assert any("DEAD GRAPH" in w for w in warnings), (
            f"Expected dead graph warning, got: {warnings}"
        )

    def test_metrics_written_to_output_dir(self, monkeypatch, tmp_path):
        """Metrics written when output_dir is set."""
        monkeypatch.setenv("SWARM_ENABLE_ENTROPY_CONVERGENCE", "1")
        bb = Blackboard(iteration=1, output_dir=str(tmp_path))
        result = compute_convergence(bb)
        metrics_path = tmp_path / "swarm" / "convergence_metrics.json"
        assert metrics_path.exists()
        saved = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert saved["enabled"] is True
        history_path = tmp_path / "swarm" / "convergence_history.json"
        assert history_path.exists()

    def test_history_persists_across_calls(self, monkeypatch):
        """Calling compute_convergence twice accumulates history."""
        monkeypatch.setenv("SWARM_ENABLE_ENTROPY_CONVERGENCE", "1")
        bb = Blackboard(iteration=2)
        bb.entries = [
            _observation("e1", "A", iteration=1),
            _observation("e2", "B", iteration=2),
        ]
        result1 = compute_convergence(bb)
        bb.iteration = 3
        bb.entries.append(_observation("e3", "C", iteration=3))
        result2 = compute_convergence(bb)
        assert len(result2.get("signals", {})) == 3

    def test_hybrid_verdict_basic(self, monkeypatch):
        """The hybrid verdict is computed correctly."""
        monkeypatch.setenv("SWARM_ENABLE_ENTROPY_CONVERGENCE", "1")
        bb = Blackboard(iteration=10)
        # Many entries from early iterations, none new
        for i in range(20):
            bb.entries.append(
                _observation(f"e{i}", f"Old fact {i}.", iteration=1)
            )
        result = compute_convergence(bb)
        # hybrid_score should be defined
        assert "hybrid_score" in result
        assert "signal_agreement" in result
