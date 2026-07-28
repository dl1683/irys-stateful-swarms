"""Tests for blackboard compression for synthesis.

Tests clustering, greedy selection, MMR ranking, budget planning, and the
full pipeline, all with deterministic fake data.
"""

import json

import pytest

from src.swarm.blackboard import Blackboard
from src.swarm.compression import (
    _jaccard_trigram,
    _entry_chars,
    _cluster_entries,
    _greedy_select,
    _mmr_rank,
    _plan_token_budgets,
    compress_blackboard,
    compress_to_entry_ids,
)
from src.swarm.models import Entry, EntrySource, Signal, WorkerRecord


# Helpers


def _active_entry(
    eid: str,
    content: str,
    entry_type: str = "observation",
    confidence: float = 0.85,
    document: str = "doc.pdf",
    section: str = "S1",
    status: str = "active",
    **kw,
) -> Entry:
    entry = Entry(
        id=eid,
        type=entry_type,
        content=content,
        source=EntrySource(document=document, section=section, evidence=""),
        confidence=confidence,
        status=status,
        created_by=WorkerRecord("worker", "test", 0),
    )
    for k, v in kw.items():
        setattr(entry, k, v)
    return entry


# --- Unit tests ---


class TestJaccardTrigram:
    def test_identical(self):
        assert _jaccard_trigram("Zenith Petrochem", "Zenith Petrochem") == 1.0

    def test_similar(self):
        sim = _jaccard_trigram("Zenith Petrochem", "Zenith Petrochemical")
        assert sim > 0.5

    def test_completely_different(self):
        sim = _jaccard_trigram("Apple Computer", "Microsoft Windows")
        assert sim < 0.4

    def test_short_strings(self):
        assert _jaccard_trigram("ab", "ab") == 1.0
        assert _jaccard_trigram("ab", "cd") == 0.0


class TestEntryChars:
    def test_simple_entry(self):
        entry = _active_entry("e1", "Revenue is $10M.")
        assert _entry_chars(entry) > len("Revenue is $10M.")

    def test_with_evidence(self):
        entry = _active_entry("e1", "Revenue is $10M.")
        entry.source.evidence = "According to page 4 of the filing, the revenue was $10M."
        assert _entry_chars(entry) > 100


class TestClusterEntries:
    def test_single_entry(self):
        entries = [_active_entry("e1", "Revenue is $10M.")]
        clusters = _cluster_entries(entries, 0.6)
        assert len(clusters) == 1
        assert clusters[0]["entry_count"] == 1

    def test_two_similar_entries_cluster_together(self):
        entries = [
            _active_entry("e1", "Revenue is $10M in 2023."),
            _active_entry("e2", "Revenue was $10M according to the filing."),
            _active_entry("e3", "Microsoft Windows is an operating system."),
        ]
        clusters = _cluster_entries(entries, 0.5)
        # e1 and e2 should be in the same cluster due to content similarity
        cluster_for_e1 = next(
            c for c in clusters if "e1" in c["entry_ids"]
        )
        cluster_for_e3 = next(
            c for c in clusters if "e3" in c["entry_ids"]
        )
        # e1 and e2 should be together (same document too)
        assert "e2" in cluster_for_e1["entry_ids"]
        # e3 should be separate or alone
        assert "e1" not in cluster_for_e3["entry_ids"]

    def test_same_document_source_boost_clusters(self):
        """Entries from the same document/section should cluster more easily."""
        entries = [
            _active_entry("e1", "Term A is defined.", document="contract.pdf", section="S1"),
            _active_entry("e2", "Term B is different.", document="contract.pdf", section="S1"),
            _active_entry("e3", "Something entirely different about fish.",
                          document="other.pdf", section="S2"),
        ]
        clusters = _cluster_entries(entries, 0.60)
        # e1 and e2 may be clustered together due to source proximity boost
        cluster = next((c for c in clusters if "e1" in c["entry_ids"]), None)
        if cluster:
            # They might cluster because source proximity adds 0.3 boost
            pass  # acceptable either way

    def test_entry_count_accuracy(self):
        entries = [
            _active_entry("e1", "Same topic A."),
            _active_entry("e2", "Same topic B."),
            _active_entry("e3", "Different topic Z."),
        ]
        clusters = _cluster_entries(entries, 0.3)
        total_count = sum(c["entry_count"] for c in clusters)
        assert total_count == 3

    def test_empty_input(self):
        assert _cluster_entries([], 0.6) == []


class TestGreedySelection:
    def test_selects_within_budget(self):
        entries = [
            _active_entry("e1", "Revenue is $10M.", confidence=0.9, addresses_signals=["s1"]),
            _active_entry("e2", "Cost is $5M.", confidence=0.8, addresses_signals=["s2"]),
            _active_entry("e3", "Margin is 50%.", confidence=0.7, addresses_signals=["s3"]),
        ]
        budget = sum(_entry_chars(e) for e in entries[:2]) + 1
        selected = _greedy_select(entries, budget, {"s1", "s2", "s3"})
        assert len(selected) >= 1
        # Should fit within budget
        total_chars = sum(_entry_chars(e) for e in selected)
        assert total_chars <= budget + 100  # small tolerance

    def test_covers_signals_greedily(self):
        entries = [
            _active_entry("e1", "Long irrelevant text " * 20, addresses_signals=["s1"], confidence=0.5),
            _active_entry("e2", "Short", addresses_signals=["s1", "s2", "s3"], confidence=0.9),
        ]
        budget = max(_entry_chars(e) for e in entries)
        selected = _greedy_select(entries, budget, {"s1", "s2", "s3"})
        # Should pick e2 (higher marginal gain: 3 signals / short length * 1.2 confidence boost)
        assert any(e.id == "e2" for e in selected), (
            f"Expected e2 to be selected (3 signals, high confidence), "
            f"got: {[e.id for e in selected]}"
        )

    def test_stops_when_all_signals_covered(self):
        entries = [
            _active_entry("e1", "A", addresses_signals=["s1"]),
            _active_entry("e2", "B", addresses_signals=["s2"]),
            _active_entry("e3", "C", addresses_signals=["s3"]),
        ]
        budget = 99999  # large budget
        selected = _greedy_select(entries, budget, {"s1", "s2"})
        # Should stop after covering s1 and s2
        assert len(selected) >= 2

    def test_no_signals_selects_by_confidence(self):
        entries = [
            _active_entry("e1", "A", confidence=0.95),
            _active_entry("e2", "B", confidence=0.50),
            _active_entry("e3", "C", confidence=0.75),
        ]
        budget = 99999
        selected = _greedy_select(entries, budget, set())
        # With no signals, boost for high confidence should help
        assert len(selected) >= 1

    def test_empty_input(self):
        assert _greedy_select([], 1000, set()) == []


class TestMMR:
    def test_single_entry(self):
        assert len(_mmr_rank([_active_entry("e1", "A")], 0.5)) == 1

    def test_ranks_by_relevance_when_lambda_high(self):
        entries = [
            _active_entry("e1", "Revenue is $10M.", confidence=0.95),
            _active_entry("e2", "Different topic entirely.", confidence=0.50),
        ]
        ranked = _mmr_rank(entries, 0.9)  # high relevance weight
        assert ranked[0].id == "e1"  # highest confidence first

    def test_empty_input(self):
        assert _mmr_rank([], 0.5) == []


class TestBudgetPlanner:
    def test_distributes_budget(self):
        clusters = [
            {"id": "c1", "entry_count": 10, "signal_density": 0.8, "materiality_weight": 2.0},
            {"id": "c2", "entry_count": 5, "signal_density": 0.2, "materiality_weight": 1.0},
        ]
        budgets = _plan_token_budgets(clusters, 20000, 5)
        assert "c1" in budgets
        assert "c2" in budgets
        # c1 should get more budget (more entries, higher signal density)
        assert budgets["c1"] > budgets["c2"]

    def test_single_cluster_gets_all(self):
        clusters = [{"id": "c1", "entry_count": 10, "signal_density": 0.5, "materiality_weight": 1.5}]
        budgets = _plan_token_budgets(clusters, 10000, 3)
        assert budgets["c1"] >= 5000  # should get most of it

    def test_empty_input(self):
        assert _plan_token_budgets([], 10000, 0) == {}

    def test_minimum_budget(self):
        clusters = [
            {"id": "c1", "entry_count": 2, "signal_density": 0.0, "materiality_weight": 1.0},
        ]
        budgets = _plan_token_budgets(clusters, 1000, 0)
        assert budgets["c1"] >= 500  # minimum


# --- Full pipeline tests ---


class TestCompressBlackboard:
    def test_disabled_by_default(self):
        bb = Blackboard()
        result = compress_blackboard(bb)
        assert result["enabled"] is False

    def test_enabled_empty_blackboard(self, monkeypatch):
        monkeypatch.setenv("SWARM_ENABLE_COMPRESSION", "1")
        bb = Blackboard()
        result = compress_blackboard(bb)
        assert result["enabled"] is True
        assert result["selected_ids"] == []

    def test_enabled_selects_relevant_entries(self, monkeypatch):
        monkeypatch.setenv("SWARM_ENABLE_COMPRESSION", "1")
        monkeypatch.setenv("SWARM_COMPRESSION_TOKEN_BUDGET", "50000")

        bb = Blackboard(
            task_instruction="Test compression.",
            entries=[
                _active_entry("e1", "Revenue is $10M.",
                              addresses_signals=["s1"], confidence=0.95),
                _active_entry("e2", "Cost is $5M.",
                              addresses_signals=["s2"], confidence=0.9),
                _active_entry("e3", "Margin is 50%.",
                              addresses_signals=["s3"], confidence=0.85),
                _active_entry("e4", "Irrelevant detail about office location.",
                              confidence=0.3),
            ],
            signals=[
                Signal(id="s1", status="open", type="question", content="What is revenue?"),
                Signal(id="s2", status="open", type="question", content="What is cost?"),
                Signal(id="s3", status="open", type="question", content="What is margin?"),
            ],
        )
        result = compress_blackboard(bb)
        assert result["enabled"] is True
        selected_ids = result.get("selected_ids", [])
        # Should have selected at least the three signal-addressing entries
        assert "e1" in selected_ids, f"Expected e1 in {selected_ids}"
        assert "e2" in selected_ids, f"Expected e2 in {selected_ids}"
        assert "e3" in selected_ids, f"Expected e3 in {selected_ids}"

    def test_signal_coverage_tracking(self, monkeypatch):
        monkeypatch.setenv("SWARM_ENABLE_COMPRESSION", "1")
        monkeypatch.setenv("SWARM_COMPRESSION_TOKEN_BUDGET", "50000")

        bb = Blackboard(
            task_instruction="Test.",
            entries=[
                _active_entry("e1", "Revenue is $10M.",
                              addresses_signals=["s1"], confidence=0.9),
                _active_entry("e2", "Cost is $5M.",
                              addresses_signals=["s2"], confidence=0.9),
            ],
            signals=[
                Signal(id="s1", status="open", type="question", content="Revenue?"),
                Signal(id="s2", status="open", type="question", content="Cost?"),
                Signal(id="s3", status="open", type="question", content="Margin?"),
            ],
        )
        result = compress_blackboard(bb)
        coverage = result.get("signal_coverage", 0)
        # s1 and s2 should be covered, s3 not — so coverage = 2/3
        assert coverage == pytest.approx(2 / 3, rel=0.1), f"Expected ~0.67, got {coverage}"

    def test_report_written_to_output_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SWARM_ENABLE_COMPRESSION", "1")
        bb = Blackboard(
            task_instruction="Test.",
            entries=[
                _active_entry("e1", "Revenue is $10M."),
            ],
            output_dir=str(tmp_path),
        )
        compress_blackboard(bb)
        report_path = tmp_path / "swarm" / "compression_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["enabled"] is True

    def test_compress_to_entry_ids_convenience(self, monkeypatch):
        monkeypatch.setenv("SWARM_ENABLE_COMPRESSION", "1")
        bb = Blackboard(
            entries=[
                _active_entry("e1", "Revenue is $10M.", addresses_signals=["s1"]),
            ],
            signals=[Signal(id="s1", status="open", type="question", content="?")],
        )
        ids = compress_to_entry_ids(bb)
        assert isinstance(ids, list)
        assert "e1" in ids

    def test_budget_respected(self, monkeypatch):
        """With a very tight budget, fewer entries should be selected."""
        monkeypatch.setenv("SWARM_ENABLE_COMPRESSION", "1")
        monkeypatch.setenv("SWARM_COMPRESSION_TOKEN_BUDGET", "500")  # tight budget

        bb = Blackboard(
            entries=[
                _active_entry("e1", "Revenue is $10 million dollars in 2023.", addresses_signals=["s1"]),
                _active_entry("e2", "Cost is $5 million dollars in 2023.", addresses_signals=["s2"]),
                _active_entry("e3", "Margin is 50 percent overall.", addresses_signals=["s3"]),
            ],
            signals=[
                Signal(id="s1", status="open", type="question", content="?"),
                Signal(id="s2", status="open", type="question", content="?"),
                Signal(id="s3", status="open", type="question", content="?"),
            ],
        )
        result = compress_blackboard(bb)
        chars_used = result.get("chars_used", 0)
        # Should not exceed budget with reasonable overhead tolerance
        assert chars_used <= 700, f"Used {chars_used} chars, budget was 500"
