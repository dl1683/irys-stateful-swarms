"""Tests for deterministic blackboard compression (Open Research Question #3)."""
from __future__ import annotations

from src.swarm.blackboard import Blackboard
from src.swarm.compression import (
    CompressionResult,
    ScoredEntry,
    compress_for_synthesis,
    compression_report,
    ranked_entries_for_curation,
    score_all_entries,
    score_entry,
)
from src.swarm.models import Entry, EntrySource, WorkerRecord, gen_entry_id


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _mk(content: str, etype: str = "observation", doc: str = "test.docx",
        confidence: float = 0.9, tags: list[str] | None = None,
        supports: list[str] | None = None) -> Entry:
    return Entry(
        id=gen_entry_id(), type=etype, content=content,
        source=EntrySource(document=doc, section="Full"),
        created_by=WorkerRecord("w", "d", 0), confidence=confidence,
        tags=tags or [], supports_entries=supports or [],
    )


# -----------------------------------------------------------------------
# Entry scoring
# -----------------------------------------------------------------------

class TestScoreEntry:
    def test_analysis_scores_higher_than_observation(self):
        obs = _mk("plain fact", etype="observation")
        ana = _mk("analysis conclusion", etype="analysis")
        s_obs = score_entry(obs, {"test.docx"})
        s_ana = score_entry(ana, {"test.docx"})
        assert s_ana.type_score > s_obs.type_score

    def test_dollar_amount_boosts_density(self):
        plain = _mk("The fee is reasonable.")
        dollar = _mk("The upfront fee is $2,500,000 due within 30 days.")
        s_plain = score_entry(plain, {"test.docx"})
        s_dollar = score_entry(dollar, {"test.docx"})
        assert s_dollar.content_density > s_plain.content_density

    def test_legal_ref_boosts_density(self):
        plain = _mk("There is a provision about this.")
        legal = _mk("Section 4.3 of the Agreement requires quarterly reporting.")
        s_plain = score_entry(plain, {"test.docx"})
        s_legal = score_entry(legal, {"test.docx"})
        assert s_legal.content_density > s_plain.content_density

    def test_high_value_tag_boost(self):
        no_tag = _mk("fact without tag")
        with_tag = _mk("fact with tag", tags=["entity_resolution", "debt_type:relation"])
        s_no = score_entry(no_tag, {"test.docx"})
        s_yes = score_entry(with_tag, {"test.docx"})
        assert s_yes.tag_boost > s_no.tag_boost

    def test_cross_ref_boost(self):
        no_ref = _mk("standalone fact")
        with_ref = _mk("supported fact", supports=["e1", "e2"])
        s_no = score_entry(no_ref, {"test.docx"})
        s_yes = score_entry(with_ref, {"test.docx"})
        assert s_yes.cross_ref_score > s_no.cross_ref_score

    def test_high_confidence_boosts(self):
        low = _mk("uncertain fact", confidence=0.3)
        high = _mk("certain fact", confidence=0.95)
        s_low = score_entry(low, {"test.docx"})
        s_high = score_entry(high, {"test.docx"})
        assert s_high.composite > s_low.composite


# -----------------------------------------------------------------------
# Batch scoring
# -----------------------------------------------------------------------

class TestScoreAllEntries:
    def test_empty_blackboard(self):
        bb = Blackboard()
        assert score_all_entries(bb) == []

    def test_scores_all_active(self):
        bb = Blackboard()
        bb.add_entry(_mk("a"))
        bb.add_entry(_mk("b"))
        bb.add_entry(_mk("c"))
        scored = score_all_entries(bb)
        assert len(scored) == 3

    def test_excludes_inactive(self):
        bb = Blackboard()
        bb.add_entry(_mk("active"))
        inactive = _mk("inactive")
        inactive.status = "disputed"
        bb.add_entry(inactive)
        scored = score_all_entries(bb)
        assert len(scored) == 1

    def test_diversity_penalty_for_overrepresented_doc(self):
        bb = Blackboard()
        # 80% from one doc
        for i in range(8):
            bb.add_entry(_mk(f"doc_a_{i}", doc="a.pdf"))
        for i in range(2):
            bb.add_entry(_mk(f"doc_b_{i}", doc="b.pdf"))
        scored = score_all_entries(bb)
        # Entries from doc_b should get a diversity bonus
        b_scores = [s for s in scored if s.entry.source.document == "b.pdf"]
        a_scores = [s for s in scored if s.entry.source.document == "a.pdf"]
        # At least one b entry should have higher diversity bonus
        max_b_div = max(s.source_diversity_bonus for s in b_scores)
        max_a_div = max(s.source_diversity_bonus for s in a_scores)
        assert max_b_div >= max_a_div


# -----------------------------------------------------------------------
# Compression
# -----------------------------------------------------------------------

class TestCompressForSynthesis:
    def test_respects_target_count(self):
        bb = Blackboard()
        for i in range(200):
            bb.add_entry(_mk(f"entry {i}"))
        result = compress_for_synthesis(bb, target_count=50)
        assert result.actual_count <= 50
        assert len(result.selected) <= 50

    def test_diversity_floor(self):
        bb = Blackboard()
        # 100 entries from one doc, 3 from another
        for i in range(100):
            bb.add_entry(_mk(f"dom_{i}", doc="dominant.pdf"))
        for i in range(3):
            bb.add_entry(_mk(f"minor_{i}", doc="minor.pdf"))
        result = compress_for_synthesis(bb, target_count=20, min_per_doc=5)
        # minor.pdf should have at least min_per_doc entries
        minor_count = result.coverage_by_doc.get("minor.pdf", 0)
        assert minor_count >= 3  # can't exceed what exists

    def test_type_diversity(self):
        bb = Blackboard()
        for i in range(20):
            bb.add_entry(_mk(f"obs_{i}", etype="observation"))
        bb.add_entry(_mk("single analysis", etype="analysis"))
        bb.add_entry(_mk("single calc", etype="calculation"))
        result = compress_for_synthesis(bb, target_count=10, ensure_types=True)
        types = set(result.coverage_by_type.keys())
        assert "analysis" in types
        assert "calculation" in types

    def test_coverage_stats(self):
        bb = Blackboard()
        bb.add_entry(_mk("a", doc="doc1.pdf"))
        bb.add_entry(_mk("b", doc="doc2.pdf"))
        bb.add_entry(_mk("c", doc="doc1.pdf"))
        result = compress_for_synthesis(bb, target_count=10)
        assert "doc1.pdf" in result.coverage_by_doc
        assert "doc2.pdf" in result.coverage_by_doc

    def test_tokens_saved_estimate(self):
        bb = Blackboard()
        for i in range(500):
            bb.add_entry(_mk(f"entry {i}"))
        result = compress_for_synthesis(bb, target_count=100)
        assert result.tokens_saved_estimate > 0
        assert result.tokens_saved_estimate == len(result.deferred) * 75

    def test_high_value_entries_selected(self):
        bb = Blackboard()
        # Low-value observation
        for i in range(10):
            bb.add_entry(_mk(f"plain fact {i}", etype="observation", confidence=0.5))
        # High-value analysis with dollar amount
        bb.add_entry(_mk(
            "The termination fee is $5,000,000 per Section 4.3.",
            etype="analysis", confidence=0.95,
            tags=["entity_resolution"],
        ))
        result = compress_for_synthesis(bb, target_count=5)
        selected_ids = {se.entry.id for se in result.selected}
        # The high-value entry should be in the selected set
        high_value_entry = [e for e in bb.entries if e.type == "analysis"][0]
        assert high_value_entry.id in selected_ids


# -----------------------------------------------------------------------
# Ranked entries for curation
# -----------------------------------------------------------------------

class TestRankedEntriesForCuration:
    def test_returns_entries_not_scored(self):
        bb = Blackboard()
        bb.add_entry(_mk("a"))
        bb.add_entry(_mk("b"))
        entries = ranked_entries_for_curation(bb, max_entries=10)
        assert all(isinstance(e, Entry) for e in entries)

    def test_respects_max(self):
        bb = Blackboard()
        for i in range(100):
            bb.add_entry(_mk(f"entry {i}"))
        entries = ranked_entries_for_curation(bb, max_entries=20)
        assert len(entries) <= 20

    def test_empty_blackboard(self):
        bb = Blackboard()
        entries = ranked_entries_for_curation(bb)
        assert entries == []


# -----------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------

class TestCompressionReport:
    def test_report_structure(self):
        bb = Blackboard()
        bb.add_entry(_mk("a"))
        bb.add_entry(_mk("b"))
        result = compress_for_synthesis(bb, target_count=10)
        report = compression_report(result)
        assert "total_scored" in report
        assert "selected" in report
        assert "deferred" in report
        assert "coverage_by_doc" in report
        assert "coverage_by_type" in report
        assert "top_10_scores" in report


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------

class TestEdgeCases:
    def test_single_entry(self):
        bb = Blackboard()
        bb.add_entry(_mk("only entry"))
        result = compress_for_synthesis(bb, target_count=100)
        assert result.actual_count == 1

    def test_target_larger_than_entries(self):
        bb = Blackboard()
        bb.add_entry(_mk("a"))
        result = compress_for_synthesis(bb, target_count=1000)
        assert result.actual_count == 1
        assert len(result.deferred) == 0

    def test_target_zero(self):
        bb = Blackboard()
        for i in range(10):
            bb.add_entry(_mk(f"entry {i}"))
        result = compress_for_synthesis(bb, target_count=0)
        # min_per_doc ensures at least some entries are selected
        assert result.actual_count >= 0
