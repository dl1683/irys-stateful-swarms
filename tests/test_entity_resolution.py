"""Tests for deterministic entity resolution (Open Research Question #1)."""
from __future__ import annotations

import pytest

from src.swarm.blackboard import Blackboard
from src.swarm.entity_resolution import (
    MatchResult,
    _cluster_matches,
    _jaro_winkler,
    _levenshtein_ratio,
    _normalize_entity,
    _is_prefix_variant,
    _token_overlap,
    _trigrams,
    resolve_entities,
    run_entity_resolution,
)
from src.swarm.models import Entry, EntrySource, WorkerRecord, gen_entry_id


# -----------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------

class TestNormalizeEntity:
    def test_strips_punctuation(self):
        assert _normalize_entity("Zenith Petrochem Industries LLC.") == "zenith petrochem industries"

    def test_strips_common_suffixes(self):
        assert _normalize_entity("Acme Corporation") == "acme"
        assert _normalize_entity("Foo Bar Inc.") == "foo bar"
        assert _normalize_entity("Baz Holdings Ltd") == "baz"

    def test_lowercases(self):
        assert _normalize_entity("NORTHBROOK CAPITAL") == "northbrook capital"

    def test_collapses_whitespace(self):
        assert _normalize_entity("  Foo   Bar  ") == "foo bar"


# -----------------------------------------------------------------------
# Similarity metrics
# -----------------------------------------------------------------------

class TestJaroWinkler:
    def test_identical(self):
        assert _jaro_winkler("hello", "hello") == 1.0

    def test_empty(self):
        assert _jaro_winkler("", "hello") == 0.0

    def test_similar(self):
        score = _jaro_winkler("martha", "marhta")
        assert score > 0.9

    def test_different(self):
        score = _jaro_winkler("abc", "xyz")
        assert score < 0.3

    def test_prefix_bonus(self):
        # "mario" vs "marion" — shared prefix should boost
        score = _jaro_winkler("mario", "marion")
        assert score > 0.85


class TestLevenshtein:
    def test_identical(self):
        assert _levenshtein_ratio("test", "test") == 1.0

    def test_one_edit(self):
        # "kitten" → "sitten" = 1 edit out of 6 chars
        score = _levenshtein_ratio("kitten", "sitten")
        assert score > 0.8

    def test_empty(self):
        assert _levenshtein_ratio("", "abc") == 0.0

    def test_completely_different(self):
        score = _levenshtein_ratio("abc", "xyz")
        assert score < 0.5


class TestTrigrams:
    def test_basic(self):
        tri = _trigrams("hello")
        assert "hel" in tri
        assert "ell" in tri
        assert "llo" in tri

    def test_short_string(self):
        tri = _trigrams("ab")
        assert tri == {"ab"}


class TestTokenOverlap:
    def test_identical(self):
        assert _token_overlap("foo bar baz", "foo bar baz") == 1.0

    def test_partial(self):
        score = _token_overlap("foo bar baz", "foo bar qux")
        assert 0.4 < score < 0.8

    def test_no_overlap(self):
        assert _token_overlap("aaa bbb", "ccc ddd") == 0.0


class TestIsPrefixVariant:
    def test_prefix(self):
        assert _is_prefix_variant("zenith petrochem", "zenith petrochemical")

    def test_not_prefix(self):
        assert not _is_prefix_variant("foo bar", "baz qux")


# -----------------------------------------------------------------------
# Real-world failure cases from the README
# -----------------------------------------------------------------------

class TestREADMEFailureCases:
    """Test the exact near-miss patterns documented in the repo's README."""

    def _make_entry(self, content: str, doc: str = "test.docx") -> Entry:
        return Entry(
            id=gen_entry_id(),
            type="observation",
            content=content,
            source=EntrySource(document=doc, section="Full Document"),
            created_by=WorkerRecord("test_worker", "test", 0),
            confidence=0.9,
        )

    def test_zenith_petrochem_variant(self):
        """'Zenith Petrochem Industries LLC' vs 'Zenith Petrochemical Industries LLC'"""
        bb = Blackboard()
        bb.add_entry(self._make_entry(
            "The exporter is Zenith Petrochem Industries LLC, located in Jebel Ali Free Zone, UAE.",
            doc="credit-agreement.docx",
        ))
        bb.add_entry(self._make_entry(
            "Zenith Petrochemical Industries LLC, Jebel Ali Free Zone, Dubai, UAE",
            doc="sanctions-screening.xlsx",
        ))
        result = resolve_entities(bb, threshold=0.65)
        assert len(result.match_pairs) >= 1, (
            "Should detect Zenith Petrochem vs Zenith Petrochemical"
        )
        assert result.match_pairs[0].composite > 0.65

    def test_pinnacle_solutions_variant(self):
        """'Pinnacle Industrial Solutions, Inc.' vs 'Pinnacle Industrial Solutions'

        After suffix stripping, both normalize to 'pinnacle industrial solutions',
        so they appear as an exact-match cluster (not a fuzzy match pair).
        """
        bb = Blackboard()
        bb.add_entry(self._make_entry(
            "Debtor: Pinnacle Industrial Solutions, Inc., a corporation organized in Ohio, Charter No. 2187650",
            doc="ucc-filing-1.pdf",
        ))
        bb.add_entry(self._make_entry(
            "Filing OH-2019-0178443 (Tristate Capital Equipment Corp.) against Pinnacle Industrial Solutions is LAPSED",
            doc="ucc-filing-2.pdf",
        ))
        result = resolve_entities(bb, threshold=0.65)
        # Exact normalization match → clusters, not match_pairs
        assert len(result.clusters) >= 1, (
            "Should detect Pinnacle Industrial Solutions, Inc. vs without Inc. "
            f"(clusters={len(result.clusters)}, match_pairs={len(result.match_pairs)})"
        )


# -----------------------------------------------------------------------
# Integration: blackboard entry creation
# -----------------------------------------------------------------------

class TestBlackboardIntegration:
    def _make_entry(self, content: str, doc: str = "test.docx") -> Entry:
        return Entry(
            id=gen_entry_id(),
            type="observation",
            content=content,
            source=EntrySource(document=doc, section="Full Document"),
            created_by=WorkerRecord("test_worker", "test", 0),
            confidence=0.9,
        )

    def test_creates_resolution_entries(self):
        bb = Blackboard()
        bb.add_entry(self._make_entry(
            "Northbrook Capital Markets, LLC commits to provide a loan.",
            doc="commitment-letter.docx",
        ))
        bb.add_entry(self._make_entry(
            "Northbrook Capital Markets LLC provided the commitment.",
            doc="credit-agreement.docx",
        ))
        initial_count = len(bb.entries)
        result = resolve_entities(bb, threshold=0.65)
        # Should have created at least one resolution entry
        assert len(bb.entries) > initial_count
        resolution_entries = [e for e in bb.entries if "entity_resolution" in (e.tags or [])]
        assert len(resolution_entries) >= 1

    def test_no_llm_tokens_used(self):
        bb = Blackboard()
        bb.add_entry(self._make_entry("Alpha Corp Holdings Inc is the buyer.", doc="a.docx"))
        bb.add_entry(self._make_entry("Alpha Corp Holdings is the acquiring party.", doc="b.docx"))
        result = resolve_entities(bb, threshold=0.65)
        assert result.tokens_used == 0

    def test_run_entity_resolution_report(self):
        bb = Blackboard()
        bb.add_entry(self._make_entry(
            "Acme Industries LLC manufactures widgets.",
            doc="contract.pdf",
        ))
        bb.add_entry(self._make_entry(
            "Acme Industries Inc. is the counterparty.",
            doc="amendment.pdf",
        ))
        report = run_entity_resolution(bb, threshold=0.65)
        assert report["tokens_used"] == 0
        assert report["schema_version"] == 1
        assert "clusters" in report
        assert "matches" in report


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_blackboard(self):
        bb = Blackboard()
        result = resolve_entities(bb)
        assert result.clusters == []
        assert result.match_pairs == []
        assert result.tokens_used == 0

    def test_single_entry_no_match(self):
        bb = Blackboard()
        bb.add_entry(Entry(
            id=gen_entry_id(), type="observation",
            content="Acme Corp is the borrower.",
            source=EntrySource(document="a.pdf"),
            created_by=WorkerRecord("w", "d", 0), confidence=0.9,
        ))
        result = resolve_entities(bb)
        assert result.match_pairs == []

    def test_different_entities_not_matched(self):
        """Completely different entities should not cluster."""
        bb = Blackboard()
        bb.add_entry(Entry(
            id=gen_entry_id(), type="observation",
            content="Alpha Corp Holdings Inc is the buyer.",
            source=EntrySource(document="a.pdf"),
            created_by=WorkerRecord("w", "d", 0), confidence=0.9,
        ))
        bb.add_entry(Entry(
            id=gen_entry_id(), type="observation",
            content="Beta Industries LLC is the seller.",
            source=EntrySource(document="b.pdf"),
            created_by=WorkerRecord("w", "d", 0), confidence=0.9,
        ))
        result = resolve_entities(bb, threshold=0.65)
        assert result.match_pairs == []

    def test_threshold_filtering(self):
        """Higher threshold should produce fewer matches."""
        bb = Blackboard()
        bb.add_entry(Entry(
            id=gen_entry_id(), type="observation",
            content="Zenith Petrochem Industries LLC is the exporter.",
            source=EntrySource(document="a.pdf"),
            created_by=WorkerRecord("w", "d", 0), confidence=0.9,
        ))
        bb.add_entry(Entry(
            id=gen_entry_id(), type="observation",
            content="Zenith Petrochemical Industries LLC is the counterparty.",
            source=EntrySource(document="b.pdf"),
            created_by=WorkerRecord("w", "d", 0), confidence=0.9,
        ))
        # Low threshold — should match
        result_low = resolve_entities(bb, threshold=0.55)
        # High threshold — might not match
        result_high = resolve_entities(bb, threshold=0.95)
        assert len(result_low.match_pairs) >= len(result_high.match_pairs)
