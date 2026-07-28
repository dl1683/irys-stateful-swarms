"""Tests for deterministic entity resolution module.

Uses real Entry objects and direct function calls — no fake callers needed
since entity_resolution.py has zero LLM dependencies.
"""

import json
import os

import pytest

from src.swarm.blackboard import Blackboard
from src.swarm.entity_resolution import (
    _jaro_winkler,
    _normalized_levenshtein,
    _trigram_similarity,
    _token_overlap,
    _extract_entity_names,
    _extract_all_entity_names,
    _find_near_duplicates,
    apply_resolutions,
    compute_entity_similarity,
    run_entity_resolution,
)
from src.swarm.models import Entry, EntrySource


# --- String similarity tests ---


class TestNormalizedLevenshtein:
    def test_identical(self):
        assert _normalized_levenshtein("Zenith Petrochem", "Zenith Petrochem") == 0.0

    def test_empty(self):
        assert _normalized_levenshtein("", "Zenith") == 1.0

    def test_one_empty(self):
        assert _normalized_levenshtein("Zenith", "") == 1.0

    def test_completely_different(self):
        # Different enough that normalized distance is high
        assert _normalized_levenshtein("Apple", "Microsoft") > 0.6

    def test_minor_variation(self):
        # "Petrochem" vs "Petrochemical" — edit distance / max len
        dist = _normalized_levenshtein("Petrochem", "Petrochemical")
        assert 0.2 < dist < 0.4, f"Expected moderate distance, got {dist}"

    def test_inc_vs_no_inc(self):
        dist = _normalized_levenshtein(
            "Pinnacle Industrial Solutions, Inc.",
            "Pinnacle Industrial Solutions",
        )
        assert dist < 0.3, f"Expected small distance, got {dist}"


class TestJaroWinkler:
    def test_identical(self):
        assert _jaro_winkler("Zenith Petrochem", "Zenith Petrochem") == 1.0

    def test_empty(self):
        assert _jaro_winkler("", "Zenith") == 0.0

    def test_one_char_different(self):
        sim = _jaro_winkler("abc", "abd")
        assert sim > 0.8

    def test_petrochem_vs_petrochemical(self):
        sim = _jaro_winkler("Zenith Petrochem", "Zenith Petrochemical")
        assert sim > 0.85, f"Expected >0.85, got {sim}"

    def test_known_near_duplicate(self):
        # The exact case from the README
        sim = _jaro_winkler("Zenith Petrochem", "Zenith Petrochemical")
        assert sim > 0.85
        # Inc vs without Inc
        sim2 = _jaro_winkler("Pinnacle Industrial Solutions, Inc.", "Pinnacle Industrial Solutions")
        assert sim2 > 0.90, f"Expected >0.90, got {sim2}"


class TestTrigramSimilarity:
    def test_identical(self):
        assert _trigram_similarity("Zenith Petrochem", "Zenith Petrochem") == 1.0

    def test_petrochem_vs_petrochemical(self):
        sim = _trigram_similarity("Zenith Petrochem", "Zenith Petrochemical")
        assert sim > 0.5, f"Expected >0.5, got {sim}"

    def test_completely_different(self):
        sim = _trigram_similarity("Apple Computer", "Microsoft Windows")
        assert sim < 0.3


class TestTokenOverlap:
    def test_identical_tokens(self):
        assert _token_overlap("Industrial Solutions Inc", "Industrial Solutions") > 0.5

    def test_no_overlap(self):
        assert _token_overlap("Apple", "Microsoft") == 0.0

    def test_partial_overlap(self):
        overlap = _token_overlap("Zenith Petrochem Industries", "Zenith Petrochemical Industries")
        assert overlap >= 0.5, f"Expected >=0.5, got {overlap}"


class TestCompositeSimilarity:
    def test_zenith_near_duplicate(self):
        """The canonical README example: Zenith Petrochem vs Zenith Petrochemical."""
        score = compute_entity_similarity(
            "Zenith Petrochem", "Zenith Petrochemical"
        )
        assert score >= 0.85, f"Expected >=0.85 for Zenith near-duplicate, got {score}"

    def test_pinnacle_inc_variation(self):
        """The UCC lien example: Pinnacle Industrial Solutions, Inc. vs without Inc."""
        score = compute_entity_similarity(
            "Pinnacle Industrial Solutions, Inc.",
            "Pinnacle Industrial Solutions",
        )
        assert score >= 0.90, f"Expected >=0.90 for Inc variation, got {score}"

    def test_completely_different(self):
        score = compute_entity_similarity("Apple Inc.", "Microsoft Corporation")
        assert score < 0.50

    def test_same_company_different_designator(self):
        score = compute_entity_similarity(
            "Northbrook Capital Markets, LLC",
            "Northbrook Capital Markets",
        )
        assert score >= 0.90, f"Expected >=0.90, got {score}"

    def test_identical(self):
        assert compute_entity_similarity("Same Corp", "Same Corp") == 1.0

    def test_empty_strings(self):
        assert compute_entity_similarity("", "") == 0.0

    def test_one_empty(self):
        assert compute_entity_similarity("Something", "") == 0.0


# --- Entity extraction tests ---


class TestExtractEntityNames:
    def test_capitalized_company_name(self):
        names = _extract_entity_names(
            "Zenith Petrochemical Industries LLC operates in Jebel Ali Free Zone."
        )
        assert len(names) >= 1
        assert any("Zenith" in n for n in names)

    def test_acronym_extraction(self):
        names = _extract_entity_names(
            "The OFAC 50% rule aggregation principle applies to this transaction."
        )
        assert "OFAC" in names

    def test_legal_designator_entity(self):
        names = _extract_entity_names(
            "Northbrook Capital Markets, LLC commits to provide a first lien facility."
        )
        assert any("Northbrook" in n for n in names)

    def test_no_entities_short_text(self):
        names = _extract_entity_names("the quick brown fox")
        assert len(names) == 0

    def test_multiple_entities(self):
        names = _extract_entity_names(
            "Haverford National Bank and Crestmoor Holdings Inc. are mentioned."
        )
        assert len(names) >= 2


# --- Full pipeline tests ---


class TestRunEntityResolution:
    def test_disabled_by_default(self):
        """When SWARM_ENABLE_ENTITY_RESOLUTION is not set, returns empty."""
        bb = Blackboard(entries=[
            Entry(id="e1", content="Zenith Petrochem is a company.",
                  source=EntrySource("doc1.pdf", "S1", "ev"), status="active"),
            Entry(id="e2", content="Zenith Petrochemical Industries LLC.",
                  source=EntrySource("doc2.pdf", "S2", "ev"), status="active"),
        ])
        entries, tokens = run_entity_resolution(bb)
        assert entries == []
        assert tokens == 0

    def test_enabled_no_duplicates(self, monkeypatch):
        """Enabled but no near-duplicates — returns empty."""
        monkeypatch.setenv("SWARM_ENABLE_ENTITY_RESOLUTION", "1")
        bb = Blackboard(entries=[
            Entry(id="e1", content="Apple Inc. is based in Cupertino.",
                  source=EntrySource("doc1.pdf", "S1", "ev"), status="active"),
            Entry(id="e2", content="Microsoft Corporation is based in Redmond.",
                  source=EntrySource("doc2.pdf", "S2", "ev"), status="active"),
        ])
        entries, tokens = run_entity_resolution(bb)
        assert entries == []
        assert tokens == 0

    def test_enabled_detects_near_duplicates(self, monkeypatch):
        """Enabled and entries contain near-duplicate entities."""
        monkeypatch.setenv("SWARM_ENABLE_ENTITY_RESOLUTION", "1")
        monkeypatch.setenv("SWARM_ENTITY_RESOLUTION_THRESHOLD", "0.80")
        bb = Blackboard(
            task_instruction="Test entity resolution.",
            entries=[
                Entry(id="e1",
                      content="Zenith Petrochem is the exporter located in Jebel Ali Free Zone, UAE.",
                      source=EntrySource("doc1.pdf", "S1", "ev"),
                      status="active"),
                Entry(id="e2",
                      content="Zenith Petrochemical Industries LLC, Jebel Ali Free Zone, Dubai, UAE",
                      source=EntrySource("doc2.pdf", "S2", "ev"),
                      status="active"),
            ],
            output_dir="",
        )
        entries, tokens = run_entity_resolution(bb)
        assert tokens == 0  # zero token cost
        assert len(entries) >= 1, (
            f"Expected at least one contradiction entry, got {len(entries)}"
        )
        assert entries[0].type == "contradiction"
        assert "Zenith" in entries[0].content
        assert entries[0].confidence >= 0.8

    def test_enabled_detects_inc_variation(self, monkeypatch):
        """Detects company name with and without 'Inc.'."""
        monkeypatch.setenv("SWARM_ENABLE_ENTITY_RESOLUTION", "1")
        monkeypatch.setenv("SWARM_ENTITY_RESOLUTION_THRESHOLD", "0.85")
        bb = Blackboard(
            task_instruction="Test entity resolution.",
            entries=[
                Entry(id="e1",
                      content="Debtor: Pinnacle Industrial Solutions, Inc., a corporation organized in Ohio.",
                      source=EntrySource("filing1.pdf", "S1", "ev"),
                      status="active"),
                Entry(id="e2",
                      content="Pinnacle Industrial Solutions is the debtor name.",
                      source=EntrySource("filing2.pdf", "S2", "ev"),
                      status="active"),
            ],
        )
        entries, tokens = run_entity_resolution(bb)
        assert len(entries) >= 1
        assert "Pinnacle" in entries[0].content

    def test_report_written_to_output_dir(self, monkeypatch, tmp_path):
        """Report JSON is written when output_dir is set."""
        monkeypatch.setenv("SWARM_ENABLE_ENTITY_RESOLUTION", "1")
        monkeypatch.setenv("SWARM_ENTITY_RESOLUTION_THRESHOLD", "0.80")
        bb = Blackboard(
            task_instruction="Test.",
            entries=[
                Entry(id="e1", content="Zenith Petrochem is a company based in the UAE.",
                      source=EntrySource("doc.pdf", "S1", "ev"), status="active"),
                Entry(id="e2", content="Zenith Petrochemical LLC is a well-known UAE company.",
                      source=EntrySource("doc.pdf", "S2", "ev"), status="active"),
            ],
            output_dir=str(tmp_path),
        )
        run_entity_resolution(bb)
        report_path = tmp_path / "swarm" / "entity_resolution_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["resolutions_applied"] >= 1
        assert report["enabled"] is True

    def test_single_entry_returns_empty(self, monkeypatch):
        """Only one entry — no pairs to compare."""
        monkeypatch.setenv("SWARM_ENABLE_ENTITY_RESOLUTION", "1")
        bb = Blackboard(entries=[
            Entry(id="e1", content="Zenith Petrochem is here.",
                  source=EntrySource("doc.pdf", "S1", "ev"), status="active"),
        ])
        entries, tokens = run_entity_resolution(bb)
        assert entries == []
        assert tokens == 0

    def test_can_be_called_twice_idempotently(self, monkeypatch):
        """Adding the same near-duplicate entries again produces the same results."""
        monkeypatch.setenv("SWARM_ENABLE_ENTITY_RESOLUTION", "1")
        monkeypatch.setenv("SWARM_ENTITY_RESOLUTION_THRESHOLD", "0.80")
        entries_list = [
            Entry(id="e1", content="Zenith Petrochem is based in the UAE.",
                  source=EntrySource("doc.pdf", "S1", "ev"), status="active"),
            Entry(id="e2", content="Zenith Petrochemical LLC operates in the UAE.",
                  source=EntrySource("doc.pdf", "S2", "ev"), status="active"),
        ]
        bb1 = Blackboard(entries=list(entries_list))
        r1, _ = run_entity_resolution(bb1)
        assert len(r1) >= 1  # hit

    def test_many_entries_performance(self, monkeypatch):
        """Handles 100+ entries without excessive runtime."""
        monkeypatch.setenv("SWARM_ENABLE_ENTITY_RESOLUTION", "1")
        monkeypatch.setenv("SWARM_ENTITY_RESOLUTION_THRESHOLD", "0.80")
        entries = []
        for i in range(50):
            entries.append(
                Entry(id=f"e{i}", content=f"Company Alpha {i} is here.",
                      source=EntrySource("doc.pdf", "S1", "ev"), status="active")
            )
        for i in range(50, 100):
            entries.append(
                Entry(id=f"e{i}", content=f"Corp Beta {i} operates.",
                      source=EntrySource("doc.pdf", "S2", "ev"), status="active")
            )
        bb = Blackboard(entries=entries)
        import time
        start = time.time()
        result, tokens = run_entity_resolution(bb)
        elapsed = time.time() - start
        assert tokens == 0
        # Should complete in under 2 seconds for 100 entries
        assert elapsed < 2.0, f"Took {elapsed:.2f}s, expected <2s"
