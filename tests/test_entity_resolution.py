from __future__ import annotations
import pytest
from src.swarm.blackboard import Blackboard
from src.swarm.entity_resolution import (
    EntityMention, are_aliases, normalise, resolve_entities,
    _edit_distance, _normalised_edit_distance, _tokenset,
)
from src.swarm.models import Entry, EntrySource, WorkerRecord, gen_entry_id


def _mention(raw: str, entry_id: str = "e1", document: str = "doc.pdf") -> EntityMention:
    norm = normalise(raw)
    return EntityMention(raw=raw, normalised=norm, tokens=_tokenset(norm),
                         entry_id=entry_id, document=document)


def _obs(content: str, doc: str = "doc_a.pdf", entry_id: str | None = None) -> Entry:
    eid = entry_id or gen_entry_id()
    return Entry(
        id=eid, type="observation", content=content,
        source=EntrySource(document=doc, section="1", evidence=content[:40]),
        confidence=0.9, status="active",
        created_by=WorkerRecord("reader", "initial_reading", 1),
    )


def _bb(*entries: Entry) -> Blackboard:
    bb = Blackboard(task_instruction="Test task", iteration=1)
    for e in entries:
        bb.entries.append(e)
        if not hasattr(bb, "_entry_index"):
            bb._entry_index = {}
        bb._entry_index[e.id] = e
    return bb


class TestNormalise:
    def test_lowercase(self):
        assert normalise("Acme Corp") == "acme"

    def test_strips_inc(self):
        assert normalise("Acme Corp Inc.") == "acme"

    def test_strips_llc(self):
        assert normalise("Widgets LLC") == "widgets"

    def test_strips_limited(self):
        # "capital" and "limited" are both legal suffixes; both are stripped
        assert normalise("Northbrook Capital Limited") == "northbrook"

    def test_preserves_content_words(self):
        result = normalise("Zenith Petrochemical Industries LLC")
        assert "zenith" in result
        assert "petrochemical" in result
        assert "industries" in result
        assert "llc" not in result

    def test_drops_punctuation(self):
        result = normalise("Smith, Jones & Partners LLP")
        assert "," not in result
        assert "&" not in result
        assert "llp" not in result

    def test_empty_string(self):
        assert normalise("") == ""

    def test_only_suffix(self):
        result = normalise("LLC")
        assert isinstance(result, str)

    def test_unicode_normalise(self):
        result = normalise("Société Générale SA")
        assert "sa" not in result
        assert "soci" in result or "societe" in result


class TestEditDistance:
    def test_identical(self):
        assert _edit_distance("hello", "hello") == 0

    def test_single_insert(self):
        assert _edit_distance("abc", "abcd") == 1

    def test_single_delete(self):
        assert _edit_distance("abcd", "abc") == 1

    def test_single_sub(self):
        assert _edit_distance("abc", "axc") == 1

    def test_petrochem_petrochemical(self):
        ed = _edit_distance("petrochem", "petrochemical")
        assert ed == 4

    def test_empty_vs_word(self):
        assert _edit_distance("", "hello") == 5

    def test_both_empty(self):
        assert _edit_distance("", "") == 0


class TestAreAliases:
    def test_exact_after_normalise(self):
        a = _mention("Zenith Petrochemical Industries LLC")
        b = _mention("Zenith Petrochemical Industries")
        assert are_aliases(a, b)

    def test_zenith_abbreviation(self):
        a = _mention("Zenith Petrochem Industries LLC")
        b = _mention("Zenith Petrochemical Industries LLC")
        assert are_aliases(a, b)

    def test_bank_with_and_without_descriptor(self):
        a = _mention("Haverford National Bank")
        b = _mention("Haverford National Bank, N.A.")
        assert are_aliases(a, b)

    def test_legal_suffix_difference(self):
        a = _mention("Pinnacle Industrial Solutions Inc.")
        b = _mention("Pinnacle Industrial Solutions")
        assert are_aliases(a, b)

    def test_abbreviation_token_overlap(self):
        a = _mention("Applied Compute")
        b = _mention("AC")
        assert not are_aliases(a, b)

    def test_different_entities(self):
        a = _mention("Goldman Sachs")
        b = _mention("Morgan Stanley")
        assert not are_aliases(a, b)

    def test_no_shared_token_not_alias(self):
        a = _mention("Northbrook Capital")
        b = _mention("Southern Finance Group")
        assert not are_aliases(a, b)

    def test_typo_one_char(self):
        a = _mention("Crestmore Capital Partners")
        b = _mention("Crestmoor Capital Partners")
        assert are_aliases(a, b)

    def test_prefix_anchor_with_shared_token(self):
        a = _mention("Northbrook Capital Markets LLC")
        b = _mention("Northbrook Capital")
        assert are_aliases(a, b)

    def test_self(self):
        a = _mention("Acme Corporation")
        assert are_aliases(a, a)


class TestResolveEntities:
    def test_empty_blackboard(self):
        bb = _bb()
        assert resolve_entities(bb) == {}

    def test_no_cross_doc_aliases(self):
        bb = _bb(
            _obs("Zenith Petrochem signed the agreement.", "doc_a.pdf", "e1"),
            _obs("Zenith Petrochemical Industries confirmed.", "doc_a.pdf", "e2"),
        )
        resolve_entities(bb)
        assert [e for e in bb.entries if e.type == "entity_alias"] == []

    def test_cross_doc_alias_emits_entry_and_signal(self):
        bb = _bb(
            _obs("The exporter is Zenith Petrochem Industries LLC, located in Jebel Ali Free Zone.",
                 "commitment-letter.docx", "e1"),
            _obs("Zenith Petrochemical Industries LLC, Jebel Ali Free Zone, Dubai, UAE.",
                 "draft-credit-agreement.docx", "e2"),
        )
        clusters = resolve_entities(bb)
        assert len(clusters) >= 1
        alias_entries = [e for e in bb.entries if e.type == "entity_alias"]
        assert len(alias_entries) == 1
        assert "Zenith" in alias_entries[0].content
        assert alias_entries[0].created_by.worker_id == "entity_resolver"
        signals = [s for s in bb.signals if s.type == "entity_inconsistency"]
        assert len(signals) == 1
        assert signals[0].priority == "high"

    def test_entity_registry_populated(self):
        bb = _bb(
            _obs("Pinnacle Industrial Solutions Inc. is the borrower.", "ucc-filing.pdf", "e1"),
            _obs("Pinnacle Industrial Solutions owes the debt.", "credit-agreement.docx", "e2"),
        )
        resolve_entities(bb)
        assert hasattr(bb, "entity_registry")
        assert len(bb.entity_registry) >= 1

    def test_no_duplicate_signals_on_rerun(self):
        # "Crestmoor Ventures" vs "Crestmoor Venture" — 2-token forms, same
        # first token, alias via edit distance (1 char diff). Must fire on
        # first run and not again on subsequent runs.
        bb = _bb(
            _obs("Crestmoor Ventures LLC agreed to the terms.", "doc_a.pdf", "e1"),
            _obs("Crestmoor Venture LLC signed the note.", "doc_b.pdf", "e2"),
        )
        resolve_entities(bb)
        first_alias = len([e for e in bb.entries if e.type == "entity_alias"])
        first_sig = len([s for s in bb.signals if s.type == "entity_inconsistency"])
        assert first_alias >= 1
        assert first_sig >= 1
        resolve_entities(bb)
        assert len([e for e in bb.entries if e.type == "entity_alias"]) == first_alias
        assert len([s for s in bb.signals if s.type == "entity_inconsistency"]) == first_sig

    def test_singleton_mentions_not_clustered(self):
        bb = _bb(_obs("Goldman Sachs provided the commitment letter.", "letter.docx", "e1"))
        clusters = resolve_entities(bb)
        assert all(len(c.mentions) >= 2 for c in clusters.values())

    def test_three_way_cluster(self):
        # All three normalise to "northbrook capital markets" (LLC/Corp/bare
        # all strip to the same form), so they form a single cluster with 3
        # mentions from 3 different documents.
        bb = _bb(
            _obs("Northbrook Capital Markets LLC arranged the facility.", "term-sheet.docx", "e1"),
            _obs("Northbrook Capital Markets provided the commitment.", "commitment.docx", "e2"),
            _obs("Northbrook Capital Markets Corp. served as agent.", "credit-agreement.docx", "e3"),
        )
        clusters = resolve_entities(bb)
        assert any(len(c.mentions) >= 3 for c in clusters.values())

    def test_non_observation_entries_skipped(self):
        bb = _bb(
            Entry(id="e1", type="gap",
                  content="Zenith Petrochem not yet identified in doc_b.",
                  source=EntrySource(document="doc_a.pdf"), status="active",
                  created_by=WorkerRecord("worker", "gap", 1)),
            Entry(id="e2", type="strategy",
                  content="Zenith Petrochemical Industries LLC is the key counterparty.",
                  source=EntrySource(document="doc_b.pdf"), status="active",
                  created_by=WorkerRecord("seed_planner", "strategy", 0)),
        )
        resolve_entities(bb)
        assert [e for e in bb.entries if e.type == "entity_alias"] == []

    def test_superseded_entries_skipped(self):
        bb = _bb(_obs("Acme Corp signed the agreement.", "doc_a.pdf", "e1"))
        bb.entries[0].status = "superseded"
        bb.entries.append(_obs("Acme Corporation is the counterparty.", "doc_b.pdf", "e2"))
        resolve_entities(bb)
        assert [e for e in bb.entries if e.type == "entity_alias"] == []
