"""Tests for language- and domain-agnostic cross-document entity resolution (ORQ #1).

Covers normalization, similarity metrics, exact + fuzzy resolution, MULTILINGUAL inputs
(German / Spanish / Japanese / Cyrillic), a NON-LEGAL domain, corpus-derived generic-term
filtering (no hardcoded blocklist), corpus-driven leading-noise stripping, and the optional
ModelCaller hybrid path. Every test is deterministic and offline.
"""
from __future__ import annotations

import pytest

from src.swarm import entity_resolution
from src.swarm.blackboard import Blackboard
from src.swarm.entity_resolution import (
    ResolutionConfig,
    _corpus_generic_tokens,
    _default_extract,
    _is_prefix_variant,
    _jaro_winkler,
    _levenshtein_ratio,
    _normalize_entity,
    _strip_generic_edges,
    _token_overlap,
    _trigrams,
    resolve_entities,
    run_entity_resolution,
)
from src.swarm.models import Entry, EntrySource, WorkerRecord, gen_entry_id


def _entry(content: str, doc: str = "a.docx", typ: str = "observation") -> Entry:
    return Entry(
        id=gen_entry_id(), type=typ, content=content,
        source=EntrySource(document=doc, section="Full Document"),
        created_by=WorkerRecord("test_worker", "test", 0),
        confidence=0.9, status="active",
    )


def _bb(*entries: Entry) -> Blackboard:
    bb = Blackboard()
    for e in entries:
        bb.add_entry(e)
    return bb


def _clusters_canon(result) -> set[str]:
    from src.swarm.entity_resolution import _canonical_name
    return {_canonical_name(c).casefold() for c in result.clusters}


def _has_cluster(result, needle: str) -> bool:
    """True if any resolved cluster's canonical name contains ``needle`` (casefolded).

    The canonical is the most frequent raw surface form (often the longest, e.g. with a
    legal suffix), so identity is checked by containment, not exact equality.
    """
    from src.swarm.entity_resolution import _canonical_name
    n = needle.casefold()
    return any(n in _canonical_name(c).casefold() for c in result.clusters)


# ---------------------------------------------------------------------------
# Normalization — unicode-correct, language-neutral
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_strips_punctuation_and_suffix(self):
        assert _normalize_entity("Zenith Petrochem Industries LLC.") == "zenith petrochem industries"

    def test_strips_common_suffixes(self):
        assert _normalize_entity("Acme Corporation") == "acme"
        assert _normalize_entity("Foo Bar Inc.") == "foo bar"
        assert _normalize_entity("Baz Holdings Ltd") == "baz"

    def test_casefolds(self):
        assert _normalize_entity("NORTHBROOK CAPITAL") == "northbrook capital"

    def test_distinguishing_tokens_not_folded(self):
        # "Capital" must survive — it distinguishes Northbrook Capital from Northbrook Group.
        assert "capital" in _normalize_entity("Northbrook Capital")

    def test_collapses_whitespace(self):
        assert _normalize_entity("  Foo   Bar  ") == "foo bar"

    def test_unicode_german_suffix(self):
        assert _normalize_entity("Zenith Petrochemie GmbH") == "zenith petrochemie"

    def test_unicode_preserves_accents(self):
        assert _normalize_entity("Industrias Álvarez") == "industrias álvarez"

    def test_unicode_cjk(self):
        # NFKC + casefold leave CJK intact; spaced legal form folds.
        assert _normalize_entity("東京エレクトロン 株式会社") == "東京エレクトロン"


# ---------------------------------------------------------------------------
# Similarity metrics (unchanged, language-agnostic over code points)
# ---------------------------------------------------------------------------

class TestSimilarity:
    def test_jw_identical(self):
        assert _jaro_winkler("hello", "hello") == 1.0

    def test_jw_empty(self):
        assert _jaro_winkler("", "hello") == 0.0

    def test_jw_similar(self):
        assert _jaro_winkler("martha", "marhta") > 0.9

    def test_jw_cyrillic(self):
        assert _jaro_winkler("роснефть", "роснефти") > 0.85

    def test_lev_one_edit(self):
        assert _levenshtein_ratio("kitten", "sitten") > 0.8

    def test_lev_empty(self):
        assert _levenshtein_ratio("", "abc") == 0.0

    def test_trigrams(self):
        assert {"hel", "ell", "llo"} <= _trigrams("hello")

    def test_token_overlap(self):
        assert _token_overlap("acme robotics", "acme systems") == pytest.approx(1 / 3)

    def test_prefix_variant(self):
        assert _is_prefix_variant("zenith petrochem", "zenith petrochemical")
        assert not _is_prefix_variant("zenith petrochem", "zenith petroleum")


# ---------------------------------------------------------------------------
# Extraction — unicode category driven, no English/legal regex
# ---------------------------------------------------------------------------

class TestExtract:
    def test_english_span(self):
        out = _default_extract("Zenith Petrochemical Industries supplies the resin.")
        assert any("Zenith Petrochemical Industries" in s for s in out)

    def test_skips_lowercase_boilerplate(self):
        # a lowercase sentence yields no proper-name spans
        assert _default_extract("the agreement was signed on the closing date") == []

    def test_german(self):
        out = _default_extract("Die Müller Maschinenbau GmbH liefert Teile.")
        assert any("Müller Maschinenbau" in s for s in out)

    def test_cyrillic(self):
        out = _default_extract("Компания Роснефть подписала договор.")
        assert any("Роснефть" in s for s in out)

    def test_cjk_present(self):
        out = _default_extract("東京エレクトロン 株式会社")
        assert any("東京エレクトロン" in s for s in out)


# ---------------------------------------------------------------------------
# Core resolution — exact / fuzzy / distinct
# ---------------------------------------------------------------------------

class TestResolution:
    def test_exact_variant_asserted(self):
        bb = _bb(
            _entry("Pinnacle Industrial Solutions Inc is the borrower.", doc="a.pdf"),
            _entry("Pinnacle Industrial Solutions provided collateral.", doc="b.pdf"),
        )
        result = resolve_entities(bb)
        assert _has_cluster(result, "pinnacle industrial solutions")
        assert result.tokens_used == 0

    def test_fuzzy_variant_is_candidate(self):
        bb = _bb(
            _entry("Zenith Petrochem refined the feedstock.", doc="a.pdf"),
            _entry("Zenith Petrochemical reported earnings.", doc="b.pdf"),
        )
        result = resolve_entities(bb)
        assert result.clusters, "expected a candidate cluster for Petrochem/Petrochemical"
        created = result.entries_created
        assert any("CANDIDATE" in e.content for e in created)
        assert all(e.confidence <= 0.7 for e in created)

    def test_distinct_prefix_not_overmerged_blindly(self):
        # Petrochem vs Petroleum share a prefix but are different; never asserted as exact.
        bb = _bb(
            _entry("Zenith Petrochem refined the feedstock.", doc="a.pdf"),
            _entry("Zenith Petroleum drilled new wells.", doc="b.pdf"),
        )
        result = resolve_entities(bb)
        for e in result.entries_created:
            assert "CANDIDATE" in e.content  # only ever surfaced for review, not asserted


# ---------------------------------------------------------------------------
# Multilingual resolution
# ---------------------------------------------------------------------------

class TestMultilingual:
    def test_german_exact(self):
        bb = _bb(
            _entry("Zenith Petrochemie GmbH ist der Lieferant.", doc="de1.pdf"),
            _entry("Zenith Petrochemie hat geliefert.", doc="de2.pdf"),
        )
        assert _has_cluster(resolve_entities(bb), "zenith petrochemie")

    def test_spanish_fuzzy(self):
        bb = _bb(
            _entry("Constructora Ibérica firmó el contrato.", doc="es1.pdf"),
            _entry("Constructora Iberica entregó la obra.", doc="es2.pdf"),
        )
        assert resolve_entities(bb).clusters  # accent variant surfaced

    def test_japanese_exact(self):
        # Full-width brackets/punctuation (as in real JP filings) isolate the entity span;
        # the spaced legal form 株式会社 folds away, leaving an exact cross-doc match.
        bb = _bb(
            _entry("主要取引先（東京エレクトロン 株式会社）と合意。", doc="jp1.pdf"),
            _entry("「東京エレクトロン」は前年比で成長。", doc="jp2.pdf"),
        )
        assert _has_cluster(resolve_entities(bb), "東京エレクトロン")

    def test_cyrillic_fuzzy(self):
        # Single-token Cyrillic near-duplicates (nominative vs genitive) share no exact token;
        # the trigram pre-filter still catches them.
        bb = _bb(
            _entry("Поставщик: Роснефть.", doc="ru1.pdf"),
            _entry("Партнёр: Роснефти.", doc="ru2.pdf"),
        )
        assert _has_cluster(resolve_entities(bb), "роснефт")


# ---------------------------------------------------------------------------
# Non-legal domain + corpus-derived generic filtering (no hardcoded blocklist)
# ---------------------------------------------------------------------------

class TestMultiDomainAndGenerics:
    def test_no_hardcoded_blocklist_symbol(self):
        # The fragile, English-legal _DEFINED_TERMS / _LEADING_NOISE / _ORG_LIKE are gone.
        assert not hasattr(entity_resolution, "_DEFINED_TERMS")
        assert not hasattr(entity_resolution, "_LEADING_NOISE")
        assert not hasattr(entity_resolution, "_ORG_LIKE")

    def test_generic_inferred_from_corpus_strategy_domain(self):
        # A non-legal (strategy) corpus: "strategic"/"priority" recur across many distinct
        # names → inferred generic; the real entity "Acme Robotics" survives and resolves.
        boiler = [
            "Strategic Priority", "Strategic Initiative", "Strategic Plan",
            "Strategic Review", "Operational Priority", "Financial Priority",
            "Corporate Priority", "Market Position", "Revenue Growth",
        ]
        entries = [_entry(f"{name} drives the roadmap.", doc=f"d{i}.md")
                   for i, name in enumerate(boiler)]
        entries.append(_entry("Acme Robotics shipped units.", doc="x.md"))
        entries.append(_entry("Acme Robotics Inc raised funding.", doc="y.md"))
        result = resolve_entities(_bb(*entries))

        assert "strategic" in result.generic_terms
        assert "priority" in result.generic_terms
        assert "acme" not in result.generic_terms
        assert "robotics" not in result.generic_terms
        assert _has_cluster(result, "acme robotics")
        # an all-generic phrase is never emitted as an entity
        assert not _has_cluster(result, "strategic priority")

    def test_leading_noise_stripped_by_corpus(self):
        # Sentence-initial determiners are inferred generic and stripped, so a name still
        # matches its bare form even when one occurrence is sentence-initial.
        entries = [
            _entry("The Northbrook Capital fund closed.", doc="a.md"),
            _entry("Investors backed Northbrook Capital again.", doc="b.md"),
        ]
        # add corpus so "the" is seen across enough names to be generic
        entries += [_entry(f"The {w} expanded operations." , doc=f"c{i}.md")
                    for i, w in enumerate(
                        ["Vortex Systems", "Helios Energy", "Orion Freight",
                         "Cobalt Mining", "Delta Pharma", "Summit Foods"])]
        result = resolve_entities(_bb(*entries))
        assert _has_cluster(result, "northbrook capital")


# ---------------------------------------------------------------------------
# Hybrid ModelCaller path (offline, monkeypatched)
# ---------------------------------------------------------------------------

class TestHybridCaller:
    def _ambiguous_bb(self):
        return _bb(
            _entry("Northwind Logistics handled the freight.", doc="a.pdf"),
            _entry("Northwynd Logistics filed the manifest.", doc="b.pdf"),
        )

    def test_model_confirms_ambiguous(self, monkeypatch):
        calls = {"n": 0}

        def fake(caller, prompt, max_tokens=256):
            calls["n"] += 1
            return {"same": True, "confidence": 0.9}, 7

        monkeypatch.setattr(entity_resolution, "_call_model", fake)
        cfg = ResolutionConfig(high_confidence=0.99)  # force the ambiguous band
        result = resolve_entities(self._ambiguous_bb(), config=cfg, caller=object())
        assert calls["n"] >= 1
        assert result.tokens_used >= 7
        assert result.clusters
        assert any(m.adjudication == "model_confirmed" for m in result.match_pairs)

    def test_model_rejects_ambiguous(self, monkeypatch):
        def fake(caller, prompt, max_tokens=256):
            return {"same": False, "confidence": 0.9}, 5

        monkeypatch.setattr(entity_resolution, "_call_model", fake)
        cfg = ResolutionConfig(high_confidence=0.99)
        result = resolve_entities(self._ambiguous_bb(), config=cfg, caller=object())
        assert result.clusters == []  # rejected pair produces no cluster

    def test_no_caller_is_zero_token(self):
        result = resolve_entities(self._ambiguous_bb())
        assert result.tokens_used == 0


# ---------------------------------------------------------------------------
# Config + report
# ---------------------------------------------------------------------------

class TestConfigAndReport:
    def test_config_threshold_override(self):
        bb = _bb(
            _entry("Zenith Petrochem refined feedstock.", doc="a.pdf"),
            _entry("Zenith Petrochemical reported earnings.", doc="b.pdf"),
        )
        # An impossibly high threshold suppresses fuzzy candidates.
        strict = resolve_entities(bb, config=ResolutionConfig(match_threshold=0.999))
        assert strict.clusters == []

    def test_report_schema(self):
        bb = _bb(
            _entry("Pinnacle Industrial Solutions Inc is the borrower.", doc="a.pdf"),
            _entry("Pinnacle Industrial Solutions provided collateral.", doc="b.pdf"),
        )
        report = run_entity_resolution(bb)
        assert report["schema_version"] == 2
        assert report["tokens_used"] == 0
        assert report["clusters_found"] >= 1
        assert "generic_terms_inferred" in report
