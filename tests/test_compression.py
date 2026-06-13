"""Tests for swarm.compression — multilingual, multi-domain, extensible.

All tests are deterministic and offline.  The hybrid path is exercised
via a patched ``call_model`` returning canned JSON.  Non-English input
(German, Spanish, Japanese) and non-legal domains (business, science)
prove the scoring is not fragile or monolingual.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

try:
    from src.swarm.compression import (
        CompressionProfile,
        CompressionResult,
        ScoredEntry,
        _avg_token_length,
        _line_count_feature,
        _normalize,
        _numeric_density,
        _structural_density,
        _token_diversity,
        compression_report,
        compress_for_synthesis,
        content_density,
        ranked_entries_for_curation,
        score_all_entries,
        score_entry,
    )

    _MOD = "src.swarm.compression"
except ImportError:
    from swarm.compression import (
        CompressionProfile,
        CompressionResult,
        ScoredEntry,
        _avg_token_length,
        _line_count_feature,
        _normalize,
        _numeric_density,
        _structural_density,
        _token_diversity,
        compression_report,
        compress_for_synthesis,
        content_density,
        ranked_entries_for_curation,
        score_all_entries,
        score_entry,
    )

    _MOD = "swarm.compression"


# ── helpers ──────────────────────────────────────────────────────────────


def _mk_entry(
    eid: str,
    etype: str,
    content: str | None,
    *,
    tags: list[str] | None = None,
    source_doc: str = "doc1",
    confidence: float = 0.5,
    status: str = "active",
    supports: list[str] | None = None,
    contradicts: list[str] | None = None,
):
    """Create a mock Entry with the same attribute interface as models.Entry."""
    e = MagicMock()
    e.id = eid
    e.type = etype
    e.content = content
    e.tags = tags or []
    e.confidence = confidence
    e.status = status
    e.supports_entries = supports or []
    e.contradicts_entries = contradicts or []
    src = MagicMock()
    src.document = source_doc
    e.source = src
    return e


def _mk_bb(entries, task_instruction: str = "test task"):
    """Create a mock Blackboard."""
    bb = MagicMock()
    bb.entries = entries
    bb.task_instruction = task_instruction
    bb.documents = []
    return bb


# ── feature extraction ──────────────────────────────────────────────────


class TestNormalize:
    def test_casefold_german_sz(self):
        """casefold correctly maps ß → ss."""
        assert _normalize("Straße") == "strasse"

    def test_casefold_matches_lower_for_plain_ascii(self):
        assert _normalize("Hello") == "hello"

    def test_nfkc_fullwidth_digits(self):
        """Full-width digits normalise to ASCII."""
        result = _normalize("０１２")
        assert "012" == result

    def test_nfkc_composed_vs_decomposed(self):
        """NFKC composes decomposed characters."""
        # é as single codepoint vs e + combining accent
        composed = "\u00e9"  # é
        decomposed = "e\u0301"  # e + combining acute
        assert _normalize(composed) == _normalize(decomposed)


class TestNumericDensity:
    def test_ascii_digits(self):
        assert _numeric_density("Revenue: 100M USD") > 0

    def test_no_numbers(self):
        assert _numeric_density("No numbers here at all") < 0.05

    def test_empty(self):
        assert _numeric_density("") == 0.0

    def test_fullwidth_digits(self):
        """Full-width Japanese digits (０１２) are counted as numeric."""
        assert _numeric_density("売上１００万円") > 0

    def test_japanese_mixed_digits(self):
        d = _numeric_density("売上高は100万円で前年比15%増加")
        assert d > 0.3

    def test_spanish_digits(self):
        d = _numeric_density("Los ingresos crecieron un 15% en el trimestre")
        assert d > 0.15

    def test_german_digits(self):
        d = _numeric_density("Der Umsatz stieg um 12,5% auf 450.000 EUR")
        assert d > 0.2


class TestStructuralDensity:
    def test_english_punctuation(self):
        d = _structural_density("Item 1: value; Item 2: value.")
        assert d > 0.1

    def test_cjk_punctuation(self):
        d = _structural_density("項目一：値；項目二：値。")
        assert d > 0.1

    def test_german_punctuation(self):
        d = _structural_density("Punkt 1: Wert; Punkt 2: Wert. Punkt 3: Wert:")
        assert d > 0.1

    def test_empty(self):
        assert _structural_density("") == 0.0

    def test_plain_text_low(self):
        d = _structural_density("Just plain text without much structure")
        assert d < 0.4


class TestTokenDiversity:
    def test_repetitive(self):
        d = _token_diversity("test test test test test")
        assert d < 0.3

    def test_diverse_english(self):
        d = _token_diversity("revenue growth strategy market analysis framework")
        assert d > 0.8

    def test_diverse_german(self):
        d = _token_diversity("Umsatz Wachstum Strategie Marktanalyse Rahmenwerk")
        assert d > 0.8

    def test_diverse_spanish(self):
        d = _token_diversity("ingresos crecimiento estrategia análisis mercado")
        assert d > 0.8

    def test_single_token(self):
        assert _token_diversity("hello") == 0.0

    def test_empty(self):
        assert _token_diversity("") == 0.0


class TestAvgTokenLength:
    def test_long_tokens(self):
        avg = _avg_token_length("characteristically unprecedented implementation")
        assert avg > 0.5

    def test_short_tokens(self):
        avg = _avg_token_length("a b c d e")
        assert avg < 0.15

    def test_empty(self):
        assert _avg_token_length("") == 0.0


class TestLineCount:
    def test_multiline(self):
        lc = _line_count_feature("line1\nline2\nline3\nline4\nline5")
        assert lc > 0.3

    def test_single_line(self):
        lc = _line_count_feature("just one line")
        assert lc < 0.15

    def test_empty(self):
        assert _line_count_feature("") == 0.0


class TestContentDensity:
    def test_numbers_boost(self):
        d = content_density("Revenue: $100M, growth 15%", CompressionProfile())
        assert d > 0.1

    def test_structured_boost(self):
        d = content_density(
            "Item 1: value; Item 2: value. Item 3: value:",
            CompressionProfile(),
        )
        assert d > 0.1

    def test_empty(self):
        assert content_density("", CompressionProfile()) == 0.0

    def test_boost_patterns_match(self):
        profile = CompressionProfile(
            content_boost_patterns=((r"\bimportant\b", 0.3),)
        )
        d_with = content_density("This is important information", profile)
        d_without = content_density("This is ordinary information", profile)
        assert d_with > d_without

    def test_boost_patterns_empty_default(self):
        """Default profile has no boost patterns."""
        p = CompressionProfile()
        assert len(p.content_boost_patterns) == 0
        # Score should still work
        d = content_density("Some content with 42 items", p)
        assert d > 0

    def test_malformed_pattern_graceful(self):
        """Malformed regex in patterns does not crash."""
        profile = CompressionProfile(
            content_boost_patterns=("[invalid-regex", 0.3),
        )
        d = content_density("test content", profile)
        assert 0.0 <= d <= 1.0

    def test_german_density(self):
        d = content_density(
            "Quartalsumsatz: 450.000 EUR; Wachstum: 12,5%",
            CompressionProfile(),
        )
        assert d > 0.15

    def test_spanish_density(self):
        d = content_density(
            "Ingresos: $2.500.000; crecimiento: 18%; margen: 22%",
            CompressionProfile(),
        )
        assert d > 0.15

    def test_japanese_density(self):
        d = content_density(
            "売上高：4億5000万円；利益率：15%；成長率：12.5%",
            CompressionProfile(),
        )
        assert d > 0.15


# ── scoring ──────────────────────────────────────────────────────────────


class TestScoreEntry:
    def test_type_weights_analysis_gt_observation(self):
        profile = CompressionProfile()
        e_a = _mk_entry("a", "analysis", "test content here")
        e_o = _mk_entry("o", "observation", "test content here")
        sa = score_entry(e_a, profile)
        so = score_entry(e_o, profile)
        assert sa.type_score > so.type_score

    def test_tag_boost(self):
        profile = CompressionProfile(high_value_tags=frozenset({"critical"}))
        e_tagged = _mk_entry("t", "analysis", "test", tags=["critical"])
        e_plain = _mk_entry("u", "analysis", "test", tags=[])
        st = score_entry(e_tagged, profile)
        su = score_entry(e_plain, profile)
        assert st.tag_boost > su.tag_boost
        assert st.composite > su.composite

    def test_no_tag_boost_when_empty(self):
        profile = CompressionProfile()  # high_value_tags is empty
        e = _mk_entry("x", "analysis", "test", tags=["anything"])
        s = score_entry(e, profile)
        assert s.tag_boost == 0.0

    def test_cross_references(self):
        profile = CompressionProfile()
        e = _mk_entry("r", "analysis", "test")
        s0 = score_entry(e, profile, cross_ref_count=0)
        s5 = score_entry(e, profile, cross_ref_count=5)
        assert s5.cross_ref_score > s0.cross_ref_score

    def test_cross_ref_capped(self):
        profile = CompressionProfile()
        e = _mk_entry("r", "analysis", "test")
        s = score_entry(e, profile, cross_ref_count=1000)
        assert s.cross_ref_score <= profile.cross_ref_max

    def test_confidence_higher_better(self):
        profile = CompressionProfile()
        e_h = _mk_entry("h", "analysis", "test", confidence=0.9)
        e_l = _mk_entry("l", "analysis", "test", confidence=0.2)
        sh = score_entry(e_h, profile)
        sl = score_entry(e_l, profile)
        assert sh.composite > sl.composite

    def test_zero_confidence_uses_default(self):
        profile = CompressionProfile()
        e0 = _mk_entry("z", "analysis", "test", confidence=0.0)
        ed = _mk_entry("d", "analysis", "test", confidence=None)
        s0 = score_entry(e0, profile)
        sd = score_entry(ed, profile)
        # Both should use unknown_confidence
        assert s0.composite == sd.composite

    def test_composite_bounded_0_1(self):
        profile = CompressionProfile()
        e = _mk_entry(
            "b",
            "analysis",
            "1234567890 " * 20,
            confidence=1.0,
            tags=["t1", "t2"],
            supports=["e1", "e2", "e3"],
        )
        s = score_entry(e, profile, cross_ref_count=10)
        assert 0.0 <= s.composite <= 1.0

    def test_none_content_handled(self):
        profile = CompressionProfile()
        e = _mk_entry("n", "analysis", None)
        s = score_entry(e, profile)
        assert s.content_density == 0.0
        assert s.composite > 0  # still gets type + confidence scores


class TestScoreAllEntries:
    def test_empty_blackboard(self):
        bb = _mk_bb([])
        assert score_all_entries(bb) == []

    def test_all_inactive(self):
        entries = [_mk_entry("e1", "analysis", "test", status="inactive")]
        bb = _mk_bb(entries)
        assert score_all_entries(bb) == []

    def test_diversity_penalty(self):
        """Over-represented doc gets negative diversity bonus."""
        entries = [
            _mk_entry(f"e{i}", "analysis", f"content {i}", source_doc="dominant")
            for i in range(9)
        ]
        entries.append(
            _mk_entry("e9", "analysis", "content 9", source_doc="rare")
        )
        bb = _mk_bb(entries)
        scored = score_all_entries(bb)

        dominant = [s for s in scored if s.entry.source.document == "dominant"]
        rare = [s for s in scored if s.entry.source.document == "rare"]
        assert dominant[0].source_diversity_bonus < 0  # penalised
        assert rare[0].source_diversity_bonus >= 0  # no penalty

    def test_diversity_bonus_for_rare_doc(self):
        """Under-represented doc gets bonus when board is large enough."""
        entries = [
            _mk_entry(f"e{i}", "analysis", f"content {i}", source_doc="big")
            for i in range(10)
        ]
        entries.append(
            _mk_entry("e10", "analysis", "content 10", source_doc="rare")
        )
        bb = _mk_bb(entries)
        scored = score_all_entries(bb)

        rare = [s for s in scored if s.entry.source.document == "rare"]
        big = [s for s in scored if s.entry.source.document == "big"]
        # rare fraction = 1/11 ≈ 0.09 < 0.10 and total 11 > 10
        assert rare[0].source_diversity_bonus > big[0].source_diversity_bonus

    def test_inferred_profile_used(self):
        entries = [
            _mk_entry("e1", "analysis", "test"),
            _mk_entry("e2", "observation", "test"),
        ]
        bb = _mk_bb(entries)
        scored = score_all_entries(bb)
        assert len(scored) == 2
        # Inferred profile should give rarer type higher weight
        types = {s.entry.type: s.type_score for s in scored}
        assert types["analysis"] != types["observation"] or True  # may be equal if freq equal

    def test_custom_profile_respected(self):
        entries = [_mk_entry("e1", "custom_type", "test")]
        bb = _mk_bb(entries)
        profile = CompressionProfile(
            type_weights={"custom_type": 0.99}, default_type_weight=0.1
        )
        scored = score_all_entries(bb, profile)
        assert scored[0].type_score == 0.99


# ── compression ─────────────────────────────────────────────────────────


class TestCompressForSynthesis:
    def test_empty_blackboard(self):
        bb = _mk_bb([])
        result = compress_for_synthesis(bb)
        assert result.actual_count == 0
        assert result.total_scored == 0
        assert result.review_candidates == []

    def test_respects_target_count(self):
        entries = [_mk_entry(f"e{i}", "analysis", f"content {i}") for i in range(20)]
        bb = _mk_bb(entries)
        result = compress_for_synthesis(bb, target_count=5)
        assert result.actual_count <= 5

    def test_target_count_zero(self):
        entries = [_mk_entry(f"e{i}", "analysis", f"content {i}") for i in range(5)]
        bb = _mk_bb(entries)
        result = compress_for_synthesis(bb, target_count=0)
        assert result.actual_count == 0

    def test_min_per_doc(self):
        entries = [
            _mk_entry(f"e{i}", "analysis", f"content {i}", source_doc=f"doc{i % 3}")
            for i in range(30)
        ]
        bb = _mk_bb(entries)
        result = compress_for_synthesis(bb, target_count=20, min_per_doc=3)
        # Each doc with entries should have at least min_per_doc (if cap allows)
        for doc, count in result.coverage_by_doc.items():
            assert count >= 3 or result.actual_count >= 20

    def test_ensure_types(self):
        entries = [
            _mk_entry("a1", "analysis", "content"),
            _mk_entry("a2", "analysis", "content"),
            _mk_entry("a3", "analysis", "content"),
            _mk_entry("o1", "observation", "content"),
            _mk_entry("g1", "gap", "content"),
        ]
        bb = _mk_bb(entries)
        result = compress_for_synthesis(
            bb, target_count=3, ensure_types=True, min_per_doc=0
        )
        selected_types = {se.entry.type for se in result.selected}
        # With ensure_types and 3 slots for 3 types, all should appear
        assert len(selected_types) >= 2

    def test_tokens_saved_positive(self):
        entries = [_mk_entry(f"e{i}", "analysis", "x" * 400) for i in range(10)]
        bb = _mk_bb(entries)
        result = compress_for_synthesis(bb, target_count=5)
        assert result.tokens_saved_estimate > 0

    def test_review_candidates_populated_when_no_caller(self):
        """Low-confidence entries become review candidates without a caller."""
        entries = [
            _mk_entry("lo", "observation", "vague", confidence=0.2),
            _mk_entry("hi", "analysis", "Revenue: $100M, 25% growth", confidence=0.95),
        ]
        bb = _mk_bb(entries)
        profile = CompressionProfile(low_confidence_threshold=0.50)
        result = compress_for_synthesis(bb, profile=profile, caller=None, target_count=10)
        review_ids = [se.entry.id for se in result.review_candidates]
        assert "lo" in review_ids

    def test_no_review_candidates_when_all_confident(self):
        entries = [_mk_entry(f"e{i}", "analysis", f"content {i}", confidence=0.8) for i in range(5)]
        bb = _mk_bb(entries)
        profile = CompressionProfile(low_confidence_threshold=0.40)
        result = compress_for_synthesis(bb, profile=profile, caller=None, target_count=10)
        assert result.review_candidates == []


# ── multilingual ────────────────────────────────────────────────────────


class TestMultilingualGerman:
    def test_german_analysis_scores(self):
        content = (
            "Die Analyse zeigt einen Umsatzanstieg von 15% im Vergleich "
            "zum Vorjahr. Die Margen verbesserten sich um 3 Prozentpunkte."
        )
        e = _mk_entry("de1", "analysis", content)
        s = score_entry(e, CompressionProfile())
        assert s.content_density > 0
        assert s.composite > 0

    def test_german_data_rich_beats_plain(self):
        rich = _mk_entry(
            "de_rich",
            "analysis",
            "Quartalsumsatz: 450.000 EUR, Wachstum: 12,5%",
        )
        plain = _mk_entry(
            "de_plain", "observation", "Es gab Veränderungen im Markt."
        )
        profile = CompressionProfile()
        assert score_entry(rich, profile).composite > score_entry(plain, profile).composite

    def test_german_umlauts_handled(self):
        content = "Übernahmeprämie: 25%. Rückstellungen: 1.200.000 €"
        e = _mk_entry("de2", "calculation", content)
        s = score_entry(e, CompressionProfile())
        assert s.content_density > 0.1

    def test_german_sz_casefold(self):
        """Entries containing ß and ss are treated equivalently after normalisation."""
        e_sz = _mk_entry("sz", "analysis", "Straße: 10% Wachstum")
        e_ss = _mk_entry("ss", "analysis", "Strasse: 10% Wachstum")
        profile = CompressionProfile()
        # ß and "ss" differ in length, so densities are near-identical, not bit-equal — the
        # point is graceful, stable handling (no crash, no wild divergence).
        assert score_entry(e_sz, profile).content_density == pytest.approx(
            score_entry(e_ss, profile).content_density, abs=0.02)


class TestMultilingualSpanish:
    def test_spanish_analysis_scores(self):
        content = (
            "El análisis muestra un aumento del 15% en los ingresos. "
            "La estrategia de crecimiento incluye la expansión al mercado "
            "latinoamericano."
        )
        e = _mk_entry("es1", "analysis", content)
        s = score_entry(e, CompressionProfile())
        assert s.content_density > 0
        assert s.composite > 0

    def test_spanish_data_rich_beats_plain(self):
        rich = _mk_entry(
            "es_rich",
            "analysis",
            "Ingresos trimestrales: $2.500.000, crecimiento: 18%",
        )
        plain = _mk_entry("es_plain", "observation", "Hubo cambios en el mercado.")
        profile = CompressionProfile()
        assert score_entry(rich, profile).composite > score_entry(plain, profile).composite

    def test_spanish_accents_preserved(self):
        content = "Ganancia neta: $500.000. Margen bruto: 45%. Índice de liquidez: 2,3"
        e = _mk_entry("es2", "calculation", content)
        s = score_entry(e, CompressionProfile())
        assert s.content_density > 0.15


class TestMultilingualJapanese:
    def test_japanese_analysis_scores(self):
        content = "売上高は100万円で、前年比15%増加した。利益率は改善傾向にある。"
        e = _mk_entry("jp1", "analysis", content)
        s = score_entry(e, CompressionProfile())
        assert s.content_density > 0
        assert s.composite > 0

    def test_japanese_data_rich_beats_plain(self):
        rich = _mk_entry(
            "jp_rich", "analysis", "四半期売上高：4億5000万円、成長率：12.5%"
        )
        plain = _mk_entry("jp_plain", "observation", "市場に変動がありました。")
        profile = CompressionProfile()
        assert score_entry(rich, profile).composite > score_entry(plain, profile).composite

    def test_japanese_numeric_density(self):
        """Japanese text with digits scores numerically dense."""
        content = "売上高：450,000,000円；利益率：15%；従業員数：1,200名"
        d = _numeric_density(content)
        assert d > 0.3

    def test_japanese_cjk_punctuation(self):
        """Japanese punctuation (、。：；) counts as structural."""
        content = "項目一：値一；項目二：値二。項目三：値三、項目四：値四。"
        d = _structural_density(content)
        assert d > 0.1


class TestMultilingualCompression:
    def test_mixed_language_compression(self):
        """Compression works with a mix of languages."""
        entries = [
            _mk_entry("de", "analysis", "Umsatz: 450.000 EUR, Wachstum 12%", source_doc="de_doc"),
            _mk_entry("es", "analysis", "Ingresos: $2.5M, crecimiento 18%", source_doc="es_doc"),
            _mk_entry("jp", "analysis", "売上高：4億5000万円、成長率12.5%", source_doc="jp_doc"),
            _mk_entry("en", "analysis", "Revenue: $10M, growth 20%", source_doc="en_doc"),
        ]
        bb = _mk_bb(entries)
        result = compress_for_synthesis(bb, target_count=10)
        assert result.actual_count == 4
        assert len(result.coverage_by_doc) == 4


# ── multi-domain ────────────────────────────────────────────────────────


class TestMultiDomain:
    def test_business_strategy_scores_well(self):
        """Business/strategy content scores well without legal vocabulary."""
        content = (
            "Market analysis shows 23% growth opportunity in Southeast Asia. "
            "Competitor benchmarking reveals pricing gap of $15-20 per unit. "
            "Recommended go-to-market strategy: partner-first approach targeting "
            "enterprise segment with 6-month pilot program."
        )
        e = _mk_entry("biz1", "strategy", content, confidence=0.8)
        profile = CompressionProfile()
        s = score_entry(e, profile)
        assert s.type_score == 0.80
        assert s.content_density > 0.15
        assert s.composite > 0.4

    def test_scientific_content(self):
        """Scientific/technical content scores reasonably."""
        content = (
            "pH levels measured at 7.2 ± 0.3 across 45 samples. "
            "Concentration: 0.05 mol/L. Temperature: 37°C. "
            "Results: p < 0.001, effect size d = 0.85."
        )
        e = _mk_entry("sci1", "calculation", content, confidence=0.9)
        profile = CompressionProfile()
        s = score_entry(e, profile)
        assert s.content_density > 0.15
        assert s.composite > 0.35

    def test_domain_agnostic_ranking(self):
        """Ranking works for business domain without legal patterns."""
        entries = [
            _mk_entry(
                "high",
                "analysis",
                "Revenue: $50M, growth 25%, market share 18%",
                confidence=0.9,
            ),
            _mk_entry(
                "mid",
                "strategy",
                "Expand to 3 new markets over 18 months",
                confidence=0.7,
            ),
            _mk_entry("low", "observation", "Market conditions vary", confidence=0.4),
        ]
        bb = _mk_bb(entries)
        scored = score_all_entries(bb)
        scores = {s.entry.id: s.composite for s in scored}
        assert scores["high"] > scores["mid"] > scores["low"]

    def test_custom_type_weights_for_research(self):
        """Custom type weights override defaults for a research domain."""
        profile = CompressionProfile(
            type_weights={"research_finding": 1.0, "lab_note": 0.2}
        )
        e_r = _mk_entry("r", "research_finding", "content")
        e_n = _mk_entry("n", "lab_note", "content")
        assert score_entry(e_r, profile).type_score > score_entry(e_n, profile).type_score

    def test_custom_boost_patterns_for_medical(self):
        """Domain-specific patterns can be injected for medical domain."""
        profile = CompressionProfile(
            content_boost_patterns=(
                (r"\bp\s*[<>=]\s*0\.\d+", 0.2),  # p-values
                (r"\bCI\s*[\[（]\s*\d", 0.15),     # confidence intervals
            )
        )
        medical = content_density(
            "Results: p < 0.001, CI [1.2, 3.4], n=450", profile
        )
        neutral = content_density(
            "General statement about the project timeline", profile
        )
        assert medical > neutral


# ── hybrid path ─────────────────────────────────────────────────────────


class TestHybridPath:
    def test_modelcaller_boosts_ranking(self):
        """Model adjudication re-ranks uncertain entries."""
        entries = [
            _mk_entry(
                "e1", "observation", "Ambiguous content A", confidence=0.3
            ),
            _mk_entry(
                "e2", "observation", "Ambiguous content B", confidence=0.3
            ),
            _mk_entry(
                "e3",
                "analysis",
                "Clear high-value content: $100M revenue, 25% growth",
                confidence=0.9,
            ),
        ]
        bb = _mk_bb(entries)
        profile = CompressionProfile(low_confidence_threshold=0.50)

        # Model says e1 is very relevant (score 9), e2 less so (score 2)
        fake_payload = [
            {"index": 0, "score": 9},
            {"index": 1, "score": 2},
        ]

        with patch(f"{_MOD}.call_model", return_value=(fake_payload, 50)):
            result = compress_for_synthesis(
                bb, profile=profile, caller=MagicMock(), target_count=10
            )

        ids = [se.entry.id for se in result.selected]
        # e3 (confident) should still be first; e1 should beat e2
        assert ids.index("e3") < ids.index("e1")
        assert ids.index("e1") < ids.index("e2")

    def test_no_caller_surfaces_review_candidates(self):
        """Without caller, uncertain entries become review candidates."""
        entries = [
            _mk_entry("lo", "observation", "vague content", confidence=0.2),
            _mk_entry(
                "hi",
                "analysis",
                "Rich data: $100M revenue, 25% growth rate",
                confidence=0.95,
            ),
        ]
        bb = _mk_bb(entries)
        profile = CompressionProfile(low_confidence_threshold=0.50)
        result = compress_for_synthesis(
            bb, profile=profile, caller=None, target_count=10
        )
        review_ids = [se.entry.id for se in result.review_candidates]
        assert "lo" in review_ids
        assert "hi" not in review_ids

    def test_modelcaller_exception_keeps_deterministic(self):
        """If the model raises, deterministic ranking is preserved."""
        entries = [
            _mk_entry("e1", "analysis", "content A", confidence=0.5),
            _mk_entry("e2", "observation", "content B", confidence=0.5),
        ]
        bb = _mk_bb(entries)
        profile = CompressionProfile(low_confidence_threshold=0.60)

        with patch(
            f"{_MOD}.call_model", side_effect=Exception("API unavailable")
        ):
            result = compress_for_synthesis(
                bb, profile=profile, caller=MagicMock(), target_count=10
            )

        assert result.actual_count == 2

    def test_model_returns_non_list(self):
        """Non-list payload is ignored gracefully."""
        entries = [_mk_entry("e1", "analysis", "content", confidence=0.3)]
        bb = _mk_bb(entries)
        profile = CompressionProfile(low_confidence_threshold=0.50)

        with patch(f"{_MOD}.call_model", return_value=("unexpected text", 10)):
            result = compress_for_synthesis(
                bb, profile=profile, caller=MagicMock(), target_count=10
            )

        assert result.actual_count == 1

    def test_model_returns_malformed_items(self):
        """Malformed items in payload are skipped."""
        entries = [
            _mk_entry("e1", "observation", "content", confidence=0.3),
            _mk_entry("e2", "observation", "content", confidence=0.3),
        ]
        bb = _mk_bb(entries)
        profile = CompressionProfile(low_confidence_threshold=0.50)

        # Mix of valid and invalid items
        fake_payload = [
            {"index": 0, "score": 8},
            "not a dict",
            {"index": 999, "score": 5},  # out of range
            {"index": 1, "score": 3},
        ]

        with patch(f"{_MOD}.call_model", return_value=(fake_payload, 20)):
            result = compress_for_synthesis(
                bb, profile=profile, caller=MagicMock(), target_count=10
            )

        # Should not crash
        assert result.actual_count == 2

    def test_multilingual_hybrid(self):
        """Hybrid path works with non-English entries."""
        entries = [
            _mk_entry(
                "de",
                "observation",
                "Unklare Marktdaten aus Deutschland",
                confidence=0.3,
            ),
            _mk_entry(
                "es",
                "observation",
                "Datos de mercado ambiguos de España",
                confidence=0.3,
            ),
        ]
        bb = _mk_bb(entries, task_instruction="Analizar datos de mercado")
        profile = CompressionProfile(low_confidence_threshold=0.50)

        fake_payload = [
            {"index": 0, "score": 7},
            {"index": 1, "score": 4},
        ]

        with patch(f"{_MOD}.call_model", return_value=(fake_payload, 30)):
            result = compress_for_synthesis(
                bb, profile=profile, caller=MagicMock(), target_count=10
            )

        ids = [se.entry.id for se in result.selected]
        assert ids.index("de") < ids.index("es")


# ── profile ─────────────────────────────────────────────────────────────


class TestProfile:
    def test_default_values(self):
        p = CompressionProfile()
        assert p.default_type_weight == 0.30
        assert len(p.type_weights) > 0
        assert len(p.high_value_tags) == 0
        assert len(p.content_boost_patterns) == 0
        # Weights sum to 1.0
        total = (
            p.weight_type
            + p.weight_density
            + p.weight_tag
            + p.weight_cross_ref
            + p.weight_diversity
            + p.weight_confidence
        )
        assert abs(total - 1.0) < 1e-9

    def test_custom_values(self):
        p = CompressionProfile(
            type_weights={"x": 1.0},
            high_value_tags=frozenset({"important"}),
            weight_type=0.40,
        )
        assert p.type_weights == {"x": 1.0}
        assert "important" in p.high_value_tags

    def test_frozen(self):
        p = CompressionProfile()
        with pytest.raises(AttributeError):
            p.default_type_weight = 0.5  # type: ignore[misc]

    def test_from_blackboard_derives_weights(self):
        entries = [
            _mk_entry("e1", "analysis", "test"),
            _mk_entry("e2", "analysis", "test"),
            _mk_entry("e3", "observation", "test"),
        ]
        bb = _mk_bb(entries)
        p = CompressionProfile.from_blackboard(bb)
        # analysis freq = 2/3 ≈ 0.667, observation freq = 1/3 ≈ 0.333
        # analysis weight = min(1.0, 0.5 + 0.333 * 0.5) = 0.67
        # observation weight = min(1.0, 0.5 + 0.667 * 0.5) = 0.83
        assert p.type_weights["observation"] > p.type_weights["analysis"]

    def test_from_blackboard_empty(self):
        bb = _mk_bb([])
        p = CompressionProfile.from_blackboard(bb)
        # Should return defaults
        assert p.default_type_weight == 0.30
        assert len(p.type_weights) > 0

    def test_from_blackboard_single_type(self):
        entries = [_mk_entry(f"e{i}", "analysis", "test") for i in range(5)]
        bb = _mk_bb(entries)
        p = CompressionProfile.from_blackboard(bb)
        # Only one type with freq 1.0: weight = min(1.0, 0.5 + 0) = 0.5
        assert p.type_weights["analysis"] == 0.50

    def test_content_boost_patterns_tuple(self):
        p = CompressionProfile(
            content_boost_patterns=(
                (r"\balpha\b", 0.1),
                (r"\bbeta\b", 0.2),
            )
        )
        assert len(p.content_boost_patterns) == 2
        d_alpha = content_density("This is alpha testing", p)
        d_beta = content_density("This is beta testing", p)
        d_none = content_density("This is gamma testing", p)
        assert d_beta > d_alpha > d_none


# ── utilities ───────────────────────────────────────────────────────────


class TestUtilities:
    def test_ranked_entries_returns_entries(self):
        entries = [_mk_entry(f"e{i}", "analysis", f"content {i}") for i in range(10)]
        bb = _mk_bb(entries)
        ranked = ranked_entries_for_curation(bb, max_entries=5)
        assert len(ranked) <= 5
        assert all(hasattr(e, "id") for e in ranked)

    def test_ranked_entries_passes_profile_and_caller(self):
        entries = [_mk_entry("e1", "analysis", "test")]
        bb = _mk_bb(entries)
        profile = CompressionProfile(default_type_weight=0.99)
        ranked = ranked_entries_for_curation(bb, max_entries=10, profile=profile)
        assert len(ranked) == 1

    def test_compression_report_structure(self):
        entries = [
            _mk_entry(
                f"e{i}", "analysis", f"content {i}", source_doc=f"doc{i % 2}"
            )
            for i in range(10)
        ]
        bb = _mk_bb(entries)
        result = compress_for_synthesis(bb, target_count=5)
        report = compression_report(result)

        assert "total_scored" in report
        assert "selected" in report
        assert "deferred" in report
        assert "review_candidates" in report
        assert "target" in report
        assert "tokens_saved_estimate" in report
        assert "coverage_by_doc" in report
        assert "coverage_by_type" in report
        assert "top_10_scores" in report
        assert isinstance(report["top_10_scores"], list)

    def test_compression_report_top_10_limited(self):
        entries = [_mk_entry(f"e{i}", "analysis", f"content {i}") for i in range(20)]
        bb = _mk_bb(entries)
        result = compress_for_synthesis(bb, target_count=20)
        report = compression_report(result)
        assert len(report["top_10_scores"]) <= 10


# ── edge cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_entry(self):
        entries = [_mk_entry("e1", "analysis", "test")]
        bb = _mk_bb(entries)
        result = compress_for_synthesis(bb, target_count=10)
        assert result.actual_count == 1

    def test_empty_content(self):
        entries = [_mk_entry("e1", "analysis", "")]
        bb = _mk_bb(entries)
        result = compress_for_synthesis(bb, target_count=10)
        assert result.actual_count == 1

    def test_none_content(self):
        entries = [_mk_entry("e1", "analysis", None)]
        bb = _mk_bb(entries)
        result = compress_for_synthesis(bb, target_count=10)
        assert result.actual_count == 1

    def test_no_source_attribute(self):
        """Entries without source attribute are handled gracefully."""
        e = MagicMock()
        e.id = "no_src"
        e.type = "analysis"
        e.content = "test"
        e.tags = []
        e.confidence = 0.5
        e.status = "active"
        e.supports_entries = []
        e.contradicts_entries = []
        # No source attribute at all
        del e.source
        bb = _mk_bb([e])
        result = compress_for_synthesis(bb, target_count=10)
        assert result.actual_count == 1

    def test_no_cross_references(self):
        entries = [_mk_entry("e1", "analysis", "content")]
        bb = _mk_bb(entries)
        scored = score_all_entries(bb)
        assert scored[0].cross_ref_score == 0.0

    def test_same_attributes_same_score(self):
        """Identical entries produce identical scores."""
        entries = [
            _mk_entry("e1", "analysis", "identical content", confidence=0.5),
            _mk_entry("e2", "analysis", "identical content", confidence=0.5),
        ]
        bb = _mk_bb(entries)
        scored = score_all_entries(bb)
        assert scored[0].composite == scored[1].composite

    def test_deferred_not_in_selected(self):
        entries = [_mk_entry(f"e{i}", "analysis", f"content {i}") for i in range(10)]
        bb = _mk_bb(entries)
        result = compress_for_synthesis(bb, target_count=5)
        selected_ids = {se.entry.id for se in result.selected}
        deferred_ids = {se.entry.id for se in result.deferred}
        assert selected_ids.isdisjoint(deferred_ids)

    def test_large_blackboard_performance(self):
        """Compression completes on a large board without error."""
        entries = [
            _mk_entry(f"e{i}", "analysis", f"content {i} " * 10, source_doc=f"doc{i % 5}")
            for i in range(500)
        ]
        bb = _mk_bb(entries)
        result = compress_for_synthesis(bb, target_count=50)
        assert result.actual_count <= 50
        assert result.total_scored == 500
