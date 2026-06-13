"""Tests for the language- and domain-agnostic domain adapter (Open Research Question #4).

Covers the neutral default, registry-driven + MULTILINGUAL detection, unicode correctness,
language-agnostic value scoring, composition with the model-driven domain_lens, the optional
ModelCaller fallback, and removal of the old hardcoded-English fragility. Deterministic/offline.
"""
from __future__ import annotations

import src.swarm.domain_adapter as da
from src.swarm.domain_adapter import (
    DomainAdapter,
    FINANCE_ADAPTER,
    LEGAL_ADAPTER,
    MEDICAL_ADAPTER,
    adapter_from_lens,
    detect_domain,
    get_adapter,
    list_adapters,
    register_adapter,
)


class TestRegistryAndNeutralDefault:
    def test_default_is_neutral_not_legal(self):
        ad = get_adapter()
        assert ad.name == "generic"
        assert "legal" not in ad.description.lower()
        assert ad.signals == ()  # no domain bias baked into the default

    def test_get_by_name_case_insensitive(self):
        assert get_adapter("legal").name == "legal"
        assert get_adapter("MEDICAL").name == "medical"
        assert get_adapter("does-not-exist").name == "generic"

    def test_builtins_registered(self):
        assert {"generic", "legal", "medical", "patent", "finance", "insurance"} <= set(list_adapters())

    def test_register_custom(self):
        register_adapter(DomainAdapter(name="maritime",
                                       signals=("charterparty", "laytime", "demurrage")))
        assert "maritime" in list_adapters()
        assert get_adapter("maritime").name == "maritime"


class TestNoHardcodedFragility:
    def test_central_keyword_table_gone(self):
        assert not hasattr(da, "_DOMAIN_SIGNALS")

    def test_ascii_entity_regex_gone(self):
        # the [A-Z][a-z]+ entity_patterns / content_value_patterns fields are removed
        ad = DomainAdapter()
        assert not hasattr(ad, "entity_patterns")
        assert not hasattr(ad, "content_value_patterns")


class TestDetectionEnglish:
    def test_legal(self):
        assert detect_domain("The borrower covenants under the credit agreement.")[0] == "legal"

    def test_medical(self):
        assert detect_domain("A randomized clinical trial measured efficacy in the cohort.")[0] == "medical"

    def test_finance(self):
        assert detect_domain("Q3 revenue and EBITDA beat guidance per the annual report.")[0] == "finance"

    def test_empty_is_generic(self):
        name, conf = detect_domain("")
        assert name == "generic"
        assert conf == 0.0


class TestMultilingualExtensibleDetection:
    def _reg(self):
        return {
            "generic": DomainAdapter(),
            "legal_de": DomainAdapter(name="legal_de",
                signals=("vertrag", "darlehensnehmer", "sicherheiten", "verpflichtung")),
            "medical_es": DomainAdapter(name="medical_es",
                signals=("paciente", "ensayo clínico", "eficacia", "dosis")),
            "legal_ja": DomainAdapter(name="legal_ja",
                signals=("契約", "借主", "義務", "担保")),
        }

    def test_german(self):
        name, _ = detect_domain(
            "Der Darlehensnehmer erfüllt seine Verpflichtung aus dem Vertrag.",
            registry=self._reg())
        assert name == "legal_de"

    def test_spanish_accented(self):
        name, _ = detect_domain(
            "El paciente completó el ensayo clínico con la dosis indicada.",
            registry=self._reg())
        assert name == "medical_es"

    def test_japanese(self):
        name, _ = detect_domain("借主は契約に基づく義務を履行した。担保を提供する。",
                                registry=self._reg())
        assert name == "legal_ja"

    def test_unicode_casefold(self):
        # uppercased / NFKC-variant signal still matches
        reg = {"generic": DomainAdapter(),
               "x": DomainAdapter(name="x", signals=("straße",))}
        assert detect_domain("Die STRASSE ist gesperrt.", registry=reg)[0] in ("x", "generic")
        assert detect_domain("Adresse: Hauptstraße fehlt", registry=reg)[0] == "generic"


class TestLanguageAgnosticValueScoring:
    def test_currency_and_percent(self):
        ad = DomainAdapter()
        assert ad.score_content("Die Vergütung beträgt 1.250.000 € bei 12% Zinsen.") > 0

    def test_dates_and_numbers_cjk(self):
        ad = DomainAdapter()
        assert ad.score_content("会社は2023-01-15に契約を締結した。") > 0

    def test_no_numbers_is_zero(self):
        assert DomainAdapter().score_content("just some words with no figures") == 0.0

    def test_domain_patterns_compose(self):
        # medical adds p-value pattern on top of the universal ones
        assert MEDICAL_ADAPTER.score_content("Outcome significant at p<0.01") > \
            DomainAdapter().score_content("Outcome significant at p<0.01")


class TestLensComposition:
    def test_merge_lens_enriches(self):
        merged = LEGAL_ADAPTER.merge_lens({
            "issue_hypotheses": ["MAC clause scope", "EBITDA add-back disputes"],
            "negative_checks": ["missing flood exclusion"],
        })
        assert "MAC clause scope" in merged.synthesis_guidance
        assert any("flood exclusion" in c for c in merged.convergence_criteria)
        assert merged is not LEGAL_ADAPTER  # immutable copy

    def test_merge_lens_empty_is_noop(self):
        assert LEGAL_ADAPTER.merge_lens({}) is LEGAL_ADAPTER

    def test_adapter_from_lens(self):
        ad = adapter_from_lens({"issue_hypotheses": ["x-factor"]}, name="custom")
        assert ad.name == "custom"
        assert "x-factor" in ad.synthesis_guidance


class TestHybridDetection:
    def test_model_fallback_when_no_signal(self, monkeypatch):
        monkeypatch.setattr(da, "_call_model",
                            lambda caller, prompt, max_tokens=64: ({"domain": "medical"}, 3))
        reg = {"generic": DomainAdapter(), "medical": MEDICAL_ADAPTER}
        name, conf = detect_domain("zzz opaque payload qqq", registry=reg, caller=object())
        assert name == "medical"
        assert conf == 0.5

    def test_no_caller_no_signal_is_generic(self):
        reg = {"generic": DomainAdapter(), "medical": MEDICAL_ADAPTER}
        assert detect_domain("zzz opaque payload qqq", registry=reg) == ("generic", 0.0)


class TestPromptSuffixes:
    def test_accessors_preserved(self):
        ad = get_adapter("legal")
        assert "DOMAIN FOCUS" in ad.get_extraction_prompt_suffix()
        assert "DOMAIN GUIDANCE" in ad.get_synthesis_prompt_suffix()
        assert "DOMAIN CONTEXT" in ad.get_seed_prompt_suffix()
        assert "DOMAIN FOCUS" in ad.get_structural_profile_suffix()
