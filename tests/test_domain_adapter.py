"""Tests for domain adapter pattern (Open Research Question #4)."""
from __future__ import annotations

from src.swarm.domain_adapter import (
    DomainAdapter,
    FINANCE_ADAPTER,
    INSURANCE_ADAPTER,
    MEDICAL_ADAPTER,
    PATENT_ADAPTER,
    detect_domain,
    get_adapter,
    list_adapters,
    register_adapter,
)


# -----------------------------------------------------------------------
# Default adapter
# -----------------------------------------------------------------------

class TestDefaultAdapter:
    def test_default_is_legal(self):
        adapter = get_adapter()
        assert adapter.name == "legal"

    def test_default_has_extraction_focus(self):
        adapter = get_adapter()
        assert "dollar amount" in adapter.extraction_focus.lower()

    def test_default_has_type_weights(self):
        adapter = get_adapter()
        assert "analysis" in adapter.type_weights
        assert adapter.type_weights["analysis"] == 1.0

    def test_default_has_entity_patterns(self):
        adapter = get_adapter()
        assert len(adapter.entity_patterns) > 0

    def test_default_has_content_value_patterns(self):
        adapter = get_adapter()
        assert len(adapter.content_value_patterns) > 0

    def test_default_has_synthesis_guidance(self):
        adapter = get_adapter()
        assert len(adapter.synthesis_guidance) > 10

    def test_default_has_convergence_criteria(self):
        adapter = get_adapter()
        assert len(adapter.convergence_criteria) > 0

    def test_prompt_suffixes_not_empty(self):
        adapter = get_adapter()
        assert len(adapter.get_extraction_prompt_suffix()) > 10
        assert len(adapter.get_synthesis_prompt_suffix()) > 10
        assert len(adapter.get_seed_prompt_suffix()) > 10
        assert len(adapter.get_structural_profile_suffix()) > 10


# -----------------------------------------------------------------------
# Built-in adapters
# -----------------------------------------------------------------------

class TestMedicalAdapter:
    def test_name(self):
        assert MEDICAL_ADAPTER.name == "medical"

    def test_extraction_focus_mentions_patients(self):
        assert "patient" in MEDICAL_ADAPTER.extraction_focus.lower()

    def test_extraction_focus_mentions_statistics(self):
        assert "p-value" in MEDICAL_ADAPTER.extraction_focus.lower()

    def test_content_patterns_include_p_values(self):
        patterns = [p for p, _ in MEDICAL_ADAPTER.content_value_patterns]
        assert any("p" in p.lower() for p in patterns)

    def test_synthesis_mentions_prisma(self):
        assert "prisma" in MEDICAL_ADAPTER.synthesis_guidance.lower()

    def test_convergence_mentions_studies(self):
        criteria_text = " ".join(MEDICAL_ADAPTER.convergence_criteria).lower()
        assert "study" in criteria_text or "studies" in criteria_text


class TestPatentAdapter:
    def test_name(self):
        assert PATENT_ADAPTER.name == "patent"

    def test_extraction_focus_mentions_claims(self):
        assert "claim" in PATENT_ADAPTER.extraction_focus.lower()

    def test_entity_patterns_include_patent_numbers(self):
        patterns = PATENT_ADAPTER.entity_patterns
        assert any("US" in p or "EP" in p for p in patterns)

    def test_synthesis_mentions_prior_art(self):
        assert "prior art" in PATENT_ADAPTER.synthesis_guidance.lower()


class TestFinanceAdapter:
    def test_name(self):
        assert FINANCE_ADAPTER.name == "finance"

    def test_extraction_focus_mentions_revenue(self):
        assert "revenue" in FINANCE_ADAPTER.extraction_focus.lower()

    def test_content_patterns_include_dollar_amounts(self):
        patterns = [p for p, _ in FINANCE_ADAPTER.content_value_patterns]
        assert any("$" in p or "\\$" in p for p in patterns)

    def test_type_weights_calculation_high(self):
        assert FINANCE_ADAPTER.type_weights["calculation"] >= 0.9

    def test_synthesis_mentions_ebitda(self):
        assert "ebitda" in FINANCE_ADAPTER.synthesis_guidance.lower()


class TestInsuranceAdapter:
    def test_name(self):
        assert INSURANCE_ADAPTER.name == "insurance"

    def test_extraction_focus_mentions_coverage(self):
        assert "coverage" in INSURANCE_ADAPTER.extraction_focus.lower()

    def test_extraction_focus_mentions_deductible(self):
        assert "deductible" in INSURANCE_ADAPTER.extraction_focus.lower()

    def test_convergence_mentions_exclusions(self):
        criteria_text = " ".join(INSURANCE_ADAPTER.convergence_criteria).lower()
        assert "exclusion" in criteria_text


# -----------------------------------------------------------------------
# Adapter registry
# -----------------------------------------------------------------------

class TestAdapterRegistry:
    def test_list_adapters(self):
        adapters = list_adapters()
        assert "legal" in adapters
        assert "medical" in adapters
        assert "patent" in adapters
        assert "finance" in adapters
        assert "insurance" in adapters

    def test_get_adapter_by_name(self):
        assert get_adapter("medical").name == "medical"
        assert get_adapter("patent").name == "patent"
        assert get_adapter("finance").name == "finance"
        assert get_adapter("insurance").name == "insurance"

    def test_get_unknown_returns_legal(self):
        adapter = get_adapter("quantum_physics")
        assert adapter.name == "legal"

    def test_case_insensitive(self):
        assert get_adapter("MEDICAL").name == "medical"
        assert get_adapter("Medical").name == "medical"

    def test_register_custom_adapter(self):
        custom = DomainAdapter(
            name="custom_test",
            description="Test domain",
            extraction_focus="Extract test things.",
        )
        register_adapter(custom)
        assert get_adapter("custom_test").name == "custom_test"
        assert "custom_test" in list_adapters()


# -----------------------------------------------------------------------
# Auto-detection
# -----------------------------------------------------------------------

class TestDetectDomain:
    def test_detect_medical(self):
        text = (
            "Analyze this clinical trial report. Extract patient demographics, "
            "efficacy endpoints, adverse events, and statistical results "
            "including p-values and confidence intervals."
        )
        domain, confidence = detect_domain(text)
        assert domain == "medical"
        assert confidence > 0.3

    def test_detect_patent(self):
        text = (
            "Review this patent application. Analyze the claims for "
            "validity in view of the cited prior art references. "
            "Identify potential obviousness issues."
        )
        domain, confidence = detect_domain(text)
        assert domain == "patent"
        assert confidence > 0.3

    def test_detect_finance(self):
        text = (
            "Prepare an analysis of Datadog's 10-K filing. Extract "
            "revenue by segment, EBITDA margins, risk factors, and "
            "management guidance for the next fiscal year."
        )
        domain, confidence = detect_domain(text)
        assert domain == "finance"
        assert confidence > 0.3

    def test_detect_insurance(self):
        text = (
            "Review this commercial property insurance policy. Extract "
            "coverage limits, deductibles, exclusions, and endorsements. "
            "Identify gaps between requested and offered coverage."
        )
        domain, confidence = detect_domain(text)
        assert domain == "insurance"
        assert confidence > 0.3

    def test_detect_legal_default(self):
        text = "Compare the merger agreement to the term sheet."
        domain, confidence = detect_domain(text)
        assert domain == "legal"

    def test_detect_empty_text(self):
        domain, confidence = detect_domain("")
        assert domain == "legal"
        assert confidence == 0.0

    def test_detect_ambiguous_returns_highest(self):
        text = (
            "This patient's insurance policy was reviewed in the clinical "
            "trial. The coverage limits and adverse events were documented."
        )
        domain, confidence = detect_domain(text)
        # Should pick whichever has more signal words
        assert domain in ("medical", "insurance")


# -----------------------------------------------------------------------
# Custom adapter
# -----------------------------------------------------------------------

class TestCustomAdapter:
    def test_custom_adapter_fields(self):
        adapter = DomainAdapter(
            name="aerospace",
            description="Aerospace engineering document analysis",
            extraction_focus="Extract thrust, specific impulse, mass ratios.",
            type_weights={"analysis": 1.0, "observation": 0.6},
            entity_patterns=[r"\b(?:TRL)\s*\d\b"],
            content_value_patterns=[(r"\b\d+\s*kN\b", 0.15)],
            synthesis_guidance="Structure by subsystem.",
            convergence_criteria=["All subsystems analyzed"],
            seed_framework_hint="Aerospace systems analysis.",
            structural_profile_hint="Count subsystems and specifications.",
        )
        assert adapter.name == "aerospace"
        assert "thrust" in adapter.extraction_prompt_suffix()
        assert "subsystem" in adapter.get_synthesis_prompt_suffix()
        assert "TRL" in adapter.entity_patterns[0]
