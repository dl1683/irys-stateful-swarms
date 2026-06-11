"""Pluggable domain adapter for multi-benchmark generalization.

Addresses Open Research Question #4: "What breaks when you run the system
on medical research papers, patent filings, or insurance underwriting
documents? Where does the architecture need domain-specific adaptation
vs. where does it generalize cleanly?"

Current state: the system uses legal-specific prompts, entry types, and
scoring. This module defines a DomainAdapter interface that allows
domain-specific customization of:

1. Extraction prompts (what to look for in documents)
2. Entry type weights (what matters most in this domain)
3. Entity patterns (domain-specific entity recognition)
4. Synthesis guidance (how to structure the deliverable)
5. Convergence criteria (what "complete" means for this domain)

The default adapter is legal (preserving current behavior). Built-in
adapters for medical, patent, finance, and insurance domains demonstrate
the pattern.

Zero LLM calls in the adapter itself. The adapters configure the prompts
that the existing LLM-calling code uses.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Domain Adapter interface
# ---------------------------------------------------------------------------

@dataclass
class DomainAdapter:
    """Configuration for domain-specific swarm behavior.

    All fields have sensible defaults that work for legal analysis.
    Override fields for other domains.
    """
    # Identity
    name: str = "legal"
    description: str = "Legal document analysis"

    # Extraction: what workers should look for
    extraction_focus: str = (
        "Extract EVERY dollar amount, percentage, date, deadline, party name "
        "with full legal entity designation, numbered/lettered items, defined "
        "terms, obligations, conditions, restrictions, representations, "
        "warranties, and payment terms."
    )

    # Entry type weights for scoring (higher = more valuable for synthesis)
    type_weights: dict[str, float] = field(default_factory=lambda: {
        "analysis": 1.0,
        "calculation": 0.95,
        "strategy": 0.80,
        "observation": 0.50,
        "gap": 0.60,
        "contradiction": 0.85,
    })

    # Entity patterns: regex patterns for domain-specific entities
    entity_patterns: list[str] = field(default_factory=lambda: [
        # Legal entity suffixes
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:\s+(?:Inc|LLC|Ltd|Corp|Corporation|Company|Holdings|Group|Partners|Associates|Bank|Capital))?)\b",
        # Legal section references
        r"(?:Section|Article|Clause|Exhibit|Schedule)\s+[\d.]+(?:\([a-z]\))?",
    ])

    # Content value patterns: regex + weight for scoring
    content_value_patterns: list[tuple[str, float]] = field(default_factory=lambda: [
        (r"\$\s*[\d,]+(?:\.\d+)?", 0.15),           # dollar amounts
        (r"\b\d+(?:\.\d+)?%", 0.10),                 # percentages
        (r"(?:Section|Article|Clause)\s+\d", 0.10),   # legal refs
        (r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", 0.05),      # dates
        (r"\b(?:shall|must|may not|prohibited)\b", 0.08),  # obligations
    ])

    # Synthesis: how to structure the deliverable
    synthesis_guidance: str = (
        "Structure with clear headings and professional legal formatting. "
        "Include EVERY specific number, date, dollar amount, percentage, "
        "party name, deadline, obligation, restriction, representation, "
        "and warranty. Show calculations with full arithmetic steps. "
        "Cite source documents when referencing facts."
    )

    # Convergence: what "complete" means for this domain
    convergence_criteria: list[str] = field(default_factory=lambda: [
        "All key questions have source-grounded answers",
        "All numbered items from source documents have been extracted",
        "Cross-document comparisons have been performed",
        "Calculations have been verified",
        "Gaps and uncertainties have been identified",
    ])

    # Seed planning: domain-specific analytical framework hints
    seed_framework_hint: str = (
        "This is a legal document analysis task. The analytical framework "
        "should identify the type of legal work (extraction, comparison, "
        "drafting, issue-flagging, calculation) and plan accordingly."
    )

    # Structural profiling: what to count in documents
    structural_profile_hint: str = (
        "Count individually numbered/lettered items (clauses, requests, "
        "conditions), data tables, and major sections."
    )

    def get_extraction_prompt_suffix(self) -> str:
        """Extra extraction instructions for this domain."""
        return f"\nDOMAIN FOCUS: {self.extraction_focus}"

    def get_synthesis_prompt_suffix(self) -> str:
        """Extra synthesis instructions for this domain."""
        return f"\nDOMAIN GUIDANCE: {self.synthesis_guidance}"

    def get_seed_prompt_suffix(self) -> str:
        """Extra seed planning instructions for this domain."""
        return f"\nDOMAIN CONTEXT: {self.seed_framework_hint}"

    def get_structural_profile_suffix(self) -> str:
        """Extra structural profiling instructions."""
        return f"\nDOMAIN FOCUS: {self.structural_profile_hint}"


# ---------------------------------------------------------------------------
# Built-in domain adapters
# ---------------------------------------------------------------------------

MEDICAL_ADAPTER = DomainAdapter(
    name="medical",
    description="Medical research and clinical document analysis",
    extraction_focus=(
        "Extract EVERY patient population characteristic (N, demographics, "
        "inclusion/exclusion criteria), intervention details (drug, dose, "
        "route, frequency, duration), outcome measures (primary, secondary, "
        "safety endpoints), statistical results (p-values, confidence intervals, "
        "effect sizes, hazard ratios), adverse events (type, incidence, severity, "
        "causality assessment), and study design elements (randomization, "
        "blinding, follow-up duration, dropout rates)."
    ),
    type_weights={
        "analysis": 1.0,
        "calculation": 0.95,
        "strategy": 0.70,
        "observation": 0.55,
        "gap": 0.65,
        "contradiction": 0.90,
    },
    entity_patterns=[
        r"\b(?:Study|Trial|Protocol)\s+(?:No\.?\s*)?[\w-]+",
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+(?:et al|trial|study|registry)",
        r"\b(?:NCT|ISRCTN|EudraCT)\d+",
        r"\b(?:ICD|CPT|SNOMED|MedDRA)\s*[-:]?\s*[\w.]+",
    ],
    content_value_patterns=[
        (r"\b\d+(?:\.\d+)?%", 0.12),
        (r"\bp\s*[<>=]\s*[\d.]+", 0.15),
        (r"\b(?:HR|OR|RR)\s*[:=]\s*[\d.]+", 0.15),
        (r"\b(?:95%?\s*CI|CI)\s*[:=]?\s*[\d.]+\s*[-–]\s*[\d.]+", 0.12),
        (r"\bN\s*=\s*\d+", 0.10),
        (r"\b\d+\s*(?:mg|mcg|mL|units|days?|weeks?|months?)\b", 0.08),
    ],
    synthesis_guidance=(
        "Structure as a systematic analysis. Include patient populations, "
        "interventions, comparators, outcomes, and study quality assessment. "
        "Report statistical results with exact p-values, confidence intervals, "
        "and effect sizes. Flag safety signals with incidence rates. "
        "Assess risk of bias per study. Use PRISMA-style formatting."
    ),
    convergence_criteria=[
        "All studies/trials referenced have been extracted",
        "Patient populations and interventions are fully characterized",
        "Primary and secondary outcomes with statistical results are captured",
        "Safety/adverse event data is complete",
        "Cross-study comparisons and inconsistencies are identified",
    ],
    seed_framework_hint=(
        "This is a medical/clinical document analysis task. Identify the "
        "study type (RCT, observational, systematic review, meta-analysis, "
        "case report), the PICO elements (Population, Intervention, "
        "Comparator, Outcomes), and plan extraction accordingly."
    ),
    structural_profile_hint=(
        "Count study arms, outcome measures, tables, figures, and "
        "supplementary materials. Identify study registration numbers."
    ),
)

PATENT_ADAPTER = DomainAdapter(
    name="patent",
    description="Patent filing and prior art analysis",
    extraction_focus=(
        "Extract EVERY claim element (independent and dependent claims), "
        "prior art reference (patent numbers, publications, dates), "
        "technical specification detail (dimensions, materials, processes, "
        "parameters), prosecution history event (office action, amendment, "
        "interview, allowance), and legal status indicator (filing date, "
        "priority date, issue date, maintenance fee status, expiration)."
    ),
    type_weights={
        "analysis": 1.0,
        "calculation": 0.80,
        "strategy": 0.85,
        "observation": 0.55,
        "gap": 0.70,
        "contradiction": 0.90,
    },
    entity_patterns=[
        r"\b(?:US|EP|WO|JP|CN)\s*\d{4,}[A-Z]?\d*\b",
        r"\b(?:Patent|Application)\s+(?:No\.?\s*)?[\d,]+",
        r"\b(?:Claim|Claims?)\s+\d+(?:\s*[-–]\s*\d+)?",
        r"\b(?:USPC|IPC|CPC)\s+[A-Z]\d{2}[A-Z]\s*\d+/\d+",
    ],
    content_value_patterns=[
        (r"\b(?:US|EP|WO)\s*\d{4,}[A-Z]?\d*\b", 0.15),
        (r"\b(?:Claim|Claims?)\s+\d+", 0.12),
        (r"\b(?:Section|Article|Paragraph)\s+\d+", 0.10),
        (r"\b\d{1,2}/\d{1,2}/\d{4}\b", 0.08),
        (r"\b(?:prior art|anticipat|obvious|novel|non-obvious)\b", 0.10),
    ],
    synthesis_guidance=(
        "Structure by claim element. Map each claim limitation to "
        "specification support and prior art. Identify claim construction "
        "issues, potential invalidity grounds, and infringement positions. "
        "Use patent-number citations. Include prosecution history events."
    ),
    convergence_criteria=[
        "All independent claims have been analyzed element-by-element",
        "Prior art references have been fully extracted and mapped",
        "Prosecution history events are captured",
        "Claim construction issues have been identified",
        "Invalidity and infringement positions have been assessed",
    ],
    seed_framework_hint=(
        "This is a patent analysis task. Identify the patent family, "
        "claim structure (independent vs dependent), technology domain, "
        "and analysis type (validity, infringement, FTO, landscape)."
    ),
    structural_profile_hint=(
        "Count claims (independent and dependent), specification sections, "
        "drawings/figures, and cited references."
    ),
)

FINANCE_ADAPTER = DomainAdapter(
    name="finance",
    description="Financial analysis, SEC filings, and due diligence",
    extraction_focus=(
        "Extract EVERY financial metric (revenue, EBITDA, margins, growth "
        "rates), balance sheet item (assets, liabilities, equity), cash flow "
        "component, risk factor, management discussion point, segment "
        "performance data, guidance figure, and material event (M&A, "
        "restatements, impairments, litigation)."
    ),
    type_weights={
        "analysis": 1.0,
        "calculation": 1.0,
        "strategy": 0.75,
        "observation": 0.50,
        "gap": 0.60,
        "contradiction": 0.85,
    },
    entity_patterns=[
        r"\b(?:NASDAQ|NYSE|AMEX):\s*[A-Z]+",
        r"\b(?:FY|Q[1-4])\s*\d{4}\b",
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+(?:Inc|Corp|Ltd|Holdings|Group)\b",
    ],
    content_value_patterns=[
        (r"\$\s*[\d,]+(?:\.\d+)?\s*(?:[BMKbmk]|million|billion|thousand)?", 0.15),
        (r"\b\d+(?:\.\d+)?%", 0.12),
        (r"\b(?:FY|Q[1-4])\s*\d{4}\b", 0.08),
        (r"\b(?:YoY|QoQ|CAGR)\b", 0.10),
        (r"\b(?:EBITDA|EPS|P/E|ROE|ROA|ROIC)\b", 0.10),
    ],
    synthesis_guidance=(
        "Structure by analytical dimension (revenue, profitability, "
        "cash flow, balance sheet, valuation, risk). Include exact "
        "financial figures with periods. Show year-over-year comparisons. "
        "Flag material changes and their drivers. Include management "
        "guidance and forward-looking metrics."
    ),
    convergence_criteria=[
        "All requested financial metrics have been extracted with periods",
        "Year-over-year and quarter-over-quarter trends are captured",
        "Risk factors have been fully enumerated",
        "Material events and their financial impact are quantified",
        "Guidance and forward-looking statements are captured",
    ],
    seed_framework_hint=(
        "This is a financial analysis task. Identify the entity type "
        "(public company, fund, SPV), filing type (10-K, 10-Q, proxy, "
        "pitch book), and analysis dimension (valuation, due diligence, "
        "risk assessment, competitive positioning)."
    ),
    structural_profile_hint=(
        "Count financial statement line items, segment breakdowns, "
        "risk factors, and tables/charts."
    ),
)

INSURANCE_ADAPTER = DomainAdapter(
    name="insurance",
    description="Insurance underwriting and policy analysis",
    extraction_focus=(
        "Extract EVERY coverage provision (limits, deductibles, exclusions, "
        "endorsements), policy condition (notice, cooperation, subrogation), "
        "premium term, claim detail (date, amount, status, reserves), "
        "underwriting factor (class, territory, experience modification), "
        "and regulatory requirement (filing, approval, compliance)."
    ),
    type_weights={
        "analysis": 1.0,
        "calculation": 0.95,
        "strategy": 0.80,
        "observation": 0.55,
        "gap": 0.65,
        "contradiction": 0.90,
    },
    entity_patterns=[
        r"\b(?:Policy|Certificate|Binder)\s+(?:No\.?\s*)?[\w-]+",
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+(?:Insurance|Assurance|Mutual|Re)\b",
        r"\b(?:ISO|AAIS)\s+form\s+[\w.]+",
    ],
    content_value_patterns=[
        (r"\$\s*[\d,]+(?:\.\d+)?", 0.15),
        (r"\b\d+(?:\.\d+)?%", 0.10),
        (r"\b(?:per occurrence|per claim|aggregate|sub-limit)\b", 0.12),
        (r"\b(?:deductible|retention|co-pay|co-insurance)\b", 0.10),
        (r"\b(?:exclusion|endorsement|rider|amendment)\b", 0.10),
    ],
    synthesis_guidance=(
        "Structure by coverage line. Include exact limits, deductibles, "
        "exclusions, and endorsements. Map coverage to specific risks. "
        "Flag gaps between requested and offered coverage. Include "
        "premium breakdown and payment terms."
    ),
    convergence_criteria=[
        "All coverage provisions have been extracted with limits",
        "Exclusions and limitations are fully enumerated",
        "Policy conditions and duties are captured",
        "Premium terms and payment schedules are complete",
        "Coverage gaps have been identified",
    ],
    seed_framework_hint=(
        "This is an insurance analysis task. Identify the line of "
        "coverage (property, casualty, liability, life, health), "
        "policy type (occurrence, claims-made), and analysis goal "
        "(underwriting review, claim analysis, coverage comparison)."
    ),
    structural_profile_hint=(
        "Count coverage sections, exclusions, endorsements, "
        "schedules, and declarations pages."
    ),
)


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

_ADAPTERS: dict[str, DomainAdapter] = {
    "legal": DomainAdapter(),  # default
    "medical": MEDICAL_ADAPTER,
    "patent": PATENT_ADAPTER,
    "finance": FINANCE_ADAPTER,
    "insurance": INSURANCE_ADAPTER,
}


def get_adapter(name: str = "legal") -> DomainAdapter:
    """Get a domain adapter by name."""
    return _ADAPTERS.get(name.lower(), DomainAdapter())


def register_adapter(adapter: DomainAdapter) -> None:
    """Register a custom domain adapter."""
    _ADAPTERS[adapter.name.lower()] = adapter


def list_adapters() -> list[str]:
    """List all registered adapter names."""
    return sorted(_ADAPTERS.keys())


# ---------------------------------------------------------------------------
# Auto-detection heuristic
# ---------------------------------------------------------------------------

_DOMAIN_SIGNALS: dict[str, list[str]] = {
    "medical": [
        "patient", "clinical", "trial", "study", "efficacy", "safety",
        "adverse event", "randomized", "placebo", "endpoint", "dosage",
        "FDA", "EMA", "regulatory submission", "IND", "NDA", "BLA",
    ],
    "patent": [
        "claim", "prior art", "patent", "invention", "specification",
        "prosecution", "office action", "examiner", "obviousness",
        "anticipation", "FTO", "freedom to operate", "patentability",
    ],
    "finance": [
        "10-K", "10-Q", "annual report", "SEC filing", "revenue",
        "EBITDA", "earnings", "balance sheet", "cash flow", "guidance",
        "material", "risk factor", "segment", "shareholder",
    ],
    "insurance": [
        "policy", "coverage", "premium", "deductible", "exclusion",
        "endorsement", "underwriting", "claim", "reserve", "actuarial",
        "occurrence", "claims-made", "aggregate limit", "sub-limit",
    ],
}


def detect_domain(text: str) -> tuple[str, float]:
    """Heuristic domain detection from task instruction or document text.

    Returns (domain_name, confidence).
    """
    lower = text.lower()
    scores: dict[str, int] = {}
    for domain, signals in _DOMAIN_SIGNALS.items():
        score = sum(1 for s in signals if s.lower() in lower)
        if score > 0:
            scores[domain] = score

    if not scores:
        return "legal", 0.0

    best = max(scores, key=scores.get)
    total_signals = sum(scores.values())
    confidence = scores[best] / max(total_signals, 1)
    return best, round(confidence, 2)
