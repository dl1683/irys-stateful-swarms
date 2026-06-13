"""Pluggable, language- and domain-agnostic domain adapter (Open Research Question #4).

ORQ #4: what breaks when the swarm runs on medical / patent / insurance / non-English
documents instead of US-legal ones, and where does the architecture need domain adaptation?

This provides a *declarative* DomainAdapter profile plus an extensible registry, so domain
behaviour (extraction focus, scoring weights, synthesis guidance, convergence criteria) is
DATA, not branching code. It is built to **complement** the model-driven ``domain_lens.py``,
never to replace it:

* The default profile is domain- and language-**neutral** — the system is not legal-biased
  out of the box (the previous version defaulted every field to US legal).
* Domain detection reads each registered profile's own ``signals`` (in *any* language) and is
  unicode-correct (NFKC + casefold); when no deterministic signal fires it can defer to a
  ``ModelCaller``. Adding a domain — in any language — is a ``register_adapter`` call, not a
  code edit and not a change to a central hardcoded keyword table.
* Value scoring uses UNIVERSAL, language-agnostic signals (numbers, currency, percentages,
  dates) by default; a domain may add its own optional patterns. No ``[A-Z][a-z]+`` ASCII
  entity regex — entity recognition is delegated to the language-agnostic extractor in
  ``entity_resolution``.
* ``DomainAdapter.merge_lens`` ingests the JSON produced by ``domain_lens.generate_domain_lens``
  so a model-derived lens enriches the deterministic profile — the two layers compose.

Zero LLM calls in the adapter itself unless a ModelCaller is explicitly passed to detection.
"""
from __future__ import annotations

import dataclasses
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Optional

try:  # optional — only used when a caller is passed to detect_domain
    from .worker_dispatch import call_model as _call_model
except Exception:  # pragma: no cover - worker_dispatch may pull heavy deps
    _call_model = None


# Universal, language-agnostic "this content carries hard facts" signals: unicode digits via
# \d, currency by symbol, percentages, numeric dates, significant numbers. No English words.
UNIVERSAL_VALUE_PATTERNS: tuple[tuple[str, float], ...] = (
    (r"\d[\d.,]*\s?%", 0.10),                                 # percentages
    (r"[$€£¥₹₽¢]\s?\d[\d.,]*", 0.15),                         # currency amounts
    (r"(?<!\d)\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}(?!\d)", 0.06),  # numeric dates (boundary by digit, not \b — CJK-safe)
    (r"(?<!\d)\d[\d.,]{2,}(?!\d)", 0.05),                     # significant numbers
)


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


# Scripts without inter-word spaces (Han, Hiragana, Katakana, Hangul) have no reliable word
# boundary, so signal matching there is substring-based; spaced scripts use a word boundary
# (so "claim" does not match inside "proclaimed").
_NO_WORD_BOUNDARY = re.compile(r"[　-鿿豈-﫿가-힯぀-ヿ]")


def _signal_hit(signal_norm: str, text_norm: str) -> bool:
    if not signal_norm:
        return False
    if _NO_WORD_BOUNDARY.search(signal_norm):
        return signal_norm in text_norm
    return re.search(rf"(?<!\w){re.escape(signal_norm)}(?!\w)", text_norm, flags=re.UNICODE) is not None


@dataclass
class DomainAdapter:
    """Declarative, language-neutral profile for domain-specific swarm behaviour.

    Every field has a domain- and language-agnostic default. Register specialised profiles
    (in any language) via :func:`register_adapter`; enrich one with model output via
    :meth:`merge_lens`.
    """
    name: str = "generic"
    description: str = "Domain- and language-neutral default profile"

    extraction_focus: str = (
        "Extract every concrete fact: quantities, amounts, percentages, dates, deadlines, "
        "named parties/entities, identifiers, defined terms, obligations, conditions, and "
        "cross-references — regardless of domain or language."
    )
    type_weights: dict[str, float] = field(default_factory=lambda: {
        "analysis": 1.0, "calculation": 0.95, "strategy": 0.80,
        "observation": 0.50, "gap": 0.60, "contradiction": 0.85,
    })
    synthesis_guidance: str = (
        "Structure with clear headings. Include every specific quantity, amount, percentage, "
        "date, named party, deadline, obligation, and cross-reference. Show calculations with "
        "full steps. Cite source documents for each fact."
    )
    convergence_criteria: list[str] = field(default_factory=lambda: [
        "All key questions have source-grounded answers",
        "All enumerated items from source documents have been extracted",
        "Cross-document comparisons have been performed",
        "Calculations have been verified",
        "Gaps and uncertainties have been identified",
    ])
    seed_framework_hint: str = (
        "Identify the task type (extraction, comparison, drafting, issue-flagging, "
        "calculation) and the subject domain from the documents, then plan accordingly."
    )
    structural_profile_hint: str = (
        "Count individually enumerated items, data tables, and major sections."
    )

    # Extensibility hooks — all optional, default empty, NO hardcoded English.
    signals: tuple[str, ...] = ()                              # terms indicating this domain (any language)
    extra_value_patterns: tuple[tuple[str, float], ...] = ()   # domain-specific scoring regex

    # ---- prompt-suffix accessors (public surface preserved) ----
    def get_extraction_prompt_suffix(self) -> str:
        return f"\nDOMAIN FOCUS: {self.extraction_focus}"

    def get_synthesis_prompt_suffix(self) -> str:
        return f"\nDOMAIN GUIDANCE: {self.synthesis_guidance}"

    def get_seed_prompt_suffix(self) -> str:
        return f"\nDOMAIN CONTEXT: {self.seed_framework_hint}"

    def get_structural_profile_suffix(self) -> str:
        return f"\nDOMAIN FOCUS: {self.structural_profile_hint}"

    # ---- language-agnostic value scoring ----
    def value_patterns(self) -> tuple[tuple[str, float], ...]:
        return UNIVERSAL_VALUE_PATTERNS + tuple(self.extra_value_patterns)

    def score_content(self, text: str) -> float:
        """Sum of capped value-pattern hits — works for any domain/language (no keywords)."""
        total = 0.0
        for pat, weight in self.value_patterns():
            if re.search(pat, text, flags=re.UNICODE):
                total += weight
        return round(min(total, 1.0), 4)

    # ---- composition with the model-driven domain_lens ----
    def merge_lens(self, lens: dict) -> "DomainAdapter":
        """Return a copy enriched by a model-generated domain lens (see ``domain_lens.py``).

        The deterministic profile is the scaffold; the lens supplies task-specific expertise.
        The two layers compose rather than compete.
        """
        if not lens or not isinstance(lens, dict):
            return self
        issues = [h.strip() for h in (lens.get("issue_hypotheses") or [])
                  if isinstance(h, str) and h.strip()]
        checks = [c.strip() for c in (lens.get("negative_checks") or [])
                  if isinstance(c, str) and c.strip()]
        guidance = self.synthesis_guidance
        if issues:
            guidance += "\nLens issues: " + "; ".join(issues[:10])
        criteria = list(self.convergence_criteria)
        criteria += [f"Lens check satisfied: {c}" for c in checks[:5]]
        return dataclasses.replace(self, synthesis_guidance=guidance, convergence_criteria=criteria)


def adapter_from_lens(lens: dict, name: str = "generic",
                      base: DomainAdapter | None = None) -> DomainAdapter:
    """Build a profile from a model-generated lens on top of a base profile (default generic)."""
    base = base or DomainAdapter(name=name)
    if name and name != base.name:
        base = dataclasses.replace(base, name=name)
    return base.merge_lens(lens or {})


# ---------------------------------------------------------------------------
# Built-in EXAMPLE profiles (illustrative, not a closed set). Each declares its own
# ``signals`` (so detection is registry-driven) and optional value patterns. None of them
# carry ASCII entity regex anymore.
# ---------------------------------------------------------------------------

LEGAL_ADAPTER = DomainAdapter(
    name="legal",
    description="Legal document analysis",
    extraction_focus=(
        "Extract every amount, percentage, date, deadline, party name with full entity "
        "designation, numbered/lettered items, defined terms, obligations, conditions, "
        "restrictions, representations, warranties, and payment terms."
    ),
    synthesis_guidance=(
        "Structure with clear headings and professional formatting. Include every number, "
        "date, amount, percentage, party, deadline, obligation, restriction, representation, "
        "and warranty. Show calculations with full arithmetic. Cite source documents."
    ),
    signals=(
        "agreement", "borrower", "lender", "covenant", "indemnification", "obligation",
        "representation", "warranty", "closing date", "governing law", "term loan",
    ),
    extra_value_patterns=((r"\b(?:shall|must|may\s+not|prohibited)\b", 0.08),),
)

MEDICAL_ADAPTER = DomainAdapter(
    name="medical",
    description="Medical research and clinical document analysis",
    extraction_focus=(
        "Extract every patient-population characteristic (N, demographics, inclusion/exclusion), "
        "intervention (drug, dose, route, frequency, duration), outcome measures, statistical "
        "results (p-values, CIs, effect sizes, hazard ratios), adverse events, and study design."
    ),
    type_weights={"analysis": 1.0, "calculation": 0.95, "strategy": 0.70,
                  "observation": 0.55, "gap": 0.65, "contradiction": 0.90},
    synthesis_guidance=(
        "Report populations, interventions, comparators, outcomes, and study-quality. Give "
        "exact p-values, confidence intervals, and effect sizes. Flag safety signals with "
        "incidence. Assess risk of bias. PRISMA-style formatting."
    ),
    signals=("patient", "clinical", "trial", "efficacy", "adverse event", "randomized",
             "placebo", "endpoint", "dosage", "cohort", "in vitro", "biomarker"),
    extra_value_patterns=(
        (r"\bp\s*[<>=]\s*[\d.]+", 0.15),
        (r"\b(?:HR|OR|RR)\s*[:=]\s*[\d.]+", 0.15),
        (r"\bN\s*=\s*\d+", 0.10),
    ),
)

PATENT_ADAPTER = DomainAdapter(
    name="patent",
    description="Patent filing and prior-art analysis",
    extraction_focus=(
        "Extract every claim element (independent/dependent), prior-art reference, technical "
        "specification detail, prosecution-history event, and legal-status indicator."
    ),
    type_weights={"analysis": 1.0, "calculation": 0.80, "strategy": 0.85,
                  "observation": 0.55, "gap": 0.70, "contradiction": 0.90},
    synthesis_guidance=(
        "Structure by claim element; map each limitation to specification support and prior art. "
        "Identify claim-construction issues, invalidity grounds, and infringement positions."
    ),
    signals=("claim", "prior art", "invention", "specification", "prosecution",
             "office action", "examiner", "obviousness", "anticipation", "patentability"),
    extra_value_patterns=((r"\b(?:claims?)\s+\d+", 0.12),),
)

FINANCE_ADAPTER = DomainAdapter(
    name="finance",
    description="Financial analysis, filings, and due diligence",
    extraction_focus=(
        "Extract every financial metric (revenue, EBITDA, margins, growth), balance-sheet and "
        "cash-flow item, risk factor, segment datum, guidance figure, and material event."
    ),
    type_weights={"analysis": 1.0, "calculation": 1.0, "strategy": 0.75,
                  "observation": 0.50, "gap": 0.60, "contradiction": 0.85},
    synthesis_guidance=(
        "Structure by dimension (revenue, profitability, cash flow, balance sheet, valuation, "
        "risk). Include exact figures with periods and year-over-year comparisons. Flag material "
        "changes and guidance."
    ),
    signals=("revenue", "ebitda", "earnings", "balance sheet", "cash flow", "guidance",
             "risk factor", "segment", "shareholder", "annual report", "valuation"),
    extra_value_patterns=((r"\b(?:EBITDA|EPS|P/E|ROE|ROA|ROIC|CAGR|YoY|QoQ)\b", 0.10),),
)

INSURANCE_ADAPTER = DomainAdapter(
    name="insurance",
    description="Insurance underwriting and policy analysis",
    extraction_focus=(
        "Extract every coverage provision (limits, deductibles, exclusions, endorsements), "
        "policy condition, premium term, claim detail, underwriting factor, and regulatory item."
    ),
    type_weights={"analysis": 1.0, "calculation": 0.95, "strategy": 0.80,
                  "observation": 0.55, "gap": 0.65, "contradiction": 0.90},
    synthesis_guidance=(
        "Structure by coverage line. Include exact limits, deductibles, exclusions, and "
        "endorsements. Map coverage to risks. Flag gaps between requested and offered coverage."
    ),
    signals=("policy", "coverage", "premium", "deductible", "exclusion", "endorsement",
             "underwriting", "reserve", "actuarial", "occurrence", "claims-made", "sub-limit"),
    extra_value_patterns=((r"\b(?:per\s+occurrence|aggregate|sub-limit|deductible|retention)\b", 0.12),),
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ADAPTERS: dict[str, DomainAdapter] = {
    "generic": DomainAdapter(),   # neutral default — NOT legal-biased
    "legal": LEGAL_ADAPTER,
    "medical": MEDICAL_ADAPTER,
    "patent": PATENT_ADAPTER,
    "finance": FINANCE_ADAPTER,
    "insurance": INSURANCE_ADAPTER,
}


def get_adapter(name: str = "generic") -> DomainAdapter:
    """Get a registered domain adapter by name (falls back to the neutral default)."""
    return _ADAPTERS.get((name or "").lower(), _ADAPTERS["generic"])


def register_adapter(adapter: DomainAdapter) -> None:
    """Register a custom domain adapter (any domain, any language)."""
    _ADAPTERS[adapter.name.lower()] = adapter


def list_adapters() -> list[str]:
    return sorted(_ADAPTERS.keys())


# ---------------------------------------------------------------------------
# Domain detection — registry-driven, unicode-correct, optional model fallback
# ---------------------------------------------------------------------------

def _detect_with_caller(text: str, names: list[str], caller: Any) -> Optional[str]:
    if caller is None or _call_model is None:
        return None
    prompt = (
        "Classify the document/task domain. Reply with strict JSON only: "
        '{"domain": "<one of: ' + ", ".join(names) + '>"}.\n\nTEXT:\n' + text[:2000]
    )
    try:
        payload, _ = _call_model(caller, prompt, max_tokens=64)
    except Exception:
        return None
    if isinstance(payload, dict):
        d = str(payload.get("domain", "")).lower().strip()
        return d or None
    return None


def detect_domain(text: str, *, registry: dict[str, DomainAdapter] | None = None,
                  caller: Any | None = None) -> tuple[str, float]:
    """Detect the domain of ``text`` from registered profiles' own signals.

    Unicode-correct and language-agnostic: each registered adapter contributes its own
    ``signals`` (in any language), so adding a domain extends detection automatically. With
    no deterministic signal and a ``ModelCaller`` supplied, detection defers to the model.

    Returns ``(domain_name, confidence)``; ``("generic", 0.0)`` when nothing matches.
    """
    reg = registry if registry is not None else _ADAPTERS
    low = _norm(text)
    scores: dict[str, int] = {}
    for name, adapter in reg.items():
        hits = sum(1 for sig in adapter.signals if _signal_hit(_norm(sig), low))
        if hits:
            scores[name] = hits

    if not scores:
        guess = _detect_with_caller(text, list(reg.keys()), caller)
        if guess and guess in reg:
            return guess, 0.5
        return "generic", 0.0

    best = max(scores, key=scores.get)
    total = sum(scores.values())
    return best, round(scores[best] / max(total, 1), 2)
