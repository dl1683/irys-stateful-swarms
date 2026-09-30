"""Synthesis obligation builder: converts blackboard knowledge into
judge-scorable obligations that get prepended to must_include.

This bridges the gap between "facts found" and "exact assertions the
deliverable must make." The #1 failure mode in the 70-89% band.
"""
from __future__ import annotations

import json

from .blackboard import Blackboard
from .models import Entry, ModelCaller
from .seed import format_task_state_map
from .verification import extract_verification_targets
from .worker_dispatch import call_model


def build_synthesis_obligations(blackboard: Blackboard, seed: dict,
                                caller: ModelCaller) -> tuple[list[dict], int]:
    """Convert blackboard knowledge into scorable obligations.

    Reads analysis/calculation/strategy/gap entries + seed plan.
    Outputs 20-60 obligations shaped like must_include items.

    Returns (obligations, tokens_used).
    """
    active = [
        e for e in blackboard.entries
        if e.status == "active" and e.type != "entity_overview"
    ]

    # Use ALL active entries — not just analytical. Raw observations contain
    # facts that need to become explicit obligations (issue flags, conclusions).
    analytical = [e for e in active if e.type in ("analysis", "calculation", "strategy", "gap")]
    # Also include high-confidence observations with source attribution
    sourced_obs = [e for e in active if e.type == "observation" and e.source and e.source.document and e.confidence >= 0.7]
    high_conf_obs = [e for e in active if e.type == "observation" and e.confidence >= 0.7]

    analytical_text = "\n".join(
        _render_analytical_entry(e) for e in analytical
    )
    # Include sourced observations — these contain facts needing explicit conclusions
    obs_text = "\n".join(
        f"[{e.id}] ({e.type}) [{e.source.document}/{e.source.section}] {e.content[:300]}"
        if e.source and e.source.document else f"[{e.id}] ({e.type}) {e.content[:300]}"
        for e in sourced_obs[:200]  # Cap to avoid prompt overflow
    )

    key_questions = seed.get("key_questions", [])
    questions_text = "\n".join(f"- {q}" for q in key_questions) if key_questions else "None."
    framework = seed.get("analytical_framework", "")
    completeness = seed.get("completeness_criteria", [])
    completeness_text = "\n".join(f"- {c}" for c in completeness) if completeness else "None."
    task_state_map = format_task_state_map(seed)

    doc_names = [d.name for d in blackboard.documents]

    prompt = f"""You are converting analytical findings into exact deliverable obligations.

TASK: {blackboard.task_instruction}

ANALYTICAL FRAMEWORK: {framework}

KEY QUESTIONS:
{questions_text}

COMPLETENESS CRITERIA:
{completeness_text}

TASK STATE MAP:
{task_state_map}

DOCUMENTS: {', '.join(doc_names)}

ANALYTICAL FINDINGS ({len(analytical)} entries — analysis, calculations, strategies, gaps):
{analytical_text[:200000]}

SOURCE-GROUNDED OBSERVATIONS ({len(sourced_obs)} entries — facts with document provenance):
{obs_text[:150000]}

YOUR JOB: Convert these findings into EXACT obligations that the final deliverable MUST contain. Each obligation should be something a judge could score as pass/fail.

CRITICAL: Look at the observations and ask "what ISSUE, CONCLUSION, or FLAG should the deliverable state based on this fact?" Raw facts alone are not enough — the deliverable must IDENTIFY issues, FLAG risks, EXPLAIN implications, and RECOMMEND actions. Every observation that reveals a problem, discrepancy, risk, or noteworthy finding should become an explicit obligation.

For each obligation, specify:
- "summary": The exact assertion, calculation, citation, or clause the deliverable must state. Be PRECISE — "$472,500, not $475,000" not "broker commission issue."
- "obligation_type": One of: exact_value | reference_framework | cross_document_link | risk_recommendation | drafting_clause | output_structure | task_state_field | task_state_relationship | task_state_closure
- "importance": critical | high | medium
- "section": Which section of the deliverable this belongs in
- "entry_id": Source entry IDs (comma-separated)
- "verification_terms": List of exact strings that should appear in the deliverable for this obligation

RULES:
- Do NOT summarize facts. Convert them into "the deliverable must say/do THIS."
- Each obligation must be judge-scorable: specific enough that a reviewer can check pass/fail.
- Include exact numbers, dates, percentages, party names, legal citations.
- Include exact calculations with arithmetic steps.
- Include exact cross-document conflicts or mismatches.
- Include exact risk flags with specific recommendations.
- Include exact drafting clauses or provisions that must appear.
- Include obligations for task-state-map rows whose fields, relationships, closure checks, or artifact form must be explicit in the final deliverable.
- Produce 20-60 obligations. Quality over quantity.

Return JSON: {{"obligations": [...]}}"""

    payload, tokens = call_model(caller, prompt, max_tokens=16384)

    obligations = payload.get("obligations", [])
    if not isinstance(obligations, list):
        obligations = []

    # Convert to must_include format for compatibility
    must_include_items = (
        _derived_calculation_obligations(active)
        + _debt_sensor_obligations(active)
    )
    for o in obligations:
        if not isinstance(o, dict):
            continue
        summary = o.get("summary", "")
        if not summary or len(summary) < 10:
            continue
        must_include_items.append({
            "entry_id": o.get("entry_id", ""),
            "importance": o.get("importance", "high"),
            "section": o.get("section", "General"),
            "summary": summary,
            "obligation_type": o.get("obligation_type", ""),
            "verification_terms": o.get("verification_terms", []),
        })

    return must_include_items, tokens


def _render_analytical_entry(e: Entry) -> str:
    parts = [f"[{e.id}] ({e.type})"]
    if e.source and e.source.document:
        parts.append(f" [{e.source.document}/{e.source.section}]")
    conv_tags = [
        t for t in (e.tags or [])
        if t.startswith((
            "state_conversion", "plan_coverage", "materiality:", "coverage:",
            "missing_work:", "derived_work:", "derived_type:", "debt_subtype:",
            "debt_type:", "severity:", "lifecycle:",
        ))
    ]
    if conv_tags:
        parts.append(f" [{','.join(conv_tags[:4])}]")
    if e.supports_entries:
        parts.append(f" supports={','.join(e.supports_entries[:5])}")
    parts.append(f" {e.content[:500]}")
    return "".join(parts)


def _derived_calculation_obligations(entries: list[Entry]) -> list[dict]:
    items = []
    for entry in entries:
        if entry.status != "active" or entry.type != "calculation":
            continue
        tags = entry.tags or []
        if not any(tag.startswith("derived_work:") for tag in tags):
            continue
        verification_terms = []
        if entry.source and entry.source.evidence:
            verification_terms.append(entry.source.evidence)
        items.append({
            "entry_id": entry.id,
            "importance": "critical",
            "section": "Calculations",
            "summary": entry.content,
            "obligation_type": "exact_value",
            "verification_terms": verification_terms,
            "source": "derived_work",
        })
    return items


def _debt_sensor_obligations(entries: list[Entry]) -> list[dict]:
    items = []
    for entry in entries:
        if entry.status != "active" or entry.type == "gap":
            continue
        tags = entry.tags or []
        if "debt_sensor" not in tags:
            continue
        debt_type = _tag_value(tags, "debt_type:")
        debt_subtype = _tag_value(tags, "debt_subtype:")
        obligation_type = {
            "relation": "cross_document_link",
            "source_object": "task_state_field",
            "severity": "risk_recommendation",
            "authority": "reference_framework",
        }.get(debt_type, "task_state_closure")
        section = {
            "relation": "Cross-Document Analysis",
            "source_object": "Source Coverage",
            "severity": "Risk and Recommendations",
            "authority": "Source Authority",
        }.get(debt_type, "Required Findings")
        verification_terms = _entry_verification_terms(entry)
        items.append({
            "entry_id": entry.id,
            "importance": "critical" if entry.confidence >= 0.85 else "high",
            "section": section,
            "summary": entry.content,
            "obligation_type": obligation_type,
            "verification_terms": verification_terms,
            "source": "debt_sensor",
            "debt_type": debt_type,
            "debt_subtype": debt_subtype,
        })
    return items


def _entry_verification_terms(entry: Entry) -> list[str]:
    terms: list[str] = []
    for target in extract_verification_targets(entry.content):
        raw = str(target.get("raw", "")).strip()
        if raw and raw not in terms:
            terms.append(raw)
    if entry.source:
        for raw in (entry.source.section, entry.source.evidence):
            value = str(raw or "").strip()
            if value and len(value) <= 160 and value not in terms:
                terms.append(value)
    return terms[:12]


def _tag_value(tags: list[str], prefix: str) -> str:
    for tag in tags:
        if isinstance(tag, str) and tag.startswith(prefix):
            return tag[len(prefix):].strip()
    return ""
