from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .blackboard import Blackboard
from .models import Entry, ModelCaller
from .placement_verifier import (
    evaluate_placements,
    get_repair_candidates,
    repair_placements,
)
from .verification import detect_placeholders
from .synthesis_packet import filter_evidence_entries
from .worker_dispatch import (
    begin_call_model_usage,
    call_model,
    end_call_model_usage,
    get_last_call_usage,
    set_last_call_usage,
)


def shadow_judge_audit_enabled() -> bool:
    return os.getenv("SWARM_ENABLE_SHADOW_JUDGE_AUDIT", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


SECTION_THRESHOLD = int(os.getenv("SWARM_SYNTHESIS_SECTION_THRESHOLD", "15"))
SECTION_CHUNK_SIZE = int(os.getenv("SWARM_SYNTHESIS_SECTION_CHUNK_SIZE", "25"))
MAX_SECTIONS = int(os.getenv("SWARM_SYNTHESIS_MAX_SECTIONS", "20"))
SECTION_DRAFT_MAX_TOKENS = int(os.getenv("SWARM_SYNTHESIS_SECTION_MAX_TOKENS", "16384"))
ASSIGNMENT_BATCH_SIZE = 50
SECTION_EVIDENCE_CHARS = int(os.getenv("SWARM_SYNTHESIS_SECTION_EVIDENCE_CHARS", "24000"))
SELECTED_ITEM_SUMMARY_CHARS = int(os.getenv("SWARM_SYNTHESIS_ITEM_SUMMARY_CHARS", "700"))
CLEAN_ASSEMBLED_DELIVERABLE = os.getenv("SWARM_SYNTHESIS_CLEAN_DELIVERABLE", "1") == "1"


def render_entry(e: Entry, max_content: int = 400) -> str:
    """Render an entry with source provenance and evidence."""
    parts = [f"[{e.id}] ({e.type})"]
    if e.source and e.source.document:
        parts.append(f"[{e.source.document}")
        if e.source.section:
            parts.append(f"/{e.source.section}")
        parts.append("]")
    conv_tags = [t for t in (e.tags or []) if t.startswith(("state_conversion", "plan_coverage", "materiality:", "answers:", "satisfies:", "coverage:", "missing_work:"))]
    if conv_tags:
        parts.append(f" [{','.join(conv_tags[:5])}]")
    if e.supports_entries:
        parts.append(f" supports={','.join(e.supports_entries[:5])}")
    parts.append(f" {e.content[:max_content]}")
    if e.source and e.source.evidence and len(e.source.evidence) > 10:
        parts.append(f" | Evidence: {e.source.evidence[:200]}")
    return "".join(parts)


def _finalize_synthesis_usage(usage_by_model: dict) -> None:
    """Combine usage collected from parallel section drafts with whatever the
    aggregate accumulated from sequential calls, then install it as the last
    call's usage so blackboard.add_tokens_from_last_call picks up everything.
    """
    end_call_model_usage()
    aggregate_usage, _, _, _ = get_last_call_usage()
    combined: dict = {}
    _merge_usage(combined, aggregate_usage)
    _merge_usage(combined, usage_by_model)
    set_last_call_usage(combined)


def synthesize_deliverable(blackboard: Blackboard, must_include: list[dict],
                           caller: ModelCaller) -> tuple[str, int]:
    begin_call_model_usage()
    try:
        active = [e for e in blackboard.entries if e.status == "active"]
        total_tokens = 0
        used_sectioned = len(must_include) > SECTION_THRESHOLD

        # If many items, use sectioned synthesis to avoid output truncation
        if used_sectioned:
            draft, draft_tokens, usage_by_model = _sectioned_synthesis(
                blackboard, must_include, active, caller,
            )
        else:
            draft, draft_tokens = _draft_synthesis(blackboard, must_include, active, caller)
            usage_by_model = {}
        total_tokens += draft_tokens

        if not must_include:
            _finalize_synthesis_usage(usage_by_model)
            return draft, total_tokens

        # Phase 2: Verify — which must_include items are missing from the draft?
        missing, verify_tokens = _verify_completeness(draft, must_include, blackboard, caller)
        total_tokens += verify_tokens

        if not missing:
            _finalize_synthesis_usage(usage_by_model)
            return draft, total_tokens

        # Phase 3: Augment — targeted repair for missing items
        if used_sectioned:
            augmented, augment_tokens = _append_missing_items(
                draft, missing, active, blackboard, caller,
            )
        else:
            augmented, augment_tokens = _augment_draft(
                draft, missing, active, blackboard, caller,
            )
        total_tokens += augment_tokens

        repaired_map, ph_tokens = _repair_placeholders(
            {"deliverable": augmented}, blackboard, caller,
        )
        total_tokens += ph_tokens
        augmented = repaired_map.get("deliverable", augmented)

        _finalize_synthesis_usage(usage_by_model)
        return augmented, total_tokens
    finally:
        end_call_model_usage()


_FILE_REQUIREMENTS_BLOCKLIST = frozenset({
    "criteria", "match_criteria", "scorer", "scores",
    "criteria_results", "rubric", "prior_score",
})


def _validate_file_requirements(requirements: list[dict]) -> None:
    """Reject any requirement dict containing benchmark/evaluator keys."""
    for req in requirements:
        if not isinstance(req, dict):
            continue
        blocked = _FILE_REQUIREMENTS_BLOCKLIST & req.keys()
        if blocked:
            raise ValueError(
                f"file_requirements contains benchmark key(s): {blocked} — "
                f"evaluator metadata must never reach synthesis"
            )


def synthesize_file_deliverables(
    blackboard: Blackboard,
    must_include: list[dict],
    deliverables_map: dict,
    file_requirements: list[dict],
    caller: ModelCaller,
) -> tuple[dict[str, str], int]:
    """Synthesize each requested output file separately.

    Multi-output tasks need purpose-built artifacts. Writing one global memo
    into every file contaminates the artifact set and can make downstream
    review context explode. This keeps the shared blackboard state but makes
    the final synthesis call file-scoped.
    """
    _validate_file_requirements(file_requirements)
    begin_call_model_usage()
    try:
        filenames = []
        for filename in deliverables_map.values():
            if isinstance(filename, str) and filename not in filenames:
                filenames.append(filename)

        outputs: dict[str, str] = {}
        total_tokens = 0
        usage_by_model: dict = {}
        if not filenames:
            _finalize_synthesis_usage(usage_by_model)
            return outputs, total_tokens

        item_pool = _numbered_must_include_pool(must_include)
        plans: dict[str, dict] = {}
        assigned_numbers: set[int] = set()
        ran_assignment_repair = False
        if len(filenames) == 1:
            filename = filenames[0]
            plans[filename] = {
                "file_requirements": _requirements_for_file(file_requirements, filename),
                "numbers": [n for n, _ in item_pool],
                "contract": _default_artifact_contract(filename),
            }
            assigned_numbers = {n for n, _ in item_pool}
        else:
            for filename in filenames:
                file_reqs = _requirements_for_file(file_requirements, filename)
                selected_numbers, contract, plan_tokens = _plan_file_deliverable(
                    blackboard, filename, file_reqs, item_pool, caller,
                )
                total_tokens += plan_tokens
                plans[filename] = {
                    "file_requirements": file_reqs,
                    "numbers": selected_numbers,
                    "contract": contract,
                }
                assigned_numbers.update(selected_numbers)

        _apply_target_file_pins(plans, item_pool, filenames)
        assigned_numbers = {
            n for plan in plans.values() for n in plan.get("numbers", [])
        }

        unassigned = [n for n, _ in item_pool if n not in assigned_numbers]
        if unassigned:
            extra_assignments, assign_tokens = _assign_unassigned_items(
                blackboard, filenames, file_requirements, item_pool, unassigned, caller,
            )
            ran_assignment_repair = True
            total_tokens += assign_tokens
            for filename, numbers in extra_assignments.items():
                if filename not in plans:
                    continue
                for n in numbers:
                    if n not in plans[filename]["numbers"]:
                        plans[filename]["numbers"].append(n)

            assigned_numbers = {
                n for plan in plans.values() for n in plan.get("numbers", [])
            }

        if _needs_assignment_rebalance(plans, item_pool, filenames):
            rebalanced, rebalance_tokens = _rebalance_file_assignments(
                blackboard, filenames, file_requirements, item_pool, plans, caller,
            )
            ran_assignment_repair = True
            total_tokens += rebalance_tokens
            if any(rebalanced.values()):
                for filename in filenames:
                    plans[filename]["numbers"] = rebalanced.get(filename, [])

                assigned_numbers = {
                    n for plan in plans.values() for n in plan.get("numbers", [])
                }

        if _needs_assignment_audit(plans, item_pool, filenames, file_requirements, ran_assignment_repair):
            audited, audit_tokens = _audit_file_assignments(
                blackboard, filenames, file_requirements, item_pool, plans, caller,
            )
            total_tokens += audit_tokens
            if any(audited.values()):
                for filename in filenames:
                    plans[filename]["numbers"] = audited.get(filename, [])

        for filename in filenames:
            plan = plans[filename]
            selected_items = _items_by_numbers(item_pool, plan["numbers"])
            selected_items = _with_file_requirement_items(
                selected_items, plan["file_requirements"], filename,
            )

            # _draft_file_deliverable's usage dict is already reflected in the
            # call_model aggregate (same thread, sequential) — don't double-count it.
            draft, draft_tokens, _ = _draft_file_deliverable(
                blackboard, filename, plan["file_requirements"], selected_items,
                plan.get("contract", _default_artifact_contract(filename)),
                caller,
            )
            total_tokens += draft_tokens

            if selected_items:
                missing, verify_tokens = _verify_completeness(
                    draft, selected_items, blackboard, caller,
                )
                total_tokens += verify_tokens
                if missing:
                    draft, augment_tokens = _append_missing_items_for_file(
                        filename, draft, missing,
                        [e for e in blackboard.entries if e.status == "active"],
                        blackboard, caller,
                    )
                    total_tokens += augment_tokens

            outputs[filename] = draft

        placements = evaluate_placements(must_include, outputs)
        repair_candidates = get_repair_candidates(placements)
        if repair_candidates:
            outputs, repair_tokens = repair_placements(
                repair_candidates, outputs, blackboard, caller,
            )
            total_tokens += repair_tokens

        outputs, ph_tokens = _repair_placeholders(outputs, blackboard, caller)
        total_tokens += ph_tokens

        _finalize_synthesis_usage(usage_by_model)
        return outputs, total_tokens
    finally:
        end_call_model_usage()


_SECTION_MERGE_JACCARD = 0.60
_SECTION_STOP_WORDS = frozenset({
    "the", "a", "an", "of", "in", "for", "and", "or", "to", "with",
    "on", "by", "from", "key", "main", "general", "additional", "other",
    "summary", "overview", "section", "details",
})


def _stem(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ses") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _normalize_sections(
    by_section: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    """Merge section names with high word-overlap to reduce fragmentation."""
    if len(by_section) <= MAX_SECTIONS:
        return by_section

    def _sig_words(name: str) -> set[str]:
        return {
            _stem(w) for w in re.findall(r"[a-z]+", name.lower())
            if len(w) > 2 and w not in _SECTION_STOP_WORDS
        }

    names = list(by_section.keys())
    sigs = {n: _sig_words(n) for n in names}
    canonical_map: dict[str, str] = {n: n for n in names}

    sorted_names = sorted(names, key=lambda n: len(by_section[n]), reverse=True)
    for i, big in enumerate(sorted_names):
        for small in sorted_names[i + 1:]:
            if canonical_map[small] != small:
                continue
            s1, s2 = sigs[big], sigs[small]
            if not s1 or not s2:
                continue
            jaccard = len(s1 & s2) / len(s1 | s2)
            if jaccard >= _SECTION_MERGE_JACCARD:
                target = canonical_map[big]
                canonical_map[small] = target

    result: dict[str, list[dict]] = {}
    for n in names:
        target = canonical_map[n]
        result.setdefault(target, []).extend(by_section[n])
    return result


def _cap_sections(
    by_section: dict[str, list[dict]],
    max_sections: int = MAX_SECTIONS,
) -> dict[str, list[dict]]:
    """Merge smallest sections when count exceeds max_sections."""
    if len(by_section) <= max_sections:
        return by_section
    ranked = sorted(by_section.items(), key=lambda kv: len(kv[1]), reverse=True)
    kept = dict(ranked[:max_sections - 1])
    overflow_items: list[dict] = []
    for name, items in ranked[max_sections - 1:]:
        overflow_items.extend(items)
    kept["Additional Items"] = overflow_items
    return kept


def _sectioned_synthesis(blackboard: Blackboard, must_include: list[dict],
                         active: list[Entry], caller: ModelCaller) -> tuple[str, int, dict]:
    """Draft deliverable in sections to avoid output truncation on large tasks."""
    total_tokens = 0

    # Group must_include by section
    by_section: dict[str, list[dict]] = {}
    for m in must_include:
        if isinstance(m, str):
            section = "General"
            by_section.setdefault(section, []).append({"summary": m, "section": section})
        elif isinstance(m, dict):
            section = m.get("section", "General")
            by_section.setdefault(section, []).append(m)

    by_section = _cap_sections(_normalize_sections(by_section))

    jobs = []
    section_order = list(by_section.keys())

    for section_name in section_order:
        items = by_section[section_name]
        item_chunks = _chunks(items, SECTION_CHUNK_SIZE)
        for chunk_index, chunk_items in enumerate(item_chunks, 1):
            chunk_name = section_name
            if len(item_chunks) > 1:
                chunk_name = f"{section_name} Part {chunk_index}"
            chunk_text = "\n".join(
                f"- {m.get('summary', '') if isinstance(m, dict) else str(m)}"
                for m in chunk_items
            )
            evidence = _selected_evidence_text(
                chunk_items, active, max_chars=SECTION_EVIDENCE_CHARS,
                include_remaining=False,
            )
            prompt = _section_prompt(
                blackboard, chunk_name, chunk_items, chunk_text, evidence,
            )
            jobs.append((len(jobs), chunk_name, prompt))

    if not jobs:
        return "", total_tokens, {}

    max_workers = min(
        len(jobs),
        max(1, int(os.getenv("SWARM_SYNTHESIS_SECTION_WORKERS", "8"))),
    )

    results: list[tuple[int, str, str, int, dict] | None] = [
        None for _ in jobs
    ]
    usage_by_model: dict = {}
    if max_workers <= 1:
        for sequence, chunk_name, prompt in jobs:
            result = _draft_section_chunk(caller, sequence, chunk_name, prompt)
            results[sequence] = result
            _, _, _, tokens, _ = result
            total_tokens += tokens
            _write_sectioned_synthesis_progress(
                blackboard, sequence + 1, len(jobs), chunk_name,
            )
    else:
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_draft_section_chunk, caller, sequence, chunk_name, prompt): (
                    sequence,
                    chunk_name,
                )
                for sequence, chunk_name, prompt in jobs
            }
            for future in as_completed(futures):
                sequence, chunk_name = futures[future]
                result = future.result()
                results[sequence] = result
                _, _, _, tokens, usage = result
                total_tokens += tokens
                _merge_usage(usage_by_model, usage)
                completed += 1
                _write_sectioned_synthesis_progress(
                    blackboard, completed, len(jobs), chunk_name,
                )

    section_drafts = []
    for result in results:
        if result is None:
            continue
        _, chunk_name, section_text, _, _ = result
        if section_text:
            section_text = _strip_redundant_section_heading(section_text, chunk_name)
            section_drafts.append(f"## {chunk_name}\n\n{section_text}")

    # Assemble deterministically. Do not ask the model to reproduce completed
    # sections; that can drop exactly the details sectioning is meant to save.
    assembled = _clean_assembled_deliverable("\n\n".join(section_drafts))
    return assembled, total_tokens, usage_by_model


def _section_prompt(
    blackboard: Blackboard,
    chunk_name: str,
    chunk_items: list[dict],
    chunk_text: str,
    evidence: str,
) -> str:
    return f"""Write the "{chunk_name}" section of a professional deliverable.

TASK: {blackboard.task_instruction}

SECTION: {chunk_name}
ITEMS THIS SECTION MUST INCLUDE ({len(chunk_items)} items):
{chunk_text}

SUPPORTING EVIDENCE:
{evidence}

PRECISION RULES:
1. For each item, state the EXACT fact: specific dollar amounts, percentages, dates, party names, statutory citations, defined terms. Never paraphrase a number or date — use the original value.
2. ENTITY NAMES AND VALUES ARE SACRED: Copy entity names, company names, person names, dollar amounts, dates, percentages, and share counts VERBATIM from the items and evidence above. Do NOT substitute with similar-sounding names or round numbers from your own knowledge. If the evidence says "$7.5M" write "$7.5M", not "$1.5M". If it says "University of Geneva" write that, not "Columbia Law School".
3. If you CANNOT find a specific entity name, person name, firm name, or address in the items and evidence above, write a bracketed placeholder like [Firm Name], [Counsel Name], [Address] instead. NEVER generate a plausible-sounding name from your own knowledge — a placeholder is always better than a hallucinated name.
4. One concise paragraph or bullet per item. Do not pad with background, context, or restating the task.
5. If two items cover the same fact, merge them into one statement with the most specific version.
6. Omit generic observations not listed as required items above and not supported by specific documents — but KEEP any required item even if it sounds generic.
7. This is one section of a larger document — focus only on this section's items.
8. Include STRATEGIC ASSESSMENTS: For each material risk, obligation, or finding in this section, state its practical significance — likelihood (high/medium/low), severity, negotiation leverage, and recommended action. Interpret, don't just list.
9. CROSS-DOCUMENT ANALYSIS: When items or evidence reference multiple source documents on the same topic, EXPLICITLY flag conflicts, inconsistencies, or gaps between them and state the practical consequence.
10. FORMAL ELEMENTS: If this section is the opening or closing of a professional deliverable, include appropriate formal elements (header, date, addressee, executive summary, signature block, confidentiality/privilege markings, rights-reservation language) as fitting for the document type."""


def _draft_section_chunk(
    caller: ModelCaller,
    sequence: int,
    chunk_name: str,
    prompt: str,
) -> tuple[int, str, str, int, dict]:
    payload, tokens = call_model(
        caller, prompt, max_tokens=SECTION_DRAFT_MAX_TOKENS, json_mode=False,
    )
    by_model, _, _, _ = get_last_call_usage()
    section_text = payload.get("text", "")
    return sequence, chunk_name, section_text, tokens, by_model or {}


def _merge_usage(target: dict, source: dict | None) -> None:
    if not isinstance(source, dict):
        return
    for model, usage in source.items():
        if model not in target:
            target[model] = {"input": 0, "output": 0, "total": 0, "calls": 0}
        target[model]["input"] += usage.get("input", 0)
        target[model]["output"] += usage.get("output", 0)
        target[model]["total"] += usage.get("total", 0)
        target[model]["calls"] += usage.get("calls", 0)


def _write_sectioned_synthesis_progress(
    blackboard: Blackboard,
    completed: int,
    total: int,
    section_name: str,
) -> None:
    if not blackboard.output_dir:
        return
    swarm_dir = Path(blackboard.output_dir) / "swarm"
    swarm_dir.mkdir(parents=True, exist_ok=True)
    progress = {
        "completed_sections": completed,
        "total_sections": total,
        "latest_section": section_name,
    }
    (swarm_dir / "sectioned_synthesis_progress.json").write_text(
        json.dumps(progress, indent=2),
        encoding="utf-8",
    )


def _numbered_must_include_pool(must_include: list[dict]) -> list[tuple[int, dict]]:
    pool = []
    for i, item in enumerate(must_include, 1):
        if isinstance(item, dict):
            pool.append((i, item))
        elif isinstance(item, str):
            pool.append((i, {"summary": item, "section": "General"}))
    return pool


def _apply_target_file_pins(
    plans: dict[str, dict],
    item_pool: list[tuple[int, dict]],
    filenames: list[str],
) -> None:
    valid_files = set(filenames)
    for number, item in item_pool:
        if not isinstance(item, dict):
            continue
        target_file = item.get("target_file")
        if not isinstance(target_file, str) or target_file not in valid_files:
            continue
        for filename in filenames:
            numbers = plans.setdefault(filename, {}).setdefault("numbers", [])
            if filename == target_file:
                if number not in numbers:
                    numbers.append(number)
            elif item.get("source") == "artifact_commitment" and number in numbers:
                plans[filename]["numbers"] = [n for n in numbers if n != number]


def _requirements_for_file(requirements: list[dict], filename: str) -> list[dict]:
    return [
        r for r in requirements
        if filename in [str(d) for d in r.get("deliverables", [])]
    ]


def _format_file_requirements(requirements: list[dict], max_count: int | None = None) -> str:
    if not requirements:
        return "No file-specific acceptance hints were provided."
    parts = []
    visible = requirements if max_count is None else requirements[:max_count]
    for r in visible:
        title = str(r.get("title", "")).strip()
        req_text = str(r.get("requirement_text", "")).strip()
        rid = str(r.get("id", "")).strip()
        parts.append(f"- {rid}: {title}\n  Requirement: {req_text}")
    if max_count is not None and len(requirements) > max_count:
        parts.append(f"- ... {len(requirements) - max_count} additional requirements omitted from prompt")
    return "\n".join(parts)


def _format_item_pool(
    item_pool: list[tuple[int, dict]],
    max_count: int | None = None,
    max_summary_chars: int = 500,
) -> str:
    if not item_pool:
        return "No mandatory item pool available."
    parts = []
    visible = item_pool if max_count is None else item_pool[:max_count]
    for number, item in visible:
        section = item.get("section", "General")
        summary = item.get("summary", "")
        entry_id = item.get("entry_id") or item.get("entry_ids", "")
        extra = _format_artifact_commitment_inline(item)
        if extra:
            extra = f" | {extra}"
        parts.append(f"{number}. [{section}] {summary[:max_summary_chars]} (ref: {entry_id}){extra}")
    if max_count is not None and len(item_pool) > max_count:
        parts.append(f"... {len(item_pool) - max_count} additional items omitted from prompt")
    return "\n".join(parts)


def _format_routing_requirements(requirements: list[dict]) -> str:
    if not requirements:
        return "No file-specific acceptance hints were provided."
    parts = []
    for r in requirements:
        title = str(r.get("title", "")).strip()
        req_text = str(r.get("requirement_text", "")).strip()
        rid = str(r.get("id", "")).strip()
        line = f"- {rid}: {title}"
        if req_text:
            line += f" | Requirement: {req_text[:180]}"
        parts.append(line)
    return "\n".join(parts)


def _plan_file_deliverable(
    blackboard: Blackboard,
    filename: str,
    file_reqs: list[dict],
    item_pool: list[tuple[int, dict]],
    caller: ModelCaller,
) -> tuple[list[int], dict, int]:
    prompt = f"""Plan the artifact contract for exactly one output file in a multi-deliverable task.

TASK:
{blackboard.task_instruction}

OUTPUT FILE TO PLAN:
{filename}

MANDATORY ITEM POOL FROM BLACKBOARD AND OBLIGATIONS:
{_format_item_pool(item_pool, max_summary_chars=220)}

Create a general artifact contract for this file. Use the task, filename, output type, and blackboard items as the source of truth. Do not assume an external rubric exists.

Choose only the mandatory item numbers that belong in this file. Do not select items just because they are important to another deliverable. Each output file must be distinct and purpose-built.

Return JSON:
{{
  "selected_item_numbers": [1, 2, 3],
  "purpose": "what this file must accomplish for the user",
  "structure": ["section, sheet, clause group, slide, or table name"],
  "format_notes": "brief format rules for this artifact",
  "closure_checks": ["general completion check for this artifact"]
}}"""
    payload, tokens = call_model(caller, prompt, max_tokens=4096)
    raw_numbers = payload.get("selected_item_numbers", [])
    numbers: list[int] = []
    if isinstance(raw_numbers, list):
        valid = {n for n, _ in item_pool}
        for raw in raw_numbers:
            try:
                n = int(raw)
            except (TypeError, ValueError):
                continue
            if n in valid and n not in numbers:
                numbers.append(n)

    contract = _normalize_artifact_contract(payload, filename)
    return numbers, contract, tokens


def _default_artifact_contract(filename: str) -> dict:
    return {
        "purpose": f"Produce the complete artifact requested for {filename}.",
        "structure": [],
        "format_notes": "",
        "closure_checks": [
            "The artifact is complete for its stated purpose.",
            "All selected state-backed items are represented in the correct format.",
        ],
    }


def _normalize_artifact_contract(
    payload: dict,
    filename: str,
) -> dict:
    contract = _default_artifact_contract(filename)

    purpose = payload.get("purpose")
    if isinstance(purpose, str) and purpose.strip():
        contract["purpose"] = purpose.strip()

    raw_structure = payload.get("structure", payload.get("outline", []))
    if isinstance(raw_structure, str):
        raw_structure = [raw_structure]
    if isinstance(raw_structure, list):
        contract["structure"] = [
            str(x).strip() for x in raw_structure if str(x).strip()
        ]

    notes = payload.get("format_notes")
    if isinstance(notes, str) and notes.strip():
        contract["format_notes"] = notes.strip()

    checks = payload.get("closure_checks", [])
    if isinstance(checks, str):
        checks = [checks]
    if isinstance(checks, list):
        normalized = [str(x).strip() for x in checks if str(x).strip()]
        if normalized:
            contract["closure_checks"] = normalized

    return contract


def _format_artifact_contract(contract: dict) -> str:
    purpose = str(contract.get("purpose", "")).strip() or "Complete the requested artifact."
    structure = contract.get("structure", [])
    notes = str(contract.get("format_notes", "")).strip()
    checks = contract.get("closure_checks", [])

    lines = [f"Purpose: {purpose}"]
    if isinstance(structure, list) and structure:
        lines.append("Planned structure:")
        lines.extend(f"- {str(item).strip()}" for item in structure if str(item).strip())
    else:
        lines.append("Planned structure: infer the best structure from the task and evidence.")
    if notes:
        lines.append(f"Format notes: {notes}")
    if isinstance(checks, list) and checks:
        lines.append("Closure checks:")
        lines.extend(f"- {str(item).strip()}" for item in checks if str(item).strip())
    return "\n".join(lines)


def _assign_unassigned_items(
    blackboard: Blackboard,
    filenames: list[str],
    file_requirements: list[dict],
    item_pool: list[tuple[int, dict]],
    unassigned_numbers: list[int],
    caller: ModelCaller,
) -> tuple[dict[str, list[int]], int]:
    assignments: dict[str, list[int]] = {filename: [] for filename in filenames}
    total_tokens = 0
    items = [
        (n, item) for n, item in item_pool if n in set(unassigned_numbers)
    ]
    for batch in _chunks(items, ASSIGNMENT_BATCH_SIZE):
        batch_assignments, tokens = _assign_item_batch_to_files(
            blackboard,
            filenames,
            file_requirements,
            batch,
            caller,
            "Assign previously unassigned mandatory items to the best output file(s).",
        )
        total_tokens += tokens
        _merge_assignments(assignments, batch_assignments)
    return assignments, total_tokens


def _assign_item_batch_to_files(
    blackboard: Blackboard,
    filenames: list[str],
    file_requirements: list[dict],
    items: list[tuple[int, dict]],
    caller: ModelCaller,
    instruction: str,
    plans: dict[str, dict] | None = None,
) -> tuple[dict[str, list[int]], int]:
    batch_numbers = {n for n, _ in items}
    file_parts = []
    for filename in filenames:
        current = ""
        if plans is not None:
            current_numbers = [
                n for n in plans.get(filename, {}).get("numbers", [])
                if n in batch_numbers
            ]
            current = (
                f"Current selected item numbers in this batch: "
                f"{', '.join(str(n) for n in current_numbers) or 'none'}\n"
            )
        file_parts.append(
            f"## {filename}\n"
            f"{current}"
            f"{_format_routing_requirements(_requirements_for_file(file_requirements, filename))}"
        )
    file_text = "\n\n".join(file_parts)

    prompt = f"""{instruction}

TASK:
{blackboard.task_instruction}

OUTPUT FILES AND THEIR ACCEPTANCE HINTS:
{file_text}

MANDATORY ITEMS TO ASSIGN:
{_format_item_pool(items, max_summary_chars=220)}

For each item, choose the filename or filenames where it belongs. Prefer one file unless the task or acceptance hints clearly require consistency across multiple deliverables. Do not assign spreadsheet/model rows to prose memos when a workbook is available. Do not assign memo-only reasoning to workbooks.

Return JSON:
{{
  "assignments": [
    {{"item_number": 1, "filenames": ["memo.docx"]}}
  ]
}}"""
    payload, tokens = call_model(caller, prompt, max_tokens=2048)
    assignments: dict[str, list[int]] = {filename: [] for filename in filenames}
    valid_files = set(filenames)
    valid_numbers = batch_numbers
    raw = payload.get("assignments", [])
    if isinstance(raw, list):
        for row in raw:
            if not isinstance(row, dict):
                continue
            try:
                number = int(row.get("item_number"))
            except (TypeError, ValueError):
                continue
            if number not in valid_numbers:
                continue
            targets = row.get("filenames", [])
            if isinstance(targets, str):
                targets = [targets]
            if not isinstance(targets, list):
                continue
            for filename in targets:
                if filename in valid_files and number not in assignments[filename]:
                    assignments[filename].append(number)
    return assignments, tokens


def _merge_assignments(target: dict[str, list[int]], source: dict[str, list[int]]) -> None:
    for filename, numbers in source.items():
        if filename not in target:
            continue
        for n in numbers:
            if n not in target[filename]:
                target[filename].append(n)


def _chunks(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _needs_assignment_rebalance(
    plans: dict[str, dict],
    item_pool: list[tuple[int, dict]],
    filenames: list[str],
) -> bool:
    if len(filenames) < 2 or not item_pool:
        return False

    valid_numbers = {n for n, _ in item_pool}
    file_sets = {
        filename: {
            n for n in plans.get(filename, {}).get("numbers", [])
            if n in valid_numbers
        }
        for filename in filenames
    }
    nonempty_sets = [numbers for numbers in file_sets.values() if numbers]
    if len(nonempty_sets) < 2:
        return False

    for i, left in enumerate(nonempty_sets):
        for right in nonempty_sets[i + 1:]:
            if left == right:
                return True

    assigned_to_count = {n: 0 for n in valid_numbers}
    for numbers in file_sets.values():
        for n in numbers:
            assigned_to_count[n] += 1
    multi_assigned = sum(1 for count in assigned_to_count.values() if count > 1)
    if multi_assigned / max(len(valid_numbers), 1) > 0.25:
        return True

    broad_files = [
        numbers for numbers in nonempty_sets
        if len(numbers) / max(len(valid_numbers), 1) > 0.7
    ]
    return len(broad_files) >= 2


def _needs_assignment_audit(
    plans: dict[str, dict],
    item_pool: list[tuple[int, dict]],
    filenames: list[str],
    file_requirements: list[dict],
    ran_assignment_repair: bool,
) -> bool:
    if ran_assignment_repair or len(filenames) < 2 or not item_pool:
        return False
    if not any(_requirements_for_file(file_requirements, filename) for filename in filenames):
        return False

    valid_numbers = {n for n, _ in item_pool}
    assigned_to_count = {n: 0 for n in valid_numbers}
    for filename in filenames:
        for n in plans.get(filename, {}).get("numbers", []):
            if n in assigned_to_count:
                assigned_to_count[n] += 1
    return all(count == 1 for count in assigned_to_count.values())


def _audit_file_assignments(
    blackboard: Blackboard,
    filenames: list[str],
    file_requirements: list[dict],
    item_pool: list[tuple[int, dict]],
    plans: dict[str, dict],
    caller: ModelCaller,
) -> tuple[dict[str, list[int]], int]:
    assignments: dict[str, list[int]] = {filename: [] for filename in filenames}
    total_tokens = 0
    for batch in _chunks(item_pool, ASSIGNMENT_BATCH_SIZE):
        current = _current_assignments_for_batch(filenames, plans, batch)
        revised, tokens = _assign_item_batch_to_files(
            blackboard,
            filenames,
            file_requirements,
            batch,
            caller,
            (
                "Audit current file assignments for this batch. Correct only "
                "items that belong in different output file(s); otherwise keep "
                "the current assignment."
            ),
            plans=plans,
        )
        total_tokens += tokens
        merged = _merge_assignment_revision(current, revised)
        _merge_assignments(assignments, merged)
    return assignments, total_tokens


def _current_assignments_for_batch(
    filenames: list[str],
    plans: dict[str, dict],
    batch: list[tuple[int, dict]],
) -> dict[str, list[int]]:
    batch_numbers = {n for n, _ in batch}
    return {
        filename: [
            n for n in plans.get(filename, {}).get("numbers", [])
            if n in batch_numbers
        ]
        for filename in filenames
    }


def _merge_assignment_revision(
    current: dict[str, list[int]],
    revised: dict[str, list[int]],
) -> dict[str, list[int]]:
    merged = {filename: list(numbers) for filename, numbers in current.items()}
    revised_numbers = {
        n for numbers in revised.values() for n in numbers
    }
    for number in revised_numbers:
        for filename in merged:
            if number in merged[filename]:
                merged[filename] = [n for n in merged[filename] if n != number]
    for filename, numbers in revised.items():
        if filename not in merged:
            continue
        for n in numbers:
            if n not in merged[filename]:
                merged[filename].append(n)
    return merged


def _rebalance_file_assignments(
    blackboard: Blackboard,
    filenames: list[str],
    file_requirements: list[dict],
    item_pool: list[tuple[int, dict]],
    plans: dict[str, dict],
    caller: ModelCaller,
) -> tuple[dict[str, list[int]], int]:
    assignments: dict[str, list[int]] = {filename: [] for filename in filenames}
    total_tokens = 0
    for batch in _chunks(item_pool, ASSIGNMENT_BATCH_SIZE):
        batch_assignments, tokens = _assign_item_batch_to_files(
            blackboard,
            filenames,
            file_requirements,
            batch,
            caller,
            (
                "Rebalance overbroad file assignments and duplicated file "
                "assignments for this batch of a multi-deliverable task."
            ),
            plans=plans,
        )
        total_tokens += tokens
        _merge_assignments(assignments, batch_assignments)
    return assignments, total_tokens


def _fallback_assignment_file(filenames: list[str]) -> str:
    for preferred in ("memo", "summary", "report", "deck", "analysis"):
        for filename in filenames:
            if preferred in filename.lower():
                return filename
    return filenames[0]


def _items_by_numbers(item_pool: list[tuple[int, dict]], numbers: list[int]) -> list[dict]:
    by_number = {n: item for n, item in item_pool}
    return [by_number[n] for n in numbers if n in by_number]


def _with_file_requirement_items(
    selected_items: list[dict], file_reqs: list[dict], filename: str,
) -> list[dict]:
    if not file_reqs:
        return selected_items

    items = list(selected_items)
    existing_criteria = {
        str(item.get("criterion_id", "")).strip()
        for item in items if isinstance(item, dict)
    }
    for req in file_reqs:
        rid = str(req.get("id", "")).strip()
        if rid and rid in existing_criteria:
            continue
        title = str(req.get("title", "")).strip()
        req_text = str(req.get("requirement_text", "")).strip()
        summary = f"{rid}: {title}" if rid else title
        if req_text:
            summary = f"{summary}. Requirement: {req_text}"
        items.append({
            "section": _criterion_component_section(req, filename),
            "summary": f"File-specific requirement for {filename}: {summary}",
            "entry_id": "",
            "criterion_id": rid,
            "importance": "critical",
            "source": "file_reqs",
        })
    return items


def _criterion_component_section(criterion: dict, filename: str) -> str:
    title = str(criterion.get("title", "")).strip()
    lower_title = title.lower()
    lower_file = filename.lower()

    import re
    issue = re.search(r"\bISSUE[_\s-]*(\d+[A-Za-z]?)\b", title, re.IGNORECASE)
    if issue:
        return f"ISSUE_{issue.group(1).upper()}"

    sections = (
        ("executive summary", "Executive Summary"),
        ("strategic rationale", "Strategic Rationale"),
        ("target overview", "Target Overview"),
        ("valuation analysis", "Valuation Analysis"),
        ("synergy analysis", "Synergy Analysis"),
        ("financial impact", "Financial Impact"),
        ("transaction structure", "Transaction Structure"),
        ("due diligence", "Due Diligence"),
        ("risk", "Risk Factors and Mitigants"),
        ("recommendation", "Recommendation"),
        ("signature", "Signatures and Certifications"),
        ("subsequent event", "Subsequent Events"),
    )
    for needle, section in sections:
        if needle in lower_title:
            return section

    if lower_title.startswith("deliverable"):
        return "Deliverable Presence"
    if lower_file.endswith(".xlsx") or "workbook" in lower_title or "tab" in lower_title:
        return "Workbook Components"
    if any(word in lower_file for word in ("redline", "markup", "rider")):
        return "Drafting and Markup Requirements"
    if "deck" in lower_file or "slide" in lower_title:
        return "Deck Required Content"
    if "policy" in lower_file or "policy" in lower_title:
        return "Policy Required Content"
    return "File Rubric Requirements"


def _format_selected_items(items: list[dict]) -> str:
    if not items:
        return "No selected mandatory items. Use the artifact contract, acceptance hints, and evidence."
    parts = []
    for i, item in enumerate(items, 1):
        section = item.get("section", "General")
        summary = _compact_selected_item_summary(
            item.get("summary", ""),
            max_chars=SELECTED_ITEM_SUMMARY_CHARS,
        )
        entry_id = item.get("entry_id") or item.get("entry_ids", "")
        block = [f"{i}. [{section}] {summary} (ref: {entry_id})"]
        native_details = _format_artifact_commitment_details(item)
        if native_details:
            block.append(native_details)
        parts.append("\n".join(block))
    return "\n".join(parts)


def _compact_selected_item_summary(summary: str, max_chars: int) -> str:
    text = str(summary or "").strip()
    if not text:
        return ""
    if text.lower().startswith("represent source-backed entry ") and ":" in text:
        text = text.split(":", 1)[1].strip()
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _format_artifact_commitment_inline(item: dict) -> str:
    if not isinstance(item, dict) or item.get("source") != "artifact_commitment":
        return ""
    pieces = []
    target = str(item.get("target_file", "") or "").strip()
    native = str(item.get("native_form", "") or "").strip()
    function = str(item.get("artifact_function", "") or "").strip()
    if target:
        pieces.append(f"target={target}")
    if native:
        pieces.append(f"native={native}")
    if function:
        pieces.append(f"function={function}")
    return ", ".join(pieces)


def _format_artifact_commitment_details(item: dict) -> str:
    if not isinstance(item, dict) or item.get("source") != "artifact_commitment":
        return ""
    lines = []
    inline = _format_artifact_commitment_inline(item)
    if inline:
        lines.append(f"   Artifact-native contract: {inline}")

    source_refs = item.get("required_source_refs") or item.get("source_refs") or []
    if isinstance(source_refs, list) and source_refs:
        labels = []
        for ref in source_refs[:3]:
            if not isinstance(ref, dict):
                continue
            label_parts = [
                str(ref.get("document", "") or "").strip(),
                str(ref.get("section", "") or "").strip(),
            ]
            evidence = str(ref.get("evidence", "") or "").strip()
            label = " / ".join(part for part in label_parts if part)
            if evidence:
                label = f"{label}: {evidence[:180]}" if label else evidence[:180]
            if label:
                labels.append(label)
        if labels:
            lines.append("   Required source refs: " + "; ".join(labels))

    conditions = item.get("satisfaction_conditions") or []
    if isinstance(conditions, list):
        clean = [str(cond).strip() for cond in conditions if str(cond).strip()]
        if clean:
            lines.append("   Satisfaction conditions:")
            lines.extend(f"   - {cond}" for cond in clean[:7])

    terms = item.get("verification_terms") or []
    if isinstance(terms, list):
        clean_terms = [str(term).strip() for term in terms if str(term).strip()]
        if clean_terms:
            lines.append("   Verification terms: " + "; ".join(clean_terms[:8]))
    return "\n".join(lines)


def _selected_evidence_text(
    selected_items: list[dict],
    active: list[Entry],
    max_chars: int,
    include_remaining: bool = True,
) -> str:
    evidence_ids: list[str] = []
    open_issue_ids: set[str] = set()
    for item in selected_items:
        if not isinstance(item, dict):
            continue
        ids = _entry_ids_from_item(item)
        if item.get("open_issue_only"):
            open_issue_ids.update(ids)
        else:
            for entry_id in ids:
                if entry_id not in evidence_ids:
                    evidence_ids.append(entry_id)

    pure_open_issue_ids = open_issue_ids - set(evidence_ids)

    by_id = {e.id: e for e in active if e.id}
    evidence_entries = [by_id[eid] for eid in evidence_ids if eid in by_id]

    parts: list[str] = []
    if evidence_entries:
        parts.append("=== SELECTED ITEM SUPPORTING ENTRIES ===")
        for e in evidence_entries:
            parts.append(render_entry(e, max_content=900))

    open_issue_entries = [by_id[eid] for eid in pure_open_issue_ids if eid in by_id]
    if open_issue_entries:
        parts.append("\n=== OPEN ISSUES (render as explicit gaps, not assertions) ===")
        for e in open_issue_entries:
            parts.append(render_entry(e, max_content=400))

    text = "\n".join(parts)
    if not include_remaining:
        return text[:max_chars]

    selected_set = set(evidence_ids) | pure_open_issue_ids
    remaining_by_doc: dict[str, list[str]] = {}
    for e in active:
        if e.id in selected_set:
            continue
        doc = e.source.document if e.source else "cross_cutting"
        remaining_by_doc.setdefault(doc or "cross_cutting", []).append(render_entry(e, max_content=450))

    remaining_parts = []
    for doc, items in remaining_by_doc.items():
        remaining_parts.append(f"=== {doc} ===\n" + "\n".join(items))

    remaining_text = "\n".join(remaining_parts)
    if text and len(text) < max_chars:
        return (text + "\n\n" + remaining_text)[:max_chars]
    if text:
        return text[:max_chars]
    return remaining_text[:max_chars]


def _file_format_guidance(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".xlsx"):
        return (
            "Produce workbook-ready text. Use '# Sheet: Sheet Name' before each "
            "sheet and Markdown pipe tables for rows and columns. Keep cells concise. "
            "Do not paste a narrative memo into the workbook."
        )
    if "redline" in lower or "markup" in lower or "rider" in lower:
        return (
            "Produce clause-by-clause drafting or markup content for this file. "
            "Include revised language, issue labels, and concise drafting notes. "
            "Do not include unrelated memo sections."
        )
    if lower.endswith(".pptx") or "deck" in lower:
        return (
            "Produce slide-ready content with numbered slides, titles, and bullets. "
            "Do not write a prose memo."
        )
    if "policy" in lower:
        return (
            "Produce the actual policy document text with operative sections, "
            "definitions, reporting channels, investigation procedures, "
            "anti-retaliation language, and jurisdiction-specific requirements. "
            "Do not write only an implementation memo or outline."
        )
    return (
        "Produce a professional document for this exact filename. Include the "
        "analysis, recommendations, and citations that belong in this file only."
    )


def _draft_file_deliverable(
    blackboard: Blackboard,
    filename: str,
    file_reqs: list[dict],
    selected_items: list[dict],
    artifact_contract: dict,
    caller: ModelCaller,
) -> tuple[str, int, dict]:
    if len(selected_items) > SECTION_THRESHOLD:
        return _sectioned_file_deliverable(
            blackboard, filename, file_reqs, selected_items,
            artifact_contract, caller,
        )

    active = [e for e in blackboard.entries if e.status == "active"]
    evidence = _selected_evidence_text(
        selected_items, active, max_chars=180000,
    )
    contract_text = _format_artifact_contract(artifact_contract)

    prompt = f"""Write exactly one deliverable file for a multi-output task.

TASK:
{blackboard.task_instruction}

OUTPUT FILE:
{filename}

FORMAT GUIDANCE:
{_file_format_guidance(filename)}

ARTIFACT CONTRACT:
{contract_text}

OPTIONAL FILE-SPECIFIC ACCEPTANCE HINTS:
{_format_file_requirements(file_reqs)}

SELECTED MANDATORY ITEMS FOR THIS FILE:
{_format_selected_items(selected_items)}

SUPPORTING BLACKBOARD EVIDENCE:
{evidence[:180000]}

CRITICAL INSTRUCTIONS:
1. Write only the content that belongs in {filename}; do not include the other deliverables.
2. Satisfy the artifact contract and use any file-specific acceptance hints as secondary checks.
3. Use source-grounded blackboard evidence; do not invent facts.
4. Make this file complete enough to stand alone for its purpose, but do not dump the entire global analysis.
5. For spreadsheets, create concise sheet/table content instead of prose paragraphs.
6. For redlines or markups, provide actual revised language and targeted drafting notes."""

    payload, tokens = call_model(caller, prompt, max_tokens=16384, json_mode=False)
    by_model, _, _, _ = get_last_call_usage()
    text = payload.get("text", "").strip()
    if not text:
        text = str(payload)
    return _clean_assembled_deliverable(text), tokens, (by_model or {})


def _sectioned_file_deliverable(
    blackboard: Blackboard,
    filename: str,
    file_reqs: list[dict],
    selected_items: list[dict],
    artifact_contract: dict,
    caller: ModelCaller,
) -> tuple[str, int, dict]:
    total_tokens = 0
    usage_by_model: dict = {}

    by_section: dict[str, list[dict]] = {}
    for item in selected_items:
        section = item.get("section", "General") if isinstance(item, dict) else "General"
        by_section.setdefault(section or "General", []).append(item)

    by_section = _cap_sections(_normalize_sections(by_section))

    active = [e for e in blackboard.entries if e.status == "active"]
    contract_text = _format_artifact_contract(artifact_contract)

    section_drafts = []
    total_sections = sum(
        len(_chunks(section_items, SECTION_CHUNK_SIZE))
        for section_items in by_section.values()
    )
    for section_name, section_items in by_section.items():
        item_chunks = _chunks(section_items, SECTION_CHUNK_SIZE)
        for chunk_index, items in enumerate(item_chunks, 1):
            chunk_name = section_name
            if len(item_chunks) > 1:
                chunk_name = f"{section_name} Part {chunk_index}"
            evidence = _selected_evidence_text(
                items, active, max_chars=SECTION_EVIDENCE_CHARS,
                include_remaining=any(
                    isinstance(item, dict) and item.get("source") == "file_reqs"
                    for item in items
                ),
            )
            prompt = f"""Write one section or sheet for a file-specific multi-deliverable output.

TASK:
{blackboard.task_instruction}

OUTPUT FILE:
{filename}

SECTION OR SHEET:
{chunk_name}

FORMAT GUIDANCE:
{_file_format_guidance(filename)}

ARTIFACT CONTRACT:
{contract_text}

OPTIONAL FILE-SPECIFIC ACCEPTANCE HINTS:
{_format_file_requirements(file_reqs)}

ITEMS THIS SECTION OR SHEET MUST INCLUDE:
{_format_selected_items(items)}

SUPPORTING BLACKBOARD EVIDENCE:
{evidence}

CRITICAL INSTRUCTIONS:
1. Write only this section or sheet for {filename}.
2. Include every selected item for this section with exact values, dates, party names, citations, and calculations.
3. ENTITY NAMES AND VALUES ARE SACRED: Copy entity names, company names, person names, dollar amounts, dates, percentages, and share counts VERBATIM from the items and evidence above. Do NOT substitute with similar-sounding names or round numbers from your own knowledge.
4. If you CANNOT find a specific entity name, person name, firm name, or address in the items and evidence above, write a bracketed placeholder like [Firm Name], [Counsel Name], [Address] instead. NEVER generate a plausible-sounding name from your own knowledge — a placeholder is always better than a hallucinated name.
5. Keep the content appropriate for {filename}; spreadsheets should use Markdown tables and '# Sheet: ...' headings.
6. One concise paragraph or bullet per item. Do not pad with background or restating the task.
7. If two items cover the same fact, merge them into one statement with the most specific version.
8. Omit generic observations not listed as required items and not supported by specific documents — but KEEP any required item even if it sounds generic."""

            _write_sectioned_synthesis_progress(
                blackboard, len(section_drafts), total_sections,
                f"starting {filename}: {chunk_name}",
            )
            payload, tokens = call_model(
                caller, prompt, max_tokens=SECTION_DRAFT_MAX_TOKENS, json_mode=False,
            )
            total_tokens += tokens
            by_model, _, _, _ = get_last_call_usage()
            _merge_usage(usage_by_model, by_model)
            text = payload.get("text", "").strip()
            if not text:
                continue
            heading = _section_heading_for_file(filename, chunk_name)
            text = _strip_redundant_section_heading(text, chunk_name)
            section_drafts.append(f"{heading}\n\n{text}")
            _write_sectioned_synthesis_progress(
                blackboard, len(section_drafts), total_sections,
                f"completed {filename}: {chunk_name}",
            )

    return (
        _clean_assembled_deliverable("\n\n".join(section_drafts)),
        total_tokens,
        usage_by_model,
    )


def _section_heading_for_file(filename: str, section_name: str) -> str:
    if filename.lower().endswith(".xlsx"):
        clean = section_name.strip() or "Analysis"
        if clean.lower().startswith("sheet:"):
            return f"# {clean}"
        return f"# Sheet: {clean}"
    return f"## {section_name.strip() or 'Section'}"


def _strip_redundant_section_heading(text: str, section_name: str) -> str:
    lines = str(text or "").splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return ""

    first = lines[0].strip()
    section_key = _heading_key(section_name)
    first_key = _heading_key(first)
    sheet_key = _heading_key(f"sheet: {section_name}")
    if first_key in {section_key, sheet_key}:
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
    return "\n".join(lines).strip()


def _clean_assembled_deliverable(text: str) -> str:
    if not CLEAN_ASSEMBLED_DELIVERABLE:
        return str(text or "").strip()
    lines = str(text or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()
    cleaned: list[str] = []
    previous_nonblank = ""
    previous_heading_key = ""

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            if cleaned and cleaned[-1].strip():
                cleaned.append("")
            continue

        current_key = _heading_key(stripped)
        if _line_is_heading_like(stripped) and current_key == _heading_key(previous_nonblank):
            continue
        if previous_heading_key and current_key == previous_heading_key:
            continue

        cleaned.append(line)
        previous_nonblank = stripped
        previous_heading_key = (
            current_key if _markdown_heading_level(stripped) is not None else ""
        )

    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    return "\n".join(cleaned).strip()


def _heading_key(text: str) -> str:
    value = str(text or "").strip()
    value = re.sub(r"^[#*\-\s\d.():]+", "", value)
    value = re.sub(r"[*_`]+", "", value)
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value


def _line_is_heading_like(line: str) -> bool:
    stripped = str(line or "").strip()
    if not stripped:
        return False
    if _markdown_heading_level(stripped) is not None:
        return True
    if len(stripped) > 90:
        return False
    if "|" in stripped:
        return False
    words = re.findall(r"[A-Za-z0-9]+", stripped)
    if not words:
        return False
    lowercase_words = sum(1 for word in words if word[:1].islower())
    return lowercase_words <= max(1, len(words) // 4)


def _markdown_heading_level(line: str) -> int | None:
    match = re.match(r"^\s*(#{1,6})\s+\S", str(line or ""))
    if not match:
        return None
    return len(match.group(1))


def _append_missing_items_for_file(
    filename: str,
    draft: str,
    missing: list[dict],
    active: list[Entry],
    blackboard: Blackboard,
    caller: ModelCaller,
) -> tuple[str, int]:
    missing_text = "\n".join(
        f"- {m.get('summary', '') if isinstance(m, dict) else str(m)}"
        f" (ref: {m.get('entry_id', '') if isinstance(m, dict) else ''})"
        for m in missing
    )

    missing_ids = {
        eid for m in missing if isinstance(m, dict) for eid in _entry_ids_from_item(m)
    }
    relevant_entries = [
        f"[{e.id}] ({e.type}) {e.content[:500]}"
        for e in active if e.id in missing_ids
    ]
    relevant_text = "\n".join(relevant_entries) if relevant_entries else "See missing items above."

    draft_excerpt = _draft_excerpt_for_verification(draft, max_chars=80_000)

    prompt = f"""The draft for one output file is missing required items. Read the current draft to understand what is already written, then add ONLY what is missing. Do not duplicate existing content.

TASK:
{blackboard.task_instruction}

OUTPUT FILE:
{filename}

FORMAT GUIDANCE:
{_file_format_guidance(filename)}

CURRENT DRAFT (already written):
{draft_excerpt}

MISSING ITEMS THAT MUST BE ADDED:
{missing_text}

SOURCE ENTRIES FOR MISSING ITEMS:
{relevant_text}

Write ONLY a supplemental addition for {filename} covering the missing items above. Do not repeat content already in the draft. Keep it appropriate for this file type. For spreadsheets, use '# Sheet: ...' and Markdown table rows. For redlines or markups, write revised clause language or drafting notes. Do not write a generic memo unless {filename} is itself a memo."""

    payload, tokens = call_model(caller, prompt, max_tokens=8192, json_mode=False)
    supplement = payload.get("text", "").strip()
    if not supplement:
        return draft, tokens
    stripped = supplement.lstrip()
    if filename.lower().endswith(".xlsx"):
        lower = stripped.lower()
        if not (
            lower.startswith("# sheet:")
            or lower.startswith("## sheet:")
            or lower.startswith("### sheet:")
        ):
            supplement = "# Sheet: Supplemental Required Items\n\n" + supplement
    elif not stripped.startswith("#"):
        supplement = "## Supplemental Required Items\n\n" + supplement
    return _clean_assembled_deliverable(f"{draft.rstrip()}\n\n{supplement}"), tokens


def _draft_synthesis(blackboard: Blackboard, must_include: list[dict],
                     active: list[Entry], caller: ModelCaller) -> tuple[str, int]:
    strategies = [e for e in active if e.type == "strategy"]

    evidence_entries, open_issue_items = filter_evidence_entries(must_include, active)

    items_text_parts = []
    for i, m in enumerate(must_include, 1):
        if isinstance(m, str):
            items_text_parts.append(f"{i}. [high] {m}")
            continue
        if not isinstance(m, dict):
            continue
        if m.get("open_issue_only"):
            continue
        imp = m.get("importance", "high")
        section = m.get("section", "General")
        summary = m.get("summary", "")
        entry_id = m.get("entry_id", "")
        items_text_parts.append(f"{i}. [{imp}] [{section}] {summary} (ref: {entry_id})")

    items_text = "\n".join(items_text_parts) or "No curated items available."
    n_items = len(items_text_parts)

    gaps_from_entries = [e.content for e in active if e.type == "gap"]
    open_issue_parts = [f"- {g}" for g in gaps_from_entries]
    for oi in open_issue_items:
        open_issue_parts.append(f"- [OPEN ISSUE] {oi.get('summary', '')}")
    gaps_text = "\n".join(open_issue_parts) if open_issue_parts else "None identified."

    strategy_text = strategies[-1].content if strategies else "Structure professionally."

    by_doc: dict[str, list[Entry]] = {}
    for e in evidence_entries:
        doc = e.source.document if e.source else "cross_cutting"
        by_doc.setdefault(doc or "cross_cutting", []).append(e)

    grouped_parts = []
    for doc_name, entries in by_doc.items():
        grouped_parts.append(f"\n=== FROM: {doc_name} ({len(entries)} findings) ===")
        for e in entries:
            grouped_parts.append(render_entry(e, max_content=600))

    all_entries_grouped = "\n".join(grouped_parts)

    prompt = f"""Produce the complete deliverable.

TASK: {blackboard.task_instruction}

STRATEGY: {strategy_text}

MANDATORY ITEMS ({n_items} items — EVERY one must appear):
{items_text}

COMPLETE FINDINGS BY DOCUMENT ({len(evidence_entries)} evidence entries — use ALL of these):
{all_entries_grouped[:200000]}

KNOWN GAPS AND OPEN ISSUES (acknowledge these honestly — do NOT assert them as facts):
{gaps_text}

CRITICAL INSTRUCTIONS:
1. The deliverable MUST include all {n_items} mandatory items AND any additional facts from the complete findings. Missing a single material fact is a failure.
2. Include EVERY specific number, date, dollar amount, percentage, party name, deadline, obligation, restriction, representation, and warranty.
3. Show calculations with full arithmetic steps.
4. Cite source documents when referencing facts.
5. Structure with clear headings and professional formatting appropriate to the task type.
6. Be EXHAUSTIVE — a thorough 10-page deliverable that covers everything beats a polished 3-page summary.
7. Every individual item from numbered lists must appear separately — do NOT summarize groups.
8. Include ALL financial terms with exact amounts and conditions.
9. Include ALL party names with full legal designations.
10. ENTITY NAMES AND VALUES ARE SACRED: Copy entity names, company names, person names, dollar amounts, dates, percentages, and share counts VERBATIM from the mandatory items and findings above. Do NOT substitute with similar-sounding names or round numbers from your own knowledge. If the findings say "$7.5M" write "$7.5M". If they say "University of Geneva" write exactly that. Your knowledge of the world is irrelevant — only the provided evidence matters.
11. If you CANNOT find a specific entity name, person name, firm name, or address in the mandatory items and findings above, write a bracketed placeholder like [Firm Name], [Counsel Name], [Address] instead. NEVER generate a plausible-sounding name from your own knowledge — a placeholder is always better than a hallucinated name.
12. Include ALL representations, warranties, covenants, conditions, and restrictions.
13. Items listed under KNOWN GAPS AND OPEN ISSUES are unresolved — flag them as open questions or caveats, never as established facts.
14. The deliverable must stand alone — a reader should get EVERY material fact without needing the source documents.
15. For drafting tasks: produce the actual document content (not a memo about the document). Include all required sections, clauses, and provisions.
16. Include STRATEGIC ASSESSMENTS: For each material risk, obligation, or finding, state its practical significance — likelihood of occurrence (high/medium/low), severity of impact, negotiation leverage implications, and recommended course of action. Do not merely list facts; INTERPRET them.
17. CROSS-DOCUMENT ANALYSIS: When multiple source documents address the same topic, EXPLICITLY flag conflicts, inconsistencies, gaps, or reinforcing provisions between them. State which documents agree, which diverge, and the practical consequence of any divergence.
18. FORMAL DOCUMENT ELEMENTS: Where the task calls for a professional deliverable (memo, letter, report, redline, draft agreement), include standard formal elements: title/header, date, addressee/recipient, subject line, executive summary, signature block or closing, confidentiality/privilege markings, and rights-reservation language as appropriate to the document type. Omit only elements that are clearly inapplicable."""

    payload, tokens = call_model(caller, prompt, max_tokens=32768, json_mode=False)
    deliverable = payload.get("text", "")
    if not deliverable:
        deliverable = str(payload)
    return deliverable, tokens


def _verify_completeness(draft: str, must_include: list[dict],
                         blackboard: Blackboard,
                         caller: ModelCaller) -> tuple[list[dict], int]:
    from .verification import verify_deterministic

    # Phase 1: Deterministic verification (zero cost, high accuracy)
    verified, unresolved = verify_deterministic(draft, must_include)

    if not unresolved:
        return [], 0

    # Phase 2: Model-based verification ONLY for unresolved items
    unresolved_items = []
    for i in unresolved:
        m = must_include[i]
        if isinstance(m, str):
            unresolved_items.append(f"{i+1}. {m}")
        elif isinstance(m, dict):
            unresolved_items.append(
                f"{i+1}. {m.get('summary', '')} (ref: {m.get('entry_id', '')})"
            )
    items_text = "\n".join(unresolved_items)
    draft_excerpt = _draft_excerpt_for_verification(draft)

    prompt = f"""You are a completeness verifier. These items could not be verified by exact-match.
Check if they are present in the draft — possibly in a different format, under a defined term, or paraphrased.

TASK: {blackboard.task_instruction}

DRAFT DELIVERABLE:
{draft_excerpt}

ITEMS TO CHECK ({len(unresolved)} items — only flag items that are TRULY absent, not just differently formatted):
{items_text}

Return JSON: {{"missing": [{{"item_number": 1, "summary": "the specific fact that is missing", "entry_id": "e1"}}], "present_count": 0, "missing_count": 0}}

IMPORTANT: These items were not found by exact-match. Some may be present under defined terms or in paraphrased form. Only flag items that are genuinely ABSENT — not just reformatted."""

    payload, tokens = call_model(caller, prompt, max_tokens=8192)
    missing = payload.get("missing", [])
    if not isinstance(missing, list):
        missing = []
    return missing, tokens


def _draft_excerpt_for_verification(draft: str, max_chars: int = 160000) -> str:
    if len(draft) <= max_chars:
        return draft

    part = max_chars // 3
    midpoint = len(draft) // 2
    middle_start = max(0, midpoint - part // 2)
    middle = draft[middle_start:middle_start + part]
    return (
        draft[:part]
        + "\n\n[... middle of long draft omitted for verifier context window ...]\n\n"
        + middle
        + "\n\n[... tail of long draft follows ...]\n\n"
        + draft[-part:]
    )


_MAX_PLACEHOLDER_REPAIRS = 3


def _repair_placeholders(
    outputs: dict[str, str],
    blackboard: Blackboard,
    caller: ModelCaller,
) -> tuple[dict[str, str], int]:
    """Scan outputs for bracket placeholders and replace them with evidence values."""
    total_tokens = 0
    repaired = dict(outputs)
    repairs_done = 0

    for filename, text in outputs.items():
        hits = detect_placeholders(text)
        if not hits or repairs_done >= _MAX_PLACEHOLDER_REPAIRS:
            continue

        active = [e for e in blackboard.entries if e.status == "active"]
        evidence_block = "\n".join(
            f"[{e.id}] ({e.type}) {e.content[:300]}"
            for e in active[:80]
        )
        placeholder_list = "\n".join(
            f"- {h['placeholder']} in: {h['context']}"
            for h in hits[:10]
        )

        prompt = (
            f"The following draft for '{filename}' contains bracket placeholders "
            f"that must be replaced with actual values from the evidence.\n\n"
            f"PLACEHOLDERS FOUND:\n{placeholder_list}\n\n"
            f"EVIDENCE:\n{evidence_block}\n\n"
            f"DRAFT (relevant sections):\n{text[:6000]}\n\n"
            f"For each placeholder, find the correct value from evidence and "
            f"return a JSON object with key \"replacements\" containing a list "
            f"of objects with \"placeholder\" (the exact bracket text), "
            f"\"value\" (the replacement from evidence), and \"confidence\" "
            f"(0.0-1.0). If no evidence supports a value, set confidence to 0 "
            f"and value to empty string. ONLY replace with values from evidence."
        )
        try:
            payload, tokens = call_model(caller, prompt, max_tokens=2048)
            total_tokens += tokens
        except Exception:
            continue

        replacements = payload.get("replacements", [])
        if not isinstance(replacements, list):
            continue

        patched = text
        for rep in replacements:
            if not isinstance(rep, dict):
                continue
            ph = rep.get("placeholder", "")
            val = str(rep.get("value", "")).strip()
            conf = float(rep.get("confidence", 0))
            if ph and val and conf >= 0.7:
                patched = patched.replace(ph, val, 1)
                repairs_done += 1

        if patched != text:
            repaired[filename] = patched

    return repaired, total_tokens


def _augment_draft(draft: str, missing: list[dict], active: list[Entry],
                   blackboard: Blackboard,
                   caller: ModelCaller) -> tuple[str, int]:
    missing_text = "\n".join(
        f"- {m.get('summary', '') if isinstance(m, dict) else str(m)}"
        f" (ref: {m.get('entry_id', '') if isinstance(m, dict) else ''})"
        for m in missing
    )

    relevant_entries = []
    missing_ids = {
        eid for m in missing if isinstance(m, dict) for eid in _entry_ids_from_item(m)
    }
    for e in active:
        if e.id in missing_ids:
            relevant_entries.append(f"[{e.id}] ({e.type}) {e.content[:500]}")

    relevant_text = "\n".join(relevant_entries) if relevant_entries else "See missing items above."

    prompt = f"""The draft deliverable is missing {len(missing)} mandatory items. Produce an IMPROVED version that includes everything from the original draft PLUS the missing items.

TASK: {blackboard.task_instruction}

ORIGINAL DRAFT:
{draft[:120000]}

MISSING ITEMS THAT MUST BE ADDED ({len(missing)} items):
{missing_text}

SOURCE ENTRIES FOR MISSING ITEMS:
{relevant_text}

INSTRUCTIONS:
1. Keep ALL content from the original draft — do not remove anything.
2. ADD the missing items in their appropriate sections.
3. If a missing item doesn't fit an existing section, create a new section for it.
4. Maintain professional formatting and document structure.
5. Include the EXACT specifics for each missing item (exact dollar amounts, dates, party names, etc.)."""

    payload, tokens = call_model(caller, prompt, max_tokens=32768, json_mode=False)
    augmented = payload.get("text", "")
    if not augmented:
        augmented = draft
    return augmented, tokens


def _append_missing_items(draft: str, missing: list[dict], active: list[Entry],
                          blackboard: Blackboard,
                          caller: ModelCaller) -> tuple[str, int]:
    """Append repairs to sectioned output without rewriting existing sections."""
    missing_text = "\n".join(
        f"- {m.get('summary', '') if isinstance(m, dict) else str(m)}"
        f" (ref: {m.get('entry_id', '') if isinstance(m, dict) else ''})"
        for m in missing
    )

    missing_ids = {
        eid for m in missing if isinstance(m, dict) for eid in _entry_ids_from_item(m)
    }
    relevant_entries = [
        f"[{e.id}] ({e.type}) {e.content[:500]}"
        for e in active if e.id in missing_ids
    ]
    relevant_text = "\n".join(relevant_entries) if relevant_entries else "See missing items above."

    prompt = f"""The sectioned deliverable is missing {len(missing)} mandatory items.

TASK: {blackboard.task_instruction}

MISSING ITEMS THAT MUST BE ADDED:
{missing_text}

SOURCE ENTRIES FOR MISSING ITEMS:
{relevant_text}

Write ONLY a supplemental section that addresses the missing items. Do not rewrite, summarize, or restate the existing deliverable. Include exact numbers, names, dates, citations, and clauses needed for these missing items."""

    payload, tokens = call_model(caller, prompt, max_tokens=16384, json_mode=False)
    supplement = payload.get("text", "").strip()
    if not supplement:
        return draft, tokens

    if not supplement.lstrip().startswith("#"):
        supplement = "## Supplemental Required Items\n\n" + supplement

    return f"{draft.rstrip()}\n\n{supplement}", tokens


def shadow_judge_audit(deliverable: str | dict[str, str], blackboard: Blackboard,
                       seed_plan: dict, caller: ModelCaller,
                       max_omissions: int = 15) -> tuple[str | dict[str, str], int]:
    """Post-synthesis shadow-judge: re-read the deliverable as a judge would,
    find source-supported omissions, and append-only repair them.

    Handles both monolithic (str) and file-scoped (dict) deliverables.
    """
    total_tokens = 0

    is_dict = isinstance(deliverable, dict)
    if is_dict:
        flat_text = "\n\n".join(
            f"--- {fname} ---\n{content}" for fname, content in deliverable.items()
        )
        primary_file = max(deliverable, key=lambda k: len(deliverable[k]))
    else:
        flat_text = deliverable

    active = [e for e in blackboard.entries if e.status == "active"]
    sourced = [
        e for e in active
        if e.source and e.source.document and e.confidence >= 0.7
    ]

    evidence_parts = []
    for e in sourced[:300]:
        parts = [f"[{e.id}] ({e.type})"]
        if e.source.document:
            parts.append(f" [{e.source.document}")
            if e.source.section:
                parts.append(f"/{e.source.section}")
            parts.append("]")
        parts.append(f" {e.content[:400]}")
        if e.source.evidence and len(e.source.evidence) > 10:
            parts.append(f" | Evidence: {e.source.evidence[:150]}")
        evidence_parts.append("".join(parts))
    evidence_text = "\n".join(evidence_parts)

    key_questions = seed_plan.get("key_questions", [])
    questions_text = "\n".join(f"- {q}" for q in key_questions) if key_questions else ""
    framework = seed_plan.get("analytical_framework", "")
    completeness = seed_plan.get("completeness_criteria", [])
    completeness_text = "\n".join(f"- {c}" for c in completeness) if completeness else ""

    prompt = f"""You are a strict legal analysis judge. Read this deliverable and identify OMISSIONS — things that SHOULD be in the deliverable based on the task, source documents, and evidence, but are ABSENT.

TASK: {blackboard.task_instruction}

ANALYTICAL FRAMEWORK: {framework}

KEY QUESTIONS THE DELIVERABLE SHOULD ANSWER:
{questions_text}

COMPLETENESS CRITERIA:
{completeness_text}

DELIVERABLE TO JUDGE:
{flat_text[:120000]}

SOURCE-GROUNDED EVIDENCE ({len(sourced)} entries from the documents):
{evidence_text[:100000]}

YOUR JOB: Find up to {max_omissions} specific, source-supported omissions. Each must be:
1. ABSENT from the deliverable (not just differently worded — genuinely missing)
2. SUPPORTED by the source evidence above (cite the entry ID)
3. Something a judge would score as a failure (material to the task, not trivial)

Focus on these common omission types:
- Exact dollar amounts, percentages, or calculations present in evidence but missing from deliverable
- Specific legal provisions, statutes, or regulatory citations that should be referenced
- Issue identifications or risk flags that the evidence supports but the deliverable doesn't state
- Cross-document conflicts or discrepancies not mentioned
- Specific recommendations or action items the task expects
- Party names, dates, deadlines, or conditions from evidence not in deliverable

DO NOT flag:
- Items that ARE present but worded differently
- Trivial formatting or structural issues
- Items not supported by the source evidence

Return JSON: {{"omissions": [{{"summary": "exact specific omission with numbers/names/citations", "entry_ids": "e1,e2", "omission_type": "exact_value|legal_citation|issue_flag|cross_doc|recommendation|calculation", "importance": "critical|high"}}]}}

Be precise. "{{'summary': 'The deliverable should mention the $2.5M termination fee in Section 4.3'}}" is good. "{{'summary': 'Missing financial details'}}" is bad."""

    payload, tokens = call_model(caller, prompt, max_tokens=8192)
    total_tokens += tokens

    omissions = payload.get("omissions", [])
    if not isinstance(omissions, list):
        return deliverable, total_tokens

    valid_omissions = [
        o for o in omissions
        if isinstance(o, dict) and len(o.get("summary", "")) >= 15
    ]

    if not valid_omissions:
        return deliverable, total_tokens

    missing_items = [
        {
            "summary": o["summary"],
            "entry_id": o.get("entry_ids", ""),
            "importance": o.get("importance", "high"),
            "section": "Shadow Judge Findings",
        }
        for o in valid_omissions[:max_omissions]
    ]

    if is_dict:
        repaired_text, repair_tokens = _append_missing_items(
            deliverable[primary_file], missing_items, active, blackboard, caller,
        )
        total_tokens += repair_tokens
        result = dict(deliverable)
        result[primary_file] = repaired_text
        return result, total_tokens

    repaired, repair_tokens = _append_missing_items(
        deliverable, missing_items, active, blackboard, caller,
    )
    total_tokens += repair_tokens

    return repaired, total_tokens


def _entry_ids_from_item(item: dict) -> list[str]:
    raw = item.get("entry_ids")
    if isinstance(raw, list):
        return [str(e).strip() for e in raw if str(e).strip()]

    raw = item.get("entry_id", "")
    if not raw:
        return []
    return [part.strip() for part in str(raw).split(",") if part.strip()]
