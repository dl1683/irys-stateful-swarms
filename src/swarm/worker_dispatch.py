from __future__ import annotations

import json
import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .blackboard import Blackboard
from .models import (
    Entry, EntrySource, EpistemicStatus, ModelCaller, WorkerOutput, WorkerRecord,
    gen_entry_id,
)
from .section_index import resolve_section_text


_usage_state = threading.local()


def parse_json_object(text: str) -> dict | None:
    """Extract the first JSON object from model output using raw_decode."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {"value": obj}
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == '{':
            try:
                obj, _ = decoder.raw_decode(text, i)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    return None


def call_model(caller: ModelCaller, prompt: str, *,
               max_tokens: int = 8192, json_mode: bool = True,
               audit_context: Any = None) -> tuple[dict, int]:
    """Returns (payload, total_tokens)."""
    if audit_context is not None:
        from .prompt_audit import audit_prompt
        audit_prompt(prompt, audit_context)
    result = caller.complete(prompt, max_tokens=max_tokens, json_mode=json_mode)
    tokens = result.tokens_total
    by_model = {
        result.model: {
            "input": result.tokens_input,
            "output": result.tokens_output,
            "total": result.tokens_total,
            "calls": 1,
        }
    }
    _usage_state.last_model = result.model
    _usage_state.last_in = result.tokens_input
    _usage_state.last_out = result.tokens_output
    _usage_state.last_by_model = by_model
    _record_aggregate_usage(result)
    text = result.text.strip()

    if not json_mode:
        return {"text": text}, tokens

    payload = parse_json_object(text)
    if payload is None:
        payload = {"findings": [], "parse_error": True}
    return payload, tokens


def begin_call_model_usage() -> None:
    _usage_state.usage_aggregate = {}


def end_call_model_usage() -> None:
    aggregate = getattr(_usage_state, "usage_aggregate", None)
    if aggregate is None:
        return
    _usage_state.last_by_model = aggregate
    if aggregate:
        first_model = next(iter(aggregate))
        _usage_state.last_model = first_model
        _usage_state.last_in = sum(v["input"] for v in aggregate.values())
        _usage_state.last_out = sum(v["output"] for v in aggregate.values())
    _usage_state.usage_aggregate = None


def get_last_call_usage() -> tuple[dict, str, int, int]:
    return (
        getattr(_usage_state, "last_by_model", None),
        getattr(_usage_state, "last_model", ""),
        getattr(_usage_state, "last_in", 0),
        getattr(_usage_state, "last_out", 0),
    )


def set_last_call_usage(by_model: dict | None) -> None:
    """Install aggregate usage for callers that coordinate parallel model calls."""
    if not by_model:
        _usage_state.last_by_model = {}
        _usage_state.last_model = ""
        _usage_state.last_in = 0
        _usage_state.last_out = 0
        return

    _usage_state.last_by_model = by_model
    _usage_state.last_model = next(iter(by_model))
    _usage_state.last_in = sum(v.get("input", 0) for v in by_model.values())
    _usage_state.last_out = sum(v.get("output", 0) for v in by_model.values())


def _record_aggregate_usage(result) -> None:
    aggregate = getattr(_usage_state, "usage_aggregate", None)
    if aggregate is None:
        return
    if result.model not in aggregate:
        aggregate[result.model] = {"input": 0, "output": 0, "total": 0, "calls": 0}
    aggregate[result.model]["input"] += result.tokens_input
    aggregate[result.model]["output"] += result.tokens_output
    aggregate[result.model]["total"] += result.tokens_total
    aggregate[result.model]["calls"] += 1


def compose_worker_prompt(task_description: str, context_entries: list[Entry],
                          document_sections: list[tuple[str, str]],
                          task_instruction: str,
                          assigned_signals: list[tuple[str, str, str]] | None = None,
                          web_search_results: str | None = None) -> str:
    parts = [task_description, f"\nOVERALL TASK: {task_instruction}"]

    if web_search_results:
        parts.append("\nWEB SEARCH RESULTS:")
        parts.append(web_search_results)

    if context_entries:
        parts.append("\nRELEVANT PRIOR FINDINGS:")
        for entry in context_entries:
            warning = ""
            if entry.epistemic and entry.epistemic.classification in (
                "adversarial_claim", "strategic"
            ):
                warning = f" [CAUTION: {entry.epistemic.classification}]"
            parts.append(
                f"  [{entry.id}] ({entry.type}, conf={entry.confidence:.1f}) "
                f"{entry.content[:300]}{warning}"
            )

    if assigned_signals:
        parts.append("\nASSIGNED OPEN SIGNALS:")
        for sig_id, priority, content in assigned_signals:
            parts.append(f"  [{sig_id}] ({priority}) {content[:500]}")
        parts.append(
            "When a finding addresses an assigned signal, include that exact "
            "signal ID in the finding's addresses_signals array."
        )

    if document_sections:
        parts.append("\nSOURCE MATERIAL:")
        for header, text in document_sections:
            parts.append(f"\n### {header}\n{text}")

    parts.append("""
OUTPUT: Return JSON with a "findings" array. Each finding:
{
  "type": "observation | analysis | calculation | strategy | gap",
  "content": "specific finding with exact numbers/citations",
  "source_document": "name or null", "source_section": "section or null",
  "evidence": "exact quote or calculation steps",
  "confidence": 0.0-1.0,
  "epistemic_classification": "fact | adversarial_claim | expert_opinion | inference | strategic",
  "epistemic_motivation": "whose interests does this serve?",
  "tags": ["topic"],
  "opens_questions": ["new questions — max 5"],
  "supports_entries": ["e3", "e7"],
  "contradicts_entries": ["entry IDs"],
  "supersedes_entries": ["entry IDs"],
  "addresses_signals": ["s2", "s5"]
}

RULES:
- Be SPECIFIC: "$2,541,500" not "a large amount". Include EXACT dollar amounts, dates, percentages.
- Show arithmetic: "35% × $7,261,428 = $2,541,500"
- ENUMERATE every individual item — list each one separately, do NOT summarize groups
- Every dollar amount, date, deadline, party name, defined term = separate finding
- Every numbered clause, schedule item, exhibit entry = separate finding
- Reference prior findings by ID when supporting or contradicting
- Flag adversarial sources
- Max 5 opens_questions per finding
- If you cannot determine something, return type "gap"
- Aim for HIGH DENSITY: 15-40 findings per worker call. Fewer than 10 means you are summarizing.
""")
    return "\n".join(parts)


def _assigned_signal_ids(task: dict, blackboard: Blackboard) -> list[str]:
    open_signal_ids = {
        s.id for s in blackboard.signals
        if s.status == "open" and s.priority in ("critical", "high")
    }
    assigned: list[str] = []
    for sig_id in task.get("addresses_signals", []):
        if not isinstance(sig_id, str):
            continue
        if sig_id in open_signal_ids and sig_id not in assigned:
            assigned.append(sig_id)
    return assigned


def _assigned_signal_details(
    signal_ids: list[str], blackboard: Blackboard,
) -> list[tuple[str, str, str]]:
    by_id = {s.id: s for s in blackboard.signals}
    details: list[tuple[str, str, str]] = []
    for sig_id in signal_ids:
        signal = by_id.get(sig_id)
        if signal:
            details.append((signal.id, signal.priority, signal.content))
    return details


def _attach_assigned_signal_ids(entries: list[Entry], signal_ids: list[str]) -> None:
    if not entries or not signal_ids:
        return
    eligible = [entry for entry in entries if passes_quality_gate(entry)]
    if not eligible:
        return
    already_addressed = {
        sig_id
        for entry in eligible
        for sig_id in entry.addresses_signals
    }
    missing_ids = [sig_id for sig_id in signal_ids if sig_id not in already_addressed]
    if not missing_ids:
        return
    target = max(
        eligible,
        key=lambda entry: (
            entry.type in ANALYTICAL_TYPES,
            entry.confidence,
            len(entry.content),
        ),
    )
    target.addresses_signals.extend(missing_ids)


def parse_worker_output(payload: dict, iteration: int,
                        worker_id: str, task_description: str,
                        valid_doc_names: set[str] | None = None) -> list[Entry]:
    findings = payload.get("findings", [])
    entries = []
    for f in findings:
        if not isinstance(f, dict):
            continue
        content = str(f.get("content", "")).strip()
        if not content or len(content) < 20:
            continue
        entry_type = f.get("type", "observation")
        if entry_type not in (
            "observation", "analysis", "calculation", "strategy", "contradiction", "gap"
        ):
            entry_type = "observation"

        source = None
        raw_doc = f.get("source_document")
        if raw_doc and valid_doc_names is not None:
            from .source_custody import source_document_is_valid
            if not source_document_is_valid(str(raw_doc), valid_doc_names):
                if entry_type == "observation":
                    continue
                raw_doc = None
                entry_type = "gap"
                content = f"[Source '{f.get('source_document')}' not in document registry — claim discarded] Needs verification: {content[:200]}"
        if raw_doc:
            source = EntrySource(
                raw_doc, f.get("source_section"),
                str(f.get("evidence", "")),
            )

        epistemic = EpistemicStatus(
            f.get("epistemic_classification", "inference"), "unknown",
            str(f.get("epistemic_motivation", "")),
        )

        try:
            conf = float(f.get("confidence", 0.5))
        except (ValueError, TypeError):
            conf = 0.5

        def _as_list(val):
            return val if isinstance(val, list) else []

        entries.append(Entry(
            id=gen_entry_id(), type=entry_type, content=content,
            source=source, epistemic=epistemic,
            created_by=WorkerRecord(worker_id, task_description, iteration),
            confidence=conf,
            tags=_as_list(f.get("tags", [])), status="active",
            opens_questions=_as_list(f.get("opens_questions", []))[:5],
            supports_entries=_as_list(f.get("supports_entries", [])),
            contradicts_entries=_as_list(f.get("contradicts_entries", [])),
            supersedes_entries=_as_list(f.get("supersedes_entries", [])),
            addresses_signals=_as_list(f.get("addresses_signals", [])),
        ))
    return entries


_NEGATIVE_PREFIXES = (
    "does not contain", "no mention of", "not found in",
    "the document does not", "there is no", "the provided text does not",
    "no information about", "the text does not", "no evidence of",
    "no reference to", "this document does not",
)


def passes_quality_gate(entry: Entry) -> bool:
    if not entry.content or len(entry.content.strip()) < 20:
        return False
    if entry.type == "observation" and (not entry.source or not entry.source.document):
        return False
    if entry.type == "calculation":
        has_nums = sum(1 for c in entry.content if c.isdigit()) >= 2
        has_op = any(op in entry.content for op in ("=", "+", "×", "*", "/", "%", "−"))
        if not (has_nums and has_op):
            return False
    if entry.type == "observation" and _is_negative_noise(entry):
        return False
    return True


def _is_negative_noise(entry: Entry) -> bool:
    low = entry.content.strip().lower()
    if not any(low.startswith(p) for p in _NEGATIVE_PREFIXES):
        return False
    if entry.addresses_signals:
        return False
    return True


ANALYTICAL_TYPES = {"analysis", "calculation", "strategy"}


def execute_workers_parallel(worker_tasks: list[dict], blackboard: Blackboard,
                             caller: ModelCaller, *,
                             analytical_caller: ModelCaller | None = None) -> list[WorkerOutput]:
    from .source_custody import _valid_document_names
    _valid_docs = _valid_document_names(blackboard)

    def run_one(task: dict) -> WorkerOutput:
        wid = f"w{blackboard.iteration}_{uuid.uuid4().hex[:4]}"
        assigned_ids = _assigned_signal_ids(task, blackboard)
        assigned_signals = _assigned_signal_details(assigned_ids, blackboard)
        context_entries = blackboard.get_entries_by_ids(
            task.get("reads_from_blackboard", [])
        )
        doc_sections: list[tuple[str, str]] = []
        sections_read: list[tuple[str, str]] = []
        for spec in task.get("reads_from_documents", []):
            if isinstance(spec, str):
                spec = {"document": spec, "sections": ["Full Document"]}
            if not isinstance(spec, dict):
                continue
            doc_name = spec.get("document", "")
            matched = None
            for ds in blackboard.documents:
                if ds.name == doc_name:
                    matched = ds
                    break
            if matched is None:
                doc_lower = doc_name.lower()
                for ds in blackboard.documents:
                    if doc_lower in ds.name.lower() or ds.name.lower() in doc_lower:
                        matched = ds
                        break
            if matched is not None:
                if not matched.is_loaded:
                    matched.materialize()
                for sec_name in spec.get("sections", ["Full Document"]):
                    text = resolve_section_text(
                        matched.text, matched.section_index, sec_name,
                    )
                    cat = matched.path_category
                    label = f"{matched.name} [{cat}] — {sec_name}" if cat else f"{matched.name} — {sec_name}"
                    doc_sections.append((label, text))
                    sections_read.append((matched.name, sec_name))

        web_results = None
        raw_queries = task.get("search_queries", [])
        if raw_queries:
            from .web_search import run_web_searches, web_search_enabled
            if web_search_enabled():
                if isinstance(raw_queries, str):
                    raw_queries = [raw_queries]
                web_results = run_web_searches(raw_queries)

        prompt = compose_worker_prompt(
            task["description"], context_entries,
            doc_sections, blackboard.task_instruction, assigned_signals,
            web_search_results=web_results,
        )
        # Route analytical workers to smarter model if available
        expected_type = task.get("expected_output_type", "observation")
        use_caller = caller
        if analytical_caller and expected_type in ANALYTICAL_TYPES:
            use_caller = analytical_caller
        result = use_caller.complete(prompt, max_tokens=8192, json_mode=True)
        t_in = result.tokens_input
        t_out = result.tokens_output
        tokens = result.tokens_total
        text = result.text.strip()
        payload = parse_json_object(text)
        if payload is None:
            payload = {"findings": [], "parse_error": True}
        entries = parse_worker_output(
            payload, blackboard.iteration, wid, task["description"],
            valid_doc_names=_valid_docs,
        )
        _attach_assigned_signal_ids(entries, assigned_ids)
        return WorkerOutput(entries, tokens, t_in, t_out, result.model, wid, task, sections_read)

    if not worker_tasks:
        return []
    max_w = min(len(worker_tasks), int(os.getenv("SWARM_MAX_WORKERS", "5")))
    with ThreadPoolExecutor(max_workers=max_w) as pool:
        futures = [pool.submit(run_one, t) for t in worker_tasks]
        results = []
        errors = []
        for f in futures:
            try:
                results.append(f.result())
            except Exception as exc:
                errors.append(exc)
        if errors and not results:
            raise RuntimeError(
                f"All {len(errors)} workers failed. "
                f"First error: {errors[0]}"
            )
        return results
